# Feedback Emission Design

**Issue**: sophia #16
**Date**: 2026-01-01
**Status**: Draft → Ready for Review

## Overview

Sophia emits structured feedback to Hermes about observation proposals, plan creation, and execution outcomes. Feedback uses Redis for guaranteed delivery with persistent retry.

## Approach

**Selected**: Option B1 - Redis-backed queue with background worker

**Why**:
- Guaranteed delivery (survives Sophia restarts)
- Non-blocking (endpoints return immediately)
- Aligns with Phase 3 Redis plans for short-term memory
- Proper retry semantics with backoff

## Components

### 1. FeedbackPayload

**Location**: `src/sophia/feedback/models.py`
**Responsibility**: Define the structure of feedback messages

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

class StepResult(BaseModel):
    """Result of a single plan step execution."""
    step_index: int
    action: str
    outcome: Literal["success", "failure", "skipped"]
    error: str | None = None
    duration_ms: int | None = None

class StateDiff(BaseModel):
    """Changes to CWM state."""
    added_nodes: list[str] = Field(default_factory=list)
    removed_nodes: list[str] = Field(default_factory=list)
    modified_nodes: list[str] = Field(default_factory=list)

class FeedbackPayload(BaseModel):
    """Feedback sent from Sophia to Hermes."""

    # Correlation (at least one required)
    correlation_id: str | None = None
    plan_id: str | None = None
    execution_id: str | None = None

    # Outcome
    feedback_type: Literal["observation", "plan", "execution", "validation"]
    outcome: Literal["accepted", "rejected", "created", "success", "failure", "partial"]
    reason: str

    # Details (optional, type-dependent)
    state_diff: StateDiff | None = None
    step_results: list[StepResult] | None = None
    node_ids_created: list[str] | None = None

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_service: str = "sophia"

    def model_post_init(self, __context) -> None:
        """Validate at least one correlation key is present."""
        if not any([self.correlation_id, self.plan_id, self.execution_id]):
            raise ValueError("At least one of correlation_id, plan_id, or execution_id required")
```

### 2. FeedbackQueue

**Location**: `src/sophia/feedback/queue.py`
**Responsibility**: Persist feedback to Redis for reliable delivery

```python
import json
import redis
from datetime import datetime
from .models import FeedbackPayload

class FeedbackQueue:
    """Redis-backed queue for feedback messages."""

    QUEUE_KEY = "sophia:feedback:pending"
    FAILED_KEY = "sophia:feedback:failed"
    MAX_RETRIES = 5

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    def enqueue(self, payload: FeedbackPayload) -> str:
        """Add feedback to queue. Returns message ID."""
        message_id = f"fb-{datetime.utcnow().timestamp()}"
        message = {
            "id": message_id,
            "payload": payload.model_dump(mode="json"),
            "attempts": 0,
            "created_at": datetime.utcnow().isoformat(),
        }
        self.redis.lpush(self.QUEUE_KEY, json.dumps(message))
        return message_id

    def dequeue(self, timeout: int = 5) -> dict | None:
        """Block-pop next message. Returns None on timeout."""
        result = self.redis.brpop(self.QUEUE_KEY, timeout=timeout)
        if result:
            return json.loads(result[1])
        return None

    def requeue_with_backoff(self, message: dict) -> None:
        """Put message back with incremented attempt count."""
        message["attempts"] += 1
        message["next_attempt_after"] = (
            datetime.utcnow().timestamp() + (2 ** message["attempts"])
        )
        self.redis.lpush(self.QUEUE_KEY, json.dumps(message))

    def move_to_failed(self, message: dict, error: str) -> None:
        """Move message to failed queue after max retries."""
        message["failed_at"] = datetime.utcnow().isoformat()
        message["final_error"] = error
        self.redis.lpush(self.FAILED_KEY, json.dumps(message))

    def pending_count(self) -> int:
        """Number of messages waiting."""
        return self.redis.llen(self.QUEUE_KEY)

    def failed_count(self) -> int:
        """Number of failed messages."""
        return self.redis.llen(self.FAILED_KEY)
```

### 3. FeedbackWorker

**Location**: `src/sophia/feedback/worker.py`
**Responsibility**: Process queue and send to Hermes with retry

```python
import asyncio
import httpx
import logging
from datetime import datetime
from .queue import FeedbackQueue
from .models import FeedbackPayload

logger = logging.getLogger(__name__)

class FeedbackWorker:
    """Background worker that sends feedback to Hermes."""

    def __init__(
        self,
        queue: FeedbackQueue,
        hermes_url: str,
        timeout: float = 10.0,
    ):
        self.queue = queue
        self.hermes_url = hermes_url.rstrip("/")
        self.timeout = timeout
        self._running = False

    async def start(self) -> None:
        """Start processing loop."""
        self._running = True
        logger.info("Feedback worker started")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while self._running:
                await self._process_one(client)

    def stop(self) -> None:
        """Signal worker to stop."""
        self._running = False
        logger.info("Feedback worker stopping")

    async def _process_one(self, client: httpx.AsyncClient) -> None:
        """Process single message from queue."""
        message = self.queue.dequeue(timeout=1)
        if not message:
            return

        # Check backoff
        next_attempt = message.get("next_attempt_after", 0)
        if datetime.utcnow().timestamp() < next_attempt:
            self.queue.requeue_with_backoff(message)
            await asyncio.sleep(0.1)
            return

        # Attempt send
        try:
            response = await client.post(
                f"{self.hermes_url}/feedback",
                json=message["payload"],
            )

            if response.status_code == 201:
                logger.info(f"Feedback {message['id']} sent successfully")
                return

            error = f"HTTP {response.status_code}: {response.text[:100]}"
            logger.warning(f"Feedback {message['id']} rejected: {error}")

        except httpx.RequestError as e:
            error = str(e)
            logger.warning(f"Feedback {message['id']} failed: {error}")

        # Handle retry or fail
        if message["attempts"] >= FeedbackQueue.MAX_RETRIES:
            self.queue.move_to_failed(message, error)
            logger.error(f"Feedback {message['id']} moved to failed after {message['attempts']} attempts")
        else:
            self.queue.requeue_with_backoff(message)
            logger.info(f"Feedback {message['id']} requeued (attempt {message['attempts'] + 1})")
```

### 4. FeedbackDispatcher

**Location**: `src/sophia/feedback/dispatcher.py`
**Responsibility**: Simple interface for endpoints to emit feedback

```python
import logging
from .models import FeedbackPayload
from .queue import FeedbackQueue

logger = logging.getLogger(__name__)

class FeedbackDispatcher:
    """Interface for emitting feedback from API endpoints."""

    def __init__(self, queue: FeedbackQueue, enabled: bool = True):
        self.queue = queue
        self.enabled = enabled

    def emit(self, payload: FeedbackPayload) -> str | None:
        """Emit feedback. Returns message ID or None if disabled."""
        if not self.enabled:
            logger.debug(f"Feedback disabled, skipping: {payload.feedback_type}")
            return None

        message_id = self.queue.enqueue(payload)
        logger.info(
            f"Feedback queued: {message_id} "
            f"type={payload.feedback_type} outcome={payload.outcome}"
        )
        return message_id
```

### 5. FeedbackConfig

**Location**: `src/sophia/feedback/config.py`
**Responsibility**: Configuration for feedback system

```python
from pydantic import BaseModel, Field

class FeedbackConfig(BaseModel):
    """Configuration for feedback emission."""

    enabled: bool = Field(
        default=True,
        description="Enable/disable feedback emission"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    hermes_url: str = Field(
        default="http://localhost:18000",
        description="Hermes base URL"
    )
    worker_timeout: float = Field(
        default=10.0,
        description="HTTP timeout for Hermes requests"
    )
```

**Environment variables**:
```bash
SOPHIA_FEEDBACK_ENABLED=true
SOPHIA_FEEDBACK_REDIS_URL=redis://localhost:6379/0
SOPHIA_FEEDBACK_HERMES_URL=http://hermes:18000
```

## Behavior Specification

### B1: Emit Observation Feedback

**Trigger**: `/ingest/hermes_proposal` endpoint processes a proposal

**Preconditions**:
- Request contains valid `HermesProposalRequest`
- `correlation_id` is present in request (added field)

**Input**:
```python
request = HermesProposalRequest(
    proposal_id="prop-123",
    correlation_id="req-abc",  # NEW FIELD
    source_service="hermes",
    llm_provider="openai",
    model="gpt-4",
    ...
)
```

**Processing**:
1. Evaluate proposal (existing logic)
2. If accepted: create nodes with provenance
3. Build FeedbackPayload with outcome
4. Call `dispatcher.emit(payload)`
5. Return response (don't wait for Hermes)

**Output**:
```python
FeedbackPayload(
    correlation_id="req-abc",
    feedback_type="observation",
    outcome="accepted",  # or "rejected"
    reason="Added observation: red_mug_on_desk",
    node_ids_created=["node-456"],
    state_diff=StateDiff(added_nodes=["node-456"]),
)
```

**Example**:
```
Input: Proposal "red mug is on desk" with correlation_id="req-abc"
Processing: Validated, created node "obs-123"
Output: Feedback queued with outcome="accepted", node_ids_created=["obs-123"]
```

### B2: Emit Plan Creation Feedback

**Trigger**: `/plan` endpoint creates a plan

**Preconditions**:
- Request contains valid `PlanRequest`
- Optional `correlation_id` in request (added field)

**Input**:
```python
request = PlanRequest(
    correlation_id="req-abc",  # NEW FIELD (optional)
    goal={"description": "move red block to bin", ...}
)
```

**Processing**:
1. Generate plan (existing logic)
2. Build FeedbackPayload
3. Call `dispatcher.emit(payload)`
4. Return plan response

**Output**:
```python
FeedbackPayload(
    correlation_id="req-abc",
    plan_id="plan-789",
    feedback_type="plan",
    outcome="created",
    reason="Generated 4-step plan: move→grasp→move→release",
)
```

**Example**:
```
Input: Goal "move red block to bin"
Processing: Generated plan with 4 steps
Output: Feedback with outcome="created", plan_id="plan-789"
```

### B3: Emit Execution Feedback

**Trigger**: `/execute` endpoint completes execution

**Preconditions**:
- Request contains valid `ExecuteRequest` with `plan_id`

**Input**:
```python
request = ExecuteRequest(
    plan_id="plan-789",
    dry_run=False,
)
```

**Processing**:
1. Execute plan (existing logic)
2. Collect step results
3. Build FeedbackPayload with step_results
4. Call `dispatcher.emit(payload)`
5. Return execution response

**Output**:
```python
FeedbackPayload(
    plan_id="plan-789",
    execution_id="exec-001",
    feedback_type="execution",
    outcome="success",  # or "failure" or "partial"
    reason="All 4 steps completed successfully",
    step_results=[
        StepResult(step_index=0, action="move", outcome="success", duration_ms=150),
        StepResult(step_index=1, action="grasp", outcome="success", duration_ms=200),
        StepResult(step_index=2, action="move", outcome="success", duration_ms=180),
        StepResult(step_index=3, action="release", outcome="success", duration_ms=100),
    ],
)
```

**Example - Failure**:
```
Input: Execute plan-789
Processing: Step 2 (grasp) failed with gripper fault
Output: Feedback with outcome="failure", step_results showing step 2 error
```

## Edge Cases & Error Handling

### E1: Redis Unavailable at Startup

**Condition**: Redis connection fails when creating FeedbackQueue

**Behavior**:
1. Log error at WARNING level
2. Set `dispatcher.enabled = False`
3. Continue Sophia startup (feedback is non-critical)
4. All `dispatcher.emit()` calls return None

**Example**:
```
Startup with REDIS_URL=redis://bad-host:6379
→ Log: "WARNING: Redis unavailable, feedback disabled"
→ Sophia starts normally, feedback silently skipped
```

### E2: Redis Unavailable During Operation

**Condition**: Redis connection lost after startup

**Behavior**:
1. `queue.enqueue()` raises `redis.ConnectionError`
2. `dispatcher.emit()` catches, logs at ERROR level
3. Returns None (feedback lost)
4. Endpoint continues normally

**Example**:
```
Emit feedback while Redis is down
→ Log: "ERROR: Failed to queue feedback: Connection refused"
→ Endpoint returns success (feedback not critical path)
```

### E3: Hermes Unavailable

**Condition**: Hermes `/feedback` endpoint unreachable

**Behavior**:
1. Worker catches `httpx.RequestError`
2. Increments attempt count
3. Requeues with exponential backoff (2^attempts seconds)
4. After 5 attempts: moves to failed queue
5. Logs at ERROR level

**Backoff schedule**:
| Attempt | Delay before retry |
|---------|-------------------|
| 1 | 2 seconds |
| 2 | 4 seconds |
| 3 | 8 seconds |
| 4 | 16 seconds |
| 5 | Move to failed |

### E4: Hermes Rejects Payload

**Condition**: Hermes returns 4xx error

**Behavior**:
1. Log response body at WARNING
2. Treat as retryable (Hermes might be updated)
3. Same retry logic as E3

### E5: Malformed Feedback Payload

**Condition**: FeedbackPayload validation fails

**Behavior**:
1. Pydantic raises `ValidationError`
2. Caller (endpoint) catches and logs at ERROR
3. Feedback not queued
4. Endpoint continues normally

**Example**:
```python
# Missing all correlation keys
FeedbackPayload(feedback_type="plan", outcome="created", reason="...")
→ ValidationError: "At least one of correlation_id, plan_id, or execution_id required"
```

### E6: Queue Grows Too Large

**Condition**: Pending queue exceeds 10,000 messages

**Behavior**:
1. Worker logs at WARNING every 100 messages processed
2. No automatic purge (messages are valuable)
3. Operator alert via metrics (future: integrate with monitoring)

## Testing Strategy

### Unit Tests

**T1: FeedbackPayload Validation**
```python
def test_payload_requires_correlation_key():
    with pytest.raises(ValidationError):
        FeedbackPayload(
            feedback_type="plan",
            outcome="created",
            reason="test",
        )

def test_payload_accepts_plan_id_only():
    payload = FeedbackPayload(
        plan_id="plan-123",
        feedback_type="plan",
        outcome="created",
        reason="test",
    )
    assert payload.plan_id == "plan-123"
```

**T2: FeedbackQueue Operations**
```python
def test_enqueue_dequeue(redis_client):
    queue = FeedbackQueue(redis_url)
    payload = FeedbackPayload(plan_id="p1", feedback_type="plan", outcome="created", reason="x")

    msg_id = queue.enqueue(payload)
    assert msg_id.startswith("fb-")

    message = queue.dequeue(timeout=1)
    assert message["payload"]["plan_id"] == "p1"

def test_requeue_increments_attempts(redis_client):
    queue = FeedbackQueue(redis_url)
    message = {"id": "fb-1", "payload": {...}, "attempts": 2}

    queue.requeue_with_backoff(message)

    requeued = queue.dequeue(timeout=1)
    assert requeued["attempts"] == 3
```

**T3: FeedbackDispatcher**
```python
def test_emit_when_disabled():
    queue = Mock()
    dispatcher = FeedbackDispatcher(queue, enabled=False)

    result = dispatcher.emit(valid_payload)

    assert result is None
    queue.enqueue.assert_not_called()
```

### Integration Tests

**T4: Full Flow - Plan Creation**
```python
@pytest.mark.integration
async def test_plan_emits_feedback(http_client, redis_client, mock_hermes):
    # Create plan
    response = await http_client.post("/plan", json={
        "correlation_id": "test-corr",
        "goal": {"description": "test", "target_state": "done"}
    })
    assert response.status_code == 201
    plan_id = response.json()["plan_id"]

    # Wait for worker to process
    await asyncio.sleep(0.5)

    # Verify Hermes received feedback
    assert mock_hermes.received_count == 1
    feedback = mock_hermes.last_payload
    assert feedback["correlation_id"] == "test-corr"
    assert feedback["plan_id"] == plan_id
    assert feedback["feedback_type"] == "plan"
    assert feedback["outcome"] == "created"
```

**T5: Retry on Hermes Failure**
```python
@pytest.mark.integration
async def test_retry_on_hermes_failure(http_client, redis_client, mock_hermes):
    mock_hermes.fail_next(2)  # Fail first 2 attempts

    await http_client.post("/plan", json={...})

    # Wait for retries
    await asyncio.sleep(10)

    assert mock_hermes.attempt_count == 3  # 2 failures + 1 success
    assert mock_hermes.received_count == 1
```

### Test Fixtures

```python
@pytest.fixture
def redis_client():
    """Real Redis for integration tests."""
    client = redis.from_url("redis://localhost:46379/15")  # Test DB
    client.flushdb()
    yield client
    client.flushdb()

@pytest.fixture
def mock_hermes():
    """Mock Hermes /feedback endpoint."""
    class MockHermes:
        def __init__(self):
            self.received = []
            self.fail_count = 0

        def fail_next(self, n):
            self.fail_count = n

        @property
        def received_count(self):
            return len(self.received)

    mock = MockHermes()
    with respx.mock:
        route = respx.post("http://localhost:18000/feedback")
        def side_effect(request):
            if mock.fail_count > 0:
                mock.fail_count -= 1
                return httpx.Response(503)
            mock.received.append(request.json())
            return httpx.Response(201)
        route.side_effect = side_effect
        yield mock
```

## Files Affected

| File | Action | Description |
|------|--------|-------------|
| `src/sophia/feedback/__init__.py` | Create | Package init |
| `src/sophia/feedback/models.py` | Create | FeedbackPayload, StepResult, StateDiff |
| `src/sophia/feedback/queue.py` | Create | FeedbackQueue |
| `src/sophia/feedback/worker.py` | Create | FeedbackWorker |
| `src/sophia/feedback/dispatcher.py` | Create | FeedbackDispatcher |
| `src/sophia/feedback/config.py` | Create | FeedbackConfig |
| `src/sophia/api/models.py` | Modify | Add correlation_id to requests |
| `src/sophia/api/app.py` | Modify | Initialize dispatcher, emit feedback |
| `containers/docker-compose.test.yml` | Modify | Add Redis service |
| `tests/unit/feedback/` | Create | Unit tests |
| `tests/integration/test_feedback.py` | Create | Integration tests |
| `pyproject.toml` | Modify | Add redis dependency |

## Infrastructure

### Redis Service (docker-compose.test.yml)

```yaml
services:
  sophia-test-redis:
    image: redis:7-alpine
    ports:
      - "46379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 3
```

Port 46379 follows sophia's 4xxxx port range.

## Out of Scope

- Capability registry (#465)
- Edge weights on graph relationships
- Hermes-side storage and querying (#17)
- Provenance edge types (OBSERVED_BY, etc.) - separate from feedback
- Metrics/monitoring integration

## Success Criteria

1. ✓ FeedbackPayload validates correlation keys
2. ✓ Queue persists across Sophia restart
3. ✓ Failed sends retry with exponential backoff
4. ✓ After 5 failures, message moves to failed queue
5. ✓ Endpoints don't block waiting for Hermes
6. ✓ Redis unavailable doesn't crash Sophia
7. ✓ Integration test proves full flow works

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Queue technology | Redis (aligns with Phase 3) |
| Retry strategy | Exponential backoff, max 5 attempts |
| What if Hermes down? | Queue, retry, eventually fail gracefully |
| Blocking vs async | Async - endpoints return immediately |

## References

- sophia #16 - This issue
- hermes #17 - Receiving side
- sophia #15 - Provenance metadata (related)
- logos #465 - Capability catalog (out of scope)
- Phase 3 spec - Redis for short-term memory
