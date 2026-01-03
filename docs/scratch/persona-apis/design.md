# Persona APIs Design Document

**Issue**: logos #246, #264
**Branch**: `feature/issue-246-persona-apis`
**Status**: Design Phase
**Date**: 2026-01-03

## Executive Summary

This document proposes moving persona management from Apollo (where it currently lives) to Sophia, where it properly belongs as part of CWM-E (emotional working memory). Sophia will own persona storage and expose APIs that Apollo consumes, following the established pattern of Sophia owning cognitive state while Apollo provides the UI layer.

---

## 1. Current State Analysis

### 1.1 Apollo's Current Persona Implementation

Apollo currently manages persona entries directly via Neo4j:

**Location**: `apollo/src/apollo/data/persona_store.py`

```python
class PersonaDiaryStore:
    """Persist and query persona diary entries in Neo4j."""

    def create_entry(self, entry: PersonaEntry) -> PersonaEntry
    def list_entries(...) -> List[PersonaEntry]
    def get_entry(entry_id: str) -> Optional[PersonaEntry]
    def recent_entries(limit: int) -> List[PersonaEntry]
    def latest_entry_timestamp() -> Optional[datetime]
```

**API Endpoints** (in `apollo/src/apollo/api/server.py`):
- `POST /api/persona/entries` - Create entry
- `GET /api/persona/entries` - List entries (with filters)
- `GET /api/persona/entries/{entry_id}` - Get specific entry

**Data Model** (`apollo/src/apollo/data/models.py`):
```python
class PersonaEntry(BaseModel):
    id: str
    timestamp: datetime
    entry_type: str          # belief, decision, observation, reflection
    content: str
    summary: Optional[str]
    sentiment: Optional[str] # positive, negative, neutral, mixed
    confidence: Optional[float]
    related_process_ids: List[str]
    related_goal_ids: List[str]
    emotion_tags: List[str]
    metadata: Dict[str, Any]
```

### 1.2 Sophia's CWM Architecture

Sophia has a well-established CWM pattern with three model types:

| Model | Purpose | Current Status |
|-------|---------|----------------|
| CWM-A | Abstract reasoning (entities, relations) | Implemented |
| CWM-G | Grounded (JEPA, sensor predictions) | Implemented |
| CWM-E | Emotional (persona, sentiment) | **Not yet implemented** |

**CWM State Service Pattern** (`sophia/src/sophia/cwm_a/state_service.py`):
- Emits `CWMState` envelopes with consistent structure
- Tracks entity/relationship changes
- Provides state history

**CWM Persistence** (`sophia/src/sophia/cwm/persistence.py`):
- Persists all CWM types to Neo4j via HCG
- Supports querying by type, timestamp, limit

### 1.3 SDK Type Expectations

The sophia-sdk already expects CWM-E data:

**`CWMESentimentData`** (from sophia-sdk):
```typescript
interface CWMESentimentData {
    sentiment?: string;      // e.g., "confident", "cautious"
    confidenceDelta?: number;
    cautionDelta?: number;
    narrative?: string;      // Short diary-style reflection
}
```

**`CWMStateLinks`** already includes:
```typescript
personaEntryId?: string;  // Link to persona entry
```

---

## 2. Proposed Architecture

### 2.1 Ownership Model

```
Apollo (UI Layer)                    Sophia (Cognitive Core)
+-------------------+                +------------------------+
|                   |                |                        |
| PersonaDiary UI   |  HTTP/REST     | CWM-E Service          |
| Chat Panel        | <------------> | - Persona Storage      |
| DiagnosticsPanel  |                | - Sentiment Analysis   |
|                   |                | - State Emissions      |
+-------------------+                +------------------------+
                                              |
                                              v
                                     +----------------+
                                     |  Neo4j HCG     |
                                     | (:CWMState)    |
                                     | type: cwm_e    |
                                     +----------------+
```

### 2.2 CWM-E State Service

Create `sophia/src/sophia/cwm_e/` module mirroring the CWM-A pattern:

```python
# sophia/src/sophia/cwm_e/__init__.py
from sophia.cwm_e.state_service import CWMEStateService
from sophia.cwm_e.models import PersonaEntry, SentimentData

__all__ = ["CWMEStateService", "PersonaEntry", "SentimentData"]
```

```python
# sophia/src/sophia/cwm_e/models.py
class PersonaEntry(BaseModel):
    """Persona diary entry - the primary CWM-E content type."""

    entry_id: str = Field(description="Unique identifier")
    entry_type: Literal["belief", "decision", "observation", "reflection"]
    content: str = Field(description="Main narrative content")
    summary: Optional[str] = None
    trigger: Optional[str] = None  # What caused this entry

    # Sentiment/Emotion
    sentiment: Optional[Literal["positive", "negative", "neutral", "mixed"]] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    emotion_tags: List[str] = Field(default_factory=list)

    # Links to HCG entities
    related_process_ids: List[str] = Field(default_factory=list)
    related_goal_ids: List[str] = Field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SentimentData(BaseModel):
    """Sentiment tracking for CWM-E state."""

    sentiment: Optional[str] = None      # Current sentiment label
    confidence_delta: float = 0.0        # Change in overall confidence
    caution_delta: float = 0.0           # Change in caution level
    narrative: Optional[str] = None      # Summary reflection

    # Aggregates
    recent_sentiment_trend: Optional[str] = None  # rising, falling, stable
    emotion_distribution: Dict[str, int] = Field(default_factory=dict)
```

```python
# sophia/src/sophia/cwm_e/state_service.py
class CWMEStateService:
    """Service for emitting CWM-E state records."""

    def __init__(self, source: str = "cwm_e_service") -> None:
        self._source = source
        self._state_history: List[CWMState] = []
        self._entries: Dict[str, PersonaEntry] = {}

    def emit_persona_entry(
        self,
        entry: PersonaEntry,
        confidence: float = 1.0,
        derivation: str = "observed",
        tags: Optional[List[str]] = None,
    ) -> CWMState:
        """Emit a CWM-E state for a persona entry."""

    def emit_sentiment_update(
        self,
        sentiment_data: SentimentData,
        triggering_entry_id: Optional[str] = None,
    ) -> CWMState:
        """Emit sentiment state change."""

    def get_entry(self, entry_id: str) -> Optional[PersonaEntry]
    def list_entries(...) -> List[PersonaEntry]
    def get_recent_entries(limit: int = 5) -> List[PersonaEntry]
    def get_sentiment_summary() -> SentimentData
    def get_state_history(limit: int = 100) -> List[CWMState]
```

---

## 3. API Design

### 3.1 Persona Entry Endpoints

Following Sophia's existing API patterns (auth required, JSON responses):

#### Create Persona Entry
```
POST /persona/entries
Authorization: Bearer <token>

Request:
{
    "entry_type": "decision",
    "content": "Selected path A over B due to shorter estimated time",
    "summary": "Path selection",
    "trigger": "plan_execution",
    "sentiment": "positive",
    "confidence": 0.87,
    "related_process_ids": ["proc_plan_123"],
    "related_goal_ids": ["goal_nav_456"],
    "emotion_tags": ["decisive", "focused"],
    "metadata": {}
}

Response (201 Created):
{
    "entry_id": "persona_abc123",
    "cwm_state_id": "cwm_e_xyz789",
    "timestamp": "2026-01-03T12:00:00Z"
}
```

#### List Persona Entries
```
GET /persona/entries
    ?entry_type=decision
    &sentiment=positive
    &related_process_id=proc_123
    &related_goal_id=goal_456
    &after_timestamp=2026-01-01T00:00:00Z
    &limit=100
    &offset=0
Authorization: Bearer <token>

Response:
{
    "entries": [...],
    "total": 42,
    "filters_applied": {
        "entry_type": "decision",
        "sentiment": "positive"
    }
}
```

#### Get Single Entry
```
GET /persona/entries/{entry_id}
Authorization: Bearer <token>

Response:
{
    "entry_id": "persona_abc123",
    "entry_type": "decision",
    "content": "...",
    ...
}
```

#### Update Entry (Partial)
```
PATCH /persona/entries/{entry_id}
Authorization: Bearer <token>

Request:
{
    "sentiment": "neutral",
    "metadata": {"updated_reason": "user correction"}
}

Response:
{
    "entry_id": "persona_abc123",
    "cwm_state_id": "cwm_e_updated_123"
}
```

#### Delete Entry
```
DELETE /persona/entries/{entry_id}
Authorization: Bearer <token>

Response: 204 No Content
```

### 3.2 Sentiment Endpoints

#### Get Current Sentiment
```
GET /persona/sentiment
Authorization: Bearer <token>

Response:
{
    "sentiment": "confident",
    "confidence_delta": 0.15,
    "caution_delta": -0.05,
    "recent_sentiment_trend": "rising",
    "emotion_distribution": {
        "confident": 12,
        "analytical": 8,
        "cautious": 3
    },
    "last_updated": "2026-01-03T12:00:00Z"
}
```

#### Get Sentiment History
```
GET /persona/sentiment/history
    ?limit=20
Authorization: Bearer <token>

Response:
{
    "snapshots": [
        {
            "timestamp": "2026-01-03T12:00:00Z",
            "sentiment": "confident",
            "confidence_delta": 0.15
        },
        ...
    ]
}
```

### 3.3 CWM-E State Endpoint

Following the existing `/cwm` endpoint pattern:

```
GET /cwm?types=cwm_e
    &after_timestamp=2026-01-01T00:00:00Z
    &limit=20
Authorization: Bearer <token>

Response:
{
    "states": [
        {
            "state_id": "cwm_e_abc123",
            "model_type": "CWM_E",
            "timestamp": "2026-01-03T12:00:00Z",
            "source": "cwm_e_service",
            "confidence": 0.87,
            "status": "observed",
            "links": {
                "persona_entry_id": "persona_xyz",
                "process_ids": ["proc_123"]
            },
            "tags": ["entry_type:decision"],
            "data": {
                "sentiment": "positive",
                ...
            }
        }
    ],
    "total": 15,
    "model_type": "cwm_e"
}
```

---

## 4. Integration Points

### 4.1 Apollo Migration Path

1. **Phase 1**: Sophia implements persona APIs (this design)
2. **Phase 2**: Apollo adds `SophiaPersonaClient` that proxies to Sophia
3. **Phase 3**: Apollo deprecates direct Neo4j access, uses client only
4. **Phase 4**: Remove Apollo's `PersonaDiaryStore`

Apollo's client pattern (already exists for other Sophia APIs):
```python
# apollo/src/apollo/client/persona_client.py
class SophiaPersonaClient:
    """Client for Sophia's persona APIs."""

    def __init__(self, config: SophiaConfig):
        self.base_url = f"http://{config.host}:{config.port}"
        self.token = config.api_key

    def create_entry(self, entry: PersonaEntryCreate) -> PersonaEntryResponse:
        ...
    def list_entries(self, filters: PersonaFilters) -> PersonaListResponse:
        ...
    def get_sentiment(self) -> SentimentResponse:
        ...
```

### 4.2 Hermes Integration

Hermes can emit persona entries via Sophia when processing LLM responses:

```python
# In hermes response processing
persona_entry = PersonaEntryCreate(
    entry_type="observation",
    content=llm_response.content,
    summary=truncate(llm_response.content, 160),
    sentiment=analyze_sentiment(llm_response.content),
    metadata={
        "llm_provider": provider,
        "model": model,
        "session_id": session_id
    }
)
sophia_client.create_persona_entry(persona_entry)
```

### 4.3 Feedback Loop

The existing `FeedbackDispatcher` in Sophia can emit persona-related feedback:

```python
# When persona state changes significantly
if sentiment_delta.confidence_delta > 0.2:
    feedback_dispatcher.emit(FeedbackPayload(
        feedback_type="persona",
        outcome="sentiment_shift",
        reason=f"Confidence increased significantly: {sentiment_delta.confidence_delta:.1%}"
    ))
```

---

## 5. Data Model Mapping

### 5.1 Neo4j Node Structure

CWM-E persona entries stored as HCG nodes:

```cypher
(:CWMState {
    uuid: "cwm_e_abc123",
    name: "cwm_e_20260103_120000",
    type: "cwm_e",
    timestamp: datetime(),
    source: "cwm_e_service",
    confidence: 0.87,
    status: "observed",
    payload: '{"entry_type":"decision","content":"...", ...}',
    links: '{"persona_entry_id":"persona_xyz"}',
    tags: ["entry_type:decision", "sentiment:positive"]
})
```

### 5.2 Backward Compatibility

Apollo's existing `(:PersonaEntry)` nodes will coexist during migration:

- Sophia writes `(:CWMState {type: "cwm_e"})` nodes
- Apollo can query both during transition
- Eventually deprecate `(:PersonaEntry)` label

---

## 6. Request/Response Models

### 6.1 Pydantic Models

```python
# sophia/src/sophia/api/persona_models.py

class PersonaEntryCreate(BaseModel):
    """Request model for creating a persona entry."""
    entry_type: Literal["belief", "decision", "observation", "reflection"]
    content: str = Field(..., min_length=1)
    summary: Optional[str] = Field(None, max_length=200)
    trigger: Optional[str] = None
    sentiment: Optional[Literal["positive", "negative", "neutral", "mixed"]] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    related_process_ids: List[str] = Field(default_factory=list)
    related_goal_ids: List[str] = Field(default_factory=list)
    emotion_tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PersonaEntryResponse(BaseModel):
    """Response model for persona entry creation."""
    entry_id: str
    cwm_state_id: str
    timestamp: datetime


class PersonaEntryFull(PersonaEntryCreate):
    """Full persona entry with ID and timestamp."""
    entry_id: str
    timestamp: datetime


class PersonaListResponse(BaseModel):
    """Response model for listing persona entries."""
    entries: List[PersonaEntryFull]
    total: int
    filters_applied: Dict[str, Any]


class PersonaEntryUpdate(BaseModel):
    """Request model for updating a persona entry."""
    summary: Optional[str] = None
    sentiment: Optional[Literal["positive", "negative", "neutral", "mixed"]] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    emotion_tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class SentimentResponse(BaseModel):
    """Current sentiment state."""
    sentiment: Optional[str]
    confidence_delta: float
    caution_delta: float
    recent_sentiment_trend: Optional[str]
    emotion_distribution: Dict[str, int]
    last_updated: datetime
```

---

## 7. Implementation Plan

### Phase 1: Core CWM-E Module (This Issue)
1. Create `sophia/src/sophia/cwm_e/` module
2. Implement `PersonaEntry` and `SentimentData` models
3. Implement `CWMEStateService`
4. Add Neo4j persistence via existing `CWMPersistence`

### Phase 2: API Endpoints
1. Add persona endpoints to `sophia/src/sophia/api/app.py`
2. Add request/response models
3. Wire up authentication
4. Add tests

### Phase 3: Apollo Migration
1. Create `SophiaPersonaClient` in Apollo
2. Update Apollo API to proxy to Sophia
3. Maintain backward compatibility
4. Deprecate direct Neo4j access

### Phase 4: Cleanup
1. Remove Apollo's `PersonaDiaryStore`
2. Migrate any remaining `(:PersonaEntry)` nodes
3. Update documentation

---

## 8. Open Questions

1. **Sentiment Aggregation**: Should Sophia compute sentiment trends automatically, or should this be triggered by Apollo/Hermes?

2. **Entry Immutability**: Should persona entries be fully mutable, or should updates create new versions with links?

3. **Retention Policy**: Should old persona entries be archived/deleted after a certain period?

4. **Access Control**: Should persona entries have per-entry access control, or is service-level auth sufficient?

5. **Real-time Updates**: Should Sophia expose a WebSocket for persona entry streaming, or should Apollo poll?

---

## 9. References

- Apollo Persona Implementation: `apollo/src/apollo/data/persona_store.py`
- CWM-A State Service: `sophia/src/sophia/cwm_a/state_service.py`
- CWM Persistence: `sophia/src/sophia/cwm/persistence.py`
- Sophia API Patterns: `sophia/src/sophia/api/app.py`
- SDK Types: `apollo/webapp/vendor/@logos/sophia-sdk/src/models/CWMESentimentData.ts`
- Persona Diary Docs: `apollo/docs/PERSONA_DIARY.md`
