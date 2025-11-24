# Shared SDK Analysis: Sophia ↔ Hermes Integration

**Date:** 2025-11-24  
**Status:** Analysis Updated - Bidirectional Communication Confirmed  
**Recommendation:** **Shared SDK is OPTIONAL - bidirectional HTTP API pattern is sufficient**

## Executive Summary

**UPDATED:** After analyzing the Sophia codebase and receiving clarification from the project owner, the integration between Sophia and Hermes is **bidirectional**, not unidirectional as initially observed. Sophia calls Hermes for certain LLM services (e.g., text-to-Cypher query generation), in addition to Hermes sending proposals to Sophia.

**However, this bidirectional communication does NOT necessarily require a shared SDK.** The services can continue communicating via HTTP APIs with standard HTTP clients. A lightweight shared SDK focused on type definitions and client utilities is **optional** and should only be created if it provides clear value over using standard HTTP clients with shared type definitions.

## Architecture Understanding

### What is Sophia?
Sophia is the **non-linguistic cognitive core** for Project LOGOS:
- **Purpose:** Planning, execution, causal reasoning, world modeling
- **Data Store:** Direct read/write access to Neo4j (HCG) and Milvus (vector store)
- **Responsibilities:**
  - Backward chaining for goal decomposition
  - Action sequencing and planning
  - State management
  - Graph-based causal reasoning
  - JEPA-based simulation

### What is Hermes (Inferred)?
Based on the `/ingest/hermes_proposal` endpoint, Hermes appears to be:
- **Purpose:** Linguistic/LLM interface layer
- **Responsibilities:**
  - Natural language processing
  - LLM interaction (OpenAI, Anthropic, etc.)
  - Converting linguistic input to structured proposals
  - Generating natural language explanations
- **Integration:** Sends structured proposals TO Sophia for evaluation and storage

### Current Integration Pattern (UPDATED)

```
┌─────────────┐
│   Hermes    │  (Linguistic Layer - LLM/NLP)
│             │
│  • NL→Cypher│◄──── GET /generate_cypher (Sophia calls Hermes)
│  • Proposals│
└──────┬──────┘
       │ POST /ingest/hermes_proposal (Hermes calls Sophia)
       │
       ▼
┌─────────────┐
│   Sophia    │  (Cognitive Core - Planning)
│             │◄─────► Neo4j (HCG)
└─────────────┘        Milvus (Vector)
```

**Key Characteristics:**
- ✅ **Bidirectional:** Hermes sends proposals TO Sophia AND Sophia calls Hermes for LLM services
- ✅ **Decoupled:** Services communicate via HTTP API
- ✅ **Clear separation:** Linguistic (Hermes) vs. Cognitive (Sophia)
- ✅ **Use Cases:**
  - Hermes → Sophia: LLM proposals, plan suggestions, imagined states
  - Sophia → Hermes: Text-to-Cypher generation, natural language explanations, query assistance

## Evidence from Codebase

### 1. The `/ingest/hermes_proposal` Endpoint

**File:** `src/sophia/api/app.py:679`

```python
@app.post(
    "/ingest/hermes_proposal",
    response_model=HermesProposalResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["ingestion"],
)
async def ingest_hermes_proposal(
    request: HermesProposalRequest,
) -> HermesProposalResponse:
    """Ingest an LLM proposal from Hermes with provenance tracking.
    
    This endpoint accepts structured proposals from Hermes (including plans,
    imagined states, diagnostics, and tool calls) and persists them to Neo4j
    with full provenance metadata. SHACL validation is applied automatically.
    
    Note: Authentication is disabled for local development.
    """
```

**What it accepts:**
- `proposal_id`: Unique identifier
- `llm_provider`: e.g., "openai", "anthropic"
- `model`: e.g., "gpt-4"
- `confidence`: Float [0.0, 1.0]
- `plan_steps`: Optional structured action plans
- `imagined_states`: Optional future state predictions
- `diagnostics`: Optional reasoning/explanation
- `tool_calls`: Optional tool invocations

**What it does:**
1. Stores proposal as `hermes_proposal` node in Neo4j
2. Links child nodes (plan steps, states, tool calls)
3. Applies SHACL validation
4. Returns stored node IDs

### 2. Bidirectional Communication (UPDATED)

**Critical Finding:** Per project owner clarification, Sophia WILL call Hermes for LLM services:

**Sophia → Hermes Use Cases:**
- **Text-to-Cypher generation**: Convert natural language queries to Cypher
- **Query assistance**: Generate complex graph queries via LLM
- **Natural language explanations**: Convert graph data to human-readable text
- **Planning assistance**: Request LLM-generated plan suggestions

**Implementation Considerations:**
- Sophia will need an HTTP client to call Hermes APIs
- Type-safe request/response models for Sophia→Hermes calls
- Error handling and retry logic for LLM service calls
- Authentication/authorization for service-to-service calls

### 3. Architectural Separation

From `.github/copilot-instructions.md`:
> Sophia is the non-linguistic cognitive core responsible for planning, execution, and direct HCG (Neo4j + Milvus) updates.

From `README.md`:
> **Non-linguistic cognitive core for Project LOGOS**

**This reinforces:** Sophia operates on structured graph data, NOT natural language.

## Revised Analysis: Bidirectional Communication Pattern

### Communication Pattern (CORRECTED)

**Hermes → Sophia:**
- Ingest LLM proposals (`POST /ingest/hermes_proposal`)
- Send plan suggestions
- Send imagined states

**Sophia → Hermes:**
- Text-to-Cypher generation (e.g., `POST /generate_cypher`)
- Natural language explanations
- Query assistance
- Other LLM-powered services

### Three Implementation Options

#### Option 1: Standard HTTP Clients (Recommended for Now) ✅

**In Sophia:**
```python
import httpx

# Call Hermes for Cypher generation
async with httpx.AsyncClient() as client:
    response = await client.post(
        f"{HERMES_URL}/generate_cypher",
        json={
            "natural_language": "Find all blocks on the table",
            "context": {"entities": ["block", "table"]}
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    cypher = response.json()["cypher"]
```

**In Hermes:**
```python
import httpx

# Call Sophia to ingest proposal
async with httpx.AsyncClient() as client:
    response = await client.post(
        f"{SOPHIA_URL}/ingest/hermes_proposal",
        json={
            "proposal_id": "hermes_001",
            "llm_provider": "openai",
            "plan_steps": [...]
        }
    )
```

**Pros:**
- ✅ No additional dependencies
- ✅ Standard Python patterns
- ✅ Services remain independent
- ✅ Simple to implement

**Cons:**
- ⚠️ Type definitions duplicated (unless shared separately)
- ⚠️ URL construction manual
- ⚠️ Error handling implemented twice

#### Option 2: Shared Type Definitions Only (Good Middle Ground) ✅

**Create:** `logos-types` package (just Pydantic models)

```python
# logos-types/src/logos_types/hermes.py
class CypherGenerationRequest(BaseModel):
    natural_language: str
    context: dict

class CypherGenerationResponse(BaseModel):
    cypher: str
    confidence: float

# logos-types/src/logos_types/sophia.py
class HermesProposalRequest(BaseModel):
    proposal_id: str
    llm_provider: str
    # ...
```

**In Sophia:**
```python
import httpx
from logos_types.hermes import CypherGenerationRequest, CypherGenerationResponse

response = await client.post(
    f"{HERMES_URL}/generate_cypher",
    json=CypherGenerationRequest(
        natural_language="Find blocks",
        context={}
    ).model_dump()
)
result = CypherGenerationResponse(**response.json())
```

**Pros:**
- ✅ Type safety
- ✅ API contract clearly defined
- ✅ Minimal additional dependency
- ✅ Services still independent

**Cons:**
- ⚠️ Still manual HTTP calls
- ⚠️ URL construction still manual

#### Option 3: Full Shared SDK (Only If Needed) ⚠️

**Create:** `logos-sdk` package (clients + types)

```python
from logos_sdk import HermesClient
from logos_sdk.models import CypherGenerationRequest

hermes = HermesClient(base_url=HERMES_URL, token=TOKEN)
result = await hermes.generate_cypher(
    CypherGenerationRequest(
        natural_language="Find blocks",
        context={}
    )
)
```

**Pros:**
- ✅ Cleanest API
- ✅ Type safety
- ✅ Centralized retry/error handling

**Cons:**
- ⚠️ Additional dependency to maintain
- ⚠️ Couples services via shared code
- ⚠️ May be over-engineering

## Correct Pattern: Lightweight Shared SDK for HTTP Communication

### Recommendation: Start Simple, Add SDK Only If Needed

**Current State:** Bidirectional HTTP API communication

**Immediate Need:** Ensure both services can call each other

**Progression Path:**

1. **Phase 1: Standard HTTP clients** (Start here)
   - Use `httpx` or `requests` in both services
   - Duplicate API contracts temporarily
   - Get bidirectional communication working

2. **Phase 2: Shared types** (Add when duplication hurts)
   - Extract Pydantic models to `logos-types` package
   - Both services import types
   - Still use standard HTTP clients

3. **Phase 3: Full SDK** (Only if clearly beneficial)
   - Add HTTP client wrappers
   - Add retry/error handling
   - Evaluate if this actually helps

### Example: Sophia Calling Hermes (Without SDK)

```python
# In Sophia's planner or query module
import httpx
from typing import Optional

async def generate_cypher_via_llm(
    natural_language: str,
    context: dict,
    hermes_url: str,
    auth_token: str
) -> Optional[str]:
    """Call Hermes to generate Cypher from natural language."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{hermes_url}/generate_cypher",
                json={
                    "natural_language": natural_language,
                    "context": context
                },
                headers={"Authorization": f"Bearer {auth_token}"},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()["cypher"]
        except httpx.HTTPError as e:
            logger.error(f"Failed to call Hermes: {e}")
            return None
```

**This works fine without an SDK.**

## Updated Recommendations

### Current Recommendation: Don't Rush Into SDK

**The fact that Sophia calls Hermes does NOT automatically mean you need a shared SDK.**

### Decision Tree

```
┌─────────────────────────────────────────┐
│ Do services need to call each other?    │
│                                         │
│ YES: Hermes→Sophia, Sophia→Hermes       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ Is HTTP API sufficient?                 │
│                                         │
│ ✅ YES: Use httpx/requests              │
│    • Simple                             │
│    • Standard Python                    │
│    • No extra dependencies              │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ Is type duplication painful?            │
│                                         │
│ ✅ SOMEWHAT: Share Pydantic models      │
│    • Create logos-types package         │
│    • Keep HTTP calls manual             │
│    • Get type safety                    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ Is HTTP boilerplate significant?        │
│                                         │
│ ⚠️  MAYBE: Consider lightweight SDK     │
│    • Only if lots of endpoints          │
│    • Only if retry logic complex        │
│    • Only if clear maintenance win      │
└─────────────────────────────────────────┘
```

### What You Should Do

1. ✅ **Implement bidirectional HTTP communication**
   - Sophia can call Hermes endpoints
   - Hermes can call Sophia endpoints
   - Use standard HTTP clients

2. ✅ **Document the API contracts**
   - OpenAPI specs for both services
   - Share specs between teams
   - Version the APIs

3. ⚠️ **Consider shared types (optional)**
   - If type duplication becomes annoying
   - Create minimal `logos-types` package
   - Just Pydantic models, nothing else

4. ⚠️ **Consider SDK later (if at all)**
   - Only if HTTP boilerplate is significant
   - Start with types-only approach first
   - Evaluate actual pain points

5. ❌ **Don't create SDK prematurely**
   - Wait until you have real pain
   - Standard HTTP clients work fine
   - SDK adds maintenance burden

## Conclusion (REVISED)

**The effort to create a shared SDK is OPTIONAL, not required.**

### Key Insight

**Bidirectional communication ≠ Need for shared SDK**

- ✅ Sophia calls Hermes for LLM services (e.g., Cypher generation)
- ✅ Hermes calls Sophia for proposal ingestion
- ✅ Both use standard HTTP APIs
- ⚠️ Shared SDK is optional convenience, not architectural requirement

### What EXISTS and Works:
✅ HTTP API contracts (defined in OpenAPI)  
✅ Standard HTTP clients (httpx, requests)  
✅ Bidirectional communication capability  
✅ Service independence maintained  

### What COULD be added (if beneficial):
⚠️ **Shared type definitions** (`logos-types` package)
- Pydantic models for API contracts
- Reduces duplication
- Improves type safety
- Minimal overhead

⚠️ **Lightweight SDK** (`logos-sdk` package)
- Only if HTTP boilerplate becomes significant
- Only if you have many endpoints
- Only if standard clients feel repetitive

### What SHOULD NOT exist:
❌ Shared business logic (planning, prompts, etc.)  
❌ Merged domain-specific dependencies  
❌ Tight coupling beyond HTTP contract  
❌ Premature abstraction  

### The Real Pattern

**Sophia calling Hermes for text-to-Cypher:**

```python
# Without SDK (perfectly fine)
response = await httpx.post(
    f"{HERMES_URL}/generate_cypher",
    json={"natural_language": "Find red blocks"}
)

# With SDK (optional convenience)
from logos_sdk import HermesClient
response = await hermes.generate_cypher("Find red blocks")
```

Both approaches work. The SDK is just sugar. **Use it only if it actually helps.**

## Action Items (UPDATED)

### Required:
1. ✅ **Enable bidirectional HTTP communication**
   - Sophia can call Hermes endpoints (add this capability)
   - Hermes can already call Sophia (exists)
   - Use standard HTTP clients initially

2. ✅ **Document API contracts**
   - Maintain OpenAPI specs for both services
   - Share specs between teams
   - Version APIs properly

### Optional (Evaluate Later):
3. ⚠️ **Consider shared types package**
   - Wait until type duplication is painful
   - Keep it minimal (just Pydantic models)
   - Don't include business logic

4. ⚠️ **Consider SDK package**
   - Wait until HTTP boilerplate is repetitive
   - Start with types-only approach first
   - Evaluate if SDK actually reduces complexity

### Not Recommended:
5. ❌ **Don't create SDK prematurely**
   - Standard HTTP clients work fine
   - SDK adds maintenance overhead
   - Wait for real pain points

6. ❌ **Don't share business logic**
   - Keep planning in Sophia
   - Keep prompts in Hermes
   - Maintain clear boundaries

## References

- Current ingestion endpoint: `src/sophia/api/app.py:679`
- API models: `src/sophia/api/models.py:274-375`
- Tests: `tests/api/test_hermes_ingestion.py`
- Architecture notes: `.github/copilot-instructions.md`
- FastAPI docs: `http://localhost:8000/docs` (when running)

---

**Verdict:** The current one-way ingestion pattern is architecturally sound. No shared SDK is needed or advisable.
