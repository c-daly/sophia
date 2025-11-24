# Shared SDK Analysis: Sophia ↔ Hermes Integration

**Date:** 2025-11-24  
**Status:** Analysis Complete  
**Recommendation:** **Do NOT create a shared SDK**

## Executive Summary

After analyzing the Sophia codebase and its integration with Hermes, **creating a shared SDK between Sophia and Hermes is NOT architecturally appropriate**. The current design already implements the correct pattern: a **one-way ingestion endpoint** from Hermes to Sophia, not bidirectional library sharing.

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

### Current Integration Pattern

```
┌─────────────┐
│   Hermes    │  (Linguistic Layer)
│  (LLM/NLP)  │
└──────┬──────┘
       │ POST /ingest/hermes_proposal
       │ (One-way data flow)
       ▼
┌─────────────┐
│   Sophia    │  (Cognitive Core)
│  (Planning) │◄─────► Neo4j (HCG)
└─────────────┘        Milvus (Vector)
```

**Key Characteristics:**
- ✅ **Unidirectional:** Hermes sends proposals TO Sophia
- ✅ **Decoupled:** Services communicate via HTTP API, not shared code
- ✅ **Clear separation:** Linguistic (Hermes) vs. Cognitive (Sophia)
- ✅ **No authentication required** for ingestion endpoint (local dev)
- ✅ **Full provenance tracking** for LLM-generated content

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

### 2. No Reverse Communication

**Critical Finding:** There is NO evidence of:
- Sophia calling Hermes APIs
- Sophia sending data back to Hermes
- Sophia needing Hermes code or libraries
- Bidirectional communication requirements

### 3. Architectural Separation

From `.github/copilot-instructions.md`:
> Sophia is the non-linguistic cognitive core responsible for planning, execution, and direct HCG (Neo4j + Milvus) updates.

From `README.md`:
> **Non-linguistic cognitive core for Project LOGOS**

**This reinforces:** Sophia operates on structured graph data, NOT natural language.

## Why a Shared SDK Would Be Wrong

### 1. **Violates Separation of Concerns**
- **Sophia:** Graph reasoning, planning, causal inference
- **Hermes:** Language understanding, LLM orchestration
- A shared SDK would blur these boundaries

### 2. **Creates Unnecessary Coupling**
Current pattern:
```
Hermes → HTTP → Sophia
(Decoupled, versioned API contract)
```

With shared SDK:
```
Hermes ← Shared Code → Sophia
(Tightly coupled, shared dependencies)
```

### 3. **No Shared Business Logic**
- Sophia: Backward chaining, SHACL validation, Neo4j queries
- Hermes: Prompt engineering, LLM calls, NLP parsing
- **Zero overlap** in algorithmic concerns

### 4. **Different Dependency Ecosystems**
- **Sophia needs:** Neo4j, Milvus, PyShacl, NetworkX
- **Hermes likely needs:** LangChain, OpenAI SDK, tokenizers
- Shared SDK forces both to carry unnecessary dependencies

### 5. **API Contract Already Exists**
The `HermesProposalRequest` Pydantic model IS the contract:
```python
class HermesProposalRequest(BaseModel):
    proposal_id: str
    llm_provider: str
    model: str
    confidence: float
    # ... etc
```

This model can be:
- Published as OpenAPI spec
- Shared via JSON Schema
- Generated as client code in any language
- Versioned independently

## Correct Pattern: API-First Communication

### What SHOULD Exist

#### 1. **API Specification** (OpenAPI/Swagger)
```yaml
# sophia-api.yaml
openapi: 3.0.0
paths:
  /ingest/hermes_proposal:
    post:
      summary: Ingest LLM proposal
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/HermesProposalRequest'
```

✅ **Already auto-generated** by FastAPI at `/docs`

#### 2. **Client Libraries** (Optional)
If needed, Hermes can use auto-generated clients:
```bash
# Generate Python client for Hermes
openapi-generator generate \
  -i http://sophia:8000/openapi.json \
  -g python \
  -o hermes-sophia-client/
```

But even this is **NOT a "shared SDK"** — it's a **generated API client**.

#### 3. **Shared Types** (If Absolutely Necessary)
IF there's a need for shared type definitions:
```
logos/
├── schemas/
│   ├── hermes_proposal.json    # JSON Schema
│   └── openapi.yaml            # API contract
```

NOT:
```
logos-shared-sdk/
├── sophia_client/
├── hermes_client/
└── shared_types/
```

## What the "Misguided Effort" Might Be

The concern about "hermes run through sophia for llm access" likely refers to:

### ❌ WRONG: Sophia calling LLMs through Hermes
```python
# In Sophia code (WRONG!)
from hermes_sdk import call_llm

response = await call_llm("Explain this plan")
```

**Why this is wrong:**
- Sophia is non-linguistic
- Sophia reasons over graphs, not text
- LLM calls belong in Hermes layer

### ✅ CORRECT: Hermes sends proposals to Sophia
```python
# In Hermes code (CORRECT!)
import httpx

proposal = {
    "proposal_id": "hermes_001",
    "llm_provider": "openai",
    "plan_steps": [...]
}

response = httpx.post(
    "http://sophia:8000/ingest/hermes_proposal",
    json=proposal
)
```

**Why this is correct:**
- Clear layer separation
- HTTP as boundary
- No code coupling

## Recommendations

### 1. ✅ Keep Current Architecture
- Maintain the `/ingest/hermes_proposal` endpoint
- Keep services decoupled via HTTP
- NO shared SDK between Sophia and Hermes

### 2. ✅ Document API Contract
```bash
cd sophia
poetry run uvicorn sophia.api.app:app --reload

# Visit http://localhost:8000/docs
# Download openapi.json
# Share with Hermes team
```

### 3. ✅ Optional: Publish API Schema
```
logos/
├── api-contracts/
│   ├── sophia-openapi.yaml
│   └── README.md
```

### 4. ✅ If Hermes Needs Client Code
Use code generation, NOT manual SDK:
```bash
# In Hermes repo
poetry add httpx  # or requests

# OR: Generate typed client
npx @openapitools/openapi-generator-cli generate \
  -i ../sophia/openapi.json \
  -g python \
  -o ./sophia_client
```

### 5. ❌ Do NOT Create
- `logos-shared-sdk` package
- `sophia-hermes-common` library
- Shared Python package with business logic

## Alternative: If "SDK" Means Something Else

If the concern is about **developer experience**, consider:

### A. Type-Safe API Clients (Generated)
```python
# hermes/sophia_client/  (auto-generated)
from sophia_client import SophiaAPI

api = SophiaAPI(base_url="http://sophia:8000")
response = await api.ingest_hermes_proposal(
    proposal_id="hermes_001",
    llm_provider="openai",
    model="gpt-4",
    confidence=0.85,
    plan_steps=[...]
)
```

✅ This is NOT a "shared SDK" — it's a generated client

### B. Shared Vocabulary (JSON-LD/Ontology)
If the shared concern is **semantic understanding**:

```
logos/
├── ontology/
│   ├── core_ontology.cypher
│   ├── action_types.ttl
│   └── state_schema.json
```

✅ Sharing ontology definitions is GOOD
✅ Sharing code libraries is BAD

## Conclusion

**The effort to create a shared SDK is indeed misguided.**

### What EXISTS and is CORRECT:
✅ HTTP API contract (`/ingest/hermes_proposal`)  
✅ Pydantic models defining request/response  
✅ OpenAPI specification (auto-generated)  
✅ Decoupled services  
✅ Clear architectural boundaries  

### What SHOULD NOT exist:
❌ Shared Python SDK package  
❌ Bidirectional code dependencies  
❌ Sophia calling Hermes (or vice versa via shared code)  
❌ Merged business logic  

### The Real Question:
**"Should Hermes route LLM access through Sophia?"**

**Answer: NO.**
- Sophia is non-linguistic
- LLM access belongs in Hermes
- Sophia consumes structured output FROM Hermes
- Not the other way around

## Action Items

1. ✅ **Document this finding** (this file)
2. ✅ **Share with team** — confirm architectural understanding
3. ✅ **If SDK effort has started** — STOP it
4. ✅ **If confusion exists** — clarify layer responsibilities:
   - **Hermes:** Language ↔ LLM ↔ Structure
   - **Sophia:** Structure ↔ Graph ↔ Planning
5. ✅ **If tooling needed** — use OpenAPI code generation, not manual SDK

## References

- Current ingestion endpoint: `src/sophia/api/app.py:679`
- API models: `src/sophia/api/models.py:274-375`
- Tests: `tests/api/test_hermes_ingestion.py`
- Architecture notes: `.github/copilot-instructions.md`
- FastAPI docs: `http://localhost:8000/docs` (when running)

---

**Verdict:** The current one-way ingestion pattern is architecturally sound. No shared SDK is needed or advisable.
