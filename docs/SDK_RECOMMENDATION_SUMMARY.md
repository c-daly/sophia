# Shared SDK Recommendation Summary

**Date:** 2025-11-24  
**Decision:** ⚠️ **SHARED SDK IS OPTIONAL**  
**Status:** ✅ **Bidirectional HTTP API is sufficient**

## TL;DR

Sophia calls Hermes for certain LLM services (e.g., text-to-Cypher), and Hermes calls Sophia for proposal ingestion. This **bidirectional communication does NOT require a shared SDK** - standard HTTP clients work fine. A shared SDK is optional convenience, not an architectural requirement.

## Quick Facts

### Current Architecture (Correct) ✅
```
Hermes (LLM/NLP) ←→ HTTP APIs ←→ Sophia (Planning/Graphs)
                                        ↓
                                  Neo4j/Milvus
```

### What You Need ✅
```
Standard HTTP clients (httpx, requests) + API contracts
```

### What's Optional ⚠️
```
Shared SDK for convenience (only if it actually helps)
```

## Why SDK is Optional

1. **Bidirectional ≠ SDK required** - Services can call each other via HTTP
2. **Standard tools work** - httpx/requests are sufficient
3. **Type sharing is easy** - Use separate types package if needed
4. **SDK adds overhead** - More code to maintain
5. **Wait for pain** - Create SDK only when HTTP boilerplate hurts

## What Already Exists (Good)

✅ HTTP API endpoints (bidirectional)  
✅ Pydantic request/response models  
✅ Auto-generated OpenAPI docs at `/docs`  
✅ SHACL validation for graph integrity  
✅ Full provenance tracking  

## Implementation Options

### Option 1: Standard HTTP (Start here) ✅
```python
# In Sophia calling Hermes
response = await httpx.post(
    f"{HERMES_URL}/generate_cypher",
    json={"natural_language": "Find red blocks"}
)
```
**Best for:** Initial implementation, simple use cases

### Option 2: Shared Types (Add if helpful) ⚠️
```python
# logos-types package
from logos_types import CypherGenerationRequest

response = await httpx.post(
    url, 
    json=CypherGenerationRequest(...).model_dump()
)
```
**Best for:** When type duplication is annoying

### Option 3: Full SDK (Only if needed) ⚠️
```python
# logos-sdk package
from logos_sdk import HermesClient

hermes = HermesClient(url, token)
result = await hermes.generate_cypher("Find red blocks")
```
**Best for:** Many endpoints, complex retry logic

## If You Need...

### Bidirectional Communication ✅
**Solution:** Standard HTTP clients
```python
# Sophia → Hermes
await httpx.post(f"{HERMES_URL}/generate_cypher", json={...})

# Hermes → Sophia  
await httpx.post(f"{SOPHIA_URL}/ingest/hermes_proposal", json={...})
```

### Type Safety ⚠️
**Solution:** Shared types package (optional)
```python
# logos-types (just Pydantic models)
from logos_types import CypherGenerationRequest
```

### Convenience Wrappers ⚠️
**Solution:** SDK (only if HTTP boilerplate is significant)
```python
# logos-sdk (full client library)
from logos_sdk import HermesClient, SophiaClient
```

## Common Misunderstandings

### ✅ "Sophia calls Hermes for certain things"
**Correct.** Sophia needs Hermes for LLM services like text-to-Cypher generation. Use standard HTTP calls.

### ❌ "Bidirectional communication requires SDK"
**Wrong.** HTTP APIs work fine. SDK is optional convenience.

### ❌ "We need shared utilities for both services"
**Mostly wrong.** Each service has different concerns. Only share types if duplication is painful.

## Action Items

1. ✅ **Implement bidirectional HTTP** - Use standard clients
2. ✅ **Document API contracts** - OpenAPI specs for both
3. ⚠️ **Consider shared types** - Only if duplication hurts
4. ⚠️ **Consider SDK later** - Only if HTTP boilerplate is significant
5. ❌ **Don't create SDK prematurely** - Wait for real pain points

## References

- Full analysis: `docs/SHARED_SDK_ANALYSIS.md`
- Ingestion endpoint: `src/sophia/api/app.py:679`
- API models: `src/sophia/api/models.py:274-375`
- Tests: `tests/api/test_hermes_ingestion.py`

---

**Bottom Line:** Sophia calling Hermes is correct and expected. This does NOT mean you need a shared SDK. Standard HTTP clients are sufficient. Add SDK only if it provides clear value.
