# Shared SDK Recommendation Summary

**Date:** 2025-11-24  
**Decision:** ❌ **DO NOT CREATE A SHARED SDK**  
**Status:** ✅ **Current architecture is correct**

## TL;DR

The `/ingest/hermes_proposal` endpoint represents the **correct architectural pattern**: a one-way HTTP API from Hermes (linguistic layer) to Sophia (cognitive core). No shared SDK is needed or advisable.

## Quick Facts

### Current Architecture (Correct) ✅
```
Hermes (LLM/NLP) → HTTP POST → Sophia (Planning/Graphs) → Neo4j/Milvus
```

### What NOT to Do ❌
```
Hermes ← shared-sdk code → Sophia
```

## Why No SDK?

1. **No bidirectional communication** - Hermes sends TO Sophia only
2. **No shared business logic** - Different algorithmic concerns
3. **Clear layer separation** - Linguistic vs. Cognitive
4. **API contract exists** - Pydantic models + OpenAPI spec
5. **Different dependencies** - Would force unnecessary bloat

## What Already Exists (Good)

✅ HTTP API endpoint: `/ingest/hermes_proposal`  
✅ Pydantic request/response models  
✅ Auto-generated OpenAPI docs at `/docs`  
✅ SHACL validation for graph integrity  
✅ Full provenance tracking  

## If You Need...

### Client Code for Hermes
Use **OpenAPI code generation**, not a manual SDK:
```bash
openapi-generator generate \
  -i http://sophia:8000/openapi.json \
  -g python \
  -o ./generated-client/
```

### Shared Type Definitions
Share **JSON Schema or OpenAPI spec**, not Python code:
```
logos/api-contracts/sophia-openapi.yaml
```

### Better Developer Experience
✅ Use typed HTTP clients (httpx, requests)  
❌ Don't create shared library packages  

## Common Misunderstandings

### ❌ "Sophia should call LLMs through Hermes SDK"
**No.** Sophia is non-linguistic. LLM access belongs in Hermes.

### ❌ "We need shared utilities for both services"
**No.** Each service has different concerns. Duplication is better than coupling.

### ❌ "A shared SDK reduces code duplication"
**No.** The only "shared" code is the API contract (OpenAPI), which should be generated.

## Action Items

1. ✅ **Keep current architecture** - Do not change it
2. ✅ **Share OpenAPI spec** - Not Python packages
3. ✅ **Use code generation** - If clients need types
4. ❌ **Stop any SDK effort** - If one has been started

## References

- Full analysis: `docs/SHARED_SDK_ANALYSIS.md`
- Ingestion endpoint: `src/sophia/api/app.py:679`
- API models: `src/sophia/api/models.py:274-375`
- Tests: `tests/api/test_hermes_ingestion.py`

---

**Bottom Line:** The concern about a shared SDK was **valid to raise**, but the current design is **already correct**. Maintain the HTTP API boundary.
