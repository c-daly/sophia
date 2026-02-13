# Handoff: Apollo Webapp Sophia Fixes

**Date:** 2026-02-01
**Status:** Investigation complete, fixes not yet started
**Source:** Log analysis of `/tmp/sophia.log`, `/tmp/apollo-api.log`, `/tmp/hermes.log`, `/tmp/apollo-webapp.log`

---

## Task 1 (High): Fix `neo4j.time.DateTime` serialization in `/hcg/snapshot`

**Symptom:** `GET /hcg/snapshot?limit=200` returns 500 Internal Server Error every time.

**Root cause:** Pydantic cannot serialize `neo4j.time.DateTime` objects returned from Neo4j queries.

```
PydanticSerializationError: Unable to serialize unknown type: <class 'neo4j.time.DateTime'>
```

**Traceback path:** `sophia/src/sophia/api/app.py:105` (dispatch) → FastAPI `serialize_response` → Pydantic `dump_python`

**Fix approach:**
- Find the HCG snapshot route handler and the Pydantic response model it uses
- Either add a custom Pydantic serializer for `neo4j.time.DateTime`, or convert `neo4j.time.DateTime` → Python `datetime` before returning from the route handler
- Check if other HCG endpoints have the same issue (e.g. `/hcg/plans` works fine, so it may only affect snapshot)

**Files to investigate:**
- `src/sophia/api/` — route handlers (look for hcg/snapshot endpoint)
- HCG client code that queries Neo4j and returns raw DateTime objects

---

## Task 2 (High): Fix `/persona/entries?limit=150` returning 422

**Symptom:** `GET /persona/entries?limit=150` returns 422 Unprocessable Content. `limit=5` works fine (200 OK).

**Root cause:** The Sophia endpoint's query parameter validation likely has a `le` (max) constraint on `limit` that is less than 150.

**Fix approach:**
- Find the persona entries route and its limit parameter validation
- Either raise the max limit in Sophia to accommodate Apollo's request, or lower Apollo's request to match
- Apollo webapp currently requests `limit=150` — check `apollo/src/` frontend code for where this is set

**Files to investigate:**
- `src/sophia/api/` — persona entries route handler
- Query parameter model/schema with limit constraint

---

## Task 3 (Medium): Redis unavailable — feedback disabled

**Log:** `Redis unavailable, feedback disabled: Error 111 connecting to localhost:6379. Connection refused.`

Sophia degrades gracefully but feedback feature is non-functional. Verify whether Redis is expected in the current dev stack.

---

## Task 4 (Medium): Milvus unavailable — Hermes persistence disabled

**Log:** `Failed to connect to Milvus: localhost:17530, illegal connection params or server unavailable`

Hermes degrades gracefully. This is a Hermes issue, not Sophia, but noted for completeness.

---

## Next Steps

1. Start with Task 1 — find the HCG snapshot endpoint and its response model
2. Add `neo4j.time.DateTime` → `datetime` conversion
3. Then Task 2 — find and adjust the persona entries limit validation
4. Test both fixes by restarting Sophia and hitting the endpoints
