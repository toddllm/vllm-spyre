# KV Connector Phase 6 Lab Note (2026-02-27)

> Internal working note for validation and iteration. Keep this in internal notes,
> not in upstream-facing roadmap docs.

## Scope
Review and verification of the Phase 6 KV connector work at `2fe2d95`, plus
targeted correctness fixes applied after review.

Reviewed files:
- `vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py`
- `vllm_spyre/distributed/kv_transfer/kv_connector/v1/metadata.py`
- `tests/v1/worker/test_kv_phase6.py`

## Findings and fixes applied

### 1) Identity load path used wrong source block IDs (fixed)

Problem:
- In load path without explicit `block_mapping`, connector used
  `enumerate(req_meta.block_ids)` and treated the enumerate index as source
  block ID.
- This fails for non-zero source blocks (for example source block `10` was
  looked up as `0`).

Fix:
- In `_load_layer`, changed implicit mapping fallback from index-based to
  identity block-ID mapping:
  - before: `i -> dest_block_id`
  - after: `block_id -> block_id`

Risk addressed:
- Silent data misses and invalid-block reporting for valid identity loads.

### 2) Incomplete scheduler-generated load mapping could silently under-load (fixed)

Problem:
- `update_state_after_alloc` built `block_mapping` by best-effort loops.
- If source registry had fewer blocks than implied by matched external tokens,
  mapping was truncated.
- Connector would load only mapped subset without marking remaining external
  blocks invalid.

Fix:
- Added strict coverage check in `update_state_after_alloc`:
  - expected mapped blocks = `num_external_tokens // block_size`
  - if built mapping count differs, fallback to `is_store=True` request
    (recompute-safe path).

Risk addressed:
- Incorrect partial reuse with no explicit error signal to scheduler.

## Regression tests added

### `tests/v1/worker/test_inmemory_spyre_connector.py`
- `test_identity_load_without_block_mapping_uses_block_ids`
  - Verifies no-mapping load works for non-zero block IDs.

### `tests/v1/worker/test_scheduler_driven_kv_reuse.py`
- `test_fallback_to_store_when_mapping_is_incomplete`
  - Verifies scheduler-side metadata generation falls back to store/recompute
    when external mapping cannot be fully constructed.

## Validation summary

### Local CPU
- `ruff check` on modified files: pass
- targeted regression tests: pass
- full worker suite: pass

### Remote CPU
- `ruff check` on modified files: pass
- targeted regression tests: pass
- full worker suite: pass

### Remote CUDA
- `ruff check` on modified files: pass
- targeted regression tests: pass
- full worker suite: pass

## Observed agent pitfalls

1. **Block ID semantics under implicit mappings**
- It assumed positional mapping (`enumerate`) where block IDs are absolute
  scheduler IDs.
- This passed existing tests because tests mostly used low/contiguous IDs.

2. **Coverage invariants between matched tokens and mapping cardinality**
- It did not enforce that externally matched blocks are fully representable in
  metadata.
- Missing this invariant allows silent partial loads.

3. **Edge-case test design**
- Existing tests were strong on happy path and explicit mapping, weak on:
  - non-zero identity mappings
  - inconsistent saved-registry state vs external token claims

## Guidance for next pass

1. Add explicit contract checks in metadata validation for load requests:
- layout whitelist
- non-negative block IDs and mapping IDs
- optional strict mode for mapping cardinality vs `token_count`

2. Keep connector async env vars separate from model-load process env vars:
- avoid overloading one environment variable with two unrelated meanings.

3. Expand fault-injection matrix:
- stale saved registry entries
- missing subset of source blocks
- malformed mapping tuples
- per-layer partial failures with async enabled

## Files modified in this review pass
- `vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py`
- `tests/v1/worker/test_inmemory_spyre_connector.py`
- `tests/v1/worker/test_scheduler_driven_kv_reuse.py`
