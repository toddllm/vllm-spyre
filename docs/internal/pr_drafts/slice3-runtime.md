# Slice 3: Runtime Hardening + Async-Ready + 2-Process Validation

Branch:
- `codex/spyre-kv-slice3-runtime`

Suggested PR base:
- `codex/spyre-kv-slice2-core` (stacked)

Compare:
- <https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-slice2-core...codex/spyre-kv-slice3-runtime>

## Scope

Hardens connector behavior with integration tests, async-ready plumbing, store limits, and phase6-level validation.

In scope:
- integration/e2e-style test coverage growth
- async-ready layer loading path and lifecycle safety
- 2-process transport test path (file-backed export/import)
- stricter metadata validation
- byte-capped store and dedicated connector async env var

Out of scope:
- Prometheus adapter wiring
- deployment/ops integration

## Included commits

- `e8ee50a` Add integration tests, LRU registry cap, and load-failure end-to-end wiring
- `eee3190` Add engine-level E2E tests, async-ready load plumbing, and SpyreConnectorStats
- `429e282` Add async layer pipeline, 2-process disaggregated E2E, schema validation, and store transport
- `9b4475d` Add byte-capped KV store, dedicated async env var, and stricter validation

## Files changed

Modified:
- `vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py`
- `vllm_spyre/distributed/kv_transfer/kv_connector/v1/metadata.py`
- `vllm_spyre/envs.py`

New tests:
- `tests/v1/worker/test_kv_integration.py`
- `tests/v1/worker/test_kv_engine_level.py`
- `tests/v1/worker/test_kv_phase6.py`

## Validation checklist

Minimal:
```bash
pytest -q tests/v1/worker/test_kv_integration.py \
         tests/v1/worker/test_kv_engine_level.py \
         tests/v1/worker/test_kv_phase6.py
```

Recommended:
```bash
pytest -q tests/v1/worker
```

## Draft PR title

`[KVConnector][Slice3] Harden runtime semantics, async-ready load path, and phase6 validation`

## Draft PR body

```markdown
## Summary
This PR extends the Spyre KV connector with runtime hardening and deeper validation.

## What changed
- Added integration and engine-level connector tests.
- Added async-ready per-layer load plumbing and lifecycle safeguards.
- Added phase6 2-process test path via file-backed store export/import.
- Added stricter metadata validation and fault handling.
- Added byte-capped store controls and dedicated async worker env knob.

## Why
After core correctness, this slice stabilizes runtime semantics and failure handling for broader testing.

## Test plan
- `pytest -q tests/v1/worker/test_kv_integration.py`
- `pytest -q tests/v1/worker/test_kv_engine_level.py`
- `pytest -q tests/v1/worker/test_kv_phase6.py`

## Non-goals
- No Prometheus adapter here.
- No production transport backend (in-memory + file-backed test path only).
```
