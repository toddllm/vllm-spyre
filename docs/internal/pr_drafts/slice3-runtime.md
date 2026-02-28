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

## What this slice is trying to prove

This slice is about runtime credibility, not new architecture.

It is meant to prove:

1. The connector logic introduced in Slice 2 still behaves correctly once the
   tests look more like engine/runtime usage instead of isolated unit cases.
2. We can add async-ready structure without lying about true async guarantees.
3. Cross-process / disaggregated-style correctness can be exercised in a small,
   testable way before introducing production transport backends.

This is where the implementation either starts to look operationally believable
or collapses under edge cases.

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

Diffstat summary:
- 6 files changed
- 3059 insertions
- 61 deletions

## File-by-file intent

`vllm_spyre/distributed/.../v1/inmemory_spyre_connector.py`
- adds stronger runtime semantics
- introduces async-ready layer load structure
- adds store limits and richer failure handling

`vllm_spyre/distributed/.../v1/metadata.py`
- adds stricter validation and stats support

`vllm_spyre/envs.py`
- adds knobs for async worker count and store limits

`tests/v1/worker/test_kv_integration.py`
- validates realistic bridge + connector interactions

`tests/v1/worker/test_kv_engine_level.py`
- validates engine-like sequencing and stats accumulation

`tests/v1/worker/test_kv_phase6.py`
- validates phase6 scenarios including 2-process transport and more fault cases

## Why this is a separate PR

This slice increases complexity substantially.

It deserves separation because it asks different review questions:
- Are failure semantics explicit enough?
- Are async-ready semantics honest and maintainable?
- Are the larger tests proving the right thing?

If reviewers are still debating the core metadata contract, do not start here.

## Core runtime invariants introduced here

1. A request should not be marked finished unless work actually completed.
2. Async-ready code paths must remain semantically correct when running
   synchronously.
3. Store eviction controls must not corrupt live-step behavior.
4. Validation failures should fail conservatively rather than silently degrade.
5. Cross-process test transport exists to prove correctness, not to claim
   production readiness.

## Reviewer focus

Primary review targets:
- async-ready path structure (`start_load_kv` / per-layer behavior / waits)
- `get_finished()` correctness under partial work
- validation hardening in metadata and connector request handling
- byte-budget and registry-limit semantics
- whether the 2-process harness actually proves what we think it proves

Secondary review targets:
- test maintainability and runtime cost
- whether env knobs remain understandable

## Main risks in this slice

1. Complexity growth:
- this is the first slice where the connector starts to accumulate operational state.

2. Test realism drift:
- larger tests can create false confidence if they only mirror the implementation.

3. Async semantics confusion:
- it is easy for readers to assume "async-ready" means "async complete"; this
  slice must remain explicit that sync semantics still dominate.

## What this slice does not solve

- production transport backend
- NIXL or offload integration for Spyre connector
- true overlap tuning on Spyre hardware
- Prometheus adapter (that is Slice 4)

## Relationship to current vs future Spyre compiler work

Current compiler / FMS path:
- useful for hardening the current integration path
- especially useful for proving lifecycle and failure semantics before any
  compiler migration

Future `torch-spyre` path:
- some test harness ideas will remain useful
- some implementation details (especially staging/store mechanics) may be replaced

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

Additional targeted checks:
```bash
pytest -q tests/v1/worker/test_kv_integration.py -k "reuse or failure"
pytest -q tests/v1/worker/test_kv_engine_level.py -k "stats or finish"
pytest -q tests/v1/worker/test_kv_phase6.py -k "schema or process or async"
```

## Suggested smoke environments

Local CPU:
- good for test-driven validation of semantics and failure behavior

Remote CUDA:
- good for testing heavier harnesses and confirming no hidden dependency on
  CPU-only assumptions

Spyre cards:
- not required before the minimal single-call POC
- useful only after current-compiler path is ready for connector-aware execution

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
