# Slice 2: Connector Core + Metadata Contract

Branch:
- `codex/spyre-kv-slice2-core`

Suggested PR base:
- `codex/spyre-kv-slice1-bridge` (stacked)

Compare:
- <https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-slice1-bridge...codex/spyre-kv-slice2-core>

## Scope

Adds the concrete in-memory connector, metadata schema, factory registration, and scheduler-driven reuse correctness.

In scope:
- `KVConnectorMetadata`-derived schema for Spyre
- in-memory connector implementation (scheduler + worker interfaces)
- connector registration
- conservative version compatibility guard
- reuse mapping correctness fixes
- core connector tests

Out of scope:
- async load worker pool behavior
- 2-process transport and phase6 runtime hardening
- prometheus adapter

## Included commits

- `ce4ae06` Add InMemorySpyreConnector, metadata schema, version guard, and factory registration
- `11ee5d4` Add scheduler-driven KV reuse with exact-prefix matching and block mapping
- `5b4069a` Fix identity load block-ID lookup and incomplete mapping fallback

## Files changed

New:
- `vllm_spyre/compat.py`
- `vllm_spyre/distributed/.../v1/metadata.py`
- `vllm_spyre/distributed/.../v1/inmemory_spyre_connector.py`
- package init files under `vllm_spyre/distributed/...`
- `tests/v1/worker/test_inmemory_spyre_connector.py`
- `tests/v1/worker/test_scheduler_driven_kv_reuse.py`

Modified:
- `vllm_spyre/__init__.py`

## Validation checklist

Minimal:
```bash
pytest -q tests/v1/worker/test_inmemory_spyre_connector.py \
         tests/v1/worker/test_scheduler_driven_kv_reuse.py
```

Recommended:
```bash
pytest -q tests/v1/worker
```

## Draft PR title

`[KVConnector][Slice2] Add Spyre in-memory connector, metadata contract, and reuse correctness`

## Draft PR body

```markdown
## Summary
This PR introduces the Spyre in-memory KV connector implementation and metadata contract, plus scheduler-driven reuse and mapping correctness fixes.

## What changed
- Added `SpyreConnectorMeta` and request metadata structures.
- Added `InMemorySpyreConnector` implementing v1 connector interfaces.
- Registered connector in plugin registration path.
- Added strict fallback when external mapping coverage is incomplete.
- Fixed identity mapping behavior to use actual scheduler block IDs.
- Added unit tests for metadata, load/save, reuse behavior, and failure handling.

## Why
This is the first fully usable connector core for correctness-first reuse testing.

## Test plan
- `pytest -q tests/v1/worker/test_inmemory_spyre_connector.py`
- `pytest -q tests/v1/worker/test_scheduler_driven_kv_reuse.py`

## Non-goals
- No async overlap tuning.
- No cross-process production transport.
- No Prometheus integration in this slice.
```
