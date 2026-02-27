# Slice 4: Prometheus Metrics Adapter

Branch:
- `codex/spyre-kv-slice4-metrics`

Suggested PR base:
- `codex/spyre-kv-slice3-runtime` (stacked)

Compare:
- <https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-slice3-runtime...codex/spyre-kv-slice4-metrics>

## Scope

Adds connector metrics adapter wiring and tests.

In scope:
- Prometheus adapter implementation for connector stats
- counter mapping from connector stats to metrics
- metrics-focused tests

Out of scope:
- connector runtime semantics changes
- scheduler/model-runner behavior changes

## Included commit

- `3448cf9` Add Prometheus metrics adapter for Spyre KV connector (P2)

## Files changed

- `vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py`
- `tests/v1/worker/test_kv_phase6.py`

## Validation checklist

Minimal:
```bash
pytest -q tests/v1/worker/test_kv_phase6.py -k "prom or metric"
```

Recommended:
```bash
pytest -q tests/v1/worker/test_kv_phase6.py
```

## Draft PR title

`[KVConnector][Slice4] Add Prometheus adapter for Spyre connector metrics`

## Draft PR body

```markdown
## Summary
This PR adds Prometheus adapter support for Spyre KV connector metrics.

## What changed
- Added connector Prometheus adapter wiring for matched/load/save/miss/eviction counters.
- Added metrics-focused tests, including smoke validation in realistic connector flow.

## Why
This slice makes connector behavior observable without changing connector runtime semantics.

## Test plan
- `pytest -q tests/v1/worker/test_kv_phase6.py -k "prom or metric"`

## Non-goals
- No scheduler/model-runner lifecycle changes.
- No new connector transport/storage behavior.
```
