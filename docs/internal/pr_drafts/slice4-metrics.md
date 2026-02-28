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

## What this slice is trying to prove

This slice is intentionally narrow.

It is meant to prove:

1. We can surface connector behavior through the upstream metrics seam without
   destabilizing connector semantics.
2. The stats contract introduced earlier is sufficient to drive counters.
3. Observability can remain an overlay rather than getting entangled with core
   connector logic.

This slice should be easy to review precisely because it is small.

## Included commit

- `3448cf9` Add Prometheus metrics adapter for Spyre KV connector (P2)

## Files changed

- `vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py`
- `tests/v1/worker/test_kv_phase6.py`

Diffstat summary:
- 2 files changed
- 408 insertions
- 3 deletions

## File-by-file intent

`vllm_spyre/distributed/.../v1/inmemory_spyre_connector.py`
- adds Prometheus adapter object
- maps connector stats into named counters
- keeps existing connector runtime semantics intact

`tests/v1/worker/test_kv_phase6.py`
- adds adapter-focused tests and smoke validation around stats-to-metrics wiring

## Why this is a separate PR

This is the cleanest slice to merge independently if observability is wanted
before the rest of the runtime hardening stack is accepted.

It also gives reviewers a narrow question:
- "Do the exported counters and adapter semantics look correct?"

## Reviewer focus

Primary review targets:
- metric names and labels
- counter mapping correctness
- whether zero/empty stats are handled safely
- whether tests validate non-trivial observations

Secondary review targets:
- whether this should stay on the connector class or move elsewhere later

## Main risks in this slice

1. Naming churn:
- if upstream naming conventions shift, this slice may need cosmetic changes.

2. Double counting:
- if stats reset/aggregation semantics are misunderstood, metrics can look right
  in tests and still be misleading in production.

## What this slice does not solve

- scrape endpoint integration across a real server deployment
- retention/export policy decisions
- any connector correctness issue

## Validation checklist

Minimal:
```bash
pytest -q tests/v1/worker/test_kv_phase6.py -k "prom or metric"
```

Recommended:
```bash
pytest -q tests/v1/worker/test_kv_phase6.py
```

Additional targeted checks:
```bash
pytest -q tests/v1/worker/test_kv_phase6.py -k "prom or observe or counter"
```

## Suggested smoke environments

Local CPU:
- sufficient for almost all validation here

Remote CUDA:
- useful only if validating that connector stats are non-zero under real traffic

Spyre cards:
- not necessary for this slice alone

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
