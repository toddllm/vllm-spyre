# Slice 1: Bridge Lifecycle Wiring

Branch:
- `codex/spyre-kv-slice1-bridge`

Suggested PR base:
- `codex/spyre-kv-base-pr759`

Compare:
- <https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-base-pr759...codex/spyre-kv-slice1-bridge>

## Scope

Implements lifecycle-correct KV connector bridge wiring in Spyre worker/model runner.

In scope:
- bridge module and lifecycle orchestration
- model runner integration points
- worker initialization and cache registration timing
- bridge unit tests

Out of scope:
- concrete connector storage logic
- scheduler-driven reuse logic
- metrics and async optimization

## Included commits

- `ce231fb` Add KV connector bridge for lifecycle-correct connector integration
- `5eb51f2` Fix test mock: set_forward_context needs valid vllm_config fields
- `a813be7` Fix KV bridge finalization and connector staging sync

## Files changed

- `vllm_spyre/v1/worker/spyre_kv_connector_bridge.py` (new)
- `vllm_spyre/v1/worker/spyre_model_runner.py`
- `vllm_spyre/v1/worker/spyre_worker.py`
- `vllm_spyre/envs.py`
- `tests/v1/worker/test_kv_connector_bridge.py` (new)

## Validation checklist

Minimal:
```bash
pytest -q tests/v1/worker/test_kv_connector_bridge.py
```

Recommended:
```bash
pytest -q tests/v1/worker
```

## Draft PR title

`[KVConnector][Slice1] Add lifecycle bridge wiring for Spyre worker/model runner`

## Draft PR body

```markdown
## Summary
This PR adds lifecycle-correct KV connector bridge wiring for Spyre v1 worker/model runner integration.

## What changed
- Added `spyre_kv_connector_bridge.py` to centralize connector lifecycle calls.
- Wired bridge calls into `spyre_model_runner` execute paths.
- Added worker-side init/registration timing updates in `spyre_worker`.
- Added bridge-focused tests.

## Why
This is the minimum integration unit needed before adding connector storage/reuse logic.

## Test plan
- `pytest -q tests/v1/worker/test_kv_connector_bridge.py`
- (optional) `pytest -q tests/v1/worker`

## Non-goals
- No connector storage/reuse policy changes.
- No metrics or async performance work.
```
