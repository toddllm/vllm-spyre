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

## What this slice is trying to prove

This slice is the control-flow foundation. It is intentionally not trying to
prove that reuse works yet. It is trying to prove three narrower things:

1. Spyre worker/model-runner can participate in the upstream KV connector
   lifecycle without forking scheduler behavior.
2. The bridge can centralize the fragile sequencing points so later connector
   logic is not spread across multiple execution paths.
3. The FMS path can safely expose a staging boundary where later load/save work
   will occur.

If this slice is unstable, everything after it becomes harder to review because
the code path itself is still moving.

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

Diffstat summary:
- 5 files changed
- 947 insertions
- 63 deletions

## File-by-file intent

`vllm_spyre/v1/worker/spyre_kv_connector_bridge.py`
- new bridge abstraction
- owns begin/load/post-finish/clear sequencing
- keeps connector lifecycle logic out of ad-hoc runner branches

`vllm_spyre/v1/worker/spyre_model_runner.py`
- invokes bridge in each execute path
- ensures no-forward and normal-forward cases both finalize correctly
- adds staging sync behavior at the FMS boundary

`vllm_spyre/v1/worker/spyre_worker.py`
- initializes transfer state
- registers KV caches after warmup / final allocation state

`vllm_spyre/envs.py`
- adds bridge enable/disable flag

`tests/v1/worker/test_kv_connector_bridge.py`
- validates lifecycle ordering
- validates cleanup invariants
- protects against premature `get_finished()` reporting

## Why this is a separate PR

This is the smallest reviewable unit that changes production control flow.

Keeping it separate gives reviewers one focused question:
- "Is the lifecycle wiring correct and safe?"

It avoids mixing that question with:
- metadata design
- storage semantics
- scheduling reuse policy
- metrics design

## Reviewer focus

Primary review targets:
- bridge call ordering in all execute paths
- cleanup on all return paths
- whether cache registration happens after final warmup allocation
- whether disabled-path behavior remains unchanged

Secondary review targets:
- env var naming and default behavior
- test realism around `set_forward_context`

## Main risks in this slice

1. Finalization missed on one branch:
- could leave stale metadata attached to the next step.

2. KV cache registration too early:
- could register stale tensor references and make later connector logic invalid.

3. FMS staging copy semantics misunderstood:
- could produce a lifecycle that looks correct but mutates the wrong storage.

## What would block merging this slice

- Any path where bridge cleanup is not guaranteed.
- Any path where disabled bridge mode changes behavior.
- Any uncertainty about cache registration timing after warmup.

## What this slice does not solve

- no actual KV reuse
- no load miss handling policy
- no cross-request prefix transfer
- no performance improvements

## Relationship to minimal Spyre POC

For a single inference-call Spyre POC, this slice is not required to prove
plain generation works. It matters once we want to validate connector-aware
execution inside the current Spyre/FMS path.

## Validation checklist

Minimal:
```bash
pytest -q tests/v1/worker/test_kv_connector_bridge.py
```

Recommended:
```bash
pytest -q tests/v1/worker
```

Additional targeted checks:
```bash
pytest -q tests/v1/worker/test_kv_connector_bridge.py -k "no_forward or final"
```

## Suggested smoke environments

Local CPU:
- good for fast lifecycle regression checks

Remote CUDA:
- useful if validating bridge behavior under real vLLM worker setup

Spyre cards:
- not required for this slice alone
- only needed once testing current-compiler integration on real hardware

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
