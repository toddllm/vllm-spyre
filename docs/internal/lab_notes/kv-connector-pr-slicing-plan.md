# KV Connector PR Slicing Plan

## Goal
Keep `codex/spyre-kv-connector` as an integration branch, while making it easy
to open focused PRs without dragging unrelated scope.

## Current stack (on top of `origin/pr-759`)
1. `51d14f0` bridge lifecycle wiring
2. `b2f7fa1` bridge test fix
3. `4b3ce55` bridge finalization + staging sync
4. `7143301` in-memory connector + metadata + version guard
5. `11da8df` scheduler-driven reuse
6. `fe1cece` integration tests + LRU cap
7. `a7f3801` engine-level tests + stats
8. `2fe2d95` async layer pipeline + 2-process E2E + schema validation
9. `d09dc6f` correctness fixes (identity mapping + incomplete mapping fallback)
10. `2d509e8` byte-capped store + dedicated async env var
11. `4c9bb8a` Prometheus metrics adapter

## Suggested PR lanes

### PR 1: Bridge-only wiring (smallest mergeable unit)
Commits:
- `51d14f0`
- `b2f7fa1`
- `4b3ce55`

Includes:
- `vllm_spyre/v1/worker/spyre_kv_connector_bridge.py`
- minimal runner/worker wiring
- bridge tests only

### PR 2: Connector core + metadata contract
Commits:
- `7143301`
- `11da8df`
- `d09dc6f`

Includes:
- connector implementation
- metadata types + validation baseline
- scheduler-driven reuse semantics

### PR 3: Runtime hardening + integration tests
Commits:
- `fe1cece`
- `a7f3801`
- `2fe2d95`
- `2d509e8`

Includes:
- failure semantics and registry caps
- async-ready load plumbing
- 2-process E2E test path
- byte-capped store controls

### PR 4: Metrics and observability
Commits:
- `4c9bb8a`

Includes:
- Prometheus adapter wiring
- metrics-focused tests

## Execution pattern for focused PR branches
Use a fresh branch from `origin/pr-759`, then cherry-pick only the needed
commits for that PR lane:

```bash
git checkout -b codex/spyre-kv-pr1-bridge origin/pr-759
git cherry-pick 51d14f0 b2f7fa1 4b3ce55
```

Repeat for PR2/PR3/PR4 with their respective commit sets.

## Hygiene rules
- Keep `docs/internal/` out of upstream PRs unless specifically requested.
- Avoid host-specific names in committed notes.
- Keep each PR lane independently testable (`pytest` subset listed in PR body).
