# KV Connector PR Slicing Plan

## Goal
Keep `codex/spyre-kv-connector` as an integration/planning branch, while making
it easy to open focused PRs without dragging unrelated scope.

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

## Prepared branches on fork

- Base mirror: `codex/spyre-kv-base-pr759`
- Slice 1: `codex/spyre-kv-slice1-bridge`
- Slice 2: `codex/spyre-kv-slice2-core`
- Slice 3: `codex/spyre-kv-slice3-runtime`
- Slice 4: `codex/spyre-kv-slice4-metrics`
- Combined testing branch: `codex/spyre-kv-combined`
- Planning/integration branch: `codex/spyre-kv-connector`

## Why the stack is split this way

The split is not arbitrary. It mirrors the actual risk boundaries:

1. Bridge wiring:
- changes control flow first
- should be reviewable without debating metadata or reuse policy

2. Connector core:
- introduces the contract and correctness semantics
- should be reviewable without runtime hardening noise

3. Runtime hardening:
- increases operational complexity
- should land only after the contract is accepted

4. Metrics:
- should remain optional and easy to separate from semantic changes

## Suggested PR lanes

### PR 1: Bridge-only wiring (smallest mergeable unit)
Branch:
- `codex/spyre-kv-slice1-bridge`

Includes:
- `vllm_spyre/v1/worker/spyre_kv_connector_bridge.py`
- minimal runner/worker wiring
- bridge tests only

Risk profile:
- medium
- control-flow-sensitive but conceptually narrow

### PR 2: Connector core + metadata contract
Branch:
- `codex/spyre-kv-slice2-core`

Includes:
- connector implementation
- metadata types + validation baseline
- scheduler-driven reuse semantics

Risk profile:
- high
- this is the contract-defining slice

### PR 3: Runtime hardening + integration tests
Branch:
- `codex/spyre-kv-slice3-runtime`

Includes:
- failure semantics and registry caps
- async-ready load plumbing
- 2-process E2E test path
- byte-capped store controls

Risk profile:
- high
- runtime complexity and broader tests

### PR 4: Metrics and observability
Branch:
- `codex/spyre-kv-slice4-metrics`

Includes:
- Prometheus adapter wiring
- metrics-focused tests

Risk profile:
- low to medium
- mostly additive, but can confuse review if mixed into core logic

## Execution pattern for focused PRs
Use the prepared slice branches directly as PR heads and set PR base to:
- Slice1 base: `codex/spyre-kv-base-pr759`
- Slice2 base: `codex/spyre-kv-slice1-bridge`
- Slice3 base: `codex/spyre-kv-slice2-core`
- Slice4 base: `codex/spyre-kv-slice3-runtime`

## Hygiene rules
- Keep `docs/internal/` out of upstream PRs unless specifically requested.
- Avoid host-specific names in committed notes.
- Keep each PR lane independently testable (`pytest` subset listed in PR body).

## When to stop and not widen scope

Stop at Slice 1 if:
- we only need to prove lifecycle wiring
- reviewers are not aligned yet on connector ownership

Stop at Slice 2 if:
- we need a correctness-first connector POC
- we want something defensible before runtime complexity grows

Only move to Slice 3 if:
- Slice 2 semantics are accepted
- we specifically need runtime hardening and broader validation

Treat Slice 4 as optional if:
- observability is useful but not part of the immediate decision
