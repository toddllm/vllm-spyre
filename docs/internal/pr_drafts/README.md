# KV Connector PR Draft Packet

This folder is the internal control plane for the KV connector branch stack.
It answers four practical questions:

1. Which branch should we test from right now?
2. Which branch should we open as a PR when we only want a narrow scope?
3. What exactly is inside each slice, in file and commit terms?
4. What should reviewers focus on so the discussion stays about the real risk?

## How to use this packet

If the goal is:

- Validate everything end to end:
  - use `codex/spyre-kv-combined`
  - see `combined-testing.md`

- Understand the architecture and how the stack is intentionally split:
  - start with `stack-overview.md`

- Open the smallest possible reviewable PR:
  - start with `codex/spyre-kv-slice1-bridge`
  - then progress slice by slice

- Understand the architecture and stack order:
  - read `../lab_notes/kv-connector-pr-slicing-plan.md`
  - then read slice docs in order: 1 -> 2 -> 3 -> 4

- Preserve notes without polluting PRs:
  - keep them on `codex/spyre-kv-connector`
  - do not merge `docs/internal/` into upstream unless explicitly needed

## Branch inventory

### Stable fork base

Base mirror (fork):
- `codex/spyre-kv-base-pr759`
- purpose: frozen compare base mirroring `origin/pr-759`
- commits on top of `origin/pr-759`: `0`
- compare link:
  - <https://github.com/toddllm/vllm-spyre/tree/codex/spyre-kv-base-pr759>

### Stacked PR slices

Slice 1 (bridge lifecycle):
- `codex/spyre-kv-slice1-bridge`
- base: `codex/spyre-kv-base-pr759`
- commits on top of `origin/pr-759`: `3`
- compare:
  - <https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-base-pr759...codex/spyre-kv-slice1-bridge>
- doc: `slice1-bridge.md`

Slice 2 (connector core + metadata + reuse correctness):
- `codex/spyre-kv-slice2-core`
- base: `codex/spyre-kv-slice1-bridge` (stacked)
- commits on top of `origin/pr-759`: `6`
- compare:
  - <https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-slice1-bridge...codex/spyre-kv-slice2-core>
- doc: `slice2-core.md`

Slice 3 (runtime hardening + async-ready + 2-process tests):
- `codex/spyre-kv-slice3-runtime`
- base: `codex/spyre-kv-slice2-core` (stacked)
- commits on top of `origin/pr-759`: `10`
- compare:
  - <https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-slice2-core...codex/spyre-kv-slice3-runtime>
- doc: `slice3-runtime.md`

Slice 4 (metrics adapter):
- `codex/spyre-kv-slice4-metrics`
- base: `codex/spyre-kv-slice3-runtime` (stacked)
- commits on top of `origin/pr-759`: `11`
- compare:
  - <https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-slice3-runtime...codex/spyre-kv-slice4-metrics>
- doc: `slice4-metrics.md`

### Full validation branches

Combined branch for validation:
- `codex/spyre-kv-combined`
- purpose: testable branch with all code slices, no internal docs
- current code state: same as slice 4
- compare to base mirror:
  - <https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-base-pr759...codex/spyre-kv-combined>
- doc: `combined-testing.md`

Planning/integration branch:
- `codex/spyre-kv-connector`
- purpose: internal notes, PR drafts, branch strategy, complete working history
- includes `docs/internal/`
- should not be treated as PR-ready payload by default

## Branch relationship model

Logical stack:

`codex/spyre-kv-base-pr759`
-> `codex/spyre-kv-slice1-bridge`
-> `codex/spyre-kv-slice2-core`
-> `codex/spyre-kv-slice3-runtime`
-> `codex/spyre-kv-slice4-metrics`

And separately:

`codex/spyre-kv-combined`
= same code payload as slice 4, but intended for testing

`codex/spyre-kv-connector`
= same general workstream plus internal documentation

## What each slice is trying to prove

Slice 1:
- We can wire the lifecycle correctly without needing a real connector backend yet.
- This proves control flow integration inside Spyre worker/model-runner.

Slice 2:
- We can implement a correctness-first connector with a clear metadata contract.
- This proves the FMS path can participate in block-id-driven load/save semantics.

Slice 3:
- The core connector can survive realistic runtime conditions, failures, and deeper test harnesses.
- This proves the design is not just unit-test-correct but operationally coherent.

Slice 4:
- Connector behavior can be surfaced through metrics without perturbing semantics.
- This proves observability can be layered on top instead of entangled into core logic.

## Review strategy by audience

If the reviewer cares most about:

- lifecycle correctness:
  - review Slice 1 first

- metadata contract and reuse semantics:
  - review Slice 2 first

- runtime behavior and failure semantics:
  - review Slice 3 first

- operational observability:
  - review Slice 4 first

## Quick strategy

If opening PRs immediately:
1. Open Slice 1 first.
2. Open Slice 2 targeting Slice 1.
3. Open Slice 3 targeting Slice 2.
4. Open Slice 4 targeting Slice 3.

If we only need to be PR-ready but do not intend to open immediately:
1. Keep testing on `codex/spyre-kv-combined`.
2. Keep notes and draft refinement on `codex/spyre-kv-connector`.
3. Leave the slice branches untouched unless a slice needs to be refreshed.

If maintainers prefer fewer PRs:
1. Collapse Slice 3 and Slice 4 together.
2. Keep Slice 1 and Slice 2 separate, because they represent the most important conceptual boundary.

If opening against upstream main after merges:
1. Rebase each slice onto the latest agreed base.
2. Retarget PR base to the branch maintainers want.
3. Keep `docs/internal/` out of upstream PR payload unless explicitly requested.

## Important constraints

- These docs exist only on `codex/spyre-kv-connector` unless we intentionally copy them elsewhere.
- The slice branches are kept code-clean on purpose.
- The combined branch is for testing, not for review clarity.
