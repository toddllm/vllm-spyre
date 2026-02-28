# KV Connector Stack Overview

This document is the architecture-oriented entry point for the branch stack.

Read this first if you want the full story without jumping immediately into a
single PR slice.

## The problem we are solving

The current `vllm-spyre` path needs a connector story that works with the
upstream scheduler-owned block IDs introduced by `pr-759`, while still using
the current FMS-based model path.

That creates three distinct concerns:

1. **Lifecycle wiring**
- Where do connector hooks run in Spyre worker/model-runner?

2. **Contract and correctness**
- What metadata moves from scheduler to worker?
- What do block IDs and mappings mean?
- How do load/store operations behave conservatively?

3. **Operational behavior**
- What happens under failures, broader tests, async-ready structure, and metrics?

The branch stack mirrors those concerns so we can review and land them
independently if needed.

## The branches, conceptually

`codex/spyre-kv-base-pr759`
- frozen review base
- no new connector work beyond `pr-759`

`codex/spyre-kv-slice1-bridge`
- adds lifecycle wiring only
- proves control flow

`codex/spyre-kv-slice2-core`
- adds connector implementation and metadata
- proves contract and correctness

`codex/spyre-kv-slice3-runtime`
- hardens behavior, adds broader tests, async-ready scaffolding
- proves runtime credibility

`codex/spyre-kv-slice4-metrics`
- adds metrics overlay
- proves observability can stay decoupled

`codex/spyre-kv-combined`
- all code slices combined
- best for testing, not best for review

`codex/spyre-kv-connector`
- planning/docs branch
- best for internal coordination, not best for PR payload

## What is stable vs likely to change

More stable:
- the need for a bridge boundary
- the need for a metadata contract
- the fact that scheduler-owned block IDs are the control-plane anchor

Less stable:
- the exact in-memory connector implementation details
- async structure details
- store/transport internals
- the final home of metrics if upstream expectations shift

## How this maps to current and future architecture

Current `vllm-spyre` + FMS:
- these slices are directly actionable now
- especially Slice 1 and Slice 2

Future `vllm-spyre-next` / `torch-spyre` direction:
- the bridge shape and metadata contract should still matter
- some implementation details in Slice 3 may be transitional

That means:
- Slice 1 and Slice 2 are the most architecturally important.
- Slice 3 and Slice 4 are more likely to be refined later.

## Decision guide

If the immediate need is:

- "Can we show a minimal, credible connector path exists?"
  - focus on Slice 1 + Slice 2

- "Can we test one branch with everything on it?"
  - use `codex/spyre-kv-combined`

- "Can we get a minimal Spyre-card demo running?"
  - use `codex/spyre-kv-combined` for code availability
  - but keep the hardware target to one simple inference call first

- "Can we open something upstream-reviewable soon?"
  - start with Slice 1

## Review anti-patterns to avoid

1. Reviewing the combined branch first
- too much scope at once

2. Debating metrics before contract semantics are agreed
- wrong order; metrics are downstream of correctness

3. Treating Slice 3 as required for the first Spyre hardware POC
- not necessary; first hardware POC should stay simple

4. Mixing internal notes into the PR payload
- makes targeted PRs harder to reason about
