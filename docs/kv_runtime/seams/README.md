# Seam Narratives

These seam pages are narrative summaries of the most important integration
boundaries in the KV runtime design.

They are not complete evidence bundles.

The pinned evidence still lives in [../source-map.md](../source-map.md). Each
page below explains one decision clearly, then points back to the relevant
anchors.

## How to use these pages

1. Read [../README.md](../README.md) first for the durable architecture.
2. Use one seam page at a time when you need to answer a concrete question.
3. Use [../source-map.md](../source-map.md) to inspect the pinned code behind
   each claim.

## Current seam set

- [01. Scheduler, Block Manager, and Placement](./01-scheduler-block-manager.md)
  - What upstream vLLM owns in the control plane, how block IDs and slot
    mapping become canonical, and what `pr-759` changes for Spyre.
- [02. Attention Backend and Slot Mapping](./02-attention-slot-mapping.md)
  - How scheduler-owned placement metadata becomes actual KV reads/writes, and
    what the backend contract requires before optimization.
- [03. KV Connector Lifecycle](./03-kv-connector-lifecycle.md)
  - How scheduler hooks, worker hooks, and recompute fallback fit together, and
    which parts belong to connector code versus runtime policy.
- [04. Old-Stack Divergences](./04-old-stack-divergences.md)
  - Why current `vllm-spyre` is still bridge-shaped around an FMS-owned data
    plane, and why that makes old-stack work useful but transitional.
- [05. Next-Stack and Runtime Direction](./05-next-stack-and-runtime-direction.md)
  - What is already real in `vllm-spyre-next` and `torch-spyre`, and which
    runtime gaps still separate the future stack from a complete KV data plane.
- [06. Identity, Placement, and Residency](./06-identity-placement-and-residency.md)
  - Why source identity, destination placement, and runtime residency are
    different layers, and where current code already begins to expose them.
- [07. Export, Import, and Transport](./07-export-import-and-transport.md)
  - How logical page decisions turn into byte movement, why transport should
    stay push/pull-neutral, and where current runtime capability gaps still sit.
- [08. Sync, Lifetime, and Materialized Coverage](./08-sync-lifetime-and-coverage.md)
  - When KV becomes exportable or consumable, why lifecycle must be explicit,
    and why “page exists” is not enough.
- [09. Failure, Invalidation, and Recompute](./09-failure-invalidation-and-recompute.md)
  - Where failure feedback is surfaced and why the only safe fallback is
    invalidation/recompute rather than silent reuse.
