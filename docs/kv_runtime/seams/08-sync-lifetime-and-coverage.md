# Seam: Sync, Lifetime, and Materialized Coverage

## Question

When is KV exportable or consumable, and why must lifecycle be reasoned about
per region/block rather than as one loose request-level state?

## Decision target

Interface decision.

## Current answer

Treat sync/lifetime as a first-class runtime seam.

The safe unit is not “the request feels done enough.” The safe unit is a region
or block that has reached a post-write state where coverage is known and export
or consumption semantics are explicit.

## What this page establishes

- Upstream already models connector lifecycle with explicit pre/after/no-forward
  phases rather than one implicit request-global state.
- Old-stack bridge code exists because lifecycle has to be made explicit when
  the data plane cannot participate natively.
- Load completion and load-error reporting are the core current evidence that
  lifetime and invalidation have to be explicit rather than inferred.

## Story snippets

### 1. Upstream lifecycle is explicit before it is optimized

Upstream worker-side lifecycle already has explicit pre-forward, post-forward,
and no-forward handling. That is strong evidence that lifecycle correctness is
not an optimization detail.

The layer-level transfer helper reinforces the same point: the system already
wants explicit moments when load/save transitions happen.

Relevant anchors in [source-map.md](../source-map.md):
[`UP-XFER-LAYER`](../source-map.md#up-xfer-layer),
[`UP-ACTIVE-PRE`](../source-map.md#up-active-pre),
[`UP-ACTIVE-POST`](../source-map.md#up-active-post),
[`UP-ACTIVE-NOFWD`](../source-map.md#up-active-nofwd).

### 2. The old-stack bridge exists to make step boundaries explicit

The bridge phases are not just scaffolding. They are the current place where
old Spyre can say:

- step begins
- forward is about to use KV
- forward has completed
- metadata can be finalized and cleared

That is exactly the kind of lifecycle bookkeeping needed before sync/lifetime
rules can ever become finer-grained.

Relevant anchors in [source-map.md](../source-map.md):
[`EXP-BRIDGE-BEGIN`](../source-map.md#exp-bridge-begin),
[`EXP-BRIDGE-AFTER`](../source-map.md#exp-bridge-after),
[`EXP-BRIDGE-FINISH`](../source-map.md#exp-bridge-finish).

### 3. Completion and load-error reporting are the current evidence for explicit lifetime state

The experimental connector has separate completion and load-error surfaces.

That matters because it shows the runtime cannot safely infer “usable” from
“someone tried to load it.” Usability must depend on explicit completion state
and explicit invalidation on failure.

Relevant anchors in [source-map.md](../source-map.md):
[`EXP-CONN-FINISH`](../source-map.md#exp-conn-finish),
[`EXP-CONN-LOADERR`](../source-map.md#exp-conn-loaderr).

### 4. Materialized coverage is the right narrowing of “extent tracking”

For the paged model, the important question is not a general-purpose prefix tree
inside the connector. The practical question is narrower:

- what KV region has actually been materialized?
- is the current tail partial?
- has it reached a state where export or consumption is safe?

Current old-stack code does not yet expose that seam cleanly, which is exactly
why this remains a target-architecture statement rather than a fully realized
implementation fact.

Relevant anchors in [source-map.md](../source-map.md):
[`OLD-FMS-KV-PATH`](../source-map.md#old-fms-kv-path),
[`UP-XFER-LAYER`](../source-map.md#up-xfer-layer).

### 5. Runtime capability gaps still matter for the future stack

Even once the architecture is right, sync and lifetime still depend on runtime
capabilities such as copy completion, stream semantics, and event handling.

That is why these rules belong at the architecture level but still need runtime
support to be implementable.

Relevant anchors in [source-map.md](../source-map.md):
[`TS-COPY`](../source-map.md#ts-copy),
[`TS-HOOKS-STREAM`](../source-map.md#ts-hooks-stream).

## What differs in old stack

Old stack currently has to reason about sync/lifetime mostly at the bridge and
worker level because the FMS path does not yet give a clean native region-level
export/import seam.

That makes current old-stack reasoning coarser than the desired future shape,
but it does not change the architecture requirement.

## What should survive into future stack

These parts should survive directly:

- explicit pre/post/no-forward lifecycle handling
- per-region or per-block readiness, not only request-global readiness
- explicit invalidation on uncertain or failed load state
- materialized coverage / partial-tail thinking rather than “page exists” as
  the only signal

## Relevant anchors in `source-map.md`

- [`UP-XFER-LAYER`](../source-map.md#up-xfer-layer)
- [`UP-ACTIVE-PRE`](../source-map.md#up-active-pre)
- [`UP-ACTIVE-POST`](../source-map.md#up-active-post)
- [`UP-ACTIVE-NOFWD`](../source-map.md#up-active-nofwd)
- [`EXP-BRIDGE-BEGIN`](../source-map.md#exp-bridge-begin)
- [`EXP-BRIDGE-AFTER`](../source-map.md#exp-bridge-after)
- [`EXP-BRIDGE-FINISH`](../source-map.md#exp-bridge-finish)
- [`EXP-CONN-FINISH`](../source-map.md#exp-conn-finish)
- [`EXP-CONN-LOADERR`](../source-map.md#exp-conn-loaderr)
- [`OLD-FMS-KV-PATH`](../source-map.md#old-fms-kv-path)
