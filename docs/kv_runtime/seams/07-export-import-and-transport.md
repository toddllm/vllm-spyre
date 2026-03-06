# Seam: Export, Import, and Transport

## Question

What is the seam between logical page decisions and actual byte movement, and
how should push/pull behavior be understood without hard-coding one transport
style into the architecture?

## Decision target

Design choice decision.

## Current answer

Keep the architecture push/pull-neutral.

The scheduler/runtime should decide what logical pages or regions need to move.
The worker/runtime should resolve that into export/import-capable state. The
transport backend should decide whether the actual movement is push, pull, or
hybrid.

## What this page establishes

- The connector contract already separates metadata/planning from worker-side
  load/save operations.
- Current experimental code already hides movement behind store/load/save seams
  rather than baking one transport direction into the metadata.
- Runtime copy and stream gaps are a capability issue, not a reason to collapse
  identity, placement, and transport into one abstraction.

## Story snippets

### 1. The connector contract is movement-oriented, not direction-hard-coded

`KVConnectorBase_V1` describes scheduler-side metadata construction and
worker-side bind/load/save operations. That is already close to the right
transport-neutral split.

The important architectural point is that the contract describes what should
move and how completion/failure is reported, not whether the movement is
implemented as push or pull.

Relevant anchors in [source-map.md](../source-map.md):
[`UP-KVCONN-BASE`](../source-map.md#up-kvconn-base),
[`UP-ACTIVE-PRE`](../source-map.md#up-active-pre),
[`UP-ACTIVE-POST`](../source-map.md#up-active-post),
[`UP-ACTIVE-NOFWD`](../source-map.md#up-active-nofwd).

### 2. Cross-layer block preference already hints that movement granularity matters

Upstream connector API explicitly exposes a `prefer_cross_layer_blocks` signal.
That matters because transport efficiency is about more than naming logical
pages: it is also about what movement unit is best.

That is a strong reason to keep transport granularity as its own design
question rather than pretending a page ID alone defines the whole byte-moving
contract.

Relevant anchors in [source-map.md](../source-map.md):
[`UP-KVCONN-CROSSLAYER`](../source-map.md#up-kvconn-crosslayer).

### 3. The experimental worker-side flow already exposes the export/import seam

The experimental connector has distinct places for:

- cache registration
- load start
- save

That is the practical seam between scheduler/runtime intent and the actual byte
movement mechanism.

Relevant anchors in [source-map.md](../source-map.md):
[`EXP-CONN-REGISTER-CACHES`](../source-map.md#exp-conn-register-caches),
[`EXP-CONN-LOAD`](../source-map.md#exp-conn-load),
[`EXP-CONN-SAVE`](../source-map.md#exp-conn-save),
[`EXP-STORE`](../source-map.md#exp-store).

### 4. The backing store abstraction proves movement can stay separate from identity

The experimental store makes it clear that a connector can present the same
logical contract while the actual backing mechanism changes.

That is the right shape for later transport evolution:

- in-memory store
- host-memory stage
- storage-backed path
- networked/disaggregated path

Relevant anchors in [source-map.md](../source-map.md):
[`EXP-STORE`](../source-map.md#exp-store),
[`EXP-META`](../source-map.md#exp-meta).

### 5. Torch-spyre runtime gaps are capability gaps, not reasons to distort the contract

Current `torch-spyre` code makes the remaining runtime issues explicit:

- copy paths still have limitations
- some copy semantics still carry FIXME/TODO comments
- stream/event hooks are not fully mature

Those are real constraints, but they should be modeled as capability gaps in
the runtime layer, not as excuses to make scheduler/control metadata
transport-specific.

Relevant anchors in [source-map.md](../source-map.md):
[`TS-COPY`](../source-map.md#ts-copy),
[`TS-COPY-FIXME`](../source-map.md#ts-copy-fixme),
[`TS-HOOKS-STREAM`](../source-map.md#ts-hooks-stream).

## What differs in old stack

Old stack currently needs more glue between logical movement intent and actual
byte access because the FMS-owned data plane is still not exposing a native
vLLM-style import/export seam.

That is exactly why transport should be kept abstract in the docs: the current
movement mechanism is not yet the durable one.

## What should survive into future stack

These parts should survive directly:

- push/pull-neutral connector semantics
- scheduler-visible movement descriptors
- runtime-local export/import handles and registrations
- explicit separation between identity, placement, and transport
- batched/coalesced movement as a transport concern rather than an identity
  concern

## Relevant anchors in `source-map.md`

- [`UP-KVCONN-BASE`](../source-map.md#up-kvconn-base)
- [`UP-KVCONN-CROSSLAYER`](../source-map.md#up-kvconn-crosslayer)
- [`EXP-CONN-REGISTER-CACHES`](../source-map.md#exp-conn-register-caches)
- [`EXP-CONN-LOAD`](../source-map.md#exp-conn-load)
- [`EXP-CONN-SAVE`](../source-map.md#exp-conn-save)
- [`EXP-STORE`](../source-map.md#exp-store)
- [`TS-COPY`](../source-map.md#ts-copy)
- [`TS-COPY-FIXME`](../source-map.md#ts-copy-fixme)
- [`TS-HOOKS-STREAM`](../source-map.md#ts-hooks-stream)
