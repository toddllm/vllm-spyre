# Seam: Identity, Placement, and Residency

## Question

What exactly names reusable KV, how is that different from destination
placement, and where should runtime track where KV lives?

## Decision target

Interface decision.

## Current answer

Treat source identity, destination placement, and residency as separate
concepts.

Source identity should name the bytes. Destination placement should describe
where those bytes go for the current step. Residency should track where the
runtime believes those bytes live now and whether they are still reusable.

## What this page establishes

- Block IDs and slot mapping are placement data, not complete source identity.
- Connector metadata and store keys are where old Spyre already starts making
  source identity and residency explicit.
- Shared-prefix reuse only stays correct if identity and residency are runtime
  concepts rather than accidental properties of the current placement.

## Story snippets

### 1. Placement is upstream-owned, but placement is not the same thing as source identity

Upstream scheduler and block-table code make placement canonical: they allocate
logical blocks and compute slot mapping for this step.

That is necessary, but it is not enough to name a reusable KV object across
time. The same source bytes may later be placed into a different destination
layout.

Relevant anchors in [source-map.md](../source-map.md):
[`UP-BT-SLOTMAP`](../source-map.md#up-bt-slotmap),
[`UP-SCHED-CONN-STEP`](../source-map.md#up-sched-conn-step).

### 2. Experimental metadata already separates reusable source state from current step placement

The experimental connector schema has distinct request-level metadata, block
mapping, and global connector metadata validation.

That is the right direction because it stops treating “current block ID” as the
complete meaning of the bytes.

Relevant anchors in [source-map.md](../source-map.md):
[`EXP-REQ-META`](../source-map.md#exp-req-meta),
[`EXP-META`](../source-map.md#exp-meta),
[`EXP-META-VALIDATE`](../source-map.md#exp-meta-validate),
[`EXP-CONN-BUILDMETA`](../source-map.md#exp-conn-buildmeta).

### 3. Store keys make source identity durable across placement changes

The experimental store does not just remember “current block 17.” It stores KV
under a separate key space designed to survive later reload and remapping.

That is the right conceptual shape for source identity even if the exact tuple
evolves.

Relevant anchors in [source-map.md](../source-map.md):
[`EXP-STOREKEY`](../source-map.md#exp-storekey),
[`EXP-STORE`](../source-map.md#exp-store).

### 4. Matching and post-allocation state are where old stack already distinguishes source from destination

The experimental connector’s scheduler-side flow has two distinct stages:

- match existing reusable state
- capture the new destination placement after allocation

That is the clearest current code evidence that identity and placement are
different layers of the system.

Relevant anchors in [source-map.md](../source-map.md):
[`EXP-CONN-MATCH`](../source-map.md#exp-conn-match),
[`EXP-CONN-AFTERALLOC`](../source-map.md#exp-conn-afteralloc),
[`EXP-CONN-BUILDMETA`](../source-map.md#exp-conn-buildmeta).

### 5. Residency is a runtime concern, not a compiler accident

Once KV can be saved, reloaded, or shared, the runtime has to answer a direct
question: where does this source KV live right now?

The experimental store layer is the current code-level stand-in for that
runtime residency concept. It is not yet the final design, but it already shows
that residency belongs above raw placement.

Relevant anchors in [source-map.md](../source-map.md):
[`EXP-STORE`](../source-map.md#exp-store),
[`EXP-META`](../source-map.md#exp-meta).

## What differs in old stack

Old stack still makes this awkward because the FMS path owns the actual live KV
bytes even while scheduler metadata is becoming more upstream-shaped.

That means old-stack experiments can validate identity, mapping, and residency
concepts, but they still need bridge-shaped data-plane glue.

## What should survive into future stack

These parts should survive directly:

- source identity distinct from destination placement
- runtime-owned residency tracking
- metadata that can describe remapping from source identity to destination
  placement
- the rule that reusability cannot be inferred only from current block IDs

## Relevant anchors in `source-map.md`

- [`UP-BT-SLOTMAP`](../source-map.md#up-bt-slotmap)
- [`UP-SCHED-CONN-STEP`](../source-map.md#up-sched-conn-step)
- [`EXP-STOREKEY`](../source-map.md#exp-storekey)
- [`EXP-REQ-META`](../source-map.md#exp-req-meta)
- [`EXP-META`](../source-map.md#exp-meta)
- [`EXP-META-VALIDATE`](../source-map.md#exp-meta-validate)
- [`EXP-STORE`](../source-map.md#exp-store)
- [`EXP-CONN-MATCH`](../source-map.md#exp-conn-match)
- [`EXP-CONN-AFTERALLOC`](../source-map.md#exp-conn-afteralloc)
- [`EXP-CONN-BUILDMETA`](../source-map.md#exp-conn-buildmeta)
