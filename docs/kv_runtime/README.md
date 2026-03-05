# KV Runtime Architecture

This note is intended to be read directly on GitHub.

It describes the durable architecture for KV offloading, KV sharing, and KV
transfer across the current `vllm-spyre` stack and the future
`vllm-spyre-next` / `torch-spyre` direction.

For exact code and line references, use the companion
[source map](./source-map.md).

For detailed state machines, capabilities, experiment paths, and open
assumptions, use the appendices linked at the end of this note.

## Why this note exists

Several discussions are easy to blur together:

- KV reuse
- KV offloading
- KV sharing across workers or instances
- disaggregated prefill/decode
- scheduler integration
- transport mechanics
- compiler/runtime constraints

The cleanest way to reason about them is to separate:

- `control / policy`
- `identity`
- `residency`
- `sync / lifetime`
- `address resolution / export`
- `transport`
- `data plane`
- `capability layer`

That separation is the main argument of this note.

## Architectural claim

KV offloading is not merely a transport feature.

It is the forcing function that moves KV residency and reuse policy into the
runtime, with the scheduler as control-plane owner.

That matters for both current and future Spyre stacks:

- on the old stack, it gives a credible experimental path to validate
  scheduler-driven lifecycle semantics and metadata contracts
- on the future stack, it aligns with the long-term goal of moving tensor
  allocation and residency control into the runtime rather than leaving KV as a
  compiler-owned exception

## Source pins

This memo is tied to pinned code snapshots, not moving branch heads.

- upstream `vllm`: `1892993bc18e243e2c05841314c5e9c06a80c70d`
- upstream `vllm-spyre` `pr-759` review base: `8a9682897aa2bd4d77cf3fdab7acd3fbfe452a72`
- experimental combined Spyre connector branch: `3448cf95710f69e0f13fac000ab170670ad5268d`

The full reference index is in [source-map.md](./source-map.md).

## First principles

KV cache is the reusable state produced during prefill and consumed during
decode.

That has four direct consequences:

1. KV dominates memory pressure for long prompts and long-running sessions.
2. Reusing KV avoids recomputation and reduces accelerator load.
3. Offloading KV frees accelerator memory and increases effective concurrency.
4. Transferring KV allows another worker or instance to continue work that did
   not start locally.

A correct KV architecture therefore has to answer these questions:

1. What exact KV object are we naming?
2. Who decides whether to keep, evict, reload, recompute, or transfer it?
3. Where does it live right now?
4. When is it safe to move or consume?
5. How do logical pages become movable regions?
6. How does execution read and write the bytes correctly?

## Core model

```text
KV RUNTIME CORE MODEL
=====================

CONTROL / POLICY
- scheduler/runtime decides what should happen

IDENTITY
- what KV object is this?

RESIDENCY
- where does it live now?

SYNC / LIFETIME
- when is it safe to move or use?

EXPORT / ADDRESS RESOLUTION
- logical pages -> export/import region handles

TRANSPORT
- move bytes between tiers

DATA PLANE
- write / read / consume KV correctly

CAPABILITY LAYER
- compiler/runtime exposes what is feasible
```

The important point is that these are distinct concerns.

The most dangerous failure mode is to collapse them into one vague “KV
connector” concept. That usually hides bugs in identity, ownership, or
lifetime.

## Source identity vs destination placement

This distinction is foundational.

```text
SOURCE IDENTITY
---------------
What bytes are these?

(prefix_key or request lineage,
 layer_id,
 logical_page_index,
 epoch/generation)

DESTINATION PLACEMENT
---------------------
Where should those bytes go now?

(device / worker,
 destination block_id,
 slot mapping,
 placement epoch)
```

A destination block ID is not the identity of the source bytes. It is only the
current placement decision.

That distinction becomes critical once pages are reloaded into a different
allocation layout.

## Shared prefixes imply sharing state and copy-on-write boundaries

Shared-prefix reuse means two requests can reference the same stable pages and
then diverge.

That implies:

- shared pages need reference tracking
- shared stable pages must not be overwritten in place
- divergence requires explicit copy-on-write boundaries
- eviction and offload must account for sharing state

In practice, this means residency tracking must carry at least:

- `refcount`
- whether a page is shareable/immutable
- whether appending would require COW

## Pages need extent and dirtiness, not just existence

A page is not just present or absent.

It also has:

- `valid_tokens` or valid byte extent
- `dirty_range` for the last page that may still be growing
- a rule for when it becomes frozen and exportable

Without extent tracking, partial-page export/import becomes ambiguous and prone
to corruption.

## Core lifecycle invariant

Lifecycle correctness is foundational, so the minimal state machine belongs in
the core note.

```text
CORE LIFECYCLE
==============

WRITING
  ->
STABLE_ON_DEVICE
  ->
EXPORT_INFLIGHT
  ->
RESIDENT_ON_HOST/REMOTE
  ->
IMPORT_INFLIGHT
  ->
STABLE_ON_DEVICE

Never export WRITING.
Never consume before IMPORT completes.
```

The detailed legal and illegal transitions are in
[appendix-lifecycle.md](./appendix-lifecycle.md).

## Region handles, not raw addresses

The durable abstraction should be exportable/importable region handles, not
bare raw addresses.

A handle may represent:

- one contiguous range
- a scatter-gather list
- registration metadata
- alignment constraints
- lifetime tokens
- sync dependencies

That is the correct place to absorb differences between deployment modes,
virtualization modes, or runtime capabilities.

The stable seam should therefore be thought of as:

```text
logical pages -> export/import region handles
```

not:

```text
logical pages -> raw physical addresses
```

## Batched movement matters

The transfer unit used for performance is not necessarily the same as the
logical identity unit.

A logical page/block may be the right identity object, while transport should
still operate on coalesced batches of pages or regions.

That is the key reason to separate:

- logical identity
- placement
- export/import region construction
- transport batching

The transport-specific reasoning and microbenchmark interpretation are in
[appendix-transport-model.md](./appendix-transport-model.md).

## What `pr-759` actually buys

The short version:

> `pr-759` makes vLLM the source of truth for block identity and block-table
> semantics, even though the current Spyre/FMS path still owns the actual KV
> bytes.

That is the key control-plane improvement.

It is not yet full KV byte ownership by upstream vLLM. It is upstream ownership
of:

- block IDs
- block-table semantics
- scheduler chunking decisions
- the ability to emit connector metadata against upstream scheduling concepts

This is why `pr-759` matters even before full offloading or disaggregation is
possible on the old stack.

## Old stack vs future stack

### Old stack

- control plane can become upstream-real once scheduler owns logical pages and
  block-table semantics
- data plane is still FMS-shaped and needs old-stack-specific hooks or
  workarounds to expose KV in a usable way
- experimental work can validate identity, residency, lifecycle, batching, and
  failure semantics
- compiler-artifact parsing and direct-address tricks should not define the
  durable contract

### Future stack

- control plane stays upstream-real
- data plane should expose cleaner export/import and consumption seams
- transport should attach to runtime-visible page/region concepts rather than
  old compiler workarounds
- compiler/runtime should provide capabilities without owning reuse/offload
  policy

## Conformance invariants

Any implementation of this model should be able to prove these invariants:

- identity is ABA-safe
- shared-prefix divergence respects COW boundaries
- WRITING pages are never exported
- decode never consumes before import completion
- uncertain or failed loads degrade to invalidation and recompute rather than
  silent reuse
- residency transitions remain legal under retry, timeout, or preemption

The detailed conformance view is in
[appendix-conformance.md](./appendix-conformance.md).

## What belongs in appendices

Use the appendices for everything that is important but more specific than the
core model:

- detailed lifecycle rules:
  [appendix-lifecycle.md](./appendix-lifecycle.md)
- identity, sharing, and copy-on-write:
  [appendix-identity-cow.md](./appendix-identity-cow.md)
- capability matrix and handle/export contract:
  [appendix-capabilities.md](./appendix-capabilities.md)
- current old-stack experimental path:
  [appendix-oldstack-experiments.md](./appendix-oldstack-experiments.md)
- transport regime and batching model:
  [appendix-transport-model.md](./appendix-transport-model.md)
- conformance invariants and test families:
  [appendix-conformance.md](./appendix-conformance.md)
- open assumptions to challenge:
  [appendix-open-assumptions.md](./appendix-open-assumptions.md)
