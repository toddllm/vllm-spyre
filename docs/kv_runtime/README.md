# KV Runtime Architecture

This note is intended to be read directly on GitHub as a complete technical
argument.

It describes the durable architecture for KV offloading, KV sharing, and KV
transfer across the current `vllm-spyre` stack and the future
`vllm-spyre-next` / `torch-spyre` direction.

For exact code and line references, use the companion
[source map](./source-map.md).

For detailed state machines, capabilities, experiment paths, and open
assumptions, use the appendices linked at the end of this note.

## How to read this note

This note is structured in three passes.

1. `first principles`
   What KV is, why it matters, and what the system must decide.
2. `panoramic architecture`
   The full runtime flow, ownership boundaries, and the old-stack vs future
   stack split.
3. `appendices`
   Detailed lifecycle rules, identity/COW rules, capability requirements,
   transport economics, and open assumptions.

The core note should be enough to understand the architecture without opening
any appendix. The appendices exist to make the design harder to misread.

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

## Why this matters technically

The same architecture has to support several technically distinct situations:

- `same-instance offload`
  - prefill happens here
  - decode later continues here
  - KV may still be temporarily offloaded to host memory
- `cross-instance continuation`
  - prefill happens on one worker or instance
  - decode later resumes elsewhere
- `cold-start reuse`
  - a new worker loads precomputed KV instead of rebuilding it
- `shared-prefix reuse`
  - multiple requests share the same stable prefix pages and later diverge

Those are not identical workloads, but they all need the same foundational
runtime contract:

- stable logical identity
- correct residency tracking
- safe export/import lifecycle
- reliable transport
- correct data-plane consumption

## Panoramic runtime flow

This is the complete architecture in one flow.

```text
END-TO-END KV RUNTIME FLOW
==========================

(1) request arrives
    |
    v
(2) scheduler allocates logical KV pages
    |
    v
(3) prefill writes KV into those pages
    |
    v
(4) runtime freezes written extent
      WRITING -> STABLE_ON_DEVICE
    |
    v
(5) runtime records residency
      identity + extent + refcount + location=device
    |
    v
(6) policy decides:
      keep / offload / share / recompute later
    |
    v
(7) if moving:
      logical pages -> export/import region handles
    |
    v
(8) transport moves batched regions
      device <-> host <-> remote
    |
    v
(9) residency updates reflect actual location/state
    |
    v
(10) later continuation or reuse arrives
     |
     v
(11) scheduler decides destination placement
      block_ids + slot mapping for this step
     |
     v
(12) destination pages are populated through import
     |
     v
(13) decode consumes imported pages safely
```

This is the architectural center of the note. Everything else exists to make
that flow safe and explain which subsystem owns which decision.

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
connector” concept. That usually hides bugs in identity, ownership, lifetime,
or placement.

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

Refcount ownership belongs to the runtime residency layer, not to compiler
internals. In a concrete implementation it may be maintained by the KV cache
manager or a closely related coordinator, but it remains a runtime concern.

Stable shared pages should be treated as immutable. A dirty tail page may still
grow until it is explicitly frozen; after freeze, it should be considered
immutable for export, reuse, and sharing purposes.

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

## Retry safety and idempotency

Export and import are not one-shot operations in a real system. Timeouts,
retries, duplicate completions, and partial failures are expected transport
behaviors.

The core rule is:

```text
export/import must be retry-safe under the same source identity + epoch
```

Duplicate retries under the same identity and epoch must not change semantics
or silently corrupt state. If correctness is uncertain, the safe fallback is to
invalidate the destination placement and force recompute rather than silently
reuse.

## Region handles, not raw addresses

The durable abstraction should be exportable/importable region handles, not
bare raw addresses.

A handle may represent:

- one contiguous range
- one or more segments in a scatter-gather structure
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

## Hook surfaces define the old-stack vs future-stack split

The cleanest way to understand the current stack split is to compare hook
surfaces.

### Upstream-native hook surface

At the model-runner level, upstream vLLM binds metadata, starts load, then
finalizes after compute:

```python
kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)
kv_connector.start_load_kv(get_forward_context())
...
kv_connector.wait_for_save()
...
kv_connector.clear_connector_metadata()
```

Source:
[vllm/v1/worker/kv_connector_model_runner_mixin.py#L77-L112](https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/worker/kv_connector_model_runner_mixin.py#L77-L112)

At the layer level, upstream attention waits for layer load and saves after the
attention work finishes:

```python
connector.wait_for_layer_load(layer_name)
result = func(*args, **kwargs)
connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
```

Source:
[vllm/attention/utils/kv_transfer_utils.py#L14-L60](https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/attention/utils/kv_transfer_utils.py#L14-L60)

### Old-stack hook surface

The old FMS-shaped path does not naturally expose that per-layer seam, so the
bridge has to reconstruct the lifecycle around `forward()`.

The underlying limitation is visible in the old-stack model runner, which still
returns a dummy FMS-derived KV spec rather than a native vLLM attention-backed
spec:

```python
# The spyre modeling code currently comes from `fms`, and does not
# integrate with vLLM's modeling classes ...
# This just returns a dummy value for now.
attn_spec = FullAttentionSpec(
    block_size=block_size,
    num_kv_heads=1,
    head_size=1,
    dtype=torch.float16,
)
return {"foo": attn_spec}
```

Source:
[vllm_spyre/v1/worker/spyre_model_runner.py#L193-L219](https://github.com/vllm-project/vllm-spyre/blob/8a9682897aa2bd4d77cf3fdab7acd3fbfe452a72/vllm_spyre/v1/worker/spyre_model_runner.py#L193-L219)

That is why old-stack work is useful but inherently transitional.

Until the data plane becomes more vLLM-native, old-stack connector integration
is therefore necessarily bulk or bridge-shaped rather than true per-layer
overlap.

## Old stack vs future stack

### Old stack

```text
OLD STACK
=========

scheduler owns logical pages and metadata
  ->
worker/model runner receives scheduler output
  ->
current execution path does not naturally expose KV pages
  ->
old-stack-specific hooks or workarounds expose movable data
  ->
transport experiment proceeds
```

Properties:

- control plane can become upstream-real
- data plane still needs old-stack-specific visibility hooks
- experimental work can validate identity, residency, lifecycle, batching, and
  failure semantics
- compiler-artifact parsing and direct-address tricks should not define the
  durable contract

### Future stack

```text
FUTURE STACK
============

scheduler owns logical pages and metadata
  ->
worker/model runner receive metadata through cleaner seams
  ->
runtime-visible page/region export/import becomes natural
  ->
transport attaches to a cleaner runtime contract
```

Properties:

- control plane stays upstream-real
- data plane should expose cleaner export/import and consumption seams
- transport should attach to runtime-visible page/region concepts rather than
  old compiler workarounds
- compiler/runtime should provide capabilities without owning reuse/offload
  policy

## What success looks like

### Old stack success

Old-stack work is successful if it validates the durable runtime contract:

- scheduler-owned metadata and page identity flow
- residency and lifecycle semantics
- safe export/import boundaries
- reliable invalidate-and-recompute fallback
- batching and transport-threshold learning

It is not necessary for old-stack work to define the final transport API or the
final production performance model.

### Future stack success

Future-stack work is successful if it removes the bridge-shaped workaround and
makes the critical seams native:

- cleaner data-plane visibility of KV
- cleaner per-layer or runtime-native export/import hooks
- address resolution as a runtime-visible capability
- transport integration without old compiler-artifact dependence

## Durable vs experimental

This is the most important filter for deciding what belongs in the architecture
and what belongs in a temporary experiment.

```text
LIKELY DURABLE
--------------
- scheduler-owned logical page identity
- source identity distinct from destination placement
- residency tracking in runtime
- sync/lifetime as a first-class contract
- region handles rather than raw-address assumptions
- batched transfer semantics
- failure -> invalidation -> recompute fallback
- shared-prefix refcount and COW rules

LIKELY EXPERIMENTAL / TRANSITIONAL
----------------------------------
- compiler-artifact parsing
- raw address reconstruction from old compiler state
- old-stack-only visibility hooks
- eager copy bridges used only to expose current-stack KV
- any direct physical-address assumption baked into the stable API
```

## Conformance invariants

Any implementation of this model should be able to prove these invariants:

- identity is ABA-safe
- shared-prefix divergence respects COW boundaries
- WRITING pages are never exported
- decode never consumes before import completion
- export/import retries under the same identity + epoch are idempotent
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
