# KV Runtime Architecture Note

This note is meant to be read directly on GitHub.

It focuses on the code and architecture of KV offloading, KV transfer, and
scheduler integration across the current `vllm-spyre` stack and the future
`vllm-spyre-next` / `torch-spyre` direction.

For exact file and line references, use the companion
[source map](./source-map.md).

## Why this note exists

Several related discussions are easy to blur together:

- KV reuse
- KV offloading
- disaggregated prefill/decode
- scheduler integration
- paged attention
- compiler/runtime constraints

The cleanest way to reason about them is to separate:

- the **control plane**
- the **data plane**
- the **transport plane**

while keeping **identity / address semantics** explicit across all three.

That separation is the main argument of this note.

## The architectural claim

KV offloading is not merely a transport feature.

It is the forcing function that moves KV residency and reuse policy into the
runtime, with the scheduler as control-plane owner.

This matters for both the current and future Spyre stacks:

- on the **old stack**, it gives us a credible experimental path to prove
  scheduler-driven lifecycle semantics and metadata contracts
- on the **new stack**, it aligns directly with the long-term goal of moving
  tensor allocation and residency control into the runtime rather than leaving
  KV as a compiler-owned exception

## Source pins

This memo is intentionally tied to pinned code snapshots, not moving branch
heads.

- **Upstream vLLM code**: `1892993bc18e243e2c05841314c5e9c06a80c70d`
- **Upstream vllm-spyre pr-759 review base**: `8a9682897aa2bd4d77cf3fdab7acd3fbfe452a72`
- **Toddllm experimental combined KV branch**: `3448cf95710f69e0f13fac000ab170670ad5268d`

The full reference index is in [source-map.md](./source-map.md).

## First principles

KV cache is the reusable state produced during prefill and consumed during
decode.

That has four consequences:

1. KV dominates memory pressure for long prompts and long-running sessions.
2. Reusing KV avoids recomputation and reduces accelerator load.
3. Offloading KV frees accelerator memory and increases effective concurrency.
4. Transferring KV makes it possible for another worker or instance to continue
   work that did not start locally.

This means a correct KV architecture must answer four questions:

1. What exact KV object are we talking about?
2. Who decides whether to keep, evict, reload, recompute, or transfer it?
3. How does the execution path read and write it?
4. How do bytes move between memory tiers?

## The model: three planes plus identity

```text
KV RUNTIME MODEL
================

                    identity / address spine
        -------------------------------------------------
        request_id | prefix_key | layer_id | block_id | mapping

control plane
- decides what should happen

data plane
- executes attention against KV

transport plane
- moves KV bytes between memory tiers

compiler/runtime capability layer
- constrains what is feasible
- should not own offload / reuse policy
```

The identity spine is not optional. Most severe KV bugs are identity bugs that
happen to surface during data movement:

- wrong block ID
- wrong source-to-destination mapping
- wrong prefix key
- wrong layer association

Transport can be correct as a byte mover and still be wrong overall if the
identity mapping is wrong.

## What `pr-759` actually buys

The short version:

> `pr-759` makes vLLM the source of truth for block identity and block-table
> semantics, even though the current Spyre/FMS path still owns the actual KV
> bytes.

That is the key control-plane improvement.

It is not yet full KV ownership by upstream vLLM. It is upstream ownership of:

- block IDs
- block-table semantics
- scheduler chunking decisions
- the ability to emit connector metadata

This is why `pr-759` matters even before full offloading or disaggregation is
possible on the old stack.

## Old stack: control is upstream-real, data is still FMS-shaped

```text
OLD STACK: CURRENT vllm-spyre + FMS
===================================

                         CONTROL / POLICY
+------------------------------------------------------------------+
| Upstream scheduler + block manager (after pr-759)                |
|                                                                  |
| - allocates block IDs                                            |
| - owns block-table semantics                                     |
| - decides chunking / reuse opportunities                         |
| - emits kv_connector_metadata                                    |
+-----------------------------------+------------------------------+
                                    |
                                    | block IDs, mappings, policy
                                    v
                         IDENTITY / ADDRESS
+------------------------------------------------------------------+
| request_id | prefix_key | layer_id | source block | dest block   |
+-----------------------------------+------------------------------+
                                    |
                                    v
                         DATA / EXECUTION
+------------------------------------------------------------------+
| Spyre worker + model runner                                      |
|                                                                  |
| - receives scheduler output                                      |
| - prepares FMS inputs                                            |
| - owns staging sync boundary                                     |
|                                                                  |
| Hook surface: around FMS forward(), not native per-layer hooks   |
+-----------------------------------+------------------------------+
                                    |
                                    v
+------------------------------------------------------------------+
| FMS model path                                                   |
|                                                                  |
| - owns live past_key_value_states                                |
| - attention internals are opaque to upstream layer hooks         |
| - no natural per-layer save/load seam                            |
+-----------------------------------+------------------------------+
                                    |
                                    v
                         TRANSPORT
+------------------------------------------------------------------+
| Connector backend                                                |
|                                                                  |
| - in-memory store                                                |
| - file-backed test transport                                     |
| - later: host offload / HostDMA / NIXL experiments               |
+------------------------------------------------------------------+
```

### Why the old stack needs a bridge

Upstream vLLM’s native path assumes the connector can participate at two
surfaces:

- a model-runner lifecycle boundary
- an attention-layer boundary

The layer-level seam is visible in upstream `maybe_transfer_kv_layer()`:

```python
connector.wait_for_layer_load(layer_name)
result = func(*args, **kwargs)
connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
```

Source:
[vllm/attention/utils/kv_transfer_utils.py#L14-L60](https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/attention/utils/kv_transfer_utils.py#L14-L60)

The current FMS-based Spyre path does not expose that seam cleanly. The old
stack therefore needs a bridge around `forward()`, not because bulk load/save is
architecturally ideal, but because that is the only reliable hook surface.

That limitation is already visible in the old-stack model runner:

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

That is the main reason old-stack work is valuable but inherently transitional.

## New stack: control, data, and transport line up better

```text
NEW STACK: UPSTREAM-ALIGNED + torch-spyre DIRECTION
===================================================

                         CONTROL / POLICY
+------------------------------------------------------------------+
| Upstream scheduler + block manager                               |
|                                                                  |
| - block IDs                                                      |
| - block tables / slot mapping                                    |
| - reuse / offload / reload / recompute policy                    |
| - connector metadata                                             |
+-----------------------------------+------------------------------+
                                    |
                                    v
                         IDENTITY / ADDRESS
+------------------------------------------------------------------+
| request_id | prefix_key | layer_id | block_id | slot mapping     |
+-----------------------------------+------------------------------+
                                    |
                                    v
                         DATA / EXECUTION
+------------------------------------------------------------------+
| Spyre worker / model runner / attention backend                  |
|                                                                  |
| - upstream-aligned forward context                               |
| - native per-layer connector hooks                               |
| - paged KV contract consumed directly                            |
+-----------------------------------+------------------------------+
                                    |
                                    v
                         TRANSPORT
+------------------------------------------------------------------+
| Connector backend                                                |
|                                                                  |
| - host DRAM offload                                              |
| - disaggregated prefill/decode                                   |
| - NIXL / DMA / remote tiers                                      |
| - retries / timeouts / invalid-block reporting                   |
+-----------------------------------+------------------------------+
                                    |
                                    v
                         CAPABILITY LAYER
+------------------------------------------------------------------+
| torch.compile / Inductor / torch-spyre                           |
|                                                                  |
| - determines what async / copy / memory mechanisms exist         |
| - should expose capabilities                                     |
| - should not own reuse / offload policy                          |
+------------------------------------------------------------------+
```

The cleaner long-term property is this:

- scheduler owns policy and identity
- attention/backend path owns execution correctness
- connector owns movement and reliability semantics
- compiler/runtime exposes capabilities, but does not decide policy

## Hook surface comparison

This is the shortest explanation for why old-stack work and new-stack work feel
so different.

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
[vllm/v1/worker/kv_connector_model_runner_mixin.py#L79-L112](https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/worker/kv_connector_model_runner_mixin.py#L79-L112)

At the layer level, upstream attention uses slot mapping to place KV into paged
storage:

```python
slot_mapping = forward_context.slot_mapping
layer_slot_mapping = slot_mapping.get(layer_name)
...
attn_layer.impl.do_kv_cache_update(
    attn_layer, key, value, kv_cache, layer_slot_mapping
)
```

Source:
[vllm/attention/layer.py#L828-L849](https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/attention/layer.py#L828-L849)

### Old-stack Spyre hook surface

The current Spyre bridge has to recreate the lifecycle around `forward()`:

```python
self._kv_connector.bind_connector_metadata(
    scheduler_output.kv_connector_metadata
)
...
self._kv_connector.start_load_kv(get_forward_context())
...
self._kv_connector.wait_for_save()
...
self._kv_connector.clear_connector_metadata()
```

Source:
[vllm_spyre/v1/worker/spyre_kv_connector_bridge.py#L135-L218](https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/v1/worker/spyre_kv_connector_bridge.py#L135-L218)

That bridge is useful, but it is a workaround for the old hook surface rather
than the final target architecture.

## Lifecycle is a state machine, not a helper method

```text
PER-STEP CONNECTOR LIFECYCLE
============================

IDLE
  |
  | scheduler emits kv_connector_metadata
  v
STEP_ACTIVE
  |
  | bind_connector_metadata()
  v
BOUND
  |
  | start_load_kv()
  v
LOADING
  |
  | wait_for_layer_load(...) or bulk wait
  v
READY_FOR_COMPUTE
  |
  | forward() / attention execution
  v
COMPUTE_DONE
  |
  | save_kv_layer(...) or bulk save path
  | wait_for_save()
  v
FINALIZING
  |
  | get_finished()
  | get_block_ids_with_load_errors()
  | collect stats
  | clear_connector_metadata()
  v
IDLE
```

This state machine matters because the common failure modes are lifecycle bugs:

- stale metadata leaking across steps
- finished requests being reported too early
- load misses not flowing back as invalid destination blocks
- no-forward steps skipping required connector cleanup

The old-stack bridge exists largely to enforce this state machine in a path that
does not naturally provide the right hooks.

## Identity / address semantics

The identity plane needs to be stated explicitly because it is what must survive
both transport and time.

At minimum, the architecture needs canonical meanings for:

- `request_id`
- `prefix_key`
- `layer_id`
- `block_id`
- `source -> destination block mapping`

The metadata contract in the experimental Spyre connector is where that becomes
explicit:

```python
@dataclass
class SpyreConnectorMeta(KVConnectorMetadata):
    schema_version: int = 1
    requests: list[SpyreConnectorRequestMeta] = field(default_factory=list)
    layer_names: list[str] = field(default_factory=list)
    block_size: int = 0
    dtype: str = ""
    layout: str = "NHD"
```

Source:
[vllm_spyre/distributed/kv_transfer/kv_connector/v1/metadata.py#L126-L153](https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/metadata.py#L126-L153)

The most important rule here is conceptual, not syntactic:

> transport can move bytes, but only identity makes those bytes meaningful.

## Prefix identity is a design choice, not a storage detail

The reusable-KV key still needs to be defined carefully.

Possible choices:

- token-id prefix key
- normalized prompt hash
- block hash chain

This choice affects:

- matching correctness
- storage growth
- eviction semantics
- cross-stack portability

The right answer does not need to be final yet, but it does need to be treated
as part of the architecture rather than an afterthought.

## Transport granularity ladder

Correctness and performance do not require the same transport unit.

```text
Level 0: whole-request KV object
Level 1: per-layer KV blob
Level 2: per-block KV
Level 3: cross-layer packed block
Level 4: sub-block / partial block
```

Why this matters:

- old-stack correctness can start with a coarser unit
- long-term performance will likely want per-block or cross-layer packed blocks
- upstream vLLM already exposes the idea of cross-layer-friendly KV layouts via
  `prefer_cross_layer_blocks` and backend stride order

Source:
[vllm/distributed/kv_transfer/kv_connector/v1/base.py#L152-L159](https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L152-L159),
[vllm/v1/worker/kv_connector_model_runner_mixin.py#L113-L177](https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/worker/kv_connector_model_runner_mixin.py#L113-L177),
[vllm/v1/attention/backends/flash_attn.py#L119-L137](https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/attention/backends/flash_attn.py#L119-L137)

## What is reusable vs transitional

### Likely reusable

- scheduler-owned block identity
- typed scheduler-to-worker metadata
- load/save/finalize lifecycle
- invalid-block feedback to scheduler
- runtime ownership of KV residency decisions

### Likely transitional

- FMS-specific bridge logic
- staging sync tricks
- old-stack bulk load/save assumptions
- anything that exists only because paged attention is not yet native there

## Near-term demo scope

The old stack should be used to prove semantics, not to overclaim production
readiness.

Reasonable near-term demo scope:

- scheduler emits connector metadata
- conservative prefix match
- save after prefill
- load before continued work
- load miss falls back to recompute
- metrics show match/load/save/miss behavior
- CPU or file-backed two-process harness is acceptable

Not required for that demo:

- final async overlap design
- production NIXL path
- final torch-spyre integration
- full old-stack paged-attention-native implementation

## Related references

- [Source map](./source-map.md)
- [Upstream vllm-spyre `pr-759`](https://github.com/vllm-project/vllm-spyre/pull/759)
- [Upstream vllm-spyre `pr-770`](https://github.com/vllm-project/vllm-spyre/pull/770)
- [Comment closing `pr-770` as not implementable in old `vllm_spyre`](https://github.com/vllm-project/vllm-spyre/pull/770#issuecomment-3993456011)
- [vLLM blog: KV Offloading Connector, January 8, 2026](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
