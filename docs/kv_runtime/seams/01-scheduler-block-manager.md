# Seam: Scheduler, Block Manager, and Placement

## Question

What is the canonical control-plane contract for KV ownership, placement, and
re-use, and what did `pr-759` actually change in old `vllm-spyre`?

## Decision target

Interface decision.

## Current answer

Treat the upstream scheduler, KV cache manager, block table, and slot mapping as
the canonical control-plane contract.

`pr-759` matters because it moves old Spyre closer to that contract by making
upstream block identity and block-table semantics the source of truth, even
though the FMS path still owns the actual KV bytes.

## What this page establishes

- Upstream vLLM models prefill and decode through one token-progress scheduler,
  not separate phase schedulers.
- Upstream `KVCacheManager` and `BlockTable` own logical block allocation and
  slot mapping.
- Old `vllm-spyre` divergence should be understood primarily as a data-plane
  divergence, plus some scheduler/policy adaptations, layered on top of this
  upstream control-plane baseline.

## Story snippets

### 1. Upstream scheduler owns the control loop

Upstream scheduling is organized around token progress and connector-aware slot
allocation, not around a hard prefill/decode split.

That matters because chunked prefill, prefix reuse, offload, and connector
metadata all hang off the same scheduling loop.

Relevant anchors in [source-map.md](../source-map.md):
[`UP-SCHED-INIT`](../source-map.md#up-sched-init),
[`UP-SCHED-KVCM-INIT`](../source-map.md#up-sched-kvcm-init),
[`UP-SCHED-SCHEDULE`](../source-map.md#up-sched-schedule),
[`UP-SCHED-KV-META`](../source-map.md#up-sched-kv-meta).

### 2. The KV cache manager allocates logical blocks, including external KV cases

`KVCacheManager.allocate_slots()` is where upstream turns scheduler decisions
into logical block ownership. That is also where external/computed-token cases
enter the flow.

This is the right place to think about offload/reload integration because the
scheduler is already deciding how many tokens are local versus externally
available.

Relevant anchors in [source-map.md](../source-map.md):
[`UP-KVCM-ALLOC`](../source-map.md#up-kvcm-alloc),
[`UP-SCHED-CONN-STEP`](../source-map.md#up-sched-conn-step).

### 3. Block table plus slot mapping is the canonical address contract

Logical block IDs are not enough by themselves. `BlockTable.compute_slot_mapping`
turns logical placement into token-level positions that the data plane can
consume.

That means any backend or connector path that bypasses slot mapping is no longer
following upstream placement semantics.

Relevant anchors in [source-map.md](../source-map.md):
[`UP-BT-SLOTMAP`](../source-map.md#up-bt-slotmap),
[`UP-ATTN-SLOTMAP`](../source-map.md#up-attn-slotmap).

### 4. `pr-759` aligns old Spyre to upstream block identity without yet changing byte ownership

The durable effect of `pr-759` is not “upstream now owns all KV bytes.” The
actual gain is narrower and more important: upstream vLLM becomes the source of
truth for block identity and block-table semantics.

That is what makes connector metadata and scheduler-owned placement credible on
old Spyre, even while the FMS path still owns the underlying KV tensors.

Relevant anchors in [source-map.md](../source-map.md):
[`UP-SCHED-CONN-STEP`](../source-map.md#up-sched-conn-step),
[`OLD-DUMMY-KVSPEC`](../source-map.md#old-dummy-kvspec),
[`OLD-FMS-KV-PATH`](../source-map.md#old-fms-kv-path).

## What differs in old stack

Old `vllm-spyre` still layers several constraints on top of the upstream
contract:

- custom scheduling policy for chunked prefill and hardware-specific limits
- model-runner shims around an FMS-owned KV path
- dummy/spec compatibility paths where the data plane is not yet fully
  vLLM-native

That is why old-stack work can validate control-plane semantics without yet
being the final data-plane shape.

## What should survive into future stack

These parts should carry forward unchanged in spirit:

- scheduler-owned logical block identity
- `KVCacheManager` as the control-plane owner of logical allocation
- block-table plus slot-mapping semantics as the placement contract
- connector metadata built from scheduler-owned state rather than local
  reconstruction in the backend

## Relevant anchors in `source-map.md`

- [`UP-SCHED-INIT`](../source-map.md#up-sched-init)
- [`UP-SCHED-KVCM-INIT`](../source-map.md#up-sched-kvcm-init)
- [`UP-SCHED-SCHEDULE`](../source-map.md#up-sched-schedule)
- [`UP-SCHED-KV-META`](../source-map.md#up-sched-kv-meta)
- [`UP-KVCM-ALLOC`](../source-map.md#up-kvcm-alloc)
- [`UP-BT-SLOTMAP`](../source-map.md#up-bt-slotmap)
- [`UP-ATTN-SLOTMAP`](../source-map.md#up-attn-slotmap)
- [`UP-SCHED-CONN-STEP`](../source-map.md#up-sched-conn-step)
- [`OLD-DUMMY-KVSPEC`](../source-map.md#old-dummy-kvspec)
- [`OLD-FMS-KV-PATH`](../source-map.md#old-fms-kv-path)
