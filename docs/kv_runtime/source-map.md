# KV Runtime Architecture Source Map

This file is the pinned reference appendix for
[KV Runtime Architecture](./README.md).

All links below use exact SHAs so the note remains stable even if branch heads
move.

## Repository pins

| Repo | Purpose | Pinned ref |
| --- | --- | --- |
| `vllm-project/vllm` | upstream runtime, scheduler, attention, connector APIs | `1892993bc18e243e2c05841314c5e9c06a80c70d` |
| `vllm-project/vllm-spyre` | old-stack `pr-759` review base | `8a9682897aa2bd4d77cf3fdab7acd3fbfe452a72` |
| `toddllm/vllm-spyre` | experimental combined Spyre connector implementation | `3448cf95710f69e0f13fac000ab170670ad5268d` |
| `toddllm/vllm-spyre` | current next-stack scaffold snapshot used by these docs | `5aeb39ec813ff01161ec8f3a832fc86f676238c8` |
| `toddllm/torch-spyre` | current torch-spyre runtime/backend snapshot used by these docs | `6782329492de7558098ec17cd1260d4aab8d93ad` |

## Traceability rules

- Coverage means that each architectural claim in
  [README.md](./README.md) or an appendix has at least one anchor showing
  where the mechanism exists in code, or where a limitation/workaround is
  visible in code.
- Some claims in README are target-architecture statements rather than current
  implementation facts. In those cases, the source map points either to the
  closest existing mechanism or to the limitation/workaround that motivates the
  target design.
- Anchors use exact SHAs plus line ranges. If a range changes, update the
  range while preserving the anchor ID where practical.
- Not every claim maps to one function. Some claims are cross-cutting and
  therefore rely on multiple anchors.

## Minimum coverage checklist

| Seam | Anchor IDs | Why this matters |
| --- | --- | --- |
| Scheduler-owned control plane | `UP-SCHED-INIT`, `UP-SCHED-KVCM-INIT`, `UP-SCHED-SCHEDULE`, `UP-SCHED-KV-META` | Shows where upstream owns connector creation, cache-manager creation, scheduling, and metadata handoff. |
| Block allocation and external KV tokens | `UP-KVCM-ALLOC`, `UP-SCHED-CONN-STEP` | Shows where external/computed tokens enter allocation and scheduling. |
| Placement and slot mapping | `UP-BT-SLOTMAP`, `UP-ATTN-SLOTMAP` | Shows how block-table placement becomes token-level slot addresses. |
| Upstream connector lifecycle | `UP-ACTIVE-PRE`, `UP-ACTIVE-POST`, `UP-ACTIVE-NOFWD` | Shows the native pre/after/no-forward lifecycle shape. |
| Old-stack limitation | `OLD-DUMMY-KVSPEC`, `OLD-FMS-KV-PATH` | Shows why old-stack integration is bridge-shaped rather than layer-native. |
| Experimental bridge lifecycle | `EXP-BRIDGE-BEGIN`, `EXP-BRIDGE-BEFORE`, `EXP-BRIDGE-AFTER`, `EXP-BRIDGE-FINISH` | Shows the explicit old-stack bridge phases. |
| Metadata and store semantics | `EXP-META`, `EXP-META-VALIDATE`, `EXP-STORE` | Shows the scheduler-to-worker schema and the backing store abstraction. |
| Experimental scheduler-side connector flow | `EXP-CONN-MATCH`, `EXP-CONN-AFTERALLOC`, `EXP-CONN-BUILDMETA` | Shows matching, allocation-state capture, and metadata construction. |
| Experimental worker-side connector flow | `EXP-CONN-BIND`, `EXP-CONN-LOAD`, `EXP-CONN-SAVE`, `EXP-CONN-FINISH` | Shows binding, load, save, and completion handling. |
| Failure and recompute fallback | `UP-SCHED-CONN-STEP`, `EXP-BRIDGE-AFTER`, `EXP-CONN-LOADERR`, `EXP-CONN-FINISH` | Shows how load errors surface as invalid block IDs and completion state rather than silent reuse. |

## Seam page mapping

| Seam page | Anchor IDs | Why this supports the page |
| --- | --- | --- |
| `seams/01-scheduler-block-manager.md` | `UP-SCHED-INIT`, `UP-SCHED-KVCM-INIT`, `UP-SCHED-SCHEDULE`, `UP-SCHED-KV-META`, `UP-KVCM-ALLOC`, `UP-BT-SLOTMAP`, `UP-ATTN-SLOTMAP`, `UP-SCHED-CONN-STEP`, `OLD-DUMMY-KVSPEC`, `OLD-FMS-KV-PATH` | Shows upstream control-plane ownership of scheduling and placement, plus the old-stack limitation that keeps byte ownership in the FMS path. |
| `seams/02-attention-slot-mapping.md` | `UP-ATTN-BACKEND`, `UP-ATTN-CPU`, `UP-BT-SLOTMAP`, `UP-ATTN-SLOTMAP`, `UP-CUDA-KV-SHAPE`, `UP-CUDA-PAGED-V1`, `UP-CUDA-PAGED-V2`, `OLD-DUMMY-KVSPEC`, `OLD-FMS-KV-PATH` | Shows the explicit backend contract, how slot mapping becomes execution, and why old stack still needs FMS-shaped translation rather than a native backend seam. |
| `seams/03-kv-connector-lifecycle.md` | `UP-SCHED-KV-META`, `UP-SCHED-CONN-STEP`, `UP-ACTIVE-PRE`, `UP-ACTIVE-POST`, `UP-ACTIVE-NOFWD`, `EXP-BRIDGE-BEGIN`, `EXP-BRIDGE-BEFORE`, `EXP-BRIDGE-AFTER`, `EXP-BRIDGE-FINISH`, `EXP-CONN-BIND`, `EXP-CONN-LOAD`, `EXP-CONN-SAVE`, `EXP-CONN-FINISH`, `EXP-CONN-LOADERR` | Shows the split between scheduler-owned lifecycle, worker-side connector phases, bridge-shaped old-stack adaptation, and explicit failure/recompute fallback. |
| `seams/04-old-stack-divergences.md` | `UP-SCHED-SCHEDULE`, `UP-SCHED-CONN-STEP`, `OLD-DUMMY-KVSPEC`, `OLD-FMS-KV-PATH`, `EXP-BRIDGE-BEGIN`, `EXP-BRIDGE-BEFORE`, `EXP-BRIDGE-AFTER`, `EXP-BRIDGE-FINISH` | Shows why old `vllm-spyre` remains bridge-shaped around an FMS-owned data plane and which parts are transitional rather than durable. |
| `seams/05-next-stack-and-runtime-direction.md` | `NEXT-PLATFORM`, `NEXT-EXAMPLE`, `TS-ENTRYPOINT`, `TS-BACKEND-REGISTER`, `TS-INDUCTOR-AUTOLOAD`, `TS-FALLBACKS`, `TS-PRELOAD`, `TS-COPY`, `TS-COPY-FIXME`, `TS-HOOKS-STREAM` | Shows that next-stack scaffolding and torch-spyre backend integration are real today, while copy/stream/event gaps still limit the future stack’s runtime maturity. |

## Upstream vLLM anchors

### Scheduler and metadata handoff

<a id="up-sched-init"></a>

- `[UP-SCHED-INIT]` `Scheduler` initialization and connector creation
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/core/sched/scheduler.py#L63-L130>

<a id="up-sched-kvcm-init"></a>

- `[UP-SCHED-KVCM-INIT]` `Scheduler` creating `KVCacheManager`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/core/sched/scheduler.py#L216-L228>

<a id="up-sched-schedule"></a>

- `[UP-SCHED-SCHEDULE]` `Scheduler.schedule()` unified token-progress model
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/core/sched/scheduler.py#L313-L383>

<a id="up-sched-kv-meta"></a>

- `[UP-SCHED-KV-META]` `SchedulerOutput.kv_connector_metadata`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/core/sched/output.py#L212-L239>

<a id="up-sched-conn-step"></a>

- `[UP-SCHED-CONN-STEP]` Scheduler interaction with connector matching, allocation, and metadata build
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/core/sched/scheduler.py#L600-L879>

### KV cache manager and block allocation

- `KVCacheBlocks`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/core/kv_cache_manager.py#L22-L91>

- `KVCacheManager`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/core/kv_cache_manager.py#L94-L204>

<a id="up-kvcm-alloc"></a>

- `[UP-KVCM-ALLOC]` `KVCacheManager.allocate_slots()` including connector/external-token notes
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/core/kv_cache_manager.py#L206-L290>

### Block table and slot mapping

- `BlockTable`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/worker/block_table.py#L16-L99>

<a id="up-bt-slotmap"></a>

- `[UP-BT-SLOTMAP]` `BlockTable.compute_slot_mapping()`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/worker/block_table.py#L133-L191>

- `MultiGroupBlockTable`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/worker/block_table.py#L253-L340>

### Connector v1 contract

<a id="up-kvconn-base"></a>

- `[UP-KVCONN-BASE]` `KVConnectorBase_V1` declaration and worker-side methods
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L147-L320>

- `prefer_cross_layer_blocks`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L152-L159>

### Model-runner lifecycle seam

- `KVConnectorModelRunnerMixin._get_kv_connector_output`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/worker/kv_connector_model_runner_mixin.py#L77-L112>

- `KVConnectorModelRunnerMixin.use_uniform_kv_cache`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/worker/kv_connector_model_runner_mixin.py#L113-L177>

### Layer-level transfer seam

- `maybe_transfer_kv_layer`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/attention/utils/kv_transfer_utils.py#L14-L60>

<a id="up-attn-slotmap"></a>

- `[UP-ATTN-SLOTMAP]` `attention.layer` KV cache update using `slot_mapping`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/attention/layer.py#L828-L849>

### GPU-side active connector reference

- `ActiveKVConnector`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/worker/gpu/kv_connector.py#L48-L125>

<a id="up-active-pre"></a>

- `[UP-ACTIVE-PRE]` `ActiveKVConnector.pre_forward()`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/worker/gpu/kv_connector.py#L62-L78>

<a id="up-active-post"></a>

- `[UP-ACTIVE-POST]` `ActiveKVConnector.post_forward()`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/worker/gpu/kv_connector.py#L79-L95>

<a id="up-active-nofwd"></a>

- `[UP-ACTIVE-NOFWD]` `ActiveKVConnector.no_forward()`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/worker/gpu/kv_connector.py#L97-L107>

### KV layout / stride reference

- `FlashAttentionBackend.get_kv_cache_stride_order`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/attention/backends/flash_attn.py#L119-L137>

<a id="up-attn-backend"></a>

- `[UP-ATTN-BACKEND]` `AttentionBackend` contract and metadata-facing methods
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/attention/backend.py#L21-L166>

<a id="up-attn-cpu"></a>

- `[UP-ATTN-CPU]` CPU backend using shared attention metadata
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/attention/backends/cpu_attn.py#L22-L121>

<a id="up-cuda-kv-shape"></a>

- `[UP-CUDA-KV-SHAPE]` CUDA kernel-level KV and block-table argument layout
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/csrc/attention/attention_kernels.cuh#L83-L114>

<a id="up-cuda-paged-v1"></a>

- `[UP-CUDA-PAGED-V1]` Paged-attention v1 launch path
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/csrc/attention/paged_attention_v1.cu#L44-L103>

<a id="up-cuda-paged-v2"></a>

- `[UP-CUDA-PAGED-V2]` Paged-attention v2 launch path
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/csrc/attention/paged_attention_v2.cu#L36-L123>

## Upstream vllm-spyre `pr-759` anchors

### `pr-759` itself

- PR page
  - <https://github.com/vllm-project/vllm-spyre/pull/759>

<a id="old-pr759-platform"></a>

### Platform control-plane wiring

- `SpyrePlatform.get_total_spyre_blocks`
  - <https://github.com/vllm-project/vllm-spyre/blob/8a9682897aa2bd4d77cf3fdab7acd3fbfe452a72/vllm_spyre/platform.py#L97-L162>

- `SpyrePlatform.check_and_update_config`
  - <https://github.com/vllm-project/vllm-spyre/blob/8a9682897aa2bd4d77cf3fdab7acd3fbfe452a72/vllm_spyre/platform.py#L165-L237>

<a id="old-pr759-scheduler"></a>

### Old-stack scheduler behavior

- `ChunkedPrefillSpyreScheduler`
  - <https://github.com/vllm-project/vllm-spyre/blob/8a9682897aa2bd4d77cf3fdab7acd3fbfe452a72/vllm_spyre/v1/core/scheduler.py#L133-L260>

### Old-stack model-runner limitation

<a id="old-dummy-kvspec"></a>

- `[OLD-DUMMY-KVSPEC]` `get_kv_cache_spec()` dummy FMS-derived spec
  - <https://github.com/vllm-project/vllm-spyre/blob/8a9682897aa2bd4d77cf3fdab7acd3fbfe452a72/vllm_spyre/v1/worker/spyre_model_runner.py#L193-L219>

<a id="old-fms-kv-path"></a>

- `[OLD-FMS-KV-PATH]` `SpyreCausalLM` owning `past_key_value_states` and passing them through the FMS forward path
  - <https://github.com/vllm-project/vllm-spyre/blob/8a9682897aa2bd4d77cf3fdab7acd3fbfe452a72/vllm_spyre/model_executor/model_loader/spyre.py#L140-L142>
  - <https://github.com/vllm-project/vllm-spyre/blob/8a9682897aa2bd4d77cf3fdab7acd3fbfe452a72/vllm_spyre/model_executor/model_loader/spyre.py#L343-L436>

## Current `vllm-spyre` next-stack scaffold anchors

<a id="next-platform"></a>

- `[NEXT-PLATFORM]` `TorchSpyrePlatform` scaffold and explicit placeholder wiring
  - <https://github.com/toddllm/vllm-spyre/blob/5aeb39ec813ff01161ec8f3a832fc86f676238c8/vllm_spyre_next/vllm_spyre_next/platform.py#L28-L82>

<a id="next-example"></a>

- `[NEXT-EXAMPLE]` next-stack example explicitly using upstream CPU path today
  - <https://github.com/toddllm/vllm-spyre/blob/5aeb39ec813ff01161ec8f3a832fc86f676238c8/vllm_spyre_next/examples/torch_spyre_inference.py#L1-L104>

## `torch-spyre` runtime/backend anchors

<a id="ts-entrypoint"></a>

- `[TS-ENTRYPOINT]` `torch.backends` entrypoint registration
  - <https://github.com/toddllm/torch-spyre/blob/6782329492de7558098ec17cd1260d4aab8d93ad/pyproject.toml#L19-L63>

<a id="ts-backend-register"></a>

- `[TS-BACKEND-REGISTER]` PrivateUse1 backend registration and device-module setup
  - <https://github.com/toddllm/torch-spyre/blob/6782329492de7558098ec17cd1260d4aab8d93ad/torch_spyre/__init__.py#L152-L179>

<a id="ts-inductor-autoload"></a>

- `[TS-INDUCTOR-AUTOLOAD]` inductor backend/device registration and patch-heavy integration
  - <https://github.com/toddllm/torch-spyre/blob/6782329492de7558098ec17cd1260d4aab8d93ad/torch_spyre/_inductor/__init__.py#L23-L111>

<a id="ts-fallbacks"></a>

- `[TS-FALLBACKS]` eager CPU fallback registration on Spyre device
  - <https://github.com/toddllm/torch-spyre/blob/6782329492de7558098ec17cd1260d4aab8d93ad/torch_spyre/fallbacks.py#L81-L196>

<a id="ts-preload"></a>

- `[TS-PRELOAD]` Inductor decomposition exclusions tied to fallback pressure
  - <https://github.com/toddllm/torch-spyre/blob/6782329492de7558098ec17cd1260d4aab8d93ad/torch_spyre/_inductor/preload.py#L15-L41>

<a id="ts-copy"></a>

- `[TS-COPY]` copy path comments and current host/device copy implementation
  - <https://github.com/toddllm/torch-spyre/blob/6782329492de7558098ec17cd1260d4aab8d93ad/torch_spyre/csrc/spyre_mem.cpp#L633-L679>

<a id="ts-copy-fixme"></a>

- `[TS-COPY-FIXME]` current Spyre-to-Spyre copy limitation
  - <https://github.com/toddllm/torch-spyre/blob/6782329492de7558098ec17cd1260d4aab8d93ad/torch_spyre/csrc/spyre_mem.cpp#L664-L676>

<a id="ts-hooks-stream"></a>

- `[TS-HOOKS-STREAM]` stream/event hooks still stubbed or placeholder
  - <https://github.com/toddllm/torch-spyre/blob/6782329492de7558098ec17cd1260d4aab8d93ad/torch_spyre/csrc/spyre_hooks.cpp#L117-L206>
  - <https://github.com/toddllm/torch-spyre/blob/6782329492de7558098ec17cd1260d4aab8d93ad/torch_spyre/csrc/spyre_hooks.cpp#L222-L255>

## Experimental Spyre connector anchors

### Branch compare

- combined experimental connector branch vs `pr-759` base
  - <https://github.com/toddllm/vllm-spyre/compare/spyre-kv-base-pr759...spyre-kv-combined>

### Bridge

- `SpyreKVConnectorBridge`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/v1/worker/spyre_kv_connector_bridge.py#L45-L259>

<a id="exp-bridge-begin"></a>

- `[EXP-BRIDGE-BEGIN]` `SpyreKVConnectorBridge.begin_step()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/v1/worker/spyre_kv_connector_bridge.py#L104-L133>

<a id="exp-bridge-before"></a>

- `[EXP-BRIDGE-BEFORE]` `SpyreKVConnectorBridge.before_forward()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/v1/worker/spyre_kv_connector_bridge.py#L135-L162>

<a id="exp-bridge-after"></a>

- `[EXP-BRIDGE-AFTER]` `SpyreKVConnectorBridge.after_forward()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/v1/worker/spyre_kv_connector_bridge.py#L164-L199>

<a id="exp-bridge-finish"></a>

- `[EXP-BRIDGE-FINISH]` `SpyreKVConnectorBridge.finish_step()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/v1/worker/spyre_kv_connector_bridge.py#L201-L218>

- `SpyreKVConnectorBridge.no_forward()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/v1/worker/spyre_kv_connector_bridge.py#L220-L233>

### Metadata contract

- `StoreKey`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/metadata.py#L38-L67>

- `SpyreConnectorRequestMeta`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/metadata.py#L100-L120>

<a id="exp-meta"></a>

- `[EXP-META]` `SpyreConnectorMeta`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/metadata.py#L126-L153>

<a id="exp-meta-validate"></a>

- `[EXP-META-VALIDATE]` `SpyreConnectorMeta.validate()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/metadata.py#L219-L293>

<a id="exp-store"></a>

- `[EXP-STORE]` `InMemoryKVStore`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/metadata.py#L305-L408>

### Connector implementation

- `InMemorySpyreConnector`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py#L220-L959>

### Connector worker-side lifecycle

- `register_kv_caches()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py#L318-L344>

<a id="exp-conn-bind"></a>

- `[EXP-CONN-BIND]` `bind_connector_metadata()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py#L350-L365>

<a id="exp-conn-load"></a>

- `[EXP-CONN-LOAD]` `start_load_kv()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py#L371-L415>

- `_load_layer()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py#L431-L504>

<a id="exp-conn-save"></a>

- `[EXP-CONN-SAVE]` `_save_kv_bulk()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py#L506-L594>

<a id="exp-conn-finish"></a>

- `[EXP-CONN-FINISH]` `get_finished()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py#L596-L620>

<a id="exp-conn-loaderr"></a>

- `[EXP-CONN-LOADERR]` `get_block_ids_with_load_errors()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py#L613-L616>

### Connector scheduler-side lifecycle

<a id="exp-conn-match"></a>

- `[EXP-CONN-MATCH]` `get_num_new_matched_tokens()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py#L622-L705>

<a id="exp-conn-afteralloc"></a>

- `[EXP-CONN-AFTERALLOC]` `update_state_after_alloc()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py#L711-L817>

<a id="exp-conn-buildmeta"></a>

- `[EXP-CONN-BUILDMETA]` `build_connector_meta()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py#L819-L842>

### Metrics

- `SpyreConnectorPromMetrics`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py#L117-L214>

- `build_prom_metrics()`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py#L922-L932>
