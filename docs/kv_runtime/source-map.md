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

## Upstream vLLM anchors

### Scheduler and metadata handoff

- `SchedulerOutput.kv_connector_metadata`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/core/sched/output.py#L212-L239>

### Connector v1 contract

- `KVConnectorBase_V1` declaration and worker-side methods
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

- `attention.layer` KV cache update using `slot_mapping`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/attention/layer.py#L828-L849>

### GPU-side active connector reference

- `ActiveKVConnector`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/worker/gpu/kv_connector.py#L48-L125>

### KV layout / stride reference

- `FlashAttentionBackend.get_kv_cache_stride_order`
  - <https://github.com/vllm-project/vllm/blob/1892993bc18e243e2c05841314c5e9c06a80c70d/vllm/v1/attention/backends/flash_attn.py#L119-L137>

## Upstream vllm-spyre `pr-759` anchors

### `pr-759` itself

- PR page
  - <https://github.com/vllm-project/vllm-spyre/pull/759>

### Platform control-plane wiring

- `SpyrePlatform.get_total_spyre_blocks`
  - <https://github.com/vllm-project/vllm-spyre/blob/8a9682897aa2bd4d77cf3fdab7acd3fbfe452a72/vllm_spyre/platform.py#L97-L162>

- `SpyrePlatform.check_and_update_config`
  - <https://github.com/vllm-project/vllm-spyre/blob/8a9682897aa2bd4d77cf3fdab7acd3fbfe452a72/vllm_spyre/platform.py#L165-L237>

### Old-stack scheduler behavior

- `ChunkedPrefillSpyreScheduler`
  - <https://github.com/vllm-project/vllm-spyre/blob/8a9682897aa2bd4d77cf3fdab7acd3fbfe452a72/vllm_spyre/v1/core/scheduler.py#L133-L260>

### Old-stack model-runner limitation

- `get_kv_cache_spec()` dummy FMS-derived spec
  - <https://github.com/vllm-project/vllm-spyre/blob/8a9682897aa2bd4d77cf3fdab7acd3fbfe452a72/vllm_spyre/v1/worker/spyre_model_runner.py#L193-L219>

## Experimental Spyre connector anchors

### Branch compare

- combined experimental connector branch vs `pr-759` base
  - <https://github.com/toddllm/vllm-spyre/compare/spyre-kv-base-pr759...spyre-kv-combined>

### Bridge

- `SpyreKVConnectorBridge`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/v1/worker/spyre_kv_connector_bridge.py#L45-L259>

### Metadata contract

- `StoreKey`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/metadata.py#L38-L67>

- `SpyreConnectorMeta`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/metadata.py#L126-L153>

### Connector implementation

- `InMemorySpyreConnector`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py#L220-L959>

### Metrics

- `SpyreConnectorPromMetrics`
  - <https://github.com/toddllm/vllm-spyre/blob/3448cf95710f69e0f13fac000ab170670ad5268d/vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py#L117-L214>
