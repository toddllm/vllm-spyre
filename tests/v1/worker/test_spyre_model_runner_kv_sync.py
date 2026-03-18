import types

import torch

from vllm_spyre.v1.worker.spyre_model_runner import ChunkedPrefillModelRunner


def test_load_sync_uses_current_model_pair_when_registered_pair_is_stale():
    runner = object.__new__(ChunkedPrefillModelRunner)
    layer_name = "model.layers.0.self_attn"

    registered_k = torch.zeros((2, 2), dtype=torch.float32)
    registered_v = torch.zeros((2, 2), dtype=torch.float32)
    current_k = torch.zeros((2, 2), dtype=torch.float32)
    current_v = torch.zeros((2, 2), dtype=torch.float32)
    staging = torch.stack(
        [
            torch.full((2, 2), 3.0, dtype=torch.float32),
            torch.full((2, 2), 7.0, dtype=torch.float32),
        ]
    )

    runner._connector_kv_staging = {layer_name: staging}
    runner._connector_kv_pairs = {layer_name: (registered_k, registered_v)}
    runner._model = types.SimpleNamespace(
        past_key_value_states=[(current_k, current_v)]
    )

    runner._sync_loaded_kv_from_staging()

    assert torch.equal(current_k, staging[0])
    assert torch.equal(current_v, staging[1])
    assert torch.equal(registered_k, torch.zeros_like(registered_k))
    assert torch.equal(registered_v, torch.zeros_like(registered_v))
    assert runner._connector_kv_pairs[layer_name] == (current_k, current_v)


def test_save_sync_uses_current_model_pair_when_registered_pair_is_stale():
    runner = object.__new__(ChunkedPrefillModelRunner)
    layer_name = "model.layers.0.self_attn"

    registered_k = torch.zeros((2, 2), dtype=torch.float32)
    registered_v = torch.zeros((2, 2), dtype=torch.float32)
    current_k = torch.full((2, 2), 5.0, dtype=torch.float32)
    current_v = torch.full((2, 2), 9.0, dtype=torch.float32)
    staging = torch.stack(
        [
            torch.zeros((2, 2), dtype=torch.float32),
            torch.zeros((2, 2), dtype=torch.float32),
        ]
    )

    runner._connector_kv_staging = {layer_name: staging}
    runner._connector_kv_pairs = {layer_name: (registered_k, registered_v)}
    runner._model = types.SimpleNamespace(
        past_key_value_states=[(current_k, current_v)]
    )

    runner._sync_fms_kv_to_staging()

    assert torch.equal(staging[0], current_k)
    assert torch.equal(staging[1], current_v)
    assert torch.equal(registered_k, torch.zeros_like(registered_k))
    assert torch.equal(registered_v, torch.zeros_like(registered_v))
    assert runner._connector_kv_pairs[layer_name] == (current_k, current_v)
