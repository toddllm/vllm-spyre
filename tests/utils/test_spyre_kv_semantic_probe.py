import json

import torch

import vllm_spyre.envs as envs_spyre
from vllm_spyre.v1.worker import spyre_kv_semantic_probe as probe


def test_parse_layer_idx_handles_expected_and_unexpected_names():
    assert probe.parse_layer_idx("model.layers.0.self_attn") == 0
    assert probe.parse_layer_idx("model.layers.17.self_attn") == 17
    assert probe.parse_layer_idx("not-a-layer-name") is None


def test_should_probe_layer_uses_env_gate(monkeypatch):
    monkeypatch.setenv("VLLM_SPYRE_KV_SEMANTIC_PROBE_ENABLED", "1")
    monkeypatch.setenv("VLLM_SPYRE_KV_SEMANTIC_PROBE_LAYER", "3")
    envs_spyre.clear_env_cache()

    assert probe.should_probe_layer(3) is True
    assert probe.should_probe_layer(2) is False


def test_emit_tensor_probe_logs_structured_payload(monkeypatch):
    monkeypatch.setenv("VLLM_SPYRE_KV_SEMANTIC_PROBE_ENABLED", "1")
    monkeypatch.setenv("VLLM_SPYRE_KV_SEMANTIC_PROBE_LAYER", "0")
    monkeypatch.setenv("VLLM_SPYRE_KV_SEMANTIC_PROBE_BLOCK", "0")
    monkeypatch.setenv("VLLM_SPYRE_KV_SEMANTIC_PROBE_HEAD", "0")
    monkeypatch.setenv("VLLM_SPYRE_KV_SEMANTIC_PROBE_ELEMS", "4")
    envs_spyre.clear_env_cache()

    records = []

    def _capture(fmt, prefix, payload):
        records.append((prefix, payload))

    monkeypatch.setattr(probe.logger, "info", _capture)

    tensor = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    compare = tensor.clone()
    probe.emit_tensor_probe(
        phase="register",
        layer_idx=0,
        layer_name="model.layers.0.self_attn",
        tensor_role="registered_k",
        tensor=tensor,
        compare_to=compare,
    )

    assert len(records) == 1
    prefix, payload_text = records[0]
    assert prefix == "[KV_SEMANTIC_PROBE]"
    payload = json.loads(payload_text)
    assert payload["phase"] == "register"
    assert payload["layer_idx"] == 0
    assert payload["tensor_role"] == "registered_k"
    assert payload["shape"] == [2, 3, 2, 2]
    assert payload["sample_values"] == [0.0, 1.0, 4.0, 5.0]
    assert payload["same_object"] is False
    assert isinstance(payload["same_data_ptr"], bool)
