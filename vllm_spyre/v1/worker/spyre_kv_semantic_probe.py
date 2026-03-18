from __future__ import annotations

import json
from typing import Any

import torch
from vllm.logger import init_logger

import vllm_spyre.envs as envs_spyre

logger = init_logger(__name__)

_PROBE_PREFIX = "[KV_SEMANTIC_PROBE]"


def probe_enabled() -> bool:
    return bool(envs_spyre.VLLM_SPYRE_KV_SEMANTIC_PROBE_ENABLED)


def should_probe_layer(layer_idx: int) -> bool:
    return probe_enabled() and layer_idx == int(envs_spyre.VLLM_SPYRE_KV_SEMANTIC_PROBE_LAYER)


def parse_layer_idx(layer_name: str) -> int | None:
    prefix = "model.layers."
    suffix = ".self_attn"
    if not layer_name.startswith(prefix) or not layer_name.endswith(suffix):
        return None
    try:
        return int(layer_name[len(prefix) : -len(suffix)])
    except ValueError:
        return None


def _safe_data_ptr(tensor: Any) -> int | None:
    try:
        return int(tensor.data_ptr())
    except Exception:
        return None


def _sample_tensor(tensor: Any) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        return {
            "sample_error": f"unsupported_tensor_type:{type(tensor).__name__}",
            "sample_values": [],
            "sample_checksum": None,
        }

    try:
        sample_elems = max(1, int(envs_spyre.VLLM_SPYRE_KV_SEMANTIC_PROBE_ELEMS))
        block_idx = max(0, int(envs_spyre.VLLM_SPYRE_KV_SEMANTIC_PROBE_BLOCK))
        head_idx = max(0, int(envs_spyre.VLLM_SPYRE_KV_SEMANTIC_PROBE_HEAD))

        view = tensor.detach()
        if view.ndim >= 4:
            block_idx = min(block_idx, int(view.shape[0]) - 1)
            head_idx = min(head_idx, int(view.shape[2]) - 1)
            flat = view[block_idx, :, head_idx, :].reshape(-1)
        else:
            flat = view.reshape(-1)

        if flat.numel() == 0:
            return {
                "sample_values": [],
                "sample_checksum": 0.0,
            }

        sample = flat[:sample_elems].cpu().to(dtype=torch.float32)
        return {
            "sample_values": [round(float(v), 6) for v in sample.tolist()],
            "sample_checksum": round(float(sample.abs().sum().item()), 6),
        }
    except Exception as exc:
        return {
            "sample_error": f"{type(exc).__name__}:{exc}",
            "sample_values": [],
            "sample_checksum": None,
        }


def _tensor_payload(tensor: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "object_id": id(tensor),
        "tensor_type": type(tensor).__name__,
        "data_ptr": _safe_data_ptr(tensor),
    }
    if isinstance(tensor, torch.Tensor):
        payload.update(
            {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }
        )
    else:
        payload.update(
            {
                "shape": None,
                "dtype": None,
                "device": None,
            }
        )
    payload.update(_sample_tensor(tensor))
    return payload


def emit_tensor_probe(
    *,
    phase: str,
    layer_idx: int,
    layer_name: str,
    tensor_role: str,
    tensor: Any,
    compare_to: Any | None = None,
    note: str | None = None,
) -> None:
    if not should_probe_layer(layer_idx):
        return

    payload: dict[str, Any] = {
        "phase": phase,
        "layer_idx": layer_idx,
        "layer_name": layer_name,
        "tensor_role": tensor_role,
    }
    if note is not None:
        payload["note"] = note

    payload.update(_tensor_payload(tensor))

    if compare_to is not None:
        payload["compare_object_id"] = id(compare_to)
        payload["compare_data_ptr"] = _safe_data_ptr(compare_to)
        payload["same_object"] = id(tensor) == id(compare_to)
        payload["same_data_ptr"] = payload["data_ptr"] == payload["compare_data_ptr"]

    logger.info("%s %s", _PROBE_PREFIX, json.dumps(payload, sort_keys=True))

