"""
Shared helpers for single-process Spyre KV reuse probes and benchmarks.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Mapping
from typing import Any


def set_local_dist_defaults() -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    if "MASTER_PORT" in os.environ:
        return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(sock.getsockname()[1])


def build_prompt(tokenizer, min_tokens: int, tail: str) -> str:
    shared = (
        "Spyre KV reuse probe. This prompt is intentionally long so we can "
        "exercise block-aligned prefix reuse through the connector path. "
    )
    prompt = shared
    while len(tokenizer.encode(prompt)) < min_tokens:
        prompt += shared
    return prompt + tail


def common_prefix_len(a: list[int], b: list[int]) -> int:
    matched = 0
    for lhs, rhs in zip(a, b):
        if lhs != rhs:
            break
        matched += 1
    return matched


def get_scheduler_connector(llm):
    engine_core = llm.llm_engine.engine_core.engine_core
    scheduler = engine_core.scheduler
    return getattr(scheduler, "connector", None)


def drain_scheduler_stats(llm) -> dict[str, int]:
    connector = get_scheduler_connector(llm)
    if connector is None:
        return {}

    stats = connector.get_kv_connector_stats()
    if stats is None:
        return {}

    if hasattr(stats, "reduce"):
        reduced = stats.reduce()
        return {str(k): int(v) for k, v in reduced.items()}

    data = getattr(stats, "data", None)
    if isinstance(data, Mapping):
        return {str(k): int(v) for k, v in data.items()}

    return {}


def get_worker_connector():
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
    )

    if not has_kv_transfer_group():
        return None

    return get_kv_transfer_group()


def get_worker_probe_state() -> tuple[dict[str, int], dict[str, Any]]:
    connector = get_worker_connector()
    if connector is None:
        return {}, {}

    cumulative = {}
    if hasattr(connector, "get_cumulative_metrics"):
        cumulative = {
            str(k): int(v)
            for k, v in connector.get_cumulative_metrics().items()
        }

    store_stats = {}
    if hasattr(connector, "get_store"):
        store_stats = dict(connector.get_store().stats())

    return cumulative, store_stats


def diff_counts(after: Mapping[str, int], before: Mapping[str, int]) -> dict[str, int]:
    keys = set(before) | set(after)
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in sorted(keys)}


def reset_probe_state(llm) -> None:
    seen_ids: set[int] = set()
    for connector in (get_scheduler_connector(llm), get_worker_connector()):
        if connector is None:
            continue

        connector_id = id(connector)
        if connector_id in seen_ids:
            continue
        seen_ids.add(connector_id)

        reset = getattr(connector, "reset_probe_state", None)
        if callable(reset):
            reset()

    # Flush any already-snapshotted scheduler stats between scenarios.
    drain_scheduler_stats(llm)


def extract_output_token_count(outputs: list[Any]) -> int:
    if not outputs:
        return 0

    request_output = outputs[0]
    candidates = getattr(request_output, "outputs", None)
    if not candidates:
        return 0

    token_ids = getattr(candidates[0], "token_ids", None)
    if token_ids is None:
        return 0

    return len(token_ids)
