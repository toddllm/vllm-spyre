"""
Probe Spyre KV connector reuse in a single-process offline LLM run.

This is intentionally narrow:
- `InMemorySpyreConnector`
- `kv_both`
- single-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`)
- built-in prefix caching disabled, so reuse comes from the connector path
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from collections.abc import Mapping
from typing import Any


def _set_local_dist_defaults() -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    if "MASTER_PORT" in os.environ:
        return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(sock.getsockname()[1])


def _build_prompt(tokenizer, min_tokens: int, tail: str) -> str:
    shared = (
        "Spyre KV reuse probe. This prompt is intentionally long so we can "
        "exercise block-aligned prefix reuse through the connector path. "
    )
    prompt = shared
    while len(tokenizer.encode(prompt)) < min_tokens:
        prompt += shared
    return prompt + tail


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    matched = 0
    for lhs, rhs in zip(a, b):
        if lhs != rhs:
            break
        matched += 1
    return matched


def _get_scheduler_connector(llm):
    engine_core = llm.llm_engine.engine_core.engine_core
    scheduler = engine_core.scheduler
    connector = getattr(scheduler, "connector", None)
    return connector


def _drain_scheduler_stats(llm) -> dict[str, int]:
    connector = _get_scheduler_connector(llm)
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


def _get_worker_probe_state() -> tuple[dict[str, int], dict[str, Any]]:
    from vllm.distributed.kv_transfer import (
        get_kv_transfer_group,
        has_kv_transfer_group,
    )

    if not has_kv_transfer_group():
        return {}, {}

    connector = get_kv_transfer_group()

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


def _diff_counts(after: Mapping[str, int], before: Mapping[str, int]) -> dict[str, int]:
    keys = set(before) | set(after)
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in sorted(keys)}


def _run_once(llm, prompt: str, sampling_params, label: str) -> dict[str, Any]:
    worker_before, store_before = _get_worker_probe_state()
    _ = llm.generate([prompt], sampling_params, use_tqdm=False)
    scheduler_stats = _drain_scheduler_stats(llm)
    worker_after, store_after = _get_worker_probe_state()

    return {
        "label": label,
        "scheduler_stats": scheduler_stats,
        "worker_delta": _diff_counts(worker_after, worker_before),
        "store_before": store_before,
        "store_after": store_after,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--backend", type=str, default="sendnn")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--shared-prefix-tokens", type=int, default=192)
    parser.add_argument("--no-assert-reuse", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("VLLM_SPYRE_DYNAMO_BACKEND", args.backend)
    os.environ.setdefault("VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _set_local_dist_defaults()

    import vllm_spyre

    vllm_spyre.register()

    from vllm import LLM, SamplingParams
    from vllm_spyre.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
        reset_global_store,
    )

    reset_global_store()

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.model,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        # Disable built-in prefix caching so reuse comes through the connector.
        "enable_prefix_caching": False,
        "kv_transfer_config": {
            "kv_connector": "InMemorySpyreConnector",
            "kv_role": "kv_both",
        },
    }
    if args.revision:
        llm_kwargs["revision"] = args.revision
        llm_kwargs["tokenizer_revision"] = args.revision

    llm = LLM(**llm_kwargs)

    try:
        tokenizer = llm.get_tokenizer()
        prompt_exact = _build_prompt(
            tokenizer,
            args.shared_prefix_tokens,
            tail="\n\nQuestion: Summarize the long prefix in one sentence.",
        )
        prompt_partial = _build_prompt(
            tokenizer,
            args.shared_prefix_tokens,
            tail="\n\nQuestion: List three keywords from the long prefix.",
        )

        prompt_exact_tokens = tokenizer.encode(prompt_exact)
        prompt_partial_tokens = tokenizer.encode(prompt_partial)
        common_prefix_tokens = _common_prefix_len(
            prompt_exact_tokens, prompt_partial_tokens
        )
        block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)

        # Discard any setup-time stats so the probe only reflects request traffic.
        _ = _drain_scheduler_stats(llm)

        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=0.0,
            ignore_eos=True,
        )

        warm = _run_once(llm, prompt_exact, sampling_params, "warm_store")
        exact = _run_once(llm, prompt_exact, sampling_params, "exact_reuse")
        partial = _run_once(llm, prompt_partial, sampling_params, "partial_reuse")

        summary = {
            "model": args.model,
            "revision": args.revision,
            "backend": args.backend,
            "block_size": block_size,
            "prompt_exact_tokens": len(prompt_exact_tokens),
            "prompt_partial_tokens": len(prompt_partial_tokens),
            "common_prefix_tokens": common_prefix_tokens,
            "warm_store": warm,
            "exact_reuse": exact,
            "partial_reuse": partial,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))

        if not args.no_assert_reuse:
            exact_loaded = exact["worker_delta"].get("blocks_loaded", 0)
            exact_missing = exact["worker_delta"].get("blocks_missing", 0)
            partial_loaded = partial["worker_delta"].get("blocks_loaded", 0)
            partial_missing = partial["worker_delta"].get("blocks_missing", 0)

            if warm["worker_delta"].get("blocks_saved", 0) <= 0:
                raise SystemExit(
                    "Probe failed: warm request did not save any connector entries."
                )
            if exact_loaded <= 0 or exact_missing > 0:
                raise SystemExit(
                    "Probe failed: exact replay did not show connector reuse."
                )
            if common_prefix_tokens >= block_size and (
                partial_loaded <= 0 or partial_missing > 0
            ):
                raise SystemExit(
                    "Probe failed: partial-prefix replay did not show connector reuse."
                )

        return 0
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
