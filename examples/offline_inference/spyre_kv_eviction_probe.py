"""
Probe request-granular KV store eviction under the current Spyre connector path.

This intentionally stays narrow:
- `InMemorySpyreConnector`
- configurable store backend under the connector
- request-granular store pressure via `VLLM_SPYRE_KV_STORE_MAX_BYTES`
- single-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`)
- built-in prefix caching disabled, so any reuse comes from the connector path
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from spyre_kv_reuse_common import (
    build_run_metadata,
    diff_counts,
    drain_scheduler_stats,
    get_worker_probe_state,
    set_local_dist_defaults,
)


def _build_distinct_prompt(tokenizer, min_tokens: int, label: str, tail: str) -> str:
    base = (
        f"Spyre KV eviction probe for request family {label}. "
        f"This prompt intentionally repeats a distinct family marker for {label}. "
    )
    prompt = base
    while len(tokenizer.encode(prompt)) < min_tokens:
        prompt += base
    return prompt + tail


def _run_once(llm, prompt: str, sampling_params, label: str) -> dict[str, Any]:
    worker_before, store_before = get_worker_probe_state()
    _ = llm.generate([prompt], sampling_params, use_tqdm=False)
    scheduler_stats = drain_scheduler_stats(llm)
    worker_after, store_after = get_worker_probe_state()
    return {
        "label": label,
        "scheduler_stats": scheduler_stats,
        "worker_delta": diff_counts(worker_after, worker_before),
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
    parser.add_argument("--store-backend", type=str, default="host_memory")
    parser.add_argument("--store-max-bytes", type=int, default=4_500_000)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt-min-tokens", type=int, default=192)
    parser.add_argument("--no-assert-eviction", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("VLLM_SPYRE_DYNAMO_BACKEND", args.backend)
    os.environ.setdefault("VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["VLLM_SPYRE_KV_STORE_BACKEND"] = args.store_backend
    os.environ["VLLM_SPYRE_KV_STORE_MAX_BYTES"] = str(args.store_max_bytes)
    set_local_dist_defaults()

    import vllm_spyre

    vllm_spyre.register()

    from vllm import LLM, SamplingParams
    from vllm_spyre.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
        reset_global_store,
    )

    reset_global_store(args.store_backend)

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.model,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
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
        prompt_a = _build_distinct_prompt(
            tokenizer,
            args.prompt_min_tokens,
            label="alpha",
            tail="\n\nQuestion: Summarize the alpha family prompt in one sentence.",
        )
        prompt_b = _build_distinct_prompt(
            tokenizer,
            args.prompt_min_tokens,
            label="beta",
            tail="\n\nQuestion: Summarize the beta family prompt in one sentence.",
        )

        prompt_a_tokens = tokenizer.encode(prompt_a)
        prompt_b_tokens = tokenizer.encode(prompt_b)
        block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)

        _ = drain_scheduler_stats(llm)

        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=0.0,
            ignore_eos=True,
        )

        warm_a = _run_once(llm, prompt_a, sampling_params, "warm_a")
        warm_b = _run_once(llm, prompt_b, sampling_params, "warm_b")
        replay_a = _run_once(llm, prompt_a, sampling_params, "replay_a_after_eviction")

        summary = {
            "run_metadata": build_run_metadata(__file__),
            "model": args.model,
            "revision": args.revision,
            "backend": args.backend,
            "store_backend": args.store_backend,
            "store_max_bytes": args.store_max_bytes,
            "block_size": block_size,
            "prompt_a_tokens": len(prompt_a_tokens),
            "prompt_b_tokens": len(prompt_b_tokens),
            "warm_a": warm_a,
            "warm_b": warm_b,
            "replay_a_after_eviction": replay_a,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))

        if not args.no_assert_eviction:
            if warm_a["worker_delta"].get("blocks_saved", 0) <= 0:
                raise SystemExit("Eviction probe failed: warm_a did not save any connector entries.")
            if warm_b["worker_delta"].get("blocks_saved", 0) <= 0:
                raise SystemExit("Eviction probe failed: warm_b did not save any connector entries.")
            if warm_b["store_after"].get("evictions", 0) <= warm_a["store_after"].get("evictions", 0):
                raise SystemExit("Eviction probe failed: warm_b did not trigger store eviction.")
            if replay_a["worker_delta"].get("blocks_loaded", 0) != 0:
                raise SystemExit("Eviction probe failed: replay_a unexpectedly loaded evicted blocks.")
            if replay_a["worker_delta"].get("blocks_missing", 0) != 0:
                raise SystemExit("Eviction probe failed: replay_a hit load misses instead of clean fallback.")
            if replay_a["worker_delta"].get("blocks_saved", 0) <= 0:
                raise SystemExit("Eviction probe failed: replay_a did not fall back to saving the request again.")

        return 0
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
