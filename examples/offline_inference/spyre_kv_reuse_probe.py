"""
Probe Spyre KV connector reuse in a single-process offline LLM run.

This is intentionally narrow:
- `InMemorySpyreConnector`
- configurable store backend under the connector
- `kv_both`
- single-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`)
- built-in prefix caching disabled, so reuse comes from the connector path
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from spyre_kv_reuse_common import (
    build_run_metadata,
    build_prompt,
    common_prefix_len,
    diff_counts,
    drain_scheduler_stats,
    get_worker_probe_state,
    set_local_dist_defaults,
)


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
    os.environ["VLLM_SPYRE_KV_STORE_BACKEND"] = args.store_backend
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
        prompt_exact = build_prompt(
            tokenizer,
            args.shared_prefix_tokens,
            tail="\n\nQuestion: Summarize the long prefix in one sentence.",
        )
        prompt_partial = build_prompt(
            tokenizer,
            args.shared_prefix_tokens,
            tail="\n\nQuestion: List three keywords from the long prefix.",
        )

        prompt_exact_tokens = tokenizer.encode(prompt_exact)
        prompt_partial_tokens = tokenizer.encode(prompt_partial)
        common_prefix_tokens = common_prefix_len(prompt_exact_tokens, prompt_partial_tokens)
        block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)

        # Discard any setup-time stats so the probe only reflects request traffic.
        _ = drain_scheduler_stats(llm)

        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=0.0,
            ignore_eos=True,
        )

        warm = _run_once(llm, prompt_exact, sampling_params, "warm_store")
        exact = _run_once(llm, prompt_exact, sampling_params, "exact_reuse")
        partial = _run_once(llm, prompt_partial, sampling_params, "partial_reuse")

        summary = {
            "run_metadata": build_run_metadata(__file__),
            "model": args.model,
            "revision": args.revision,
            "backend": args.backend,
            "store_backend": args.store_backend,
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
