"""
Sequential prefill/decode handoff demo for the Spyre KV connector.

This keeps one AIU-owning process at a time while persisting KV state and
saved-request metadata in a long-lived node-local service.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from spyre_kv_reuse_common import (
    build_aligned_reuse_token_prompts,
    build_run_metadata,
    common_prefix_len,
    diff_counts,
    drain_scheduler_stats,
    extract_output_token_count,
    get_demo_prompt_template_names,
    get_worker_probe_state,
    round_down_to_block,
    round_up_to_block,
    set_local_dist_defaults,
)

SPYRE_DEMO_BLOCK_SIZE = 64


def _run_once(llm, prompt_input, sampling_params, label: str) -> dict[str, Any]:
    worker_before, store_before = get_worker_probe_state()
    start_t = time.perf_counter()
    outputs = llm.generate([prompt_input], sampling_params, use_tqdm=False)
    elapsed_s = time.perf_counter() - start_t
    scheduler_stats = drain_scheduler_stats(llm)
    worker_after, store_after = get_worker_probe_state()

    return {
        "label": label,
        "request_elapsed_s": elapsed_s,
        "output_tokens": extract_output_token_count(outputs),
        "scheduler_stats": scheduler_stats,
        "worker_delta": diff_counts(worker_after, worker_before),
        "store_before": store_before,
        "store_after": store_after,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--backend", type=str, default="sendnn")
    parser.add_argument(
        "--service-socket",
        type=str,
        default="/tmp/spyre-kv-persistent.sock",
    )
    parser.add_argument("--store-max-bytes", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--shared-prefix-tokens", type=int, default=128)
    parser.add_argument("--partial-tail-tokens", type=int, default=16)
    parser.add_argument(
        "--demo-template",
        choices=get_demo_prompt_template_names(),
        default="science_fair_invite",
    )
    parser.add_argument(
        "--demo-scenario",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--demo-question",
        type=str,
        default=None,
    )
    parser.add_argument("--demo-partial-tail", type=str, default=None)
    parser.add_argument(
        "--decode-variant",
        choices=("exact", "partial"),
        default="exact",
    )
    parser.add_argument("--clear-service", action="store_true")
    parser.add_argument("--no-assert-reuse", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("VLLM_SPYRE_DYNAMO_BACKEND", args.backend)
    os.environ.setdefault("VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["VLLM_SPYRE_KV_STORE_BACKEND"] = "serialized_shared_memory_service"
    os.environ["VLLM_SPYRE_KV_STORE_MAX_BYTES"] = str(args.store_max_bytes)
    os.environ["VLLM_SPYRE_KV_SERVICE_SOCKET"] = args.service_socket
    set_local_dist_defaults()

    import vllm_spyre

    vllm_spyre.register()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm_spyre.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
        reset_global_store,
    )
    from vllm_spyre.distributed.kv_transfer.kv_connector.v1.persistent_kv_service import (
        PersistentKVServiceClient,
    )

    if args.clear_service:
        client = PersistentKVServiceClient(args.service_socket)
        try:
            client.clear()
        finally:
            client.close()

    reset_global_store("serialized_shared_memory_service")

    planned_aligned_shared_prefix_tokens = round_down_to_block(
        args.shared_prefix_tokens,
        SPYRE_DEMO_BLOCK_SIZE,
    )
    if planned_aligned_shared_prefix_tokens == 0:
        planned_aligned_shared_prefix_tokens = SPYRE_DEMO_BLOCK_SIZE

    planned_partial_prompt_tokens = (
        planned_aligned_shared_prefix_tokens + max(0, args.partial_tail_tokens)
    )
    requested_max_num_batched_tokens = max(
        args.max_num_batched_tokens,
        planned_partial_prompt_tokens,
    )
    effective_max_num_batched_tokens = round_up_to_block(
        requested_max_num_batched_tokens,
        SPYRE_DEMO_BLOCK_SIZE,
    )

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.model,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": effective_max_num_batched_tokens,
        "enable_prefix_caching": False,
        "kv_transfer_config": {
            "kv_connector": "InMemorySpyreConnector",
            "kv_role": "kv_both",
        },
    }
    if args.revision:
        llm_kwargs["revision"] = args.revision
        llm_kwargs["tokenizer_revision"] = args.revision

    llm_init_start_t = time.perf_counter()
    llm = LLM(**llm_kwargs)
    llm_init_elapsed_s = time.perf_counter() - llm_init_start_t

    try:
        tokenizer = llm.get_tokenizer()
        block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)
        if block_size != SPYRE_DEMO_BLOCK_SIZE:
            raise SystemExit(
                f"Unexpected block size {block_size}; sequential handoff demo assumes {SPYRE_DEMO_BLOCK_SIZE}."
            )
        prompt_data = build_aligned_reuse_token_prompts(
            tokenizer,
            requested_shared_prefix_tokens=args.shared_prefix_tokens,
            block_size=block_size,
            partial_tail_tokens=args.partial_tail_tokens,
            template_name=args.demo_template,
            scenario_text=args.demo_scenario,
            question_text=args.demo_question,
            partial_tail_text=args.demo_partial_tail,
        )
        prompt_prefill_tokens = prompt_data["prefill_prompt_token_ids"]
        prompt_partial_tokens = prompt_data["partial_prompt_token_ids"]
        common_prefix_tokens = common_prefix_len(
            prompt_prefill_tokens,
            prompt_partial_tokens,
        )

        _ = drain_scheduler_stats(llm)

        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=0.0,
            ignore_eos=True,
        )

        if args.role == "prefill":
            result = _run_once(
                llm,
                TokensPrompt(prompt_token_ids=prompt_prefill_tokens),
                sampling_params,
                "prefill_store",
            )
        else:
            decode_prompt_tokens = (
                prompt_prefill_tokens
                if args.decode_variant == "exact"
                else prompt_partial_tokens
            )
            result = _run_once(
                llm,
                TokensPrompt(prompt_token_ids=decode_prompt_tokens),
                sampling_params,
                f"{args.decode_variant}_reuse",
            )

        summary = {
            "run_metadata": build_run_metadata(__file__),
            "engine_init_elapsed_s": llm_init_elapsed_s,
            "role": args.role,
            "decode_variant": args.decode_variant if args.role == "decode" else None,
            "model": args.model,
            "revision": args.revision,
            "backend": args.backend,
            "store_backend": "serialized_shared_memory_service",
            "service_socket": args.service_socket,
            "store_max_bytes": args.store_max_bytes,
            "block_size": block_size,
            "requested_max_num_batched_tokens": requested_max_num_batched_tokens,
            "effective_max_num_batched_tokens": effective_max_num_batched_tokens,
            "prefill_prompt_tokens": len(prompt_prefill_tokens),
            "partial_prompt_tokens": len(prompt_partial_tokens),
            "requested_shared_prefix_tokens": prompt_data["requested_shared_prefix_tokens"],
            "aligned_shared_prefix_tokens": prompt_data["aligned_shared_prefix_tokens"],
            "demo_template": prompt_data["demo_template"],
            "demo_template_display_name": prompt_data["demo_template_display_name"],
            "resolved_instruction_text": prompt_data["instruction_text"],
            "resolved_partial_tail_text": prompt_data["partial_tail_text"],
            "common_prefix_tokens": common_prefix_tokens,
            "partial_tail_tokens": prompt_data["partial_tail_tokens"],
            "exact_prompt_recompute_tokens": prompt_data["exact_prompt_recompute_tokens"],
            "partial_prompt_recompute_tokens": prompt_data["partial_prompt_recompute_tokens"],
            "result": result,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))

        if args.role == "prefill":
            if result["worker_delta"].get("blocks_saved", 0) <= 0:
                raise SystemExit(
                    "Sequential handoff prefill failed: request did not save connector entries."
                )
        elif not args.no_assert_reuse:
            loaded = result["worker_delta"].get("blocks_loaded", 0)
            missing = result["worker_delta"].get("blocks_missing", 0)
            if loaded <= 0 or missing > 0:
                raise SystemExit(
                    "Sequential handoff decode failed: request did not reuse connector entries."
                )

        return 0
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
