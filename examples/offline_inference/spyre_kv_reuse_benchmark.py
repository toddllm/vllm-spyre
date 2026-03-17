"""
Benchmark cold vs connector-backed KV reuse for a single-process Spyre LLM.

This intentionally stays narrow:
- `InMemorySpyreConnector`
- configurable store backend under the connector
- `kv_both`
- single-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`)
- built-in prefix caching disabled, so reuse comes from the connector path
- request latency, not TTFT, because the offline API does not expose TTFT directly
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any

from spyre_kv_reuse_common import (
    build_aligned_reuse_token_prompts,
    build_run_metadata,
    build_prompt,
    common_prefix_len,
    diff_counts,
    drain_scheduler_stats,
    extract_output_token_count,
    get_worker_probe_state,
    reset_probe_state,
    set_local_dist_defaults,
)


def _clear_store_backend(store_backend: str, service_socket: str | None) -> None:
    from vllm_spyre.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
        reset_global_store,
    )
    from vllm_spyre.distributed.kv_transfer.kv_connector.v1.persistent_kv_service import (
        PersistentKVServiceClient,
    )

    if service_socket:
        client = PersistentKVServiceClient(service_socket)
        try:
            client.clear()
        finally:
            client.close()

    reset_global_store(store_backend)


def _prompt_token_count(prompt_input: Any, tokenizer) -> int:
    if hasattr(prompt_input, "get"):
        prompt_token_ids = prompt_input.get("prompt_token_ids")
        if prompt_token_ids is not None:
            return len(prompt_token_ids)

    if isinstance(prompt_input, str):
        return len(tokenizer.encode(prompt_input))

    raise TypeError(f"Unsupported prompt input type: {type(prompt_input)!r}")


def _effective_max_num_batched_tokens(
    *,
    requested_tokens: int,
    required_tokens: int,
    block_size: int,
    aligned_prompts: bool,
) -> int:
    effective = max(requested_tokens, required_tokens)
    if aligned_prompts:
        return ((effective + block_size - 1) // block_size) * block_size
    return effective


def _format_live_result_line(
    *,
    stage: str,
    turn_index: int,
    total_turns: int,
    run: dict[str, Any],
    baseline_latency_seconds: float | None = None,
) -> str:
    worker_delta = run["worker_delta"]
    parts = [
        f"{stage}[{turn_index}/{total_turns}]",
        f"latency_s={float(run['latency_seconds']):.6f}",
        f"output_tokens={int(run['output_tokens'])}",
        f"blocks_loaded={int(worker_delta.get('blocks_loaded', 0))}",
        f"blocks_missing={int(worker_delta.get('blocks_missing', 0))}",
        f"blocks_saved={int(worker_delta.get('blocks_saved', 0))}",
    ]
    if baseline_latency_seconds is not None and float(run["latency_seconds"]) > 0:
        saved_ms = 1000.0 * (baseline_latency_seconds - float(run["latency_seconds"]))
        parts.append(f"saved_ms={saved_ms:.3f}")
        parts.append(
            "speedup_vs_baseline="
            f"{(baseline_latency_seconds / float(run['latency_seconds'])):.3f}x"
        )
    return " ".join(parts)


def _run_timed_request(llm, prompt_input: Any, sampling_params, label: str) -> dict[str, Any]:
    worker_before, store_before = get_worker_probe_state()
    started = time.perf_counter()
    outputs = llm.generate([prompt_input], sampling_params, use_tqdm=False)
    latency_seconds = time.perf_counter() - started
    scheduler_stats = drain_scheduler_stats(llm)
    worker_after, store_after = get_worker_probe_state()
    output_tokens = extract_output_token_count(outputs)

    return {
        "label": label,
        "latency_seconds": latency_seconds,
        "output_tokens": output_tokens,
        "tokens_per_second": (
            output_tokens / latency_seconds if latency_seconds > 0 else 0.0
        ),
        "scheduler_stats": scheduler_stats,
        "worker_delta": diff_counts(worker_after, worker_before),
        "store_before": store_before,
        "store_after": store_after,
    }


def _metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    latency_values = [float(run["latency_seconds"]) for run in runs]
    throughput_values = [float(run["tokens_per_second"]) for run in runs]
    loaded_values = [int(run["worker_delta"].get("blocks_loaded", 0)) for run in runs]
    missing_values = [int(run["worker_delta"].get("blocks_missing", 0)) for run in runs]
    saved_values = [int(run["worker_delta"].get("blocks_saved", 0)) for run in runs]

    return {
        "runs": runs,
        "latency_seconds": _metric_summary(latency_values),
        "tokens_per_second": _metric_summary(throughput_values),
        "output_tokens": _metric_summary([float(run["output_tokens"]) for run in runs]),
        "blocks_loaded": _metric_summary([float(value) for value in loaded_values]),
        "blocks_missing": _metric_summary([float(value) for value in missing_values]),
        "blocks_saved": _metric_summary([float(value) for value in saved_values]),
    }


def _compare_latency(cold_runs: list[dict[str, Any]], reuse_runs: list[dict[str, Any]]) -> dict[str, float]:
    cold_mean = statistics.fmean(float(run["latency_seconds"]) for run in cold_runs)
    reuse_mean = statistics.fmean(float(run["latency_seconds"]) for run in reuse_runs)
    reduction = cold_mean - reuse_mean
    return {
        "cold_mean_seconds": cold_mean,
        "reuse_mean_seconds": reuse_mean,
        "latency_delta_seconds": reduction,
        "latency_reduction_percent": (100.0 * reduction / cold_mean) if cold_mean > 0 else 0.0,
        "speedup": (cold_mean / reuse_mean) if reuse_mean > 0 else 0.0,
    }


def _validate_run(run: dict[str, Any], *, expect_load: bool, expect_save: bool, label: str) -> None:
    loaded = int(run["worker_delta"].get("blocks_loaded", 0))
    missing = int(run["worker_delta"].get("blocks_missing", 0))
    saved = int(run["worker_delta"].get("blocks_saved", 0))

    if expect_load:
        if loaded <= 0 or missing > 0:
            raise SystemExit(f"Benchmark failed: {label} did not show clean connector reuse.")
    else:
        if loaded != 0 or missing != 0:
            raise SystemExit(f"Benchmark failed: {label} unexpectedly loaded connector entries.")

    if expect_save and saved <= 0:
        raise SystemExit(f"Benchmark failed: {label} did not save any connector entries.")


def _run_exact_live_demo(
    *,
    llm,
    prompt_exact_input: Any,
    sampling_params,
    store_backend: str,
    service_socket: str | None,
    warmup_runs: int,
    reuse_turns: int,
    print_live: bool,
) -> dict[str, Any]:
    warmup_runs_data: list[dict[str, Any]] = []
    for warmup_idx in range(warmup_runs):
        _clear_store_backend(store_backend, service_socket)
        reset_probe_state(llm)
        warmup_run = _run_timed_request(
            llm,
            prompt_exact_input,
            sampling_params,
            "exact_warmup",
        )
        warmup_runs_data.append(warmup_run)
        if print_live:
            print(
                _format_live_result_line(
                    stage="exact_warmup",
                    turn_index=warmup_idx + 1,
                    total_turns=warmup_runs,
                    run=warmup_run,
                )
            )

    _clear_store_backend(store_backend, service_socket)
    reset_probe_state(llm)
    warm_baseline_run = _run_timed_request(
        llm,
        prompt_exact_input,
        sampling_params,
        "exact_warm_baseline",
    )
    if print_live:
        print(
            _format_live_result_line(
                stage="exact_warm_baseline",
                turn_index=1,
                total_turns=1,
                run=warm_baseline_run,
            )
        )

    exact_reuse_runs: list[dict[str, Any]] = []
    baseline_latency_seconds = float(warm_baseline_run["latency_seconds"])
    for reuse_idx in range(reuse_turns):
        reset_probe_state(llm)
        exact_reuse = _run_timed_request(
            llm,
            prompt_exact_input,
            sampling_params,
            "exact_reuse",
        )
        exact_reuse_runs.append(exact_reuse)
        if print_live:
            print(
                _format_live_result_line(
                    stage="exact_reuse",
                    turn_index=reuse_idx + 1,
                    total_turns=reuse_turns,
                    run=exact_reuse,
                    baseline_latency_seconds=baseline_latency_seconds,
                )
            )

    return {
        "warmup_runs": warmup_runs_data,
        "warm_baseline": warm_baseline_run,
        "reuse_runs": exact_reuse_runs,
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
    parser.add_argument("--service-socket", type=str, default=None)
    parser.add_argument("--store-max-bytes", type=int, default=0)
    parser.add_argument("--clear-service", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--shared-prefix-tokens", type=int, default=192)
    parser.add_argument("--partial-tail-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--demo-mode",
        choices=("paired_benchmark", "warm_baseline_then_reuse"),
        default="paired_benchmark",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--aligned-prompts", action="store_true")
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--print-live", action="store_true")
    parser.add_argument("--no-assert-reuse", action="store_true")
    args = parser.parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be >= 1")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be >= 0")

    os.environ.setdefault("VLLM_SPYRE_DYNAMO_BACKEND", args.backend)
    os.environ.setdefault("VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["VLLM_SPYRE_KV_STORE_BACKEND"] = args.store_backend
    os.environ["VLLM_SPYRE_KV_STORE_MAX_BYTES"] = str(args.store_max_bytes)
    if args.service_socket:
        os.environ["VLLM_SPYRE_KV_SERVICE_SOCKET"] = args.service_socket
    set_local_dist_defaults()

    import vllm_spyre

    vllm_spyre.register()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    if args.clear_service or args.demo_mode == "warm_baseline_then_reuse":
        _clear_store_backend(args.store_backend, args.service_socket)
    else:
        _clear_store_backend(args.store_backend, None)

    tokenizer_probe_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.model,
    }
    if args.revision:
        tokenizer_probe_kwargs["revision"] = args.revision
        tokenizer_probe_kwargs["tokenizer_revision"] = args.revision

    probe_llm = LLM(
        **{
            **tokenizer_probe_kwargs,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_prefix_caching": False,
            "kv_transfer_config": {
                "kv_connector": "InMemorySpyreConnector",
                "kv_role": "kv_both",
            },
        }
    )
    try:
        tokenizer = probe_llm.get_tokenizer()
        probe_block_size = int(probe_llm.llm_engine.vllm_config.cache_config.block_size)
        exact_prompt_recompute_tokens = None
        partial_prompt_recompute_tokens = None
        requested_shared_prefix_tokens = args.shared_prefix_tokens

        if args.aligned_prompts:
            prompt_data = build_aligned_reuse_token_prompts(
                tokenizer,
                requested_shared_prefix_tokens=args.shared_prefix_tokens,
                block_size=probe_block_size,
                partial_tail_tokens=args.partial_tail_tokens,
            )
            prompt_exact_tokens = prompt_data["prefill_prompt_token_ids"]
            prompt_partial_tokens = prompt_data["partial_prompt_token_ids"]
            prompt_exact_input: Any = TokensPrompt(prompt_token_ids=prompt_exact_tokens)
            prompt_partial_input: Any = TokensPrompt(prompt_token_ids=prompt_partial_tokens)
            requested_shared_prefix_tokens = prompt_data["requested_shared_prefix_tokens"]
            exact_prompt_recompute_tokens = prompt_data["exact_prompt_recompute_tokens"]
            partial_prompt_recompute_tokens = prompt_data["partial_prompt_recompute_tokens"]
        else:
            prompt_exact_input = build_prompt(
                tokenizer,
                args.shared_prefix_tokens,
                tail="\n\nQuestion: Summarize the long prefix in one sentence.",
            )
            prompt_partial_input = build_prompt(
                tokenizer,
                args.shared_prefix_tokens,
                tail="\n\nQuestion: List three keywords from the long prefix.",
            )
            prompt_exact_tokens = tokenizer.encode(prompt_exact_input)
            prompt_partial_tokens = tokenizer.encode(prompt_partial_input)
    finally:
        probe_llm.llm_engine.engine_core.shutdown()

    required_prompt_tokens = len(prompt_exact_tokens)
    if not args.exact_only:
        required_prompt_tokens = max(required_prompt_tokens, len(prompt_partial_tokens))
    effective_max_num_batched_tokens = _effective_max_num_batched_tokens(
        requested_tokens=args.max_num_batched_tokens,
        required_tokens=required_prompt_tokens,
        block_size=probe_block_size,
        aligned_prompts=args.aligned_prompts,
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

    llm_init_started = time.perf_counter()
    llm = LLM(**llm_kwargs)
    llm_init_elapsed_s = time.perf_counter() - llm_init_started

    try:
        common_prefix_tokens = common_prefix_len(prompt_exact_tokens, prompt_partial_tokens)
        block_size = int(llm.llm_engine.vllm_config.cache_config.block_size)
        aligned_common_prefix_tokens = (common_prefix_tokens // block_size) * block_size

        _ = drain_scheduler_stats(llm)

        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=0.0,
            ignore_eos=True,
        )

        exact_cold_runs: list[dict[str, Any]] = []
        exact_reuse_runs: list[dict[str, Any]] = []
        partial_cold_runs: list[dict[str, Any]] = []
        partial_seed_runs: list[dict[str, Any]] = []
        partial_reuse_runs: list[dict[str, Any]] = []

        live_demo = None
        if args.demo_mode == "warm_baseline_then_reuse":
            live_demo = _run_exact_live_demo(
                llm=llm,
                prompt_exact_input=prompt_exact_input,
                sampling_params=sampling_params,
                store_backend=args.store_backend,
                service_socket=args.service_socket,
                warmup_runs=args.warmup_runs,
                reuse_turns=args.repeats,
                print_live=args.print_live,
            )
            exact_cold_runs = [live_demo["warm_baseline"]]
            exact_reuse_runs = list(live_demo["reuse_runs"])
        else:
            for repeat_idx in range(args.repeats):
                reset_probe_state(llm)
                exact_cold = _run_timed_request(
                    llm,
                    prompt_exact_input,
                    sampling_params,
                    "exact_cold",
                )
                exact_reuse = _run_timed_request(
                    llm,
                    prompt_exact_input,
                    sampling_params,
                    "exact_reuse",
                )
                exact_cold_runs.append(exact_cold)
                exact_reuse_runs.append(exact_reuse)
                if args.print_live:
                    print(
                        _format_live_result_line(
                            stage="exact_cold",
                            turn_index=repeat_idx + 1,
                            total_turns=args.repeats,
                            run=exact_cold,
                        )
                    )
                    print(
                        _format_live_result_line(
                            stage="exact_reuse",
                            turn_index=repeat_idx + 1,
                            total_turns=args.repeats,
                            run=exact_reuse,
                            baseline_latency_seconds=float(exact_cold["latency_seconds"]),
                        )
                    )
                if args.exact_only:
                    continue

                reset_probe_state(llm)
                partial_cold = _run_timed_request(
                    llm,
                    prompt_partial_input,
                    sampling_params,
                    "partial_cold",
                )
                partial_cold_runs.append(partial_cold)

                reset_probe_state(llm)
                partial_seed = _run_timed_request(
                    llm,
                    prompt_exact_input,
                    sampling_params,
                    "partial_seed",
                )
                partial_reuse = _run_timed_request(
                    llm,
                    prompt_partial_input,
                    sampling_params,
                    "partial_reuse",
                )
                partial_seed_runs.append(partial_seed)
                partial_reuse_runs.append(partial_reuse)
                if args.print_live:
                    print(
                        _format_live_result_line(
                            stage="partial_cold",
                            turn_index=repeat_idx + 1,
                            total_turns=args.repeats,
                            run=partial_cold,
                        )
                    )
                    print(
                        _format_live_result_line(
                            stage="partial_reuse",
                            turn_index=repeat_idx + 1,
                            total_turns=args.repeats,
                            run=partial_reuse,
                            baseline_latency_seconds=float(partial_cold["latency_seconds"]),
                        )
                    )

        if not args.no_assert_reuse:
            for run in exact_cold_runs:
                _validate_run(run, expect_load=False, expect_save=True, label=run["label"])
            for run in exact_reuse_runs:
                _validate_run(run, expect_load=True, expect_save=False, label=run["label"])
            if not args.exact_only and args.demo_mode != "warm_baseline_then_reuse":
                for run in partial_cold_runs:
                    _validate_run(run, expect_load=False, expect_save=True, label=run["label"])
                for run in partial_seed_runs:
                    _validate_run(run, expect_load=False, expect_save=True, label=run["label"])
                if aligned_common_prefix_tokens >= block_size:
                    for run in partial_reuse_runs:
                        _validate_run(run, expect_load=True, expect_save=False, label=run["label"])

        summary = {
            "run_metadata": build_run_metadata(__file__),
            "engine_init_elapsed_s": llm_init_elapsed_s,
            "model": args.model,
            "revision": args.revision,
            "backend": args.backend,
            "store_backend": args.store_backend,
            "service_socket": args.service_socket,
            "store_max_bytes": args.store_max_bytes,
            "aligned_prompts": args.aligned_prompts,
            "exact_only": args.exact_only,
            "demo_mode": args.demo_mode,
            "warmup_runs": args.warmup_runs,
            "block_size": block_size,
            "repeats": args.repeats,
            "requested_max_num_batched_tokens": args.max_num_batched_tokens,
            "effective_max_num_batched_tokens": effective_max_num_batched_tokens,
            "prompt_exact_tokens": _prompt_token_count(prompt_exact_input, tokenizer),
            "prompt_partial_tokens": _prompt_token_count(prompt_partial_input, tokenizer),
            "common_prefix_tokens": common_prefix_tokens,
            "aligned_common_prefix_tokens": aligned_common_prefix_tokens,
            "requested_shared_prefix_tokens": requested_shared_prefix_tokens,
            "exact_prompt_recompute_tokens": exact_prompt_recompute_tokens,
            "partial_prompt_recompute_tokens": partial_prompt_recompute_tokens,
            "scenarios": {
                "exact_cold": _summarize_runs(exact_cold_runs),
                "exact_reuse": _summarize_runs(exact_reuse_runs),
            },
            "comparisons": {
                "exact_reuse_vs_exact_cold": _compare_latency(exact_cold_runs, exact_reuse_runs),
            },
        }
        if live_demo is not None:
            summary["live_demo"] = {
                "warmup_runs": _summarize_runs(live_demo["warmup_runs"])
                if live_demo["warmup_runs"]
                else None,
                "warm_baseline": live_demo["warm_baseline"],
                "reuse_runs": _summarize_runs(live_demo["reuse_runs"]),
            }
        if not args.exact_only:
            summary["scenarios"]["partial_cold"] = _summarize_runs(partial_cold_runs)
            summary["scenarios"]["partial_seed"] = _summarize_runs(partial_seed_runs)
            summary["scenarios"]["partial_reuse"] = _summarize_runs(partial_reuse_runs)
            summary["comparisons"]["partial_reuse_vs_partial_cold"] = _compare_latency(
                partial_cold_runs,
                partial_reuse_runs,
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
