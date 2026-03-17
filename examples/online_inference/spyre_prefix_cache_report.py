#!/usr/bin/env python3
"""Run a small server-path prefix-caching probe and emit a report.

This script is intentionally simple:

- connect to a running ``vllm serve`` Spyre server
- warm it once so compile/startup noise is out of band
- reset prefix cache counters
- run a small exact-prefix and partial-prefix matrix
- collect prefix-cache metrics from the OpenAI metrics endpoint
- print a compact human-readable summary followed by embedded JSON

The JSON block at the end is meant to be saved in a raw artifact with:

    python examples/online_inference/spyre_prefix_cache_report.py ... 2>&1 | tee out.raw.txt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI
from transformers import AutoTokenizer


OFFLINE_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "offline_inference"
if str(OFFLINE_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(OFFLINE_EXAMPLES_DIR))

from spyre_kv_reuse_common import build_run_metadata  # noqa: E402


DEFAULT_EXACT_CASES = [
    (53, 0),
    (127, 0),
    (144, 64),
    (250, 128),
    (299, 192),
    (350, 256),
    (420, 320),
]

DEFAULT_PARTIAL_CASES = [
    (144, 0),
    (250, 0),
    (299, 64),
    (350, 128),
    (420, 192),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Server-path prefix-cache probe for Spyre AIU reporting."
    )
    parser.add_argument(
        "--server-root",
        default="http://localhost:8000",
        help="Root URL of the running vLLM OpenAI server, without /v1.",
    )
    parser.add_argument(
        "--api-key",
        default="token-abc123",
        help="API key for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to request. If omitted, the first served model is used.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer name or path. Defaults to the served model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help="Number of output tokens to request in each probe call.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Expected chunk size used by chunked prefill.",
    )
    parser.add_argument(
        "--warmup-prompt-tokens",
        type=int,
        default=32,
        help="Warmup prompt token length. Set to 0 to skip warmup.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for deterministic prompt-token generation.",
    )
    parser.add_argument(
        "--exact-cases",
        default=",".join(f"{length}:{hits}" for length, hits in DEFAULT_EXACT_CASES),
        help="Comma-separated prompt_len:expected_hit_tokens cases for exact-prefix runs.",
    )
    parser.add_argument(
        "--partial-cases",
        default=",".join(f"{length}:{hits}" for length, hits in DEFAULT_PARTIAL_CASES),
        help="Comma-separated prompt_len:expected_hit_tokens cases for partial-prefix runs.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the JSON summary.",
    )
    parser.add_argument(
        "--markdown-out",
        default=None,
        help="Optional path to write the Markdown summary.",
    )
    return parser.parse_args()


def _parse_cases(spec: str) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    if not spec.strip():
        return cases
    for item in spec.split(","):
        prompt_len, expected_hits = item.split(":")
        cases.append((int(prompt_len), int(expected_hits)))
    return cases


def _server_url(root: str, *parts: str) -> str:
    base = root.rstrip("/")
    if not parts:
        return base
    return base + "/" + "/".join(parts)


def _get_metrics(server_root: str) -> dict[str, float]:
    response = requests.get(_server_url(server_root, "metrics"), timeout=60)
    response.raise_for_status()
    metrics: dict[str, float] = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        if " " not in line:
            continue
        name, value = line.rsplit(" ", 1)
        base_name = name.split("{", 1)[0]
        try:
            metrics[base_name] = metrics.get(base_name, 0.0) + float(value)
        except ValueError:
            continue
    return metrics


def _metric_delta(
    before: dict[str, float],
    after: dict[str, float],
    *candidate_names: str,
) -> int:
    for metric_name in candidate_names:
        if metric_name in before or metric_name in after:
            return int(after.get(metric_name, 0.0) - before.get(metric_name, 0.0))
    return 0


def _reset_prefix_cache(server_root: str) -> None:
    response = requests.post(_server_url(server_root, "reset_prefix_cache"), timeout=60)
    response.raise_for_status()


def _resolve_model(client: OpenAI, requested_model: str | None) -> str:
    if requested_model:
        return requested_model
    return client.models.list().data[0].id


def _valid_token_ids(tokenizer) -> list[int]:
    special_ids = set(tokenizer.all_special_ids)
    return [token_id for token_id in range(tokenizer.vocab_size) if token_id not in special_ids]


def _random_prompt_token_ids(
    *,
    tokenizer,
    prompt_len: int,
    seed: int,
    valid_ids: list[int],
) -> list[int]:
    rng = random.Random(seed)
    return [rng.choice(valid_ids) for _ in range(prompt_len)]


def _preview_text(text: str, limit: int = 80) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _run_completion(
    client: OpenAI,
    *,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    response = client.completions.create(
        model=model,
        prompt=prompt_token_ids,
        max_tokens=max_tokens,
        temperature=0,
    )
    latency = time.perf_counter() - start
    text = response.choices[0].text
    usage = response.usage
    return {
        "latency_seconds": latency,
        "output_preview": _preview_text(text),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }


def _ratio(observed: int, expected: int) -> float:
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return observed / expected


def _run_case(
    *,
    client: OpenAI,
    tokenizer,
    valid_ids: list[int],
    model: str,
    server_root: str,
    prompt_len: int,
    expected_hit_tokens: int,
    max_tokens: int,
    chunk_size: int,
    seed: int,
    partial: bool,
) -> dict[str, Any]:
    prompt_token_ids = _random_prompt_token_ids(
        tokenizer=tokenizer,
        prompt_len=prompt_len,
        seed=seed,
        valid_ids=valid_ids,
    )
    second_prompt = (
        prompt_token_ids[:-chunk_size] if partial else list(prompt_token_ids)
    )

    metrics_before = _get_metrics(server_root)
    first_request = _run_completion(
        client,
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_tokens=max_tokens,
    )
    second_request = _run_completion(
        client,
        model=model,
        prompt_token_ids=second_prompt,
        max_tokens=max_tokens,
    )
    metrics_after = _get_metrics(server_root)

    observed_queries = _metric_delta(
        metrics_before,
        metrics_after,
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_queries",
    )
    observed_hits = _metric_delta(
        metrics_before,
        metrics_after,
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_hits",
    )

    return {
        "kind": "partial" if partial else "exact",
        "prompt_len_tokens": prompt_len,
        "second_prompt_len_tokens": len(second_prompt),
        "expected_hit_tokens": expected_hit_tokens,
        "observed_hit_tokens": observed_hits,
        "observed_query_tokens": observed_queries,
        "hit_ratio_vs_expected": _ratio(observed_hits, expected_hit_tokens),
        "pass": observed_hits == expected_hit_tokens,
        "first_request": first_request,
        "second_request": second_request,
    }


def _summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {
            "count": 0,
            "passed": 0,
            "failed": 0,
            "mean_second_request_latency_s": 0.0,
        }

    return {
        "count": len(cases),
        "passed": sum(1 for case in cases if case["pass"]),
        "failed": sum(1 for case in cases if not case["pass"]),
        "mean_second_request_latency_s": sum(
            case["second_request"]["latency_seconds"] for case in cases
        )
        / len(cases),
    }


def _bar(value: float, *, max_value: float, width: int = 12) -> str:
    if max_value <= 0:
        return ""
    filled = max(0, min(width, round(width * (value / max_value))))
    return "█" * filled + "░" * (width - filled)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Prefix Cache Probe Summary",
        "",
        f"- Model: `{report['model']}`",
        f"- Chunk size: `{report['chunk_size']}`",
        f"- Max new tokens: `{report['max_tokens']}`",
        f"- Warmup prompt tokens: `{report['warmup_prompt_tokens']}`",
        "",
        "## Exact Prefix Cases",
        "",
        "| Prompt tokens | Expected hit tokens | Observed hit tokens | Query tokens | Req 1 (s) | Req 2 (s) | Pass |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]

    max_exact_hits = max((case["expected_hit_tokens"] for case in report["exact_cases"]), default=0)
    for case in report["exact_cases"]:
        lines.append(
            "| {prompt_len_tokens} | {expected_hit_tokens} | {observed_hit_tokens} | "
            "{observed_query_tokens} | {req1:.3f} | {req2:.3f} | {passed} |".format(
                prompt_len_tokens=case["prompt_len_tokens"],
                expected_hit_tokens=case["expected_hit_tokens"],
                observed_hit_tokens=case["observed_hit_tokens"],
                observed_query_tokens=case["observed_query_tokens"],
                req1=case["first_request"]["latency_seconds"],
                req2=case["second_request"]["latency_seconds"],
                passed="yes" if case["pass"] else "no",
            )
        )

    lines.extend(
        [
            "",
            "```text",
            "Exact hit tokens",
        ]
    )
    for case in report["exact_cases"]:
        lines.append(
            f"{case['prompt_len_tokens']:>4}t {_bar(case['observed_hit_tokens'], max_value=max_exact_hits)} {case['observed_hit_tokens']}"
        )
    lines.extend(
        [
            "```",
            "",
            "## Partial Prefix Cases",
            "",
            "| Prompt tokens | Second prompt tokens | Expected hit tokens | Observed hit tokens | Query tokens | Req 1 (s) | Req 2 (s) | Pass |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
    )

    max_partial_hits = max(
        (case["expected_hit_tokens"] for case in report["partial_cases"]),
        default=0,
    )
    for case in report["partial_cases"]:
        lines.append(
            "| {prompt_len_tokens} | {second_prompt_len_tokens} | {expected_hit_tokens} | "
            "{observed_hit_tokens} | {observed_query_tokens} | {req1:.3f} | {req2:.3f} | {passed} |".format(
                prompt_len_tokens=case["prompt_len_tokens"],
                second_prompt_len_tokens=case["second_prompt_len_tokens"],
                expected_hit_tokens=case["expected_hit_tokens"],
                observed_hit_tokens=case["observed_hit_tokens"],
                observed_query_tokens=case["observed_query_tokens"],
                req1=case["first_request"]["latency_seconds"],
                req2=case["second_request"]["latency_seconds"],
                passed="yes" if case["pass"] else "no",
            )
        )

    lines.extend(["", "```text", "Partial hit tokens"])
    for case in report["partial_cases"]:
        lines.append(
            f"{case['prompt_len_tokens']:>4}t {_bar(case['observed_hit_tokens'], max_value=max_partial_hits)} {case['observed_hit_tokens']}"
        )
    lines.extend(
        [
            "```",
            "",
            "## Summary",
            "",
            f"- Exact cases passed: `{report['summary']['exact']['passed']}/{report['summary']['exact']['count']}`",
            f"- Partial cases passed: `{report['summary']['partial']['passed']}/{report['summary']['partial']['count']}`",
            f"- All cases passed: `{report['summary']['all_passed']}`",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    exact_cases = _parse_cases(args.exact_cases)
    partial_cases = _parse_cases(args.partial_cases)

    client = OpenAI(
        api_key=args.api_key,
        base_url=_server_url(args.server_root, "v1"),
        max_retries=0,
        timeout=600,
    )
    model = _resolve_model(client, args.model)
    tokenizer_name = args.tokenizer or model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    valid_ids = _valid_token_ids(tokenizer)

    if args.warmup_prompt_tokens > 0:
        warmup_prompt = _random_prompt_token_ids(
            tokenizer=tokenizer,
            prompt_len=args.warmup_prompt_tokens,
            seed=args.seed,
            valid_ids=valid_ids,
        )
        _run_completion(
            client,
            model=model,
            prompt_token_ids=warmup_prompt,
            max_tokens=args.max_tokens,
        )

    _reset_prefix_cache(args.server_root)

    exact_results = []
    for idx, (prompt_len, expected_hits) in enumerate(exact_cases):
        exact_results.append(
            _run_case(
                client=client,
                tokenizer=tokenizer,
                valid_ids=valid_ids,
                model=model,
                server_root=args.server_root,
                prompt_len=prompt_len,
                expected_hit_tokens=expected_hits,
                max_tokens=args.max_tokens,
                chunk_size=args.chunk_size,
                seed=args.seed + idx,
                partial=False,
            )
        )

    _reset_prefix_cache(args.server_root)

    partial_results = []
    for idx, (prompt_len, expected_hits) in enumerate(partial_cases):
        partial_results.append(
            _run_case(
                client=client,
                tokenizer=tokenizer,
                valid_ids=valid_ids,
                model=model,
                server_root=args.server_root,
                prompt_len=prompt_len,
                expected_hit_tokens=expected_hits,
                max_tokens=args.max_tokens,
                chunk_size=args.chunk_size,
                seed=args.seed + 1000 + idx,
                partial=True,
            )
        )

    report = {
        "model": model,
        "tokenizer": tokenizer_name,
        "server_root": args.server_root.rstrip("/"),
        "chunk_size": args.chunk_size,
        "max_tokens": args.max_tokens,
        "warmup_prompt_tokens": args.warmup_prompt_tokens,
        "exact_cases": exact_results,
        "partial_cases": partial_results,
        "summary": {
            "exact": _summarize_cases(exact_results),
            "partial": _summarize_cases(partial_results),
        },
        "run_metadata": build_run_metadata(__file__),
    }
    report["summary"]["all_passed"] = (
        report["summary"]["exact"]["failed"] == 0
        and report["summary"]["partial"]["failed"] == 0
    )

    markdown = render_markdown(report)
    print(markdown)
    print()
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        Path(args.markdown_out).write_text(markdown + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
