"""
Shared helpers for single-process Spyre KV reuse probes and benchmarks.
"""

from __future__ import annotations

import datetime
import os
import socket
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


BASELINE_ENV_VARS = (
    "HF_HUB_OFFLINE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "VLLM_LOGGING_LEVEL",
    "SPYRE_DEVICES",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "VLLM_SPYRE_DYNAMO_BACKEND",
    "VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE",
    "VLLM_SPYRE_KV_STORE_BACKEND",
    "VLLM_SPYRE_KV_STORE_MAX_BYTES",
    "VLLM_SPYRE_KV_SERVICE_SOCKET",
)

NON_DELTA_METRIC_KEYS = frozenset({"saved_requests_count"})

INSTRUCTION_PROMPT_HEADER = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request. Be polite in your response to the "
    "user."
)

DEMO_PROMPT_TEMPLATES: dict[str, dict[str, Any]] = {
    "science_fair_invite": {
        "display_name": "Science fair invitation",
        "scenario_text": "a neighborhood science fair for families",
        "instruction_text": (
            "Write a short invitation that makes the event sound welcoming, "
            "practical, and fun."
        ),
        "partial_tail_text": (
            "Mention one hands-on activity, one practical detail, and one "
            "reason a parent might say yes."
        ),
        "context_chunks": [
            "The team is preparing materials for {scenario_text}. ",
            "Visitors want a clear sense of what will happen, who it is for, and why it is worth showing up. ",
            "The tone should be friendly, concrete, and easy to picture. ",
            "Good answers mention atmosphere, a useful detail, and one memorable reason to attend. ",
        ],
    },
    "chicken_soup": {
        "display_name": "Chicken soup instructions",
        "instruction_text": "Provide a list of instructions for preparing chicken soup.",
        "partial_tail_text": (
            "Include one tip for a richer broth and one simple side dish that pairs well."
        ),
        "context_chunks": [
            "The cook wants clear steps, common ingredients, and a warm, practical tone. ",
            "Strong answers mention prepping ingredients, simmering the broth, seasoning carefully, and serving suggestions. ",
            "The result should feel useful to a home cook, not like a restaurant recipe. ",
        ],
    },
    "bedtime_story": {
        "display_name": "Bedtime story",
        "instruction_text": (
            "Tell a gentle bedtime story about a fox who learns to share."
        ),
        "partial_tail_text": (
            "Keep the ending calm, add one memorable image, and avoid anything scary."
        ),
        "context_chunks": [
            "The story is meant for a young child at bedtime. ",
            "Good answers use simple language, a calm rhythm, and reassuring imagery. ",
            "The tone should feel soft, warm, and easy to read aloud. ",
        ],
    },
    "power_query_columns": {
        "display_name": "Power Query help",
        "instruction_text": (
            "Explain how to add multiple new columns in M for Power Query or Power BI."
        ),
        "partial_tail_text": (
            "Explain it in very simple terms and include one short example."
        ),
        "context_chunks": [
            "The user is a beginner and wants a plain-language explanation. ",
            "Helpful answers break the task into steps, mention the UI and the M code, and keep the example short. ",
            "The response should reduce confusion and sound encouraging. ",
        ],
    },
    "java_char_to_string": {
        "display_name": "Java char to string",
        "instruction_text": "Explain how to convert a char to a string in Java.",
        "partial_tail_text": (
            "Include one tiny code example and one common mistake to avoid."
        ),
        "context_chunks": [
            "The reader wants a quick, practical programming answer. ",
            "Good answers mention the main conversion options, show one clear example, and keep jargon light. ",
            "The response should be concise and easy to copy into code review comments or team chat. ",
        ],
    },
}


def set_local_dist_defaults() -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    if "MASTER_PORT" in os.environ:
        return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(sock.getsockname()[1])


def _git_output(start_dir: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(start_dir), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    value = result.stdout.strip()
    return value or None


def build_run_metadata(script_path: str, argv: list[str] | None = None) -> dict[str, Any]:
    script = Path(script_path).resolve()
    git_root = _git_output(script.parent, "rev-parse", "--show-toplevel")
    git_branch = _git_output(script.parent, "symbolic-ref", "--short", "HEAD")
    git_commit = _git_output(script.parent, "rev-parse", "HEAD")

    git_info: dict[str, str] = {}
    if git_root is not None:
        git_info["root"] = git_root
    if git_branch is not None:
        git_info["branch"] = git_branch
    if git_commit is not None:
        git_info["commit"] = git_commit

    return {
        "artifact_format_version": 1,
        "timestamp_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "script": str(script),
        "argv": list(argv or sys.argv),
        "python_version": sys.version.split()[0],
        "git": git_info,
        "env": {
            key: os.environ[key]
            for key in BASELINE_ENV_VARS
            if key in os.environ
        },
    }


def build_prompt(tokenizer, min_tokens: int, tail: str) -> str:
    shared = (
        "Spyre KV reuse probe. This prompt is intentionally long so we can "
        "exercise block-aligned prefix reuse through the connector path. "
    )
    prompt = shared
    while len(tokenizer.encode(prompt)) < min_tokens:
        prompt += shared
    return prompt + tail


def get_demo_prompt_template_names() -> tuple[str, ...]:
    return tuple(DEMO_PROMPT_TEMPLATES.keys())


def resolve_demo_prompt_template(
    template_name: str,
    *,
    scenario_text: str | None = None,
    question_text: str | None = None,
    partial_tail_text: str | None = None,
) -> dict[str, Any]:
    if template_name not in DEMO_PROMPT_TEMPLATES:
        valid = ", ".join(get_demo_prompt_template_names())
        raise ValueError(f"Unknown demo template {template_name!r}. Expected one of: {valid}")

    base = DEMO_PROMPT_TEMPLATES[template_name]
    resolved_scenario = scenario_text or base.get("scenario_text")
    resolved_instruction = question_text or base["instruction_text"]
    resolved_tail = partial_tail_text or base["partial_tail_text"]

    context_chunks = [
        chunk.format(
            scenario_text=resolved_scenario,
            instruction_text=resolved_instruction,
        )
        for chunk in base["context_chunks"]
    ]

    return {
        "template_name": template_name,
        "display_name": base["display_name"],
        "scenario_text": resolved_scenario,
        "instruction_text": resolved_instruction,
        "partial_tail_text": resolved_tail,
        "context_chunks": context_chunks,
        "background_header_text": f"{INSTRUCTION_PROMPT_HEADER}\n\n### Background:\n",
        "instruction_prefix_text": "\n\n### Instruction:\n",
        "partial_tail_prefix_text": "\n\n### Extra detail to include:\n",
    }


def build_templated_prompt(
    tokenizer,
    min_tokens: int,
    *,
    template_name: str,
    scenario_text: str | None = None,
    question_text: str | None = None,
    partial_tail_text: str | None = None,
    include_partial_tail: bool = False,
) -> str:
    template = resolve_demo_prompt_template(
        template_name,
        scenario_text=scenario_text,
        question_text=question_text,
        partial_tail_text=partial_tail_text,
    )

    prompt = template["background_header_text"]
    context_idx = 0
    while len(tokenizer.encode(prompt)) < min_tokens:
        prompt += template["context_chunks"][context_idx % len(template["context_chunks"])]
        context_idx += 1

    prompt += (
        template["instruction_prefix_text"]
        + template["instruction_text"]
    )
    if include_partial_tail:
        prompt += (
            template["partial_tail_prefix_text"]
            + template["partial_tail_text"]
        )
    return prompt


def round_down_to_block(tokens: int, block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return max(0, (tokens // block_size) * block_size)


def round_up_to_block(tokens: int, block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if tokens <= 0:
        return 0
    return ((tokens + block_size - 1) // block_size) * block_size


def build_token_sequence(tokenizer, num_tokens: int, seed_text: str) -> list[int]:
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    if num_tokens == 0:
        return []

    seed_ids = list(tokenizer.encode(seed_text, add_special_tokens=False))
    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    seed_ids = [token_id for token_id in seed_ids if token_id not in special_ids]
    if not seed_ids:
        raise ValueError("seed_text must tokenize to at least one non-special token")

    repeats, remainder = divmod(num_tokens, len(seed_ids))
    return seed_ids * repeats + seed_ids[:remainder]


def build_token_sequence_from_chunks(
    tokenizer,
    num_tokens: int,
    text_chunks: list[str],
) -> list[int]:
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    if num_tokens == 0:
        return []
    if not text_chunks:
        raise ValueError("text_chunks must not be empty")

    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    token_ids: list[int] = []
    chunk_idx = 0
    while len(token_ids) < num_tokens:
        chunk = text_chunks[chunk_idx % len(text_chunks)]
        chunk_token_ids = [
            token_id
            for token_id in tokenizer.encode(chunk, add_special_tokens=False)
            if token_id not in special_ids
        ]
        if not chunk_token_ids:
            raise ValueError("Each text chunk must tokenize to at least one non-special token")
        token_ids.extend(chunk_token_ids)
        chunk_idx += 1

    return token_ids[:num_tokens]


def build_aligned_reuse_token_prompts(
    tokenizer,
    *,
    requested_shared_prefix_tokens: int,
    block_size: int,
    partial_tail_tokens: int,
    template_name: str = "science_fair_invite",
    scenario_text: str | None = None,
    question_text: str | None = None,
    partial_tail_text: str | None = None,
) -> dict[str, Any]:
    aligned_shared_prefix_tokens = round_down_to_block(
        requested_shared_prefix_tokens,
        block_size,
    )
    if aligned_shared_prefix_tokens == 0:
        aligned_shared_prefix_tokens = block_size

    template = resolve_demo_prompt_template(
        template_name,
        scenario_text=scenario_text,
        question_text=question_text,
        partial_tail_text=partial_tail_text,
    )

    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    header_ids = [
        token_id
        for token_id in tokenizer.encode(
            template["background_header_text"],
            add_special_tokens=False,
        )
        if token_id not in special_ids
    ]
    instruction_ids = [
        token_id
        for token_id in tokenizer.encode(
            template["instruction_prefix_text"] + template["instruction_text"],
            add_special_tokens=False,
        )
        if token_id not in special_ids
    ]
    if not instruction_ids:
        raise ValueError("instruction_text must tokenize to at least one non-special token")

    if len(header_ids) + len(instruction_ids) >= aligned_shared_prefix_tokens:
        shared_prefix_ids = (header_ids + instruction_ids)[:aligned_shared_prefix_tokens]
    else:
        context_budget = aligned_shared_prefix_tokens - len(header_ids) - len(instruction_ids)
        context_ids = build_token_sequence_from_chunks(
            tokenizer,
            context_budget,
            template["context_chunks"],
        )
        shared_prefix_ids = header_ids + context_ids + instruction_ids

    partial_tail_ids = build_token_sequence(
        tokenizer,
        max(0, partial_tail_tokens),
        template["partial_tail_prefix_text"] + template["partial_tail_text"],
    )

    prefill_prompt_ids = list(shared_prefix_ids)
    partial_prompt_ids = list(shared_prefix_ids) + list(partial_tail_ids)

    return {
        "demo_template": template["template_name"],
        "demo_template_display_name": template["display_name"],
        "scenario_text": template["scenario_text"],
        "instruction_text": template["instruction_text"],
        "partial_tail_text": template["partial_tail_text"],
        "requested_shared_prefix_tokens": requested_shared_prefix_tokens,
        "aligned_shared_prefix_tokens": aligned_shared_prefix_tokens,
        "prefill_prompt_token_ids": prefill_prompt_ids,
        "partial_prompt_token_ids": partial_prompt_ids,
        "partial_tail_tokens": len(partial_tail_ids),
        "exact_prompt_recompute_tokens": 0,
        "partial_prompt_recompute_tokens": len(partial_tail_ids),
    }


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
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(keys)
        if key not in NON_DELTA_METRIC_KEYS
    }


def reset_probe_state(
    llm,
    *,
    clear_store: bool = True,
    clear_saved_requests: bool = True,
    clear_metrics: bool = True,
) -> None:
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
            reset(
                clear_store=clear_store,
                clear_saved_requests=clear_saved_requests,
                clear_metrics=clear_metrics,
            )

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


def extract_output_text(outputs: list[Any]) -> str:
    if not outputs:
        return ""

    request_output = outputs[0]
    candidates = getattr(request_output, "outputs", None)
    if not candidates:
        return ""

    text = getattr(candidates[0], "text", None)
    if not isinstance(text, str):
        return ""

    return text
