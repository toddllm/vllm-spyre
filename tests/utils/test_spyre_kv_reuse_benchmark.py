from pathlib import Path
import sys


EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "offline_inference"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from spyre_kv_reuse_benchmark import (  # noqa: E402
    _effective_max_num_batched_tokens,
    _format_live_result_line,
    _preview_text,
    _prompt_token_count,
    _prompt_work_status,
    _resolve_demo_settings,
)


class _FakeTokenizer:
    def encode(self, text):
        return [1, 2, 3, 4]


def test_prompt_token_count_supports_string_prompt():
    tokenizer = _FakeTokenizer()

    assert _prompt_token_count("demo prompt", tokenizer) == 4


def test_prompt_token_count_supports_token_prompt_mapping():
    tokenizer = _FakeTokenizer()

    assert _prompt_token_count({"prompt_token_ids": [10, 11, 12]}, tokenizer) == 3


def test_format_live_result_line_includes_speedup_when_baseline_provided():
    line = _format_live_result_line(
        stage="exact_reuse",
        turn_index=2,
        total_turns=5,
        run={
            "latency_seconds": 0.1,
            "output_tokens": 3,
            "worker_delta": {
                "blocks_loaded": 16,
                "blocks_missing": 0,
                "blocks_saved": 0,
            },
        },
        baseline_latency_seconds=0.4,
    )

    assert "exact_reuse[2/5]" in line
    assert "latency_s=0.100000" in line
    assert "blocks_loaded=16" in line
    assert "saved_ms=300.000" in line
    assert "speedup_vs_baseline=4.000x" in line


def test_format_live_result_line_demo_style_is_cleaner():
    line = _format_live_result_line(
        stage="exact_reuse",
        turn_index=1,
        total_turns=4,
        run={
            "latency_seconds": 0.084474,
            "output_tokens": 1,
            "worker_delta": {
                "blocks_loaded": 56,
                "blocks_missing": 0,
                "blocks_saved": 56,
            },
        },
        baseline_latency_seconds=0.186547,
        cumulative_saved_ms=102.073,
        style="demo",
        prompt_work_status="saved prompt work reused",
    )

    assert line == (
        "Repeat 1/4: 0.084s | saved prompt work reused | 102.1 ms faster | 2.21x faster | total saved 102.1 ms"
    )


def test_format_live_result_line_includes_cumulative_saved_time_when_provided():
    line = _format_live_result_line(
        stage="exact_reuse",
        turn_index=3,
        total_turns=6,
        run={
            "latency_seconds": 0.2,
            "output_tokens": 1,
            "worker_delta": {
                "blocks_loaded": 48,
                "blocks_missing": 0,
                "blocks_saved": 48,
            },
        },
        baseline_latency_seconds=0.5,
        cumulative_saved_ms=725.0,
    )

    assert "saved_ms=300.000" in line
    assert "cumulative_saved_ms=725.000" in line


def test_effective_max_num_batched_tokens_rounds_up_for_aligned_prompts():
    assert (
        _effective_max_num_batched_tokens(
            requested_tokens=128,
            required_tokens=385,
            block_size=64,
            aligned_prompts=True,
            fit_prompt_in_single_prefill=True,
        )
        == 448
    )


def test_effective_max_num_batched_tokens_keeps_requested_non_aligned_value():
    assert (
        _effective_max_num_batched_tokens(
            requested_tokens=256,
            required_tokens=180,
            block_size=64,
            aligned_prompts=False,
            fit_prompt_in_single_prefill=True,
        )
        == 256
    )


def test_effective_max_num_batched_tokens_keeps_requested_value_by_default():
    assert (
        _effective_max_num_batched_tokens(
            requested_tokens=128,
            required_tokens=385,
            block_size=64,
            aligned_prompts=True,
            fit_prompt_in_single_prefill=False,
        )
        == 128
    )


def test_resolve_demo_settings_applies_explicit_overrides():
    class _Args:
        demo_prompt_tokens = 512
        demo_response_tokens = 2
        demo_turns = 5
        demo_pause_seconds = 1.75
        shared_prefix_tokens = 192
        max_new_tokens = 8
        repeats = 3
        sleep_between_live_lines = 0.0

    args = _Args()

    resolved = _resolve_demo_settings(args)

    assert resolved["overrides"] == {
        "shared_prefix_tokens": 512,
        "max_new_tokens": 2,
        "repeats": 5,
        "sleep_between_live_lines": 1.75,
    }
    assert args.shared_prefix_tokens == 512
    assert args.max_new_tokens == 2
    assert args.repeats == 5
    assert args.sleep_between_live_lines == 1.75


def test_prompt_work_status_distinguishes_normal_vs_reused():
    assert (
        _prompt_work_status(
            {"worker_delta": {"blocks_loaded": 0, "blocks_missing": 0, "blocks_saved": 48}},
            prompt_recompute_tokens=0,
        )
        == "normal run"
    )
    assert (
        _prompt_work_status(
            {"worker_delta": {"blocks_loaded": 48, "blocks_missing": 0, "blocks_saved": 0}},
            prompt_recompute_tokens=0,
        )
        == "saved prompt work reused"
    )
    assert (
        _prompt_work_status(
            {"worker_delta": {"blocks_loaded": 24, "blocks_missing": 0, "blocks_saved": 24}},
            prompt_recompute_tokens=16,
        )
        == "partly reused saved prompt work"
    )


def test_preview_text_collapses_whitespace_and_truncates():
    assert _preview_text("hello   world", 20) == "hello world"
    assert _preview_text("abcdefghij", 7) == "abcd..."
