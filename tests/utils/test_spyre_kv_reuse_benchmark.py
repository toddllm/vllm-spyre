from pathlib import Path
import sys


EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "offline_inference"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from spyre_kv_reuse_benchmark import (  # noqa: E402
    _effective_max_num_batched_tokens,
    _format_live_result_line,
    _prompt_token_count,
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
