from pathlib import Path
import sys


EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "offline_inference"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from spyre_kv_reuse_benchmark import (  # noqa: E402
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
    assert "speedup_vs_cold=4.000x" in line
