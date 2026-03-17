from pathlib import Path
import sys


EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "online_inference"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from spyre_prefix_cache_report import (  # noqa: E402
    _parse_cases,
    _ratio,
    render_markdown,
)


def test_parse_cases_parses_prompt_and_hit_lengths():
    assert _parse_cases("144:64,250:128") == [(144, 64), (250, 128)]


def test_ratio_handles_zero_expected_hits():
    assert _ratio(0, 0) == 1.0
    assert _ratio(1, 0) == 0.0
    assert _ratio(96, 192) == 0.5


def test_render_markdown_includes_tables_and_summary():
    report = {
        "model": "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        "chunk_size": 128,
        "max_tokens": 1,
        "warmup_prompt_tokens": 32,
        "exact_cases": [
            {
                "prompt_len_tokens": 250,
                "expected_hit_tokens": 128,
                "observed_hit_tokens": 128,
                "observed_query_tokens": 250,
                "first_request": {"latency_seconds": 0.4},
                "second_request": {"latency_seconds": 0.2},
                "pass": True,
            }
        ],
        "partial_cases": [
            {
                "prompt_len_tokens": 350,
                "second_prompt_len_tokens": 222,
                "expected_hit_tokens": 128,
                "observed_hit_tokens": 128,
                "observed_query_tokens": 222,
                "first_request": {"latency_seconds": 0.5},
                "second_request": {"latency_seconds": 0.3},
                "pass": True,
            }
        ],
        "summary": {
            "exact": {"count": 1, "passed": 1},
            "partial": {"count": 1, "passed": 1},
            "all_passed": True,
        },
    }

    rendered = render_markdown(report)

    assert "# Prefix Cache Probe Summary" in rendered
    assert "| Prompt tokens | Expected hit tokens | Observed hit tokens |" in rendered
    assert "| 250 | 128 | 128 | 250 | 0.400 | 0.200 | yes |" in rendered
    assert "| 350 | 222 | 128 | 128 | 222 | 0.500 | 0.300 | yes |" in rendered
    assert "- All cases passed: `True`" in rendered
