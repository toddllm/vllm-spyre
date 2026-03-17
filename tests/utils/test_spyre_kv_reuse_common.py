from pathlib import Path
import sys

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "offline_inference"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from spyre_kv_reuse_common import (
    build_aligned_reuse_token_prompts,
    build_templated_prompt,
    build_token_sequence_from_chunks,
    build_token_sequence,
    get_demo_prompt_template_names,
    reset_probe_state,
    resolve_demo_prompt_template,
    round_down_to_block,
    round_up_to_block,
)


class _FakeTokenizer:
    all_special_ids = [0, 1]

    def __init__(self):
        self._vocab = {}

    def encode(self, text, add_special_tokens=False):
        tokens = text.split()
        ids = []
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab) + 10
            ids.append(self._vocab[token])
        return ids


def test_round_down_to_block():
    assert round_down_to_block(0, 64) == 0
    assert round_down_to_block(63, 64) == 0
    assert round_down_to_block(64, 64) == 64
    assert round_down_to_block(129, 64) == 128


def test_round_up_to_block():
    assert round_up_to_block(0, 64) == 0
    assert round_up_to_block(1, 64) == 64
    assert round_up_to_block(64, 64) == 64
    assert round_up_to_block(129, 64) == 192


def test_build_token_sequence_returns_exact_token_count():
    tokenizer = _FakeTokenizer()

    token_ids = build_token_sequence(tokenizer, 10, "aligned shared prefix seed")

    assert len(token_ids) == 10
    assert token_ids[:4] == token_ids[4:8]
    assert token_ids[8:] == token_ids[:2]


def test_build_token_sequence_from_chunks_returns_exact_token_count():
    tokenizer = _FakeTokenizer()

    token_ids = build_token_sequence_from_chunks(
        tokenizer,
        9,
        ["hello there", "general kenobi now"],
    )

    assert len(token_ids) == 9


def test_build_aligned_reuse_token_prompts_produces_exact_reuse_case():
    tokenizer = _FakeTokenizer()

    prompt_data = build_aligned_reuse_token_prompts(
        tokenizer,
        requested_shared_prefix_tokens=199,
        block_size=64,
        partial_tail_tokens=15,
    )

    assert prompt_data["requested_shared_prefix_tokens"] == 199
    assert prompt_data["aligned_shared_prefix_tokens"] == 192
    assert prompt_data["demo_template"] == "science_fair_invite"
    assert len(prompt_data["prefill_prompt_token_ids"]) == 192
    assert len(prompt_data["partial_prompt_token_ids"]) == 207
    assert (
        prompt_data["partial_prompt_token_ids"][:192]
        == prompt_data["prefill_prompt_token_ids"]
    )
    assert prompt_data["prefill_prompt_token_ids"][-1] in prompt_data["prefill_prompt_token_ids"]
    assert prompt_data["exact_prompt_recompute_tokens"] == 0
    assert prompt_data["partial_prompt_recompute_tokens"] == 15


def test_demo_prompt_template_catalog_exposes_internal_and_demo_friendly_options():
    template_names = get_demo_prompt_template_names()

    assert "science_fair_invite" in template_names
    assert "chicken_soup" in template_names
    assert "bedtime_story" in template_names


def test_resolve_demo_prompt_template_applies_defaults_and_overrides():
    template = resolve_demo_prompt_template("chicken_soup")
    assert template["display_name"] == "Chicken soup instructions"
    assert "preparing chicken soup" in template["instruction_text"]

    overridden = resolve_demo_prompt_template(
        "science_fair_invite",
        scenario_text="a robotics club open house",
        question_text="Write a short invitation for local families.",
        partial_tail_text="Mention one demo and one reason to attend.",
    )
    assert overridden["scenario_text"] == "a robotics club open house"
    assert overridden["instruction_text"] == "Write a short invitation for local families."
    assert overridden["partial_tail_text"] == "Mention one demo and one reason to attend."


def test_build_templated_prompt_uses_selected_template_text():
    tokenizer = _FakeTokenizer()

    prompt = build_templated_prompt(
        tokenizer,
        20,
        template_name="bedtime_story",
    )

    assert "### Background:" in prompt
    assert "### Instruction:" in prompt
    assert "fox who learns to share" in prompt


def test_round_down_to_block_rejects_non_positive_block_size():
    with pytest.raises(ValueError, match="block_size must be positive"):
        round_down_to_block(128, 0)


def test_round_up_to_block_rejects_non_positive_block_size():
    with pytest.raises(ValueError, match="block_size must be positive"):
        round_up_to_block(128, 0)


def test_reset_probe_state_forwards_clear_flags(monkeypatch):
    scheduler_calls = []
    worker_calls = []

    class _FakeConnector:
        def __init__(self, calls):
            self.calls = calls

        def reset_probe_state(
            self,
            *,
            clear_store=True,
            clear_saved_requests=True,
            clear_metrics=True,
        ):
            self.calls.append(
                {
                    "clear_store": clear_store,
                    "clear_saved_requests": clear_saved_requests,
                    "clear_metrics": clear_metrics,
                }
            )

    monkeypatch.setattr(
        "spyre_kv_reuse_common.get_scheduler_connector",
        lambda llm: _FakeConnector(scheduler_calls),
    )
    monkeypatch.setattr(
        "spyre_kv_reuse_common.get_worker_connector",
        lambda: _FakeConnector(worker_calls),
    )
    monkeypatch.setattr("spyre_kv_reuse_common.drain_scheduler_stats", lambda llm: {})

    reset_probe_state(
        object(),
        clear_store=False,
        clear_saved_requests=False,
        clear_metrics=True,
    )

    expected = [
        {
            "clear_store": False,
            "clear_saved_requests": False,
            "clear_metrics": True,
        }
    ]
    assert scheduler_calls == expected
    assert worker_calls == expected
