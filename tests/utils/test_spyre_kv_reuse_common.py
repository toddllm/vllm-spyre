from pathlib import Path
import sys

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "offline_inference"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from spyre_kv_reuse_common import (
    build_aligned_reuse_token_prompts,
    build_token_sequence,
    round_down_to_block,
    round_up_to_block,
)


class _FakeTokenizer:
    all_special_ids = [0, 1]

    def encode(self, text, add_special_tokens=False):
        if "aligned shared prefix" in text:
            return [11, 12, 13, 14]
        if "Divergent partial tail" in text:
            return [21, 22, 23]
        return [31, 32]


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

    token_ids = build_token_sequence(tokenizer, 10, "aligned shared prefix")

    assert len(token_ids) == 10
    assert token_ids == [11, 12, 13, 14, 11, 12, 13, 14, 11, 12]


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
    assert len(prompt_data["prefill_prompt_token_ids"]) == 192
    assert len(prompt_data["partial_prompt_token_ids"]) == 207
    assert (
        prompt_data["partial_prompt_token_ids"][:192]
        == prompt_data["prefill_prompt_token_ids"]
    )
    assert prompt_data["exact_prompt_recompute_tokens"] == 0
    assert prompt_data["partial_prompt_recompute_tokens"] == 15


def test_round_down_to_block_rejects_non_positive_block_size():
    with pytest.raises(ValueError, match="block_size must be positive"):
        round_down_to_block(128, 0)


def test_round_up_to_block_rejects_non_positive_block_size():
    with pytest.raises(ValueError, match="block_size must be positive"):
        round_up_to_block(128, 0)
