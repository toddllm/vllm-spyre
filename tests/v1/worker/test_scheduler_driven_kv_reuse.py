"""End-to-end scheduler-driven KV reuse tests.

Validates the full flow:
  1. Request A prefill → scheduler side records → worker side saves KV
  2. Request B with shared prefix → scheduler matches → worker loads KV
  3. Loaded data matches what was saved
  4. Block mapping is correct
  5. Load misses produce get_block_ids_with_load_errors
  6. No premature finished_* reporting
  7. Non-matching prompt returns 0
  8. Partial prefix (not block-aligned) returns block-aligned count
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)

from vllm_spyre.distributed.kv_transfer.kv_connector.v1.metadata import (
    InMemoryKVStore,
    KVKind,
    SpyreConnectorMeta,
    SpyreConnectorRequestMeta,
    StoreKey,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vllm_config_mock(block_size: int = 4):
    cfg = MagicMock()
    cfg.cache_config.block_size = block_size
    cfg.kv_transfer_config = MagicMock()
    cfg.kv_transfer_config.is_kv_transfer_instance = True
    cfg.parallel_config.data_parallel_size = 1
    cfg.compilation_config.fast_moe_cold_start = False
    cfg.compilation_config.static_forward_context = {}
    cfg.speculative_config = None
    return cfg


def _make_connector(
    store=None, role=KVConnectorRole.SCHEDULER, block_size=4
):
    from vllm_spyre.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
        InMemorySpyreConnector,
    )
    cfg = _make_vllm_config_mock(block_size=block_size)
    store = store or InMemoryKVStore()
    return InMemorySpyreConnector(
        vllm_config=cfg, role=role,
        kv_cache_config=None, store=store,
    )


def _make_staging_caches(
    num_layers=2, num_blocks=8, block_size=2,
    num_kv_heads=1, head_dim=2,
):
    caches = {}
    for i in range(num_layers):
        name = f"model.layers.{i}.self_attn"
        caches[name] = torch.zeros(
            (2, num_blocks, block_size, num_kv_heads, head_dim),
            dtype=torch.float32,
        )
    return caches


def _make_request(req_id, prompt_token_ids, all_token_ids=None):
    """Create a Request-like mock with the fields the connector reads."""
    req = MagicMock()
    req.request_id = req_id
    req.prompt_token_ids = list(prompt_token_ids)
    req.all_token_ids = list(all_token_ids or prompt_token_ids)
    req.num_computed_tokens = 0
    req.num_external_computed_tokens = 0
    return req


def _make_blocks(block_ids):
    """Create a KVCacheBlocks-like mock. Single KV cache group."""
    blocks = MagicMock()
    blocks.get_block_ids.return_value = (list(block_ids),)
    return blocks


def _make_scheduler_output():
    so = MagicMock()
    so.finished_req_ids = set()
    so.kv_connector_metadata = None
    return so


# ---------------------------------------------------------------------------
# Full End-to-End Reuse Test
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestEndToEndKVReuse:
    """Request A prefill saves, Request B loads via scheduler matching."""

    def test_full_reuse_flow(self):
        """
        Complete flow:
        1. Create scheduler-side + worker-side connectors sharing a store
        2. Req A (prompt [1,2,3,4,5,6,7,8]) → scheduler side: update_state → build_meta
        3. Worker side: bind meta → forward → wait_for_save → saves KV
        4. Scheduler side: request_finished records A
        5. Req B (same prompt) → get_num_new_matched_tokens finds A
        6. Scheduler allocates new blocks for B → update_state → build_meta
        7. Worker side: bind meta → start_load_kv → loads A's KV into B's blocks
        8. Verify B's staging has A's data
        """
        block_size = 4
        store = InMemoryKVStore()

        # Scheduler-side connector
        sched = _make_connector(store=store, role=KVConnectorRole.SCHEDULER, block_size=block_size)
        # Worker-side connector
        worker = _make_connector(store=store, role=KVConnectorRole.WORKER, block_size=block_size)

        # Register staging caches on worker
        staging = _make_staging_caches(
            num_layers=2, num_blocks=8, block_size=block_size,
            num_kv_heads=1, head_dim=2,
        )
        worker.register_kv_caches(staging)

        # --- Step 1: Request A prefill ---
        prompt_a = [10, 20, 30, 40, 50, 60, 70, 80]  # 8 tokens = 2 blocks
        req_a = _make_request("req-A", prompt_a)
        blocks_a = _make_blocks([0, 1])

        # Scheduler side
        matched, is_async = sched.get_num_new_matched_tokens(req_a, 0)
        assert matched == 0  # No saved requests yet
        assert is_async is False

        sched.update_state_after_alloc(req_a, blocks_a, num_external_tokens=0)
        so_a = _make_scheduler_output()
        meta_a = sched.build_connector_meta(so_a)

        # Verify it's a store request
        assert isinstance(meta_a, SpyreConnectorMeta)
        assert len(meta_a.requests) == 1
        assert meta_a.requests[0].is_store is True
        assert meta_a.requests[0].req_id == "req-A"
        assert meta_a.requests[0].block_ids == [0, 1]

        # Worker side: fill staging with "computed" data, then save
        for layer_name in staging:
            staging[layer_name][0][0].fill_(100.0)  # K, block 0
            staging[layer_name][1][0].fill_(200.0)  # V, block 0
            staging[layer_name][0][1].fill_(300.0)  # K, block 1
            staging[layer_name][1][1].fill_(400.0)  # V, block 1

        worker.bind_connector_metadata(meta_a)
        worker.wait_for_save()
        worker.clear_connector_metadata()

        # Verify data is in the store
        assert store.size > 0
        k_entry = store.get(StoreKey("req-A", 0, 0, KVKind.K))
        assert k_entry is not None

        # Scheduler side: request_finished records A
        sched.request_finished(req_a, [0, 1])

        # --- Step 2: Request B with same prompt prefix ---
        prompt_b = [10, 20, 30, 40, 50, 60, 70, 80]  # Exact same
        req_b = _make_request("req-B", prompt_b)

        # Scheduler matches
        matched, is_async = sched.get_num_new_matched_tokens(req_b, 0)
        assert matched == 8  # All 8 tokens matched (2 blocks * 4 tokens/block)
        assert is_async is False

        # Scheduler allocates new blocks for B
        blocks_b = _make_blocks([4, 5])
        sched.update_state_after_alloc(req_b, blocks_b, num_external_tokens=8)
        so_b = _make_scheduler_output()
        meta_b = sched.build_connector_meta(so_b)

        # Verify it's a load request with correct mapping
        assert len(meta_b.requests) == 1
        load_req = meta_b.requests[0]
        assert load_req.is_store is False
        assert load_req.req_id == "req-B"
        assert load_req.source_req_id == "req-A"
        assert load_req.block_ids == [4, 5]
        # Block mapping: source block 0 -> dest block 4, source block 1 -> dest block 5
        assert load_req.block_mapping == [(0, 4), (1, 5)]

        # Worker side: zero staging, bind load meta, execute load
        for layer_name in staging:
            staging[layer_name].zero_()

        worker.bind_connector_metadata(meta_b)
        worker.start_load_kv(MagicMock())

        # Verify B's staging blocks have A's data
        for layer_name in sorted(staging.keys()):
            assert torch.equal(
                staging[layer_name][0][4],
                torch.full((block_size, 1, 2), 100.0),
            ), f"K block 4 mismatch in {layer_name}"
            assert torch.equal(
                staging[layer_name][1][4],
                torch.full((block_size, 1, 2), 200.0),
            ), f"V block 4 mismatch in {layer_name}"
            assert torch.equal(
                staging[layer_name][0][5],
                torch.full((block_size, 1, 2), 300.0),
            ), f"K block 5 mismatch in {layer_name}"
            assert torch.equal(
                staging[layer_name][1][5],
                torch.full((block_size, 1, 2), 400.0),
            ), f"V block 5 mismatch in {layer_name}"

        # No load errors
        assert worker.get_block_ids_with_load_errors() == set()

        # get_finished reports recving
        _, recving = worker.get_finished({"req-B"})
        assert recving == {"req-B"}

        worker.clear_connector_metadata()


# ---------------------------------------------------------------------------
# Partial Prefix + Block Alignment Tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestBlockAlignedMatching:
    """Verify get_num_new_matched_tokens returns block-aligned counts."""

    def test_exact_block_aligned_match(self):
        """8 tokens with block_size=4 → 8 matched."""
        sched = _make_connector(block_size=4)
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        req_a = _make_request("req-A", prompt)
        sched.request_finished(req_a, [0, 1])

        req_b = _make_request("req-B", prompt)
        matched, _ = sched.get_num_new_matched_tokens(req_b, 0)
        assert matched == 8

    def test_partial_prefix_block_aligned(self):
        """Source has 8 tokens, dest has 12 tokens → 8 matched (prefix match)."""
        sched = _make_connector(block_size=4)
        prompt_a = [1, 2, 3, 4, 5, 6, 7, 8]
        req_a = _make_request("req-A", prompt_a)
        sched.request_finished(req_a, [0, 1])

        prompt_b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        req_b = _make_request("req-B", prompt_b)
        matched, _ = sched.get_num_new_matched_tokens(req_b, 0)
        assert matched == 8  # A's full 8 tokens matched as prefix of B

    def test_non_block_aligned_saved_request(self):
        """Source has 6 tokens with block_size=4 → 4 matched (round down)."""
        sched = _make_connector(block_size=4)
        prompt_a = [1, 2, 3, 4, 5, 6]  # 6 tokens, only 1 full block
        req_a = _make_request("req-A", prompt_a)
        sched.request_finished(req_a, [0, 1])

        req_b = _make_request("req-B", [1, 2, 3, 4, 5, 6, 7, 8])
        matched, _ = sched.get_num_new_matched_tokens(req_b, 0)
        assert matched == 4  # Round down 6 to 4 (1 block)

    def test_no_match_different_prompt(self):
        """Completely different prompt → 0 matched."""
        sched = _make_connector(block_size=4)
        prompt_a = [1, 2, 3, 4, 5, 6, 7, 8]
        req_a = _make_request("req-A", prompt_a)
        sched.request_finished(req_a, [0, 1])

        req_b = _make_request("req-B", [99, 98, 97, 96])
        matched, _ = sched.get_num_new_matched_tokens(req_b, 0)
        assert matched == 0

    def test_no_match_when_no_saved_requests(self):
        """Empty registry → 0 matched."""
        sched = _make_connector(block_size=4)
        req = _make_request("req-1", [1, 2, 3, 4])
        matched, _ = sched.get_num_new_matched_tokens(req, 0)
        assert matched == 0

    def test_shorter_prompt_than_saved_partial_match(self):
        """B's prompt is shorter than A's but is a prefix → matched up to B's length."""
        sched = _make_connector(block_size=4)
        prompt_a = [1, 2, 3, 4, 5, 6, 7, 8]
        req_a = _make_request("req-A", prompt_a)
        sched.request_finished(req_a, [0, 1])

        req_b = _make_request("req-B", [1, 2, 3, 4])
        matched, _ = sched.get_num_new_matched_tokens(req_b, 0)
        assert matched == 4  # A[:4] == B[:4], block-aligned to 4

    def test_sub_block_prefix_returns_zero(self):
        """Prefix shorter than one block → 0 matched."""
        sched = _make_connector(block_size=4)
        prompt_a = [1, 2]  # 2 tokens < block_size
        req_a = _make_request("req-A", prompt_a)
        sched.request_finished(req_a, [0])

        req_b = _make_request("req-B", [1, 2, 3, 4, 5, 6])
        matched, _ = sched.get_num_new_matched_tokens(req_b, 0)
        assert matched == 0  # 2 tokens rounds down to 0

    def test_side_effect_free(self):
        """Calling get_num_new_matched_tokens twice returns the same result."""
        sched = _make_connector(block_size=4)
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        req_a = _make_request("req-A", prompt)
        sched.request_finished(req_a, [0, 1])

        req_b = _make_request("req-B", prompt)
        m1, _ = sched.get_num_new_matched_tokens(req_b, 0)
        m2, _ = sched.get_num_new_matched_tokens(req_b, 0)
        assert m1 == m2 == 8

    def test_subtracts_local_computed_tokens(self):
        """Reported external tokens should exclude already local tokens."""
        sched = _make_connector(block_size=4)
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        req_a = _make_request("req-A", prompt)
        sched.request_finished(req_a, [0, 1])

        req_b = _make_request("req-B", prompt)
        matched, _ = sched.get_num_new_matched_tokens(req_b, num_computed_tokens=4)
        assert matched == 4

    def test_non_aligned_local_tokens_do_not_overreport(self):
        """If local tokens are not block-aligned, keep external report conservative."""
        sched = _make_connector(block_size=4)
        prompt = [1, 2, 3, 4]
        req_a = _make_request("req-A", prompt)
        sched.request_finished(req_a, [0])

        req_b = _make_request("req-B", prompt)
        matched, _ = sched.get_num_new_matched_tokens(req_b, num_computed_tokens=2)
        assert matched == 0


# ---------------------------------------------------------------------------
# Load Miss / Failure Policy Tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestLoadMissFailurePolicy:
    """Verify load miss behavior and error reporting."""

    def test_load_miss_reports_error_blocks(self):
        """When store has no data, load misses are reported."""
        store = InMemoryKVStore()
        worker = _make_connector(store=store, role=KVConnectorRole.WORKER, block_size=4)
        staging = _make_staging_caches(
            num_layers=1, num_blocks=8, block_size=4,
            num_kv_heads=1, head_dim=2,
        )
        worker.register_kv_caches(staging)

        # Load request for data that doesn't exist in store
        meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-B", block_ids=[4, 5],
                    is_store=False, source_req_id="req-A",
                    block_mapping=[(0, 4), (1, 5)],
                ),
            ],
            layer_names=sorted(staging.keys()),
        )
        worker.bind_connector_metadata(meta)
        worker.start_load_kv(MagicMock())

        # Both dest blocks should be in error set
        errors = worker.get_block_ids_with_load_errors()
        assert 4 in errors
        assert 5 in errors

        # Request still reported as finished (even with errors)
        _, recving = worker.get_finished({"req-B"})
        assert recving == {"req-B"}

        worker.clear_connector_metadata()

    def test_partial_load_miss(self):
        """When some blocks exist and some don't, only missing ones error."""
        store = InMemoryKVStore()
        worker = _make_connector(store=store, role=KVConnectorRole.WORKER, block_size=4)
        staging = _make_staging_caches(
            num_layers=1, num_blocks=8, block_size=4,
            num_kv_heads=1, head_dim=2,
        )
        worker.register_kv_caches(staging)

        # Pre-populate store with data for block 0 only (layer 0)
        store.put(
            StoreKey("req-A", 0, 0, KVKind.K),
            torch.randn(4, 1, 2),
        )
        store.put(
            StoreKey("req-A", 0, 0, KVKind.V),
            torch.randn(4, 1, 2),
        )
        # Block 1 is missing

        meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-B", block_ids=[4, 5],
                    is_store=False, source_req_id="req-A",
                    block_mapping=[(0, 4), (1, 5)],
                ),
            ],
            layer_names=sorted(staging.keys()),
        )
        worker.bind_connector_metadata(meta)
        worker.start_load_kv(MagicMock())

        errors = worker.get_block_ids_with_load_errors()
        # Block 4 loaded OK, block 5 has missing source
        assert 4 not in errors
        assert 5 in errors

        worker.clear_connector_metadata()

    def test_no_premature_finished_recving(self):
        """A load request should not be in finished_recving before start_load_kv."""
        store = InMemoryKVStore()
        worker = _make_connector(store=store, role=KVConnectorRole.WORKER, block_size=4)
        staging = _make_staging_caches(num_layers=1, num_blocks=8)
        worker.register_kv_caches(staging)

        meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-B", block_ids=[4],
                    is_store=False, source_req_id="req-A",
                ),
            ],
            layer_names=sorted(staging.keys()),
        )
        worker.bind_connector_metadata(meta)

        # Before start_load_kv: should NOT report finished
        _, recving = worker.get_finished({"req-B"})
        assert recving is None

        # After start_load_kv: should report finished
        worker.start_load_kv(MagicMock())
        _, recving = worker.get_finished({"req-B"})
        assert recving == {"req-B"}

        worker.clear_connector_metadata()

    def test_no_premature_finished_sending(self):
        """A store request should not be in finished_sending before wait_for_save."""
        store = InMemoryKVStore()
        worker = _make_connector(store=store, role=KVConnectorRole.WORKER, block_size=4)
        staging = _make_staging_caches(num_layers=1, num_blocks=8)
        worker.register_kv_caches(staging)

        meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-A", block_ids=[0],
                    is_store=True,
                ),
            ],
            layer_names=sorted(staging.keys()),
        )
        worker.bind_connector_metadata(meta)

        # Before wait_for_save: should NOT report finished
        sending, _ = worker.get_finished({"req-A"})
        assert sending is None

        # After wait_for_save: should report finished
        worker.wait_for_save()
        sending, _ = worker.get_finished({"req-A"})
        assert sending == {"req-A"}

        worker.clear_connector_metadata()


# ---------------------------------------------------------------------------
# Scheduler-side update_state_after_alloc correctness
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestUpdateStateAfterAlloc:
    """Verify update_state_after_alloc produces correct metadata."""

    def test_store_request_when_no_external_tokens(self):
        sched = _make_connector(block_size=4)
        req = _make_request("req-A", [1, 2, 3, 4])
        blocks = _make_blocks([0])
        sched.update_state_after_alloc(req, blocks, num_external_tokens=0)

        so = _make_scheduler_output()
        meta = sched.build_connector_meta(so)
        assert len(meta.requests) == 1
        assert meta.requests[0].is_store is True
        assert meta.requests[0].block_ids == [0]

    def test_load_request_with_external_tokens(self):
        sched = _make_connector(block_size=4)

        # First, register a saved request
        req_a = _make_request("req-A", [1, 2, 3, 4, 5, 6, 7, 8])
        sched.request_finished(req_a, [0, 1])

        # Now request B matches
        req_b = _make_request("req-B", [1, 2, 3, 4, 5, 6, 7, 8])
        matched, _ = sched.get_num_new_matched_tokens(req_b, 0)
        assert matched == 8

        blocks_b = _make_blocks([3, 4])
        sched.update_state_after_alloc(req_b, blocks_b, num_external_tokens=8)

        so = _make_scheduler_output()
        meta = sched.build_connector_meta(so)
        assert len(meta.requests) == 1
        load_req = meta.requests[0]
        assert load_req.is_store is False
        assert load_req.source_req_id == "req-A"
        assert load_req.block_mapping == [(0, 3), (1, 4)]

    def test_load_mapping_offsets_by_local_computed_blocks(self):
        """External mapping should start after local hit blocks."""
        sched = _make_connector(block_size=4)

        # Source has 3 blocks available in store registry.
        req_a = _make_request("req-A", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        sched.request_finished(req_a, [10, 11, 12])

        # New request shares the same prompt, but has 1 local block already.
        req_b = _make_request("req-B", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        matched, _ = sched.get_num_new_matched_tokens(req_b, num_computed_tokens=4)
        assert matched == 8

        # Flat blocks include local block at index 0, then external-loaded blocks.
        blocks_b = _make_blocks([200, 201, 202, 203])
        sched.update_state_after_alloc(req_b, blocks_b, num_external_tokens=8)

        so = _make_scheduler_output()
        meta = sched.build_connector_meta(so)
        load_req = meta.requests[0]

        assert load_req.is_store is False
        assert load_req.source_req_id == "req-A"
        # Must map source blocks [11, 12] to destination [201, 202].
        assert load_req.block_mapping == [(11, 201), (12, 202)]

    def test_fallback_to_store_when_source_missing(self):
        """If update_state_after_alloc is called with external tokens but
        no source was recorded, fall back to a store request."""
        sched = _make_connector(block_size=4)
        req = _make_request("req-B", [1, 2, 3, 4])
        blocks = _make_blocks([0])
        # Directly call with external_tokens > 0, no prior match
        sched.update_state_after_alloc(req, blocks, num_external_tokens=4)

        so = _make_scheduler_output()
        meta = sched.build_connector_meta(so)
        assert len(meta.requests) == 1
        assert meta.requests[0].is_store is True  # Fell back to store

    def test_fallback_to_store_when_mapping_is_incomplete(self):
        """If external token count cannot be fully mapped, fall back to store."""
        sched = _make_connector(block_size=4)

        # Deliberately inconsistent saved entry:
        # prompt length implies 2 external blocks, but only 1 source block id.
        req_a = _make_request("req-A", [1, 2, 3, 4, 5, 6, 7, 8])
        sched.request_finished(req_a, [10])

        req_b = _make_request("req-B", [1, 2, 3, 4, 5, 6, 7, 8])
        matched, _ = sched.get_num_new_matched_tokens(req_b, 0)
        assert matched == 8

        blocks_b = _make_blocks([20, 21])
        sched.update_state_after_alloc(req_b, blocks_b, num_external_tokens=8)

        so = _make_scheduler_output()
        meta = sched.build_connector_meta(so)
        assert len(meta.requests) == 1
        assert meta.requests[0].is_store is True


# ---------------------------------------------------------------------------
# Best-match selection
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestBestMatchSelection:
    """When multiple saved requests match, use the longest prefix."""

    def test_longest_prefix_wins(self):
        sched = _make_connector(block_size=4)

        # Save two requests with different prefix lengths
        req_short = _make_request("req-short", [1, 2, 3, 4])
        sched.request_finished(req_short, [0])

        req_long = _make_request("req-long", [1, 2, 3, 4, 5, 6, 7, 8])
        sched.request_finished(req_long, [0, 1])

        # New request matches both
        req_new = _make_request("req-new", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        matched, _ = sched.get_num_new_matched_tokens(req_new, 0)
        assert matched == 8  # Long prefix wins (8 tokens vs 4)
