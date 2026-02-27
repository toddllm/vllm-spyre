"""Phase 4: Runtime integration tests for KV connector.

Tests the full scheduler -> bridge -> worker flow using InstrumentedModelRunner
(a test harness that wires the real InMemorySpyreConnector through the real
SpyreKVConnectorBridge), NOT handcrafted metadata.

Validates:
  1. Scheduler-driven save + reuse through the bridge lifecycle
  2. Load-failure policy propagates through KVConnectorOutput.invalid_block_ids
  3. LRU eviction in the saved-request registry
  4. Mixed local + external token accounting across scheduler steps
  5. No premature finished_* and no stuck request state
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm_spyre.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
    InMemorySpyreConnector,
)
from vllm_spyre.distributed.kv_transfer.kv_connector.v1.metadata import (
    InMemoryKVStore,
)
from vllm_spyre.v1.worker.spyre_kv_connector_bridge import (
    SpyreKVConnectorBridge,
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


def _make_request(req_id, prompt_token_ids, all_token_ids=None):
    req = MagicMock()
    req.request_id = req_id
    req.prompt_token_ids = list(prompt_token_ids)
    req.all_token_ids = list(all_token_ids or prompt_token_ids)
    req.num_computed_tokens = 0
    req.num_external_computed_tokens = 0
    return req


def _make_blocks(block_ids):
    blocks = MagicMock()
    blocks.get_block_ids.return_value = (list(block_ids),)
    return blocks


def _make_scheduler_output(meta=None, finished_req_ids=None):
    so = MagicMock()
    so.kv_connector_metadata = meta
    so.finished_req_ids = finished_req_ids or set()
    so.preempted_req_ids = set()
    return so


def _make_staging_caches(
    num_layers=2, num_blocks=16, block_size=4,
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


class InstrumentedModelRunner:
    """Test harness wiring a real scheduler connector + bridge + worker connector.

    Exercises the same code paths as the production model runner, but without
    needing an actual model or real scheduler.  All connector methods are
    called through the public API, never patched.
    """

    def __init__(
        self,
        block_size: int = 4,
        num_layers: int = 2,
        num_blocks: int = 16,
        store: InMemoryKVStore | None = None,
    ):
        self.block_size = block_size
        self.store = store or InMemoryKVStore()
        cfg = _make_vllm_config_mock(block_size=block_size)

        # Scheduler-side connector (drives token matching + metadata production)
        self.sched = InMemorySpyreConnector(
            vllm_config=cfg, role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None, store=self.store,
        )

        # Worker-side connector (executes load/save on staging tensors)
        self.worker = InMemorySpyreConnector(
            vllm_config=cfg, role=KVConnectorRole.WORKER,
            kv_cache_config=None, store=self.store,
        )

        # Staging caches
        self.staging = _make_staging_caches(
            num_layers=num_layers, num_blocks=num_blocks,
            block_size=block_size,
        )
        self.worker.register_kv_caches(self.staging)

        # Bridge wrapping the worker connector
        self.bridge = SpyreKVConnectorBridge(cfg)
        self.bridge._kv_connector = self.worker

    # -- scheduler helpers --

    def scheduler_match(self, request, num_computed_tokens=0):
        """Run scheduler-side token matching."""
        return self.sched.get_num_new_matched_tokens(
            request, num_computed_tokens,
        )

    def scheduler_alloc(self, request, blocks, num_external_tokens):
        """Run scheduler-side block allocation recording."""
        self.sched.update_state_after_alloc(
            request, blocks, num_external_tokens,
        )

    def scheduler_build_meta(self):
        """Build connector metadata from scheduler pending state."""
        so = _make_scheduler_output()
        return self.sched.build_connector_meta(so)

    def scheduler_finish_request(self, request, block_ids):
        """Notify scheduler that a request completed generation."""
        return self.sched.request_finished(request, block_ids)

    # -- bridge/worker helpers --

    def execute_step(self, meta, finished_req_ids=None, fill_data=None):
        """Drive the bridge lifecycle for one step.

        Args:
            meta: SpyreConnectorMeta from scheduler_build_meta().
            finished_req_ids: Set of req IDs the scheduler says are finished.
            fill_data: Optional dict of {(layer_name, kv_dim, block_id): value}
                       to fill staging before save. If None, staging is not
                       modified before save.

        Returns:
            KVConnectorOutput from the bridge.
        """
        so = _make_scheduler_output(
            meta=meta, finished_req_ids=finished_req_ids or set(),
        )

        active = self.bridge.begin_step(so)
        if not active:
            return self.bridge.finish_step()

        # Optionally fill staging with "computed" data before forward
        if fill_data:
            for (layer_name, kv_dim, block_id), value in fill_data.items():
                self.staging[layer_name][kv_dim][block_id].fill_(value)

        self.bridge.before_forward(so)
        # [simulate model forward]
        self.bridge.after_forward(so)
        return self.bridge.finish_step()


# ---------------------------------------------------------------------------
# Integration: scheduler → bridge → worker save + reuse
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestIntegrationSaveAndReuse:
    """Full scheduler→bridge→worker flow for save + prefix reuse."""

    def test_save_then_reuse_via_bridge(self):
        """
        Req A prefill → scheduler produces store meta → bridge saves KV.
        Req B same prompt → scheduler matches → bridge loads KV.
        Assert: metadata is load-mode, worker reports finished_recving,
        loaded data matches saved data.
        """
        runner = InstrumentedModelRunner(block_size=4, num_blocks=16)

        # --- Request A: prefill (save) ---
        prompt = [10, 20, 30, 40, 50, 60, 70, 80]
        req_a = _make_request("req-A", prompt)
        blocks_a = _make_blocks([0, 1])

        matched, _ = runner.scheduler_match(req_a)
        assert matched == 0

        runner.scheduler_alloc(req_a, blocks_a, num_external_tokens=0)
        meta_a = runner.scheduler_build_meta()

        assert len(meta_a.requests) == 1
        assert meta_a.requests[0].is_store is True

        # Fill staging with known data and execute save via bridge
        fill = {}
        for layer in runner.staging:
            fill[(layer, 0, 0)] = 100.0  # K, block 0
            fill[(layer, 1, 0)] = 200.0  # V, block 0
            fill[(layer, 0, 1)] = 300.0  # K, block 1
            fill[(layer, 1, 1)] = 400.0  # V, block 1

        output_a = runner.execute_step(
            meta_a, finished_req_ids={"req-A"}, fill_data=fill,
        )
        assert output_a is not None
        assert output_a.finished_sending == {"req-A"}

        # Register A in the saved-request registry
        runner.scheduler_finish_request(req_a, [0, 1])

        # --- Request B: same prompt (reuse) ---
        req_b = _make_request("req-B", prompt)

        matched, is_async = runner.scheduler_match(req_b)
        assert matched == 8
        assert is_async is False

        blocks_b = _make_blocks([4, 5])
        runner.scheduler_alloc(req_b, blocks_b, num_external_tokens=8)
        meta_b = runner.scheduler_build_meta()

        # Verify scheduler produced load metadata
        assert len(meta_b.requests) == 1
        load_req = meta_b.requests[0]
        assert load_req.is_store is False
        assert load_req.source_req_id == "req-A"
        assert load_req.block_mapping == [(0, 4), (1, 5)]

        # Zero staging, then execute load via bridge
        for layer in runner.staging:
            runner.staging[layer].zero_()

        output_b = runner.execute_step(
            meta_b, finished_req_ids={"req-B"},
        )
        assert output_b is not None
        assert output_b.finished_recving == {"req-B"}
        assert output_b.invalid_block_ids == set()

        # Verify loaded data matches saved data
        for layer in sorted(runner.staging.keys()):
            assert torch.equal(
                runner.staging[layer][0][4],
                torch.full((4, 1, 2), 100.0),
            ), f"K block 4 mismatch in {layer}"
            assert torch.equal(
                runner.staging[layer][1][4],
                torch.full((4, 1, 2), 200.0),
            ), f"V block 4 mismatch in {layer}"
            assert torch.equal(
                runner.staging[layer][0][5],
                torch.full((4, 1, 2), 300.0),
            ), f"K block 5 mismatch in {layer}"
            assert torch.equal(
                runner.staging[layer][1][5],
                torch.full((4, 1, 2), 400.0),
            ), f"V block 5 mismatch in {layer}"

    def test_reuse_avoids_recompute_for_matched_blocks(self):
        """When B reuses A's prefix, the metadata signals load (not store),
        meaning the worker loads pre-computed KV rather than recomputing."""
        runner = InstrumentedModelRunner(block_size=4)

        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        req_a = _make_request("req-A", prompt)
        blocks_a = _make_blocks([0, 1])

        runner.scheduler_alloc(req_a, blocks_a, num_external_tokens=0)
        meta_a = runner.scheduler_build_meta()
        runner.execute_step(meta_a, finished_req_ids={"req-A"})
        runner.scheduler_finish_request(req_a, [0, 1])

        # B with same prefix
        req_b = _make_request("req-B", prompt)
        matched, _ = runner.scheduler_match(req_b)
        assert matched == 8  # All tokens matched

        blocks_b = _make_blocks([4, 5])
        runner.scheduler_alloc(req_b, blocks_b, num_external_tokens=8)
        meta_b = runner.scheduler_build_meta()

        # The key assertion: B's metadata is load-mode (not store),
        # which means no recompute for those matched blocks.
        assert meta_b.requests[0].is_store is False
        assert meta_b.requests[0].token_count == 8

    def test_no_match_produces_store_metadata(self):
        """When no prefix match, scheduler produces store metadata."""
        runner = InstrumentedModelRunner(block_size=4)

        req = _make_request("req-A", [1, 2, 3, 4])
        blocks = _make_blocks([0])

        matched, _ = runner.scheduler_match(req)
        assert matched == 0

        runner.scheduler_alloc(req, blocks, num_external_tokens=0)
        meta = runner.scheduler_build_meta()

        assert len(meta.requests) == 1
        assert meta.requests[0].is_store is True


# ---------------------------------------------------------------------------
# Integration: load-failure policy end-to-end through bridge
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestIntegrationLoadFailure:
    """Verify load failures propagate through bridge to KVConnectorOutput."""

    def test_missing_blocks_produce_invalid_block_ids(self):
        """Force partial missing blocks → verify invalid_block_ids in output."""
        runner = InstrumentedModelRunner(block_size=4)

        # Save request A with data for blocks 0, 1
        prompt = [10, 20, 30, 40, 50, 60, 70, 80]
        req_a = _make_request("req-A", prompt)
        blocks_a = _make_blocks([0, 1])
        runner.scheduler_alloc(req_a, blocks_a, num_external_tokens=0)
        meta_a = runner.scheduler_build_meta()

        # Only fill block 0, leave block 1 empty in staging
        fill = {}
        for layer in runner.staging:
            fill[(layer, 0, 0)] = 111.0
            fill[(layer, 1, 0)] = 222.0
            # block 1 is NOT filled → will be stored as zeros
        runner.execute_step(meta_a, finished_req_ids={"req-A"}, fill_data=fill)
        runner.scheduler_finish_request(req_a, [0, 1])

        # Now manually delete block 1 from the store to force a miss
        keys_to_remove = [
            k for k in list(runner.store._store.keys())
            if k.block_id == 1
        ]
        for k in keys_to_remove:
            del runner.store._store[k]

        # Request B tries to reuse A
        req_b = _make_request("req-B", prompt)
        matched, _ = runner.scheduler_match(req_b)
        assert matched == 8

        blocks_b = _make_blocks([4, 5])
        runner.scheduler_alloc(req_b, blocks_b, num_external_tokens=8)
        meta_b = runner.scheduler_build_meta()

        output_b = runner.execute_step(
            meta_b, finished_req_ids={"req-B"},
        )

        # Block 5 (dest for missing source block 1) should be in errors
        assert output_b is not None
        assert 5 in output_b.invalid_block_ids
        # Block 4 (dest for source block 0 which exists) should NOT error
        assert 4 not in output_b.invalid_block_ids

    def test_complete_miss_all_blocks_invalid(self):
        """When store has no data at all, all dest blocks are invalid."""
        runner = InstrumentedModelRunner(block_size=4)

        # Register a saved request without actually saving any data
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        # Manually inject into saved-request registry (bypassing save)
        from vllm_spyre.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
            _SavedRequest,
        )
        runner.sched._saved_requests["req-A"] = _SavedRequest(
            req_id="req-A",
            prompt_token_ids=tuple(prompt),
            block_ids=[0, 1],
            num_tokens=8,
        )

        req_b = _make_request("req-B", prompt)
        matched, _ = runner.scheduler_match(req_b)
        assert matched == 8

        blocks_b = _make_blocks([6, 7])
        runner.scheduler_alloc(req_b, blocks_b, num_external_tokens=8)
        meta_b = runner.scheduler_build_meta()

        output_b = runner.execute_step(
            meta_b, finished_req_ids={"req-B"},
        )

        assert output_b is not None
        assert 6 in output_b.invalid_block_ids
        assert 7 in output_b.invalid_block_ids

    def test_no_premature_finished_before_load(self):
        """Finished_recving must not be reported until start_load_kv runs."""
        runner = InstrumentedModelRunner(block_size=4)

        prompt = [1, 2, 3, 4]
        req_a = _make_request("req-A", prompt)
        blocks_a = _make_blocks([0])
        runner.scheduler_alloc(req_a, blocks_a, num_external_tokens=0)
        meta_a = runner.scheduler_build_meta()
        runner.execute_step(meta_a, finished_req_ids={"req-A"})
        runner.scheduler_finish_request(req_a, [0])

        # B matches A
        req_b = _make_request("req-B", prompt)
        matched, _ = runner.scheduler_match(req_b)
        assert matched == 4

        blocks_b = _make_blocks([2])
        runner.scheduler_alloc(req_b, blocks_b, num_external_tokens=4)
        meta_b = runner.scheduler_build_meta()

        # Bind metadata but don't call start_load_kv yet
        runner.worker.bind_connector_metadata(meta_b)
        _, recving = runner.worker.get_finished({"req-B"})
        assert recving is None  # Not yet loaded

        # Now do the load
        runner.worker.start_load_kv(MagicMock())
        _, recving = runner.worker.get_finished({"req-B"})
        assert recving == {"req-B"}

        runner.worker.clear_connector_metadata()

    def test_no_stuck_request_after_load_failure(self):
        """Even with load failures, request is still reported as finished
        (the scheduler should handle recompute via invalid_block_ids)."""
        runner = InstrumentedModelRunner(block_size=4)

        # Fake a saved request with no actual store data
        from vllm_spyre.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
            _SavedRequest,
        )
        runner.sched._saved_requests["req-A"] = _SavedRequest(
            req_id="req-A",
            prompt_token_ids=(1, 2, 3, 4),
            block_ids=[0],
            num_tokens=4,
        )

        req_b = _make_request("req-B", [1, 2, 3, 4])
        matched, _ = runner.scheduler_match(req_b)
        assert matched == 4

        blocks_b = _make_blocks([3])
        runner.scheduler_alloc(req_b, blocks_b, num_external_tokens=4)
        meta_b = runner.scheduler_build_meta()

        output_b = runner.execute_step(
            meta_b, finished_req_ids={"req-B"},
        )

        # Request is NOT stuck — it's in finished_recving
        assert output_b is not None
        assert output_b.finished_recving == {"req-B"}
        # But blocks are invalid
        assert 3 in output_b.invalid_block_ids


# ---------------------------------------------------------------------------
# LRU eviction tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestLRUEviction:
    """Verify saved-request registry respects size cap."""

    def _make_connector_with_cap(self, max_size, block_size=4):
        with patch.dict(
            "os.environ",
            {"VLLM_SPYRE_KV_REUSE_REGISTRY_MAX_SIZE": str(max_size)},
        ), patch("vllm_spyre.envs._cache", {}):
            cfg = _make_vllm_config_mock(block_size=block_size)
            return InMemorySpyreConnector(
                vllm_config=cfg, role=KVConnectorRole.SCHEDULER,
                kv_cache_config=None, store=InMemoryKVStore(),
            )

    def test_eviction_at_cap(self):
        """When cap is 3, adding a 4th entry evicts the oldest."""
        sched = self._make_connector_with_cap(max_size=3)

        for i in range(4):
            req = _make_request(f"req-{i}", [1, 2, 3, 4])
            sched.request_finished(req, [i])

        assert len(sched._saved_requests) == 3
        # req-0 should have been evicted (oldest)
        assert "req-0" not in sched._saved_requests
        # req-1, req-2, req-3 should remain
        assert "req-1" in sched._saved_requests
        assert "req-2" in sched._saved_requests
        assert "req-3" in sched._saved_requests

    def test_eviction_preserves_match_correctness(self):
        """After eviction, matching still works for retained entries."""
        sched = self._make_connector_with_cap(max_size=2)

        req_a = _make_request("req-A", [1, 2, 3, 4])
        sched.request_finished(req_a, [0])

        req_b = _make_request("req-B", [10, 20, 30, 40])
        sched.request_finished(req_b, [1])

        # At cap of 2. Add a third — evicts req-A
        req_c = _make_request("req-C", [100, 200, 300, 400])
        sched.request_finished(req_c, [2])

        # Trying to match against evicted req-A's prompt → 0
        req_new = _make_request("req-new", [1, 2, 3, 4, 5, 6, 7, 8])
        matched, _ = sched.get_num_new_matched_tokens(req_new, 0)
        assert matched == 0

        # Matching against retained req-B → still works
        req_new2 = _make_request("req-new2", [10, 20, 30, 40, 50, 60, 70, 80])
        matched2, _ = sched.get_num_new_matched_tokens(req_new2, 0)
        assert matched2 == 4

    def test_update_existing_moves_to_end(self):
        """Re-finishing an existing request refreshes its LRU position."""
        sched = self._make_connector_with_cap(max_size=3)

        for i in range(3):
            req = _make_request(f"req-{i}", [1, 2, 3, 4])
            sched.request_finished(req, [i])

        # Re-finish req-0 → moves it to end
        req_refresh = _make_request("req-0", [1, 2, 3, 4])
        sched.request_finished(req_refresh, [0])

        # Now add req-3 → should evict req-1 (oldest), not req-0
        req_new = _make_request("req-3", [10, 20, 30, 40])
        sched.request_finished(req_new, [3])

        assert len(sched._saved_requests) == 3
        assert "req-1" not in sched._saved_requests  # evicted
        assert "req-0" in sched._saved_requests  # refreshed
        assert "req-2" in sched._saved_requests
        assert "req-3" in sched._saved_requests

    def test_unlimited_when_cap_is_zero(self):
        """When max_size is 0, no eviction occurs."""
        sched = self._make_connector_with_cap(max_size=0)

        for i in range(100):
            req = _make_request(f"req-{i}", [1, 2, 3, 4])
            sched.request_finished(req, [i])

        assert len(sched._saved_requests) == 100

    def test_cap_of_one(self):
        """Edge case: only the most recent request is retained."""
        sched = self._make_connector_with_cap(max_size=1)

        req_a = _make_request("req-A", [1, 2, 3, 4])
        sched.request_finished(req_a, [0])
        assert len(sched._saved_requests) == 1

        req_b = _make_request("req-B", [10, 20, 30, 40])
        sched.request_finished(req_b, [1])
        assert len(sched._saved_requests) == 1
        assert "req-B" in sched._saved_requests
        assert "req-A" not in sched._saved_requests


# ---------------------------------------------------------------------------
# Mixed local + external token regression
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestMixedLocalExternal:
    """Verify offset mapping when local hits exist alongside external loads."""

    def test_partial_local_partial_external(self):
        """A has 12 tokens (3 blocks). B matches all 12 but has 4 local.
        External = 8 tokens → 2 blocks loaded, offset by 1 local block."""
        runner = InstrumentedModelRunner(block_size=4, num_blocks=32)

        # Save request A: 12 tokens = 3 blocks
        prompt = list(range(1, 13))
        req_a = _make_request("req-A", prompt)
        blocks_a = _make_blocks([0, 1, 2])
        runner.scheduler_alloc(req_a, blocks_a, num_external_tokens=0)
        meta_a = runner.scheduler_build_meta()

        fill = {}
        for layer in runner.staging:
            fill[(layer, 0, 0)] = 10.0
            fill[(layer, 1, 0)] = 20.0
            fill[(layer, 0, 1)] = 30.0
            fill[(layer, 1, 1)] = 40.0
            fill[(layer, 0, 2)] = 50.0
            fill[(layer, 1, 2)] = 60.0
        runner.execute_step(meta_a, finished_req_ids={"req-A"}, fill_data=fill)
        runner.scheduler_finish_request(req_a, [0, 1, 2])

        # Request B: same 12-token prompt, but 4 tokens already local
        req_b = _make_request("req-B", prompt)
        matched, _ = runner.scheduler_match(req_b, num_computed_tokens=4)
        assert matched == 8  # 12 total - 4 local = 8 external

        # Blocks: local block at idx 0, then 2 external + remainder
        blocks_b = _make_blocks([10, 11, 12, 13])
        runner.scheduler_alloc(req_b, blocks_b, num_external_tokens=8)
        meta_b = runner.scheduler_build_meta()

        load_req = meta_b.requests[0]
        assert load_req.is_store is False
        assert load_req.source_req_id == "req-A"
        # External mapping starts after 1 local block:
        # source block_ids[1]=1 -> dest block_ids[1]=11
        # source block_ids[2]=2 -> dest block_ids[2]=12
        assert load_req.block_mapping == [(1, 11), (2, 12)]

        # Execute load via bridge
        for layer in runner.staging:
            runner.staging[layer].zero_()
        output_b = runner.execute_step(
            meta_b, finished_req_ids={"req-B"},
        )

        assert output_b is not None
        assert output_b.finished_recving == {"req-B"}
        assert output_b.invalid_block_ids == set()

        # Verify loaded data at dest blocks
        for layer in sorted(runner.staging.keys()):
            assert torch.equal(
                runner.staging[layer][0][11],
                torch.full((4, 1, 2), 30.0),
            ), f"K block 11 mismatch in {layer}"
            assert torch.equal(
                runner.staging[layer][1][11],
                torch.full((4, 1, 2), 40.0),
            ), f"V block 11 mismatch in {layer}"
            assert torch.equal(
                runner.staging[layer][0][12],
                torch.full((4, 1, 2), 50.0),
            ), f"K block 12 mismatch in {layer}"
            assert torch.equal(
                runner.staging[layer][1][12],
                torch.full((4, 1, 2), 60.0),
            ), f"V block 12 mismatch in {layer}"

    def test_all_local_no_external_load(self):
        """When local tokens cover the full match, external = 0 → store path."""
        runner = InstrumentedModelRunner(block_size=4)

        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        req_a = _make_request("req-A", prompt)
        blocks_a = _make_blocks([0, 1])
        runner.scheduler_alloc(req_a, blocks_a, num_external_tokens=0)
        meta_a = runner.scheduler_build_meta()
        runner.execute_step(meta_a, finished_req_ids={"req-A"})
        runner.scheduler_finish_request(req_a, [0, 1])

        # B has 8 tokens, all 8 already computed locally
        req_b = _make_request("req-B", prompt)
        matched, _ = runner.scheduler_match(req_b, num_computed_tokens=8)
        assert matched == 0  # All tokens are local

    def test_multi_step_accounting(self):
        """Verify offset accounting stays correct across multiple scheduler steps."""
        runner = InstrumentedModelRunner(block_size=4, num_blocks=32)

        # Step 1: Save A (8 tokens)
        prompt_a = [1, 2, 3, 4, 5, 6, 7, 8]
        req_a = _make_request("req-A", prompt_a)
        blocks_a = _make_blocks([0, 1])
        runner.scheduler_alloc(req_a, blocks_a, num_external_tokens=0)
        meta1 = runner.scheduler_build_meta()
        runner.execute_step(meta1, finished_req_ids={"req-A"})
        runner.scheduler_finish_request(req_a, [0, 1])

        # Step 2: Save B (different prompt, 8 tokens)
        prompt_b = [10, 20, 30, 40, 50, 60, 70, 80]
        req_b = _make_request("req-B", prompt_b)
        blocks_b = _make_blocks([2, 3])
        runner.scheduler_alloc(req_b, blocks_b, num_external_tokens=0)
        meta2 = runner.scheduler_build_meta()
        runner.execute_step(meta2, finished_req_ids={"req-B"})
        runner.scheduler_finish_request(req_b, [2, 3])

        # Step 3: C matches A's prefix, D matches B's prefix
        req_c = _make_request("req-C", prompt_a + [9, 10, 11, 12])
        matched_c, _ = runner.scheduler_match(req_c)
        assert matched_c == 8

        req_d = _make_request("req-D", prompt_b + [90, 91, 92, 93])
        matched_d, _ = runner.scheduler_match(req_d)
        assert matched_d == 8

        blocks_c = _make_blocks([10, 11, 12])
        runner.scheduler_alloc(req_c, blocks_c, num_external_tokens=8)
        blocks_d = _make_blocks([13, 14, 15])
        runner.scheduler_alloc(req_d, blocks_d, num_external_tokens=8)

        meta3 = runner.scheduler_build_meta()

        # Both should be load requests
        assert len(meta3.requests) == 2
        load_c = next(r for r in meta3.requests if r.req_id == "req-C")
        load_d = next(r for r in meta3.requests if r.req_id == "req-D")

        assert load_c.is_store is False
        assert load_c.source_req_id == "req-A"
        assert load_c.block_mapping == [(0, 10), (1, 11)]

        assert load_d.is_store is False
        assert load_d.source_req_id == "req-B"
        assert load_d.block_mapping == [(2, 13), (3, 14)]
