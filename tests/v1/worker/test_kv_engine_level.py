"""Phase 5: Engine-level E2E tests, async-ready semantics, and metrics.

Validates:
  1. SchedulerSimulator drives the real connector call sequence
  2. Strict "no premature finish" invariant at engine level
  3. Async-ready per-layer load tracking
  4. SpyreConnectorStats accumulation, reset, and propagation
  5. Metrics counters match actual operations
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)

from vllm_spyre.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
    InMemorySpyreConnector,
    _SavedRequest,
)
from vllm_spyre.distributed.kv_transfer.kv_connector.v1.metadata import (
    InMemoryKVStore,
    SpyreConnectorStats,
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


class SchedulerSimulator:
    """Simulates the real vLLM scheduler's connector call sequence.

    Mirrors the actual call sites in vllm/v1/core/sched/scheduler.py:
      1. get_num_new_matched_tokens  (line ~609)
      2. update_state_after_alloc    (line ~735)
      3. build_connector_meta        (line ~876)
      4. request_finished            (line ~1880)

    The worker side is driven through SpyreKVConnectorBridge matching
    the real model runner pattern.
    """

    def __init__(self, block_size=4, num_layers=2, num_blocks=16):
        self.block_size = block_size
        self.store = InMemoryKVStore()
        cfg = _make_vllm_config_mock(block_size=block_size)

        self.sched = InMemorySpyreConnector(
            vllm_config=cfg, role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None, store=self.store,
        )
        self.worker = InMemorySpyreConnector(
            vllm_config=cfg, role=KVConnectorRole.WORKER,
            kv_cache_config=None, store=self.store,
        )
        self.staging = _make_staging_caches(
            num_layers=num_layers, num_blocks=num_blocks,
            block_size=block_size,
        )
        self.worker.register_kv_caches(self.staging)

        self.bridge = SpyreKVConnectorBridge(cfg)
        self.bridge._kv_connector = self.worker

        # Track all steps for invariant checking
        self._step_history: list[dict] = []

    def schedule_and_execute(
        self,
        request,
        block_ids,
        num_computed_tokens=0,
        fill_staging=None,
        is_finished=True,
    ):
        """Run one full scheduler→worker step for a single request.

        Returns dict with keys: matched, is_store, output, meta
        """
        # --- Scheduler side (mirrors real scheduler.py) ---
        matched, is_async = self.sched.get_num_new_matched_tokens(
            request, num_computed_tokens,
        )

        blocks = _make_blocks(block_ids)
        self.sched.update_state_after_alloc(
            request, blocks, num_external_tokens=matched,
        )

        so_build = MagicMock()
        meta = self.sched.build_connector_meta(so_build)

        # --- Worker side via bridge ---
        so = MagicMock()
        so.kv_connector_metadata = meta
        so.finished_req_ids = {request.request_id} if is_finished else set()
        so.preempted_req_ids = set()

        active = self.bridge.begin_step(so)
        output = None
        if active:
            if fill_staging:
                for (layer, kv_dim, bid), val in fill_staging.items():
                    self.staging[layer][kv_dim][bid].fill_(val)

            self.bridge.before_forward(so)
            self.bridge.after_forward(so)
            output = self.bridge.finish_step()

        # --- Scheduler side: request_finished ---
        if is_finished:
            self.sched.request_finished(request, block_ids)

        is_store = meta.requests[0].is_store if meta.requests else True

        result = {
            "matched": matched,
            "is_store": is_store,
            "output": output,
            "meta": meta,
        }
        self._step_history.append(result)
        return result


# ---------------------------------------------------------------------------
# Engine-level E2E with SchedulerSimulator
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestEngineE2E:
    """E2E tests using SchedulerSimulator driving real connector call sequence."""

    def test_two_requests_same_prefix_reuse(self):
        """Request A saves, Request B reuses via scheduler matching."""
        sim = SchedulerSimulator(block_size=4)

        prompt = [10, 20, 30, 40, 50, 60, 70, 80]

        # Request A: no match, saves
        req_a = _make_request("req-A", prompt)
        fill = {}
        for layer in sim.staging:
            fill[(layer, 0, 0)] = 1.0
            fill[(layer, 1, 0)] = 2.0
            fill[(layer, 0, 1)] = 3.0
            fill[(layer, 1, 1)] = 4.0

        res_a = sim.schedule_and_execute(
            req_a, [0, 1], fill_staging=fill,
        )
        assert res_a["matched"] == 0
        assert res_a["is_store"] is True
        assert res_a["output"].finished_sending == {"req-A"}

        # Request B: same prompt, matches
        req_b = _make_request("req-B", prompt)
        for layer in sim.staging:
            sim.staging[layer].zero_()

        res_b = sim.schedule_and_execute(req_b, [4, 5])
        assert res_b["matched"] == 8
        assert res_b["is_store"] is False
        assert res_b["output"].finished_recving == {"req-B"}
        assert res_b["output"].invalid_block_ids == set()

        # Verify data integrity
        for layer in sorted(sim.staging.keys()):
            assert torch.equal(
                sim.staging[layer][0][4],
                torch.full((4, 1, 2), 1.0),
            )
            assert torch.equal(
                sim.staging[layer][1][5],
                torch.full((4, 1, 2), 4.0),
            )

    def test_get_num_new_matched_tokens_drives_external_reuse(self):
        """Prove get_num_new_matched_tokens controls whether external
        tokens are loaded (load-mode) vs recomputed (store-mode)."""
        sim = SchedulerSimulator(block_size=4)

        prompt = [1, 2, 3, 4]

        # A saves
        req_a = _make_request("req-A", prompt)
        res_a = sim.schedule_and_execute(req_a, [0])
        assert res_a["matched"] == 0
        assert res_a["is_store"] is True

        # B with same prompt → matched > 0 → load mode
        req_b = _make_request("req-B", prompt)
        res_b = sim.schedule_and_execute(req_b, [1])
        assert res_b["matched"] == 4
        assert res_b["is_store"] is False
        assert res_b["meta"].requests[0].source_req_id == "req-A"

        # C with different prompt → matched == 0 → store mode
        req_c = _make_request("req-C", [99, 98, 97, 96])
        res_c = sim.schedule_and_execute(req_c, [2])
        assert res_c["matched"] == 0
        assert res_c["is_store"] is True

    def test_three_request_chain(self):
        """A saves, B reuses A, C reuses B (both match against A's prefix)."""
        sim = SchedulerSimulator(block_size=4)

        prompt = [1, 2, 3, 4, 5, 6, 7, 8]

        req_a = _make_request("req-A", prompt)
        sim.schedule_and_execute(req_a, [0, 1])

        req_b = _make_request("req-B", prompt)
        res_b = sim.schedule_and_execute(req_b, [2, 3])
        assert res_b["matched"] == 8

        # B finishes and saves too. C can match B or A.
        req_c = _make_request("req-C", prompt + [9, 10, 11, 12])
        res_c = sim.schedule_and_execute(req_c, [4, 5, 6])

        # C should match 8 tokens from A or B (both have 8-token prefix)
        assert res_c["matched"] == 8
        assert res_c["is_store"] is False


# ---------------------------------------------------------------------------
# Strict "no premature finish" invariants
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestNoPrematureFinishInvariant:
    """Engine-level invariant: finished_* only after actual load/save."""

    def test_finished_recving_only_after_start_load_kv(self):
        """Through the bridge, finished_recving appears only when
        start_load_kv has actually been called."""
        sim = SchedulerSimulator(block_size=4)

        prompt = [1, 2, 3, 4]
        req_a = _make_request("req-A", prompt)
        sim.schedule_and_execute(req_a, [0])

        # B will be a load request
        req_b = _make_request("req-B", prompt)
        matched, _ = sim.sched.get_num_new_matched_tokens(req_b, 0)
        assert matched == 4

        blocks_b = _make_blocks([1])
        sim.sched.update_state_after_alloc(req_b, blocks_b, num_external_tokens=4)
        so_build = MagicMock()
        meta = sim.sched.build_connector_meta(so_build)

        # Bind metadata only (no load yet)
        sim.worker.bind_connector_metadata(meta)

        # INVARIANT: No finished_recving before load
        _, recving = sim.worker.get_finished({"req-B"})
        assert recving is None

        # Now load
        sim.worker.start_load_kv(MagicMock())

        # INVARIANT: finished_recving appears after load
        _, recving = sim.worker.get_finished({"req-B"})
        assert recving == {"req-B"}

        sim.worker.clear_connector_metadata()

    def test_finished_sending_only_after_wait_for_save(self):
        """finished_sending only after wait_for_save (bulk save trigger)."""
        sim = SchedulerSimulator(block_size=4)

        prompt = [1, 2, 3, 4]
        req_a = _make_request("req-A", prompt)
        blocks_a = _make_blocks([0])

        sim.sched.update_state_after_alloc(req_a, blocks_a, num_external_tokens=0)
        so_build = MagicMock()
        meta = sim.sched.build_connector_meta(so_build)

        sim.worker.bind_connector_metadata(meta)

        # INVARIANT: No finished_sending before save
        sending, _ = sim.worker.get_finished({"req-A"})
        assert sending is None

        sim.worker.wait_for_save()

        # INVARIANT: finished_sending after save
        sending, _ = sim.worker.get_finished({"req-A"})
        assert sending == {"req-A"}

        sim.worker.clear_connector_metadata()

    def test_bridge_enforces_invariants(self):
        """Through the full bridge lifecycle, invariants hold."""
        sim = SchedulerSimulator(block_size=4)

        prompt = [1, 2, 3, 4]
        req_a = _make_request("req-A", prompt)

        # Execute a store step
        res = sim.schedule_and_execute(req_a, [0])

        # After bridge step, sending is reported
        assert res["output"].finished_sending == {"req-A"}

        # Execute a load step
        req_b = _make_request("req-B", prompt)
        res_b = sim.schedule_and_execute(req_b, [1])

        assert res_b["output"].finished_recving == {"req-B"}


# ---------------------------------------------------------------------------
# Async-ready per-layer load tracking
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestAsyncReadyPlumbing:
    """Verify per-layer load tracking is correctly wired."""

    def test_layer_load_done_after_start_load(self):
        """All layers marked done after synchronous start_load_kv."""
        sim = SchedulerSimulator(block_size=4, num_layers=3)

        prompt = [1, 2, 3, 4]
        req_a = _make_request("req-A", prompt)
        sim.schedule_and_execute(req_a, [0])

        # After the save step, do a load step
        req_b = _make_request("req-B", prompt)
        matched, _ = sim.sched.get_num_new_matched_tokens(req_b, 0)
        blocks_b = _make_blocks([1])
        sim.sched.update_state_after_alloc(req_b, blocks_b, num_external_tokens=4)
        so = MagicMock()
        meta = sim.sched.build_connector_meta(so)

        sim.worker.bind_connector_metadata(meta)
        sim.worker.start_load_kv(MagicMock())

        # All 3 layers should be marked done
        for layer_name in sim.worker._layer_names:
            assert sim.worker._layer_load_done.get(layer_name) is True

        sim.worker.clear_connector_metadata()

    def test_wait_for_layer_load_is_safe(self):
        """wait_for_layer_load doesn't crash even when called multiple times."""
        sim = SchedulerSimulator(block_size=4)

        prompt = [1, 2, 3, 4]
        req_a = _make_request("req-A", prompt)
        sim.schedule_and_execute(req_a, [0])

        req_b = _make_request("req-B", prompt)
        matched, _ = sim.sched.get_num_new_matched_tokens(req_b, 0)
        blocks_b = _make_blocks([1])
        sim.sched.update_state_after_alloc(req_b, blocks_b, num_external_tokens=4)
        so = MagicMock()
        meta = sim.sched.build_connector_meta(so)

        sim.worker.bind_connector_metadata(meta)
        sim.worker.start_load_kv(MagicMock())

        # Calling wait_for_layer_load should be a no-op (sync backend)
        for layer_name in sim.worker._layer_names:
            sim.worker.wait_for_layer_load(layer_name)

        # Also safe for unknown layers
        sim.worker.wait_for_layer_load("nonexistent.layer")

        sim.worker.clear_connector_metadata()

    def test_load_layer_factored_correctly(self):
        """_load_layer produces correct per-layer counts."""
        store = InMemoryKVStore()
        cfg = _make_vllm_config_mock(block_size=4)
        worker = InMemorySpyreConnector(
            vllm_config=cfg, role=KVConnectorRole.WORKER,
            kv_cache_config=None, store=store,
        )
        staging = _make_staging_caches(
            num_layers=2, num_blocks=8, block_size=4,
        )
        worker.register_kv_caches(staging)

        # Populate store with data for layer 0 only
        from vllm_spyre.distributed.kv_transfer.kv_connector.v1.metadata import (
            KVKind,
            SpyreConnectorMeta,
            SpyreConnectorRequestMeta,
            StoreKey,
        )
        store.put(StoreKey("req-A", 0, 0, KVKind.K), torch.randn(4, 1, 2))
        store.put(StoreKey("req-A", 0, 0, KVKind.V), torch.randn(4, 1, 2))
        # Layer 1 has no data

        meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-B", block_ids=[2],
                    is_store=False, source_req_id="req-A",
                    block_mapping=[(0, 2)],
                ),
            ],
            layer_names=sorted(staging.keys()),
        )

        # Test _load_layer directly
        layer0_load, layer0_miss = worker._load_layer(
            meta, 0, "model.layers.0.self_attn",
        )
        assert layer0_load == 2  # K + V
        assert layer0_miss == 0

        layer1_load, layer1_miss = worker._load_layer(
            meta, 1, "model.layers.1.self_attn",
        )
        assert layer1_load == 0
        assert layer1_miss == 2  # K + V both missing


# ---------------------------------------------------------------------------
# SpyreConnectorStats tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestSpyreConnectorStats:
    """Verify SpyreConnectorStats accumulation and lifecycle."""

    def test_stats_dataclass_basics(self):
        """Fresh stats are empty, record increments."""
        stats = SpyreConnectorStats()
        assert stats.is_empty()

        stats.record("matched_tokens", 8)
        assert not stats.is_empty()
        assert stats.data["matched_tokens"] == 8

        stats.record("matched_tokens", 4)
        assert stats.data["matched_tokens"] == 12

    def test_stats_reset(self):
        """reset() zeroes all counters."""
        stats = SpyreConnectorStats()
        stats.record("saved_blocks", 10)
        stats.record("load_misses", 3)
        stats.reset()
        assert stats.is_empty()
        assert stats.data["saved_blocks"] == 0

    def test_stats_aggregate(self):
        """aggregate() sums counters from another stats object."""
        a = SpyreConnectorStats()
        a.record("loaded_blocks", 5)
        a.record("saved_blocks", 3)

        b = SpyreConnectorStats()
        b.record("loaded_blocks", 7)
        b.record("evictions", 2)

        result = a.aggregate(b)
        assert result.data["loaded_blocks"] == 12
        assert result.data["saved_blocks"] == 3
        assert result.data["evictions"] == 2

    def test_stats_reduce(self):
        """reduce() returns a dict suitable for logging."""
        stats = SpyreConnectorStats()
        stats.record("matched_tokens", 16)
        stats.record("loaded_blocks", 4)

        reduced = stats.reduce()
        assert reduced["matched_tokens"] == 16
        assert reduced["loaded_blocks"] == 4
        assert reduced["saved_blocks"] == 0

    def test_build_kv_connector_stats(self):
        """Class method builds stats from serialized data dict."""
        data = {"matched_tokens": 10, "loaded_blocks": 5,
                "saved_blocks": 3, "load_misses": 1,
                "evictions": 0, "match_attempts": 2}
        stats = InMemorySpyreConnector.build_kv_connector_stats(data)
        assert isinstance(stats, SpyreConnectorStats)
        assert stats.data["matched_tokens"] == 10


# ---------------------------------------------------------------------------
# Metrics assertions: counters match actual operations
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestMetricsCounters:
    """Verify connector metrics track operations accurately."""

    def test_save_increments_saved_blocks(self):
        """saved_blocks counter matches actual save operations."""
        sim = SchedulerSimulator(block_size=4, num_layers=2)

        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        req = _make_request("req-A", prompt)
        sim.schedule_and_execute(req, [0, 1])

        # 2 blocks * 2 layers * 2 (K+V) = 8 store entries
        metrics = sim.worker.get_cumulative_metrics()
        assert metrics["blocks_saved"] == 8

    def test_load_increments_loaded_blocks(self):
        """loaded_blocks counter matches actual load operations."""
        sim = SchedulerSimulator(block_size=4, num_layers=2)

        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        req_a = _make_request("req-A", prompt)
        sim.schedule_and_execute(req_a, [0, 1])

        req_b = _make_request("req-B", prompt)
        sim.schedule_and_execute(req_b, [4, 5])

        # 2 blocks * 2 layers * 2 (K+V) = 8 loaded entries
        metrics = sim.worker.get_cumulative_metrics()
        assert metrics["blocks_loaded"] == 8

    def test_load_miss_increments_blocks_missing(self):
        """blocks_missing counter tracks load failures."""
        sim = SchedulerSimulator(block_size=4, num_layers=1)

        # Fake a saved request without actual data
        sim.sched._saved_requests["req-A"] = _SavedRequest(
            req_id="req-A",
            prompt_token_ids=(1, 2, 3, 4),
            block_ids=[0],
            num_tokens=4,
        )

        req_b = _make_request("req-B", [1, 2, 3, 4])
        sim.schedule_and_execute(req_b, [2])

        metrics = sim.worker.get_cumulative_metrics()
        # 1 block * 1 layer * 2 (K+V) = 2 misses
        assert metrics["blocks_missing"] == 2

    def test_stats_propagate_through_bridge_output(self):
        """Stats from get_kv_connector_stats appear in bridge output."""
        sim = SchedulerSimulator(block_size=4, num_layers=2)

        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        req_a = _make_request("req-A", prompt)
        res_a = sim.schedule_and_execute(req_a, [0, 1])

        # The bridge output should have stats from the save
        stats = res_a["output"].kv_connector_stats
        assert stats is not None
        assert isinstance(stats, SpyreConnectorStats)
        assert stats.data["saved_blocks"] == 8  # 2 blocks * 2 layers * 2

    def test_stats_reset_after_collection(self):
        """After get_kv_connector_stats, counters reset for next interval."""
        sim = SchedulerSimulator(block_size=4, num_layers=1)

        prompt = [1, 2, 3, 4]
        req_a = _make_request("req-A", prompt)
        sim.schedule_and_execute(req_a, [0])

        # First collection should have data
        sim.worker.get_kv_connector_stats()
        # Stats were already collected by bridge, so worker stats are reset
        # Let's do another save to generate fresh stats
        req_b = _make_request("req-B", [10, 20, 30, 40])
        # Don't go through bridge — call connector directly
        sim.sched.update_state_after_alloc(
            req_b, _make_blocks([1]), num_external_tokens=0,
        )
        so = MagicMock()
        meta = sim.sched.build_connector_meta(so)
        sim.worker.bind_connector_metadata(meta)
        sim.worker.wait_for_save()

        stats = sim.worker.get_kv_connector_stats()
        assert stats is not None
        assert stats.data["saved_blocks"] == 2  # 1 block * 1 layer * 2

        # Second call should be empty (reset happened)
        stats2 = sim.worker.get_kv_connector_stats()
        assert stats2 is None

        sim.worker.clear_connector_metadata()

    def test_matched_tokens_counter(self):
        """matched_tokens stat tracks tokens matched by scheduler."""
        sim = SchedulerSimulator(block_size=4)

        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        req_a = _make_request("req-A", prompt)
        sim.schedule_and_execute(req_a, [0, 1])

        # Match happens on scheduler side — check sched stats
        req_b = _make_request("req-B", prompt)
        sim.sched.get_num_new_matched_tokens(req_b, 0)

        stats = sim.sched.get_kv_connector_stats()
        assert stats is not None
        assert stats.data["matched_tokens"] == 8
        assert stats.data["match_attempts"] >= 1

    def test_eviction_counter(self):
        """evictions stat tracks LRU evictions."""
        with patch.dict(
            "os.environ",
            {"VLLM_SPYRE_KV_REUSE_REGISTRY_MAX_SIZE": "2"},
        ), patch("vllm_spyre.envs._cache", {}):
            cfg = _make_vllm_config_mock(block_size=4)
            sched = InMemorySpyreConnector(
                vllm_config=cfg, role=KVConnectorRole.SCHEDULER,
                kv_cache_config=None, store=InMemoryKVStore(),
            )

        for i in range(4):
            req = _make_request(f"req-{i}", [1, 2, 3, 4])
            sched.request_finished(req, [i])

        # 2 evictions (cap=2, added 4 items)
        stats = sched.get_kv_connector_stats()
        assert stats is not None
        assert stats.data["evictions"] == 2

    def test_cumulative_metrics_survive_stats_reset(self):
        """Cumulative metrics (blocks_saved etc) persist across stats resets."""
        sim = SchedulerSimulator(block_size=4, num_layers=1)

        prompt = [1, 2, 3, 4]
        req_a = _make_request("req-A", prompt)
        sim.schedule_and_execute(req_a, [0])

        # Collect (and reset) interval stats
        sim.worker.get_kv_connector_stats()

        # Cumulative should still reflect the work done
        metrics = sim.worker.get_cumulative_metrics()
        assert metrics["blocks_saved"] == 2  # 1 block * 1 layer * 2 (K+V)

        # Do more work
        req_b = _make_request("req-B", [10, 20, 30, 40])
        sim.sched.update_state_after_alloc(
            req_b, _make_blocks([1]), num_external_tokens=0,
        )
        so = MagicMock()
        meta = sim.sched.build_connector_meta(so)
        sim.worker.bind_connector_metadata(meta)
        sim.worker.wait_for_save()
        sim.worker.clear_connector_metadata()

        # Cumulative keeps growing
        metrics2 = sim.worker.get_cumulative_metrics()
        assert metrics2["blocks_saved"] == 4  # 2 + 2
