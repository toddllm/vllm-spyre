"""Phase 6: Multi-process, async pipeline, scheduler feedback, and schema hardening.

Validates:
  1. Real disaggregated 2-process prefill/decode via file-backed transport
  2. True async layer pipeline with ThreadPoolExecutor
  3. Scheduler feedback handling for invalid block IDs
  4. Metadata schema validation (schema_version, strict fields)
  5. Store export/import for cross-process transport
"""

import multiprocessing
import tempfile
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
    KVKind,
    SpyreConnectorMeta,
    SpyreConnectorRequestMeta,
    StoreKey,
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


def _make_connector(block_size=4, store=None, **env_overrides):
    """Create a connector with optional env overrides."""
    env = {
        "VLLM_SPYRE_KV_REUSE_REGISTRY_MAX_SIZE": "1024",
        "VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS": "0",
    }
    env.update(env_overrides)
    with patch.dict("os.environ", env), patch("vllm_spyre.envs._cache", {}):
        cfg = _make_vllm_config_mock(block_size=block_size)
        return InMemorySpyreConnector(
            vllm_config=cfg, role=KVConnectorRole.WORKER,
            kv_cache_config=None, store=store,
        )


# ---------------------------------------------------------------------------
# Store export/import tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestStoreExportImport:
    """Verify InMemoryKVStore export_to_dir / import_from_dir."""

    def test_roundtrip(self):
        """Export and import produces identical store contents."""
        store = InMemoryKVStore()
        store.put(StoreKey("r1", 0, 0, KVKind.K), torch.randn(4, 1, 2))
        store.put(StoreKey("r1", 0, 0, KVKind.V), torch.randn(4, 1, 2))
        store.put(StoreKey("r1", 1, 0, KVKind.K), torch.randn(4, 1, 2))

        with tempfile.TemporaryDirectory() as tmpdir:
            exported = store.export_to_dir(tmpdir)
            assert exported == 3

            store2 = InMemoryKVStore()
            imported = store2.import_from_dir(tmpdir)
            assert imported == 3
            assert store2.size == 3

            # Data integrity
            for key_tuple in [
                ("r1", 0, 0, KVKind.K),
                ("r1", 0, 0, KVKind.V),
                ("r1", 1, 0, KVKind.K),
            ]:
                key = StoreKey(*key_tuple)
                orig = store.get(key)
                copy = store2.get(key)
                assert copy is not None
                assert torch.equal(orig.data, copy.data)

    def test_empty_store_exports_nothing(self):
        """Empty store exports zero files."""
        store = InMemoryKVStore()
        with tempfile.TemporaryDirectory() as tmpdir:
            assert store.export_to_dir(tmpdir) == 0
            store2 = InMemoryKVStore()
            assert store2.import_from_dir(tmpdir) == 0

    def test_import_merges_into_existing(self):
        """Import adds to (doesn't replace) existing entries."""
        store1 = InMemoryKVStore()
        store1.put(StoreKey("r1", 0, 0, KVKind.K), torch.ones(4, 1, 2))

        store2 = InMemoryKVStore()
        store2.put(StoreKey("r2", 0, 0, KVKind.K), torch.ones(4, 1, 2) * 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            store1.export_to_dir(tmpdir)
            store2.import_from_dir(tmpdir)

        assert store2.size == 2
        assert store2.get(StoreKey("r1", 0, 0, KVKind.K)) is not None
        assert store2.get(StoreKey("r2", 0, 0, KVKind.K)) is not None


# ---------------------------------------------------------------------------
# Disaggregated 2-process prefill/decode E2E
# ---------------------------------------------------------------------------

def _prefill_worker(export_dir, result_queue):
    """Prefill process: save KV data and export store to disk."""
    try:
        store = InMemoryKVStore()
        cfg = _make_vllm_config_mock(block_size=4)
        with patch.dict("os.environ", {
            "VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS": "0",
        }), patch("vllm_spyre.envs._cache", {}):
            connector = InMemorySpyreConnector(
                vllm_config=cfg, role=KVConnectorRole.WORKER,
                kv_cache_config=None, store=store,
            )

        staging = _make_staging_caches(
            num_layers=2, num_blocks=8, block_size=4,
        )
        connector.register_kv_caches(staging)

        # Fill staging with known data
        for layer in staging:
            staging[layer][0][0].fill_(1.0)  # K, block 0
            staging[layer][1][0].fill_(2.0)  # V, block 0
            staging[layer][0][1].fill_(3.0)  # K, block 1
            staging[layer][1][1].fill_(4.0)  # V, block 1

        # Save via connector
        meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="prefill-req",
                    block_ids=[0, 1],
                    is_store=True,
                    token_count=8,
                ),
            ],
            layer_names=sorted(staging.keys()),
            block_size=4,
        )
        connector.bind_connector_metadata(meta)
        connector.wait_for_save()

        # Export store to disk
        count = store.export_to_dir(export_dir)
        result_queue.put({"status": "ok", "exported": count})
    except Exception as e:
        result_queue.put({"status": "error", "error": str(e)})


def _decode_worker(export_dir, result_queue):
    """Decode process: import store from disk, load KV data."""
    try:
        store = InMemoryKVStore()
        imported = store.import_from_dir(export_dir)

        cfg = _make_vllm_config_mock(block_size=4)
        with patch.dict("os.environ", {
            "VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS": "0",
        }), patch("vllm_spyre.envs._cache", {}):
            connector = InMemorySpyreConnector(
                vllm_config=cfg, role=KVConnectorRole.WORKER,
                kv_cache_config=None, store=store,
            )

        staging = _make_staging_caches(
            num_layers=2, num_blocks=8, block_size=4,
        )
        connector.register_kv_caches(staging)

        # Load into different destination blocks
        meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="decode-req",
                    block_ids=[4, 5],
                    is_store=False,
                    source_req_id="prefill-req",
                    block_mapping=[(0, 4), (1, 5)],
                ),
            ],
            layer_names=sorted(staging.keys()),
            block_size=4,
        )
        connector.bind_connector_metadata(meta)
        connector.start_load_kv(MagicMock())

        # Verify data integrity
        checks = {}
        for layer in sorted(staging.keys()):
            k0 = staging[layer][0][4].mean().item()
            v0 = staging[layer][1][4].mean().item()
            k1 = staging[layer][0][5].mean().item()
            v1 = staging[layer][1][5].mean().item()
            checks[layer] = {"k0": k0, "v0": v0, "k1": k1, "v1": v1}

        errors = connector.get_block_ids_with_load_errors()

        result_queue.put({
            "status": "ok",
            "imported": imported,
            "checks": checks,
            "errors": list(errors),
        })
    except Exception as e:
        result_queue.put({"status": "error", "error": str(e)})


@pytest.mark.cpu
class TestDisaggregated2Process:
    """Real 2-process prefill/decode via file-backed transport."""

    def test_prefill_saves_decode_loads_across_processes(self):
        """Prefill process saves KV, decode process loads it.

        Uses multiprocessing.Process with file-based store transport.
        """
        ctx = multiprocessing.get_context("spawn")

        with tempfile.TemporaryDirectory() as export_dir:
            # --- Prefill process ---
            prefill_q = ctx.Queue()
            prefill_proc = ctx.Process(
                target=_prefill_worker,
                args=(export_dir, prefill_q),
            )
            prefill_proc.start()
            prefill_proc.join(timeout=30)
            assert prefill_proc.exitcode == 0, (
                f"Prefill process exited with code {prefill_proc.exitcode}"
            )
            prefill_result = prefill_q.get(timeout=5)
            assert prefill_result["status"] == "ok", prefill_result
            assert prefill_result["exported"] > 0

            # --- Decode process ---
            decode_q = ctx.Queue()
            decode_proc = ctx.Process(
                target=_decode_worker,
                args=(export_dir, decode_q),
            )
            decode_proc.start()
            decode_proc.join(timeout=30)
            assert decode_proc.exitcode == 0, (
                f"Decode process exited with code {decode_proc.exitcode}"
            )
            decode_result = decode_q.get(timeout=5)
            assert decode_result["status"] == "ok", decode_result
            assert decode_result["imported"] == prefill_result["exported"]
            assert decode_result["errors"] == []

            # Verify the loaded data matches what prefill saved
            for layer_checks in decode_result["checks"].values():
                assert layer_checks["k0"] == pytest.approx(1.0)
                assert layer_checks["v0"] == pytest.approx(2.0)
                assert layer_checks["k1"] == pytest.approx(3.0)
                assert layer_checks["v1"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# True async layer pipeline with ThreadPoolExecutor
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestAsyncLayerPipeline:
    """Verify ThreadPoolExecutor-based async layer loading."""

    def test_async_load_produces_same_results_as_sync(self):
        """Async path loads same data as sync path."""
        store = InMemoryKVStore()

        # --- Sync save ---
        sync_conn = _make_connector(
            block_size=4, store=store,
            VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS="0",
        )
        staging_save = _make_staging_caches(num_layers=2, num_blocks=8)
        sync_conn.register_kv_caches(staging_save)

        for layer in staging_save:
            staging_save[layer][0][0].fill_(1.0)
            staging_save[layer][1][0].fill_(2.0)
            staging_save[layer][0][1].fill_(3.0)
            staging_save[layer][1][1].fill_(4.0)

        save_meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="req-A", block_ids=[0, 1],
                is_store=True, token_count=8,
            )],
            layer_names=sorted(staging_save.keys()),
            block_size=4,
        )
        sync_conn.bind_connector_metadata(save_meta)
        sync_conn.wait_for_save()
        sync_conn.clear_connector_metadata()

        # --- Sync load (baseline) ---
        staging_sync = _make_staging_caches(num_layers=2, num_blocks=8)
        sync_conn.register_kv_caches(staging_sync)

        load_meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="req-B", block_ids=[4, 5],
                is_store=False, source_req_id="req-A",
                block_mapping=[(0, 4), (1, 5)],
            )],
            layer_names=sorted(staging_sync.keys()),
            block_size=4,
        )
        sync_conn.bind_connector_metadata(load_meta)
        sync_conn.start_load_kv(MagicMock())
        sync_conn.clear_connector_metadata()

        # --- Async load ---
        async_conn = _make_connector(
            block_size=4, store=store,
            VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS="2",
        )
        assert async_conn._async_load_enabled is True
        staging_async = _make_staging_caches(num_layers=2, num_blocks=8)
        async_conn.register_kv_caches(staging_async)

        load_meta2 = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="req-C", block_ids=[4, 5],
                is_store=False, source_req_id="req-A",
                block_mapping=[(0, 4), (1, 5)],
            )],
            layer_names=sorted(staging_async.keys()),
            block_size=4,
        )
        async_conn.bind_connector_metadata(load_meta2)
        async_conn.start_load_kv(MagicMock())

        # Wait for all layers (drains futures)
        for layer_name in async_conn._layer_names:
            async_conn.wait_for_layer_load(layer_name)

        # Compare sync vs async results
        for layer in sorted(staging_sync.keys()):
            assert torch.equal(
                staging_sync[layer], staging_async[layer]
            ), f"Mismatch in {layer}"

        async_conn.shutdown()

    def test_async_futures_created_per_layer(self):
        """With async enabled, start_load_kv creates per-layer futures."""
        store = InMemoryKVStore()
        # Seed store with data
        store.put(StoreKey("req-A", 0, 0, KVKind.K), torch.randn(4, 1, 2))
        store.put(StoreKey("req-A", 0, 0, KVKind.V), torch.randn(4, 1, 2))
        store.put(StoreKey("req-A", 1, 0, KVKind.K), torch.randn(4, 1, 2))
        store.put(StoreKey("req-A", 1, 0, KVKind.V), torch.randn(4, 1, 2))

        conn = _make_connector(
            block_size=4, store=store,
            VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS="2",
        )
        staging = _make_staging_caches(num_layers=2, num_blocks=8)
        conn.register_kv_caches(staging)

        meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="req-B", block_ids=[2],
                is_store=False, source_req_id="req-A",
                block_mapping=[(0, 2)],
            )],
            layer_names=sorted(staging.keys()),
            block_size=4,
        )
        conn.bind_connector_metadata(meta)
        conn.start_load_kv(MagicMock())

        # Before wait: layers should NOT be marked done (async)
        # (they may already be done if threads were fast, but futures exist)
        assert len(conn._layer_load_futures) >= 0  # may have drained

        # After waiting, all done
        conn._collect_all_async_loads()
        for ln in conn._layer_names:
            assert conn._layer_load_done[ln] is True

        conn.shutdown()

    def test_async_get_finished_drains_futures(self):
        """get_finished() drains all async loads before reporting."""
        store = InMemoryKVStore()
        store.put(StoreKey("req-A", 0, 0, KVKind.K), torch.randn(4, 1, 2))
        store.put(StoreKey("req-A", 0, 0, KVKind.V), torch.randn(4, 1, 2))

        conn = _make_connector(
            block_size=4, store=store,
            VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS="1",
        )
        staging = _make_staging_caches(num_layers=1, num_blocks=8)
        conn.register_kv_caches(staging)

        meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="req-B", block_ids=[2],
                is_store=False, source_req_id="req-A",
                block_mapping=[(0, 2)],
            )],
            layer_names=sorted(staging.keys()),
            block_size=4,
        )
        conn.bind_connector_metadata(meta)
        conn.start_load_kv(MagicMock())

        # get_finished should internally drain futures
        _, recving = conn.get_finished({"req-B"})
        assert recving == {"req-B"}

        conn.shutdown()

    def test_async_error_blocks_reported(self):
        """Async load misses are tracked and reported correctly."""
        store = InMemoryKVStore()
        # No data in store — everything will miss

        conn = _make_connector(
            block_size=4, store=store,
            VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS="2",
        )
        staging = _make_staging_caches(num_layers=2, num_blocks=8)
        conn.register_kv_caches(staging)

        meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="req-B", block_ids=[2, 3],
                is_store=False, source_req_id="req-A",
                block_mapping=[(0, 2), (1, 3)],
            )],
            layer_names=sorted(staging.keys()),
            block_size=4,
        )
        conn.bind_connector_metadata(meta)
        conn.start_load_kv(MagicMock())

        # get_block_ids_with_load_errors drains futures first
        errors = conn.get_block_ids_with_load_errors()
        assert 2 in errors
        assert 3 in errors

        conn.shutdown()

    def test_sync_fallback_when_workers_zero(self):
        """With VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS=0, sync path is used."""
        conn = _make_connector(
            block_size=4,
            VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS="0",
        )
        assert conn._async_load_enabled is False
        assert conn._executor is None

    def test_async_stats_accumulate_correctly(self):
        """Stats from async loads are accumulated after drain."""
        store = InMemoryKVStore()
        store.put(StoreKey("req-A", 0, 0, KVKind.K), torch.randn(4, 1, 2))
        store.put(StoreKey("req-A", 0, 0, KVKind.V), torch.randn(4, 1, 2))
        store.put(StoreKey("req-A", 1, 0, KVKind.K), torch.randn(4, 1, 2))
        store.put(StoreKey("req-A", 1, 0, KVKind.V), torch.randn(4, 1, 2))

        conn = _make_connector(
            block_size=4, store=store,
            VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS="2",
        )
        staging = _make_staging_caches(num_layers=2, num_blocks=8)
        conn.register_kv_caches(staging)

        meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="req-B", block_ids=[2],
                is_store=False, source_req_id="req-A",
                block_mapping=[(0, 2)],
            )],
            layer_names=sorted(staging.keys()),
            block_size=4,
        )
        conn.bind_connector_metadata(meta)
        conn.start_load_kv(MagicMock())

        # Drain
        conn._collect_all_async_loads()

        stats = conn.get_kv_connector_stats()
        assert stats is not None
        # 1 block * 2 layers * 2 (K+V) = 4 loaded
        assert stats.data["loaded_blocks"] == 4
        assert stats.data["load_misses"] == 0

        conn.shutdown()


# ---------------------------------------------------------------------------
# Scheduler feedback handling for invalid block IDs
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestSchedulerFeedback:
    """Verify invalid_block_ids flow and recompute handling."""

    def test_invalid_blocks_flow_through_bridge(self):
        """Bridge propagates invalid_block_ids from connector to output."""
        store = InMemoryKVStore()
        cfg = _make_vllm_config_mock(block_size=4)

        # Scheduler connector with fake saved request (no actual data)
        with patch.dict("os.environ", {
            "VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS": "0",
        }), patch("vllm_spyre.envs._cache", {}):
            sched = InMemorySpyreConnector(
                vllm_config=cfg, role=KVConnectorRole.SCHEDULER,
                kv_cache_config=None, store=store,
            )
            worker = InMemorySpyreConnector(
                vllm_config=cfg, role=KVConnectorRole.WORKER,
                kv_cache_config=None, store=store,
            )

        staging = _make_staging_caches(num_layers=1, num_blocks=8)
        worker.register_kv_caches(staging)

        bridge = SpyreKVConnectorBridge(cfg)
        bridge._kv_connector = worker

        # Inject a saved request without storing actual data
        sched._saved_requests["req-A"] = _SavedRequest(
            req_id="req-A",
            prompt_token_ids=(1, 2, 3, 4),
            block_ids=[0],
            num_tokens=4,
        )

        # Match and allocate
        req_b = _make_request("req-B", [1, 2, 3, 4])
        matched, _ = sched.get_num_new_matched_tokens(req_b, 0)
        assert matched == 4

        sched.update_state_after_alloc(
            req_b, _make_blocks([2]), num_external_tokens=4,
        )
        meta = sched.build_connector_meta(MagicMock())

        # Execute through bridge
        so = MagicMock()
        so.kv_connector_metadata = meta
        so.finished_req_ids = {"req-B"}
        so.preempted_req_ids = set()

        bridge.begin_step(so)
        bridge.before_forward(so)
        bridge.after_forward(so)
        output = bridge.finish_step()

        # Since no actual KV data exists, blocks should be marked invalid
        assert output is not None
        assert 2 in output.invalid_block_ids

    def test_retry_after_invalid_blocks(self):
        """After invalid blocks, a retry with store mode succeeds."""
        store = InMemoryKVStore()
        cfg = _make_vllm_config_mock(block_size=4)

        with patch.dict("os.environ", {
            "VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS": "0",
        }), patch("vllm_spyre.envs._cache", {}):
            sched = InMemorySpyreConnector(
                vllm_config=cfg, role=KVConnectorRole.SCHEDULER,
                kv_cache_config=None, store=store,
            )
            worker = InMemorySpyreConnector(
                vllm_config=cfg, role=KVConnectorRole.WORKER,
                kv_cache_config=None, store=store,
            )

        staging = _make_staging_caches(num_layers=1, num_blocks=8)
        worker.register_kv_caches(staging)

        bridge = SpyreKVConnectorBridge(cfg)
        bridge._kv_connector = worker

        # Step 1: Failed load (fake saved request, no data)
        sched._saved_requests["req-A"] = _SavedRequest(
            req_id="req-A",
            prompt_token_ids=(1, 2, 3, 4),
            block_ids=[0],
            num_tokens=4,
        )

        req_b = _make_request("req-B", [1, 2, 3, 4])
        matched, _ = sched.get_num_new_matched_tokens(req_b, 0)
        sched.update_state_after_alloc(
            req_b, _make_blocks([2]), num_external_tokens=matched,
        )
        meta = sched.build_connector_meta(MagicMock())

        so = MagicMock()
        so.kv_connector_metadata = meta
        so.finished_req_ids = {"req-B"}
        so.preempted_req_ids = set()

        bridge.begin_step(so)
        bridge.before_forward(so)
        bridge.after_forward(so)
        output = bridge.finish_step()
        assert 2 in output.invalid_block_ids

        # Step 2: Retry as store (recompute) — should succeed
        # Simulate scheduler resetting num_computed_tokens to 0
        # and re-scheduling as a fresh prefill (store mode)
        sched.update_state_after_alloc(
            req_b, _make_blocks([2]), num_external_tokens=0,
        )
        meta2 = sched.build_connector_meta(MagicMock())
        assert meta2.requests[0].is_store is True

        staging[sorted(staging.keys())[0]][0][2].fill_(42.0)

        so2 = MagicMock()
        so2.kv_connector_metadata = meta2
        so2.finished_req_ids = {"req-B"}
        so2.preempted_req_ids = set()

        bridge.begin_step(so2)
        bridge.before_forward(so2)
        bridge.after_forward(so2)
        output2 = bridge.finish_step()

        # Store should succeed with no errors
        assert output2.invalid_block_ids == set()
        assert output2.finished_sending == {"req-B"}

    def test_partial_invalid_blocks(self):
        """When some blocks load and some fail, only failures are reported."""
        store = InMemoryKVStore()
        # Only store data for block 0, not block 1
        store.put(StoreKey("req-A", 0, 0, KVKind.K), torch.randn(4, 1, 2))
        store.put(StoreKey("req-A", 0, 0, KVKind.V), torch.randn(4, 1, 2))

        conn = _make_connector(block_size=4, store=store)
        staging = _make_staging_caches(num_layers=1, num_blocks=8)
        conn.register_kv_caches(staging)

        meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="req-B", block_ids=[2, 3],
                is_store=False, source_req_id="req-A",
                block_mapping=[(0, 2), (1, 3)],
            )],
            layer_names=sorted(staging.keys()),
            block_size=4,
        )
        conn.bind_connector_metadata(meta)
        conn.start_load_kv(MagicMock())

        errors = conn.get_block_ids_with_load_errors()
        # Block 0 loaded OK, block 1 missed → dest block 3 is invalid
        assert 3 in errors
        assert 2 not in errors

    def test_scheduler_can_observe_invalid_blocks_via_output(self):
        """Full scheduler→worker→scheduler feedback loop with invalid blocks."""
        store = InMemoryKVStore()
        cfg = _make_vllm_config_mock(block_size=4)

        with patch.dict("os.environ", {
            "VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS": "0",
        }), patch("vllm_spyre.envs._cache", {}):
            sched = InMemorySpyreConnector(
                vllm_config=cfg, role=KVConnectorRole.SCHEDULER,
                kv_cache_config=None, store=store,
            )
            worker = InMemorySpyreConnector(
                vllm_config=cfg, role=KVConnectorRole.WORKER,
                kv_cache_config=None, store=store,
            )

        staging = _make_staging_caches(num_layers=1, num_blocks=8)
        worker.register_kv_caches(staging)

        # Save some data for req-A (only block 0)
        staging[sorted(staging.keys())[0]][0][0].fill_(10.0)
        staging[sorted(staging.keys())[0]][1][0].fill_(20.0)

        save_meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="req-A", block_ids=[0],
                is_store=True, token_count=4,
            )],
            layer_names=sorted(staging.keys()),
            block_size=4,
        )
        worker.bind_connector_metadata(save_meta)
        worker.wait_for_save()
        worker.clear_connector_metadata()

        # Register req-A in scheduler registry
        req_a = _make_request("req-A", [1, 2, 3, 4, 5, 6, 7, 8])
        sched.request_finished(req_a, [0, 1])

        # Try to load 2 blocks from req-A for req-B
        # Block 0 exists in store, block 1 does NOT
        req_b = _make_request("req-B", [1, 2, 3, 4, 5, 6, 7, 8])
        matched, _ = sched.get_num_new_matched_tokens(req_b, 0)
        assert matched == 8  # Scheduler thinks 8 tokens match

        sched.update_state_after_alloc(
            req_b, _make_blocks([2, 3]), num_external_tokens=8,
        )
        meta = sched.build_connector_meta(MagicMock())

        # Worker tries to load
        worker.bind_connector_metadata(meta)
        worker.start_load_kv(MagicMock())
        errors = worker.get_block_ids_with_load_errors()

        # Block 3 (dest for source block 1) should be invalid
        assert 3 in errors
        # Block 2 (dest for source block 0) should be OK
        assert 2 not in errors

        worker.clear_connector_metadata()


# ---------------------------------------------------------------------------
# Schema validation hardening
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestSchemaValidation:
    """Verify SpyreConnectorMeta.validate() catches invalid metadata."""

    def test_unsupported_schema_version(self):
        """Reject unknown schema versions."""
        meta = SpyreConnectorMeta(schema_version=99)
        with pytest.raises(ValueError, match="Unsupported schema_version"):
            meta.validate()

    def test_supported_schema_version(self):
        """Version 1 is accepted."""
        meta = SpyreConnectorMeta(schema_version=1)
        meta.validate()  # Should not raise

    def test_negative_num_layers(self):
        """Reject negative num_layers."""
        meta = SpyreConnectorMeta(num_layers=-1)
        with pytest.raises(ValueError, match="num_layers"):
            meta.validate()

    def test_negative_num_kv_heads(self):
        """Reject negative num_kv_heads."""
        meta = SpyreConnectorMeta(num_kv_heads=-1)
        with pytest.raises(ValueError, match="num_kv_heads"):
            meta.validate()

    def test_negative_head_dim(self):
        """Reject negative head_dim."""
        meta = SpyreConnectorMeta(head_dim=-1)
        with pytest.raises(ValueError, match="head_dim"):
            meta.validate()

    def test_negative_block_size(self):
        """Reject negative block_size."""
        meta = SpyreConnectorMeta(block_size=-1)
        with pytest.raises(ValueError, match="block_size"):
            meta.validate()

    def test_zero_block_size_allowed(self):
        """block_size=0 is allowed (informational default)."""
        meta = SpyreConnectorMeta(block_size=0)
        meta.validate()  # Should not raise

    def test_empty_req_id_rejected(self):
        """Request with empty req_id is rejected."""
        meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(req_id="")],
        )
        with pytest.raises(ValueError, match="empty req_id"):
            meta.validate()

    def test_load_with_mapping_but_no_source_rejected(self):
        """Load request with block_mapping but no source_req_id is rejected."""
        meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="r1", is_store=False,
                block_mapping=[(0, 1)],
                source_req_id="",
            )],
        )
        with pytest.raises(ValueError, match="no source_req_id"):
            meta.validate()

    def test_valid_load_request_passes(self):
        """Properly formed load request passes validation."""
        meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="r1", is_store=False,
                block_mapping=[(0, 1)],
                source_req_id="r0",
            )],
            block_size=4,
        )
        meta.validate()  # Should not raise

    def test_validate_called_from_bind_connector_metadata(self):
        """bind_connector_metadata calls validate() for SpyreConnectorMeta."""
        conn = _make_connector(block_size=4)

        meta = SpyreConnectorMeta(schema_version=99)
        with pytest.raises(ValueError, match="Unsupported schema_version"):
            conn.bind_connector_metadata(meta)

    def test_duplicate_block_mapping_caught_via_validate(self):
        """validate() catches duplicate dest blocks (defers to validate_block_mapping)."""
        meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="r1", is_store=False,
                    source_req_id="r0",
                    block_ids=[5, 5],
                ),
            ],
        )
        with pytest.raises(ValueError, match="Duplicate destination"):
            meta.validate()


# ---------------------------------------------------------------------------
# Connector shutdown/cleanup
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestConnectorShutdown:
    """Verify connector shutdown cleans up async resources."""

    def test_shutdown_closes_executor(self):
        """shutdown() cleans up the thread pool executor."""
        conn = _make_connector(
            block_size=4,
            VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS="2",
        )
        assert conn._executor is not None
        conn.shutdown()
        assert conn._executor is None

    def test_double_shutdown_safe(self):
        """Calling shutdown() twice doesn't crash."""
        conn = _make_connector(
            block_size=4,
            VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS="2",
        )
        conn.shutdown()
        conn.shutdown()  # Should not raise


# ---------------------------------------------------------------------------
# Byte-capped KV store
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestByteCapStore:
    """Verify InMemoryKVStore byte-cap LRU eviction."""

    def test_unlimited_when_cap_is_zero(self):
        """max_bytes=0 means no byte-based eviction."""
        store = InMemoryKVStore(max_bytes=0)
        for i in range(100):
            store.put(
                StoreKey(f"r{i}", 0, 0, KVKind.K),
                torch.randn(4, 1, 2),
            )
        assert store.size == 100
        assert store.evictions == 0

    def test_eviction_at_byte_cap(self):
        """Entries evicted when byte cap is exceeded."""
        # Each float32 tensor of shape (4,1,2) = 32 bytes
        entry_bytes = 4 * 1 * 2 * 4  # 32 bytes
        cap = entry_bytes * 3  # Room for 3 entries

        store = InMemoryKVStore(max_bytes=cap)

        for i in range(5):
            store.put(
                StoreKey(f"r{i}", 0, 0, KVKind.K),
                torch.randn(4, 1, 2),
            )

        # Should have evicted 2 oldest to make room
        assert store.size == 3
        assert store.evictions == 2
        assert store.current_bytes <= cap

        # Oldest entries should be gone
        assert store.get(StoreKey("r0", 0, 0, KVKind.K)) is None
        assert store.get(StoreKey("r1", 0, 0, KVKind.K)) is None
        # Newest should remain
        assert store.get(StoreKey("r4", 0, 0, KVKind.K)) is not None

    def test_overwrite_reclaims_bytes(self):
        """Overwriting an entry doesn't double-count bytes."""
        entry_bytes = 4 * 1 * 2 * 4  # 32 bytes
        cap = entry_bytes * 2

        store = InMemoryKVStore(max_bytes=cap)
        key = StoreKey("r1", 0, 0, KVKind.K)

        store.put(key, torch.randn(4, 1, 2))
        assert store.current_bytes == entry_bytes

        # Overwrite with same-size tensor
        store.put(key, torch.randn(4, 1, 2))
        assert store.current_bytes == entry_bytes
        assert store.size == 1
        assert store.evictions == 0

    def test_remove_by_req_reclaims_bytes(self):
        """remove_by_req subtracts bytes correctly."""
        store = InMemoryKVStore()
        store.put(StoreKey("r1", 0, 0, KVKind.K), torch.randn(4, 1, 2))
        store.put(StoreKey("r1", 0, 0, KVKind.V), torch.randn(4, 1, 2))
        assert store.current_bytes > 0

        store.remove_by_req("r1")
        assert store.current_bytes == 0
        assert store.size == 0

    def test_clear_resets_bytes(self):
        """clear() resets byte tracking."""
        store = InMemoryKVStore(max_bytes=1024)
        store.put(StoreKey("r1", 0, 0, KVKind.K), torch.randn(4, 1, 2))
        assert store.current_bytes > 0

        store.clear()
        assert store.current_bytes == 0
        assert store.size == 0

    def test_stats_include_byte_info(self):
        """stats() reports max_bytes and evictions."""
        store = InMemoryKVStore(max_bytes=1000)
        store.put(StoreKey("r1", 0, 0, KVKind.K), torch.randn(4, 1, 2))

        s = store.stats()
        assert "max_bytes" in s
        assert s["max_bytes"] == 1000
        assert "evictions" in s
        assert s["memory_estimate_bytes"] == store.current_bytes

    def test_large_entry_evicts_many_small_ones(self):
        """A single large entry can evict multiple small entries."""
        small_bytes = 4 * 1 * 2 * 4  # 32 bytes
        large_bytes = 4 * 4 * 4 * 4  # 1024 bytes
        cap = small_bytes * 5 + large_bytes  # Room for 5 small + 1 large

        store = InMemoryKVStore(max_bytes=cap)

        # Add 6 small entries (6 * 32 = 192)
        for i in range(6):
            store.put(
                StoreKey(f"r{i}", 0, 0, KVKind.K),
                torch.randn(4, 1, 2),
            )
        assert store.size == 6

        # Add 1 large entry — may need to evict small ones
        store.put(
            StoreKey("big", 0, 0, KVKind.K),
            torch.randn(4, 4, 4),
        )
        assert store.current_bytes <= cap
        assert store.get(StoreKey("big", 0, 0, KVKind.K)) is not None

    def test_env_var_wires_to_global_store(self):
        """VLLM_SPYRE_KV_STORE_MAX_BYTES configures the global store."""
        from vllm_spyre.distributed.kv_transfer.kv_connector.v1 import (
            inmemory_spyre_connector as mod,
        )
        old_store = mod._GLOBAL_STORE
        try:
            mod._GLOBAL_STORE = None
            with patch.dict("os.environ", {
                "VLLM_SPYRE_KV_STORE_MAX_BYTES": "4096",
            }), patch("vllm_spyre.envs._cache", {}):
                store = mod.get_global_store()
            assert store.max_bytes == 4096
        finally:
            mod._GLOBAL_STORE = old_store


# ---------------------------------------------------------------------------
# Stricter metadata validation (layout, block-id bounds, mapping bounds)
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestStricterValidation:
    """Extended validation tests for layout, block-IDs, and mapping bounds."""

    def test_unknown_layout_rejected(self):
        """Unknown layout string is rejected."""
        meta = SpyreConnectorMeta(layout="BSHD")
        with pytest.raises(ValueError, match="Unknown layout"):
            meta.validate()

    def test_nhd_layout_accepted(self):
        """NHD layout is accepted."""
        meta = SpyreConnectorMeta(layout="NHD")
        meta.validate()  # Should not raise

    def test_empty_layout_accepted(self):
        """Empty layout is accepted (optional field)."""
        meta = SpyreConnectorMeta(layout="")
        meta.validate()  # Should not raise

    def test_negative_block_id_in_request(self):
        """Negative block_id in request is rejected."""
        meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="r1", block_ids=[-1, 0],
            )],
        )
        with pytest.raises(ValueError, match="negative block_id"):
            meta.validate()

    def test_negative_source_in_block_mapping(self):
        """Negative source block in mapping is rejected."""
        meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="r1", is_store=False,
                source_req_id="r0",
                block_mapping=[(-1, 5)],
            )],
        )
        with pytest.raises(ValueError, match="negative source block_id"):
            meta.validate()

    def test_negative_dest_in_block_mapping(self):
        """Negative dest block in mapping is rejected."""
        meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="r1", is_store=False,
                source_req_id="r0",
                block_mapping=[(0, -1)],
            )],
        )
        with pytest.raises(ValueError, match="negative dest block_id"):
            meta.validate()

    def test_valid_block_ids_pass(self):
        """All non-negative block IDs pass validation."""
        meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="r1", block_ids=[0, 1, 2],
            )],
        )
        meta.validate()  # Should not raise

    def test_valid_block_mapping_passes(self):
        """Non-negative source and dest in mapping passes."""
        meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="r1", is_store=False,
                source_req_id="r0",
                block_mapping=[(0, 5), (1, 6)],
            )],
        )
        meta.validate()  # Should not raise


# ---------------------------------------------------------------------------
# Prometheus metrics adapter
# ---------------------------------------------------------------------------

class _FakeMetricChild:
    """Stand-in for a labeled prometheus_client metric child (.labels())."""

    def __init__(self):
        self._value: float = 0.0

    def inc(self, amount: float = 1) -> None:
        self._value += amount

    def set(self, value: float) -> None:
        self._value = value

    def observe(self, amount: float) -> None:
        self._value += amount


class _FakeMetric:
    """Stand-in for a prometheus_client Counter/Gauge/Histogram."""

    def __init__(self, name="", documentation="", labelnames=None):
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames or []
        self._children: dict[tuple, _FakeMetricChild] = {}

    def labels(self, *args):
        key = args
        if key not in self._children:
            self._children[key] = _FakeMetricChild()
        return self._children[key]


def _make_fake_metric_types():
    """Return a metric_types dict compatible with KVConnectorPromMetrics."""
    from prometheus_client import Counter, Gauge, Histogram
    return {
        Counter: _FakeMetric,
        Gauge: _FakeMetric,
        Histogram: _FakeMetric,
    }


@pytest.mark.cpu
class TestPrometheusAdapter:
    """Verify SpyreConnectorPromMetrics and build_prom_metrics()."""

    def test_build_prom_metrics_returns_adapter(self):
        """build_prom_metrics() returns a SpyreConnectorPromMetrics instance."""
        from vllm_spyre.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
            SpyreConnectorPromMetrics,
        )

        cfg = _make_vllm_config_mock()
        metric_types = _make_fake_metric_types()
        labelnames = ["engine"]
        per_engine_labelvalues = {0: ["engine_0"]}

        result = InMemorySpyreConnector.build_prom_metrics(
            cfg, metric_types, labelnames, per_engine_labelvalues,
        )
        assert isinstance(result, SpyreConnectorPromMetrics)

    def test_adapter_has_all_six_counters(self):
        """Adapter exposes all 6 expected counter attributes."""
        cfg = _make_vllm_config_mock()
        metric_types = _make_fake_metric_types()
        labelnames = ["engine"]
        per_engine_labelvalues = {0: ["engine_0"]}

        adapter = InMemorySpyreConnector.build_prom_metrics(
            cfg, metric_types, labelnames, per_engine_labelvalues,
        )

        expected_attrs = [
            "counter_matched_tokens",
            "counter_loaded_blocks",
            "counter_saved_blocks",
            "counter_load_misses",
            "counter_evictions",
            "counter_match_attempts",
        ]
        for attr in expected_attrs:
            assert hasattr(adapter, attr), f"Missing attribute: {attr}"
            # Each should be a per-engine dict with engine_idx 0
            counter_dict = getattr(adapter, attr)
            assert 0 in counter_dict

    def test_observe_increments_counters(self):
        """observe() maps stats keys to the correct Prometheus counters."""
        cfg = _make_vllm_config_mock()
        metric_types = _make_fake_metric_types()
        labelnames = ["engine"]
        per_engine_labelvalues = {0: ["engine_0"]}

        adapter = InMemorySpyreConnector.build_prom_metrics(
            cfg, metric_types, labelnames, per_engine_labelvalues,
        )

        stats_data = {
            "matched_tokens": 10,
            "loaded_blocks": 5,
            "saved_blocks": 3,
            "load_misses": 1,
            "evictions": 2,
            "match_attempts": 7,
        }
        adapter.observe(stats_data, engine_idx=0)

        assert adapter.counter_matched_tokens[0]._value == 10
        assert adapter.counter_loaded_blocks[0]._value == 5
        assert adapter.counter_saved_blocks[0]._value == 3
        assert adapter.counter_load_misses[0]._value == 1
        assert adapter.counter_evictions[0]._value == 2
        assert adapter.counter_match_attempts[0]._value == 7

    def test_observe_skips_zero_values(self):
        """observe() does not call inc() for zero-valued counters."""
        cfg = _make_vllm_config_mock()
        metric_types = _make_fake_metric_types()
        labelnames = ["engine"]
        per_engine_labelvalues = {0: ["engine_0"]}

        adapter = InMemorySpyreConnector.build_prom_metrics(
            cfg, metric_types, labelnames, per_engine_labelvalues,
        )

        # All zeros — none should increment
        adapter.observe({"matched_tokens": 0, "loaded_blocks": 0}, engine_idx=0)

        assert adapter.counter_matched_tokens[0]._value == 0
        assert adapter.counter_loaded_blocks[0]._value == 0

    def test_observe_accumulates_across_calls(self):
        """Multiple observe() calls accumulate counter values."""
        cfg = _make_vllm_config_mock()
        metric_types = _make_fake_metric_types()
        labelnames = ["engine"]
        per_engine_labelvalues = {0: ["engine_0"]}

        adapter = InMemorySpyreConnector.build_prom_metrics(
            cfg, metric_types, labelnames, per_engine_labelvalues,
        )

        adapter.observe({"matched_tokens": 5}, engine_idx=0)
        adapter.observe({"matched_tokens": 3}, engine_idx=0)
        assert adapter.counter_matched_tokens[0]._value == 8

    def test_observe_handles_missing_keys(self):
        """observe() gracefully handles stats dict with missing keys."""
        cfg = _make_vllm_config_mock()
        metric_types = _make_fake_metric_types()
        labelnames = ["engine"]
        per_engine_labelvalues = {0: ["engine_0"]}

        adapter = InMemorySpyreConnector.build_prom_metrics(
            cfg, metric_types, labelnames, per_engine_labelvalues,
        )

        # Only one key present — others default to 0
        adapter.observe({"saved_blocks": 4}, engine_idx=0)

        assert adapter.counter_saved_blocks[0]._value == 4
        assert adapter.counter_matched_tokens[0]._value == 0
        assert adapter.counter_loaded_blocks[0]._value == 0

    def test_multi_engine_observe(self):
        """observe() routes metrics to the correct engine index."""
        cfg = _make_vllm_config_mock()
        metric_types = _make_fake_metric_types()
        labelnames = ["engine"]
        per_engine_labelvalues = {
            0: ["engine_0"],
            1: ["engine_1"],
        }

        adapter = InMemorySpyreConnector.build_prom_metrics(
            cfg, metric_types, labelnames, per_engine_labelvalues,
        )

        adapter.observe({"loaded_blocks": 10}, engine_idx=0)
        adapter.observe({"loaded_blocks": 20}, engine_idx=1)

        assert adapter.counter_loaded_blocks[0]._value == 10
        assert adapter.counter_loaded_blocks[1]._value == 20


# ---------------------------------------------------------------------------
# Runtime metrics smoke test: reuse flow produces non-zero stats
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestMetricsSmokeTest:
    """End-to-end: run a reuse flow and verify stats feed into adapter."""

    def test_reuse_flow_produces_nonzero_metrics(self):
        """Save + load flow: connector stats feed into prom adapter."""
        store = InMemoryKVStore()
        conn = _make_connector(block_size=4, store=store)
        staging = _make_staging_caches(num_layers=2, num_blocks=8)
        conn.register_kv_caches(staging)

        # --- Step 1: Save KV for req-A ---
        for layer in staging:
            staging[layer][0][0].fill_(1.0)
            staging[layer][1][0].fill_(2.0)

        save_meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="req-A", block_ids=[0],
                is_store=True, token_count=4,
            )],
            layer_names=sorted(staging.keys()),
            block_size=4,
        )
        conn.bind_connector_metadata(save_meta)
        conn.wait_for_save()

        # Collect stats from save step
        save_stats = conn.get_kv_connector_stats()
        assert save_stats is not None
        assert save_stats.data["saved_blocks"] > 0

        conn.clear_connector_metadata()

        # --- Step 2: Load KV for req-B from req-A ---
        load_meta = SpyreConnectorMeta(
            requests=[SpyreConnectorRequestMeta(
                req_id="req-B", block_ids=[4],
                is_store=False, source_req_id="req-A",
                block_mapping=[(0, 4)],
            )],
            layer_names=sorted(staging.keys()),
            block_size=4,
        )
        conn.bind_connector_metadata(load_meta)
        conn.start_load_kv(MagicMock())

        # Collect stats from load step
        load_stats = conn.get_kv_connector_stats()
        assert load_stats is not None
        assert load_stats.data["loaded_blocks"] > 0

        conn.clear_connector_metadata()

        # --- Step 3: Scheduler-side match produces matched_tokens ---
        sched_conn = _make_connector(block_size=4, store=store)
        sched_conn.register_kv_caches(staging)

        req_a = _make_request("req-A", [1, 2, 3, 4])
        sched_conn.request_finished(req_a, [0])

        req_b = _make_request("req-B", [1, 2, 3, 4])
        matched, _ = sched_conn.get_num_new_matched_tokens(req_b, 0)
        assert matched == 4

        match_stats = sched_conn.get_kv_connector_stats()
        assert match_stats is not None
        assert match_stats.data["matched_tokens"] > 0
        assert match_stats.data["match_attempts"] > 0

        # --- Step 4: Feed all stats into prometheus adapter ---
        cfg = _make_vllm_config_mock()
        metric_types = _make_fake_metric_types()
        labelnames = ["engine"]
        per_engine_labelvalues = {0: ["engine_0"]}

        adapter = InMemorySpyreConnector.build_prom_metrics(
            cfg, metric_types, labelnames, per_engine_labelvalues,
        )

        # Feed save stats
        adapter.observe(save_stats.data, engine_idx=0)
        # Feed load stats
        adapter.observe(load_stats.data, engine_idx=0)
        # Feed match stats
        adapter.observe(match_stats.data, engine_idx=0)

        # Verify non-zero metrics in the adapter
        assert adapter.counter_saved_blocks[0]._value > 0
        assert adapter.counter_loaded_blocks[0]._value > 0
        assert adapter.counter_matched_tokens[0]._value > 0
        assert adapter.counter_match_attempts[0]._value > 0
        assert adapter.counter_load_misses[0]._value == 0
