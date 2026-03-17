from unittest.mock import MagicMock

import pytest
import torch
from multiprocessing import shared_memory

import vllm_spyre.envs as envs_spyre
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.outputs import KVConnectorOutput

from vllm_spyre.distributed.kv_transfer.kv_connector.v1.metadata import (
    HostMemoryKVStoreBackend,
    InMemoryKVStore,
    KVKind,
    SerializedSharedMemoryKVStoreBackend,
    SerializedUDSProcessKVStoreBackend,
    SerializedHostMemoryKVStoreBackend,
    SpyreConnectorMeta,
    SpyreConnectorRequestMeta,
    build_spyre_kv_store_backend,
    StoreKey,
)


def _make_vllm_config_mock(block_size: int = 64):
    vllm_config = MagicMock()
    vllm_config.cache_config.block_size = block_size
    vllm_config.kv_transfer_config = MagicMock()
    vllm_config.kv_transfer_config.is_kv_transfer_instance = True
    vllm_config.kv_transfer_config.kv_connector = "InMemorySpyreConnector"
    vllm_config.parallel_config.data_parallel_size = 1
    vllm_config.compilation_config.fast_moe_cold_start = False
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.speculative_config = None
    return vllm_config


def _make_connector(store=None, role=KVConnectorRole.WORKER, block_size=64):
    from vllm_spyre.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
        InMemorySpyreConnector,
    )

    vllm_config = _make_vllm_config_mock(block_size=block_size)
    store = store or InMemoryKVStore()
    return InMemorySpyreConnector(
        vllm_config=vllm_config,
        role=role,
        kv_cache_config=None,
        store=store,
    )


def _make_staging_caches(
    num_layers: int = 2,
    num_blocks: int = 4,
    block_size: int = 2,
    num_kv_heads: int = 2,
    head_dim: int = 4,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    caches = {}
    for i in range(num_layers):
        layer_name = f"model.layers.{i}.self_attn"
        caches[layer_name] = torch.zeros(
            (2, num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
        )
    return caches


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


def _store_complete_block(
    store,
    req_id: str,
    block_id: int,
    *,
    layer_idx: int = 0,
    dtype: torch.dtype = torch.float32,
):
    k = torch.full((2, 2, 4), float(block_id + 1), dtype=dtype)
    v = torch.full((2, 2, 4), float(block_id + 101), dtype=dtype)
    store.put(
        StoreKey(req_id=req_id, layer_idx=layer_idx, block_id=block_id, kv_kind=KVKind.K),
        k,
        source_req=req_id,
    )
    store.put(
        StoreKey(req_id=req_id, layer_idx=layer_idx, block_id=block_id, kv_kind=KVKind.V),
        v,
        source_req=req_id,
    )


def _make_fake_scheduler_output(kv_connector_metadata=None, finished_req_ids=None):
    so = MagicMock()
    so.kv_connector_metadata = kv_connector_metadata
    so.finished_req_ids = finished_req_ids or set()
    so.total_num_scheduled_tokens = 0
    so.preempted_req_ids = set()
    return so


@pytest.mark.cpu
class TestFactoryRegistration:
    def test_duplicate_registration_is_harmless(self):
        import vllm_spyre as pkg

        old_flag = pkg._connector_registered
        pkg._connector_registered = False
        try:
            pkg._register_kv_connector()
            pkg._register_kv_connector()
        finally:
            pkg._connector_registered = old_flag

    def test_connector_class_loads_via_factory(self):
        import vllm_spyre as pkg

        old_flag = pkg._connector_registered
        pkg._connector_registered = False
        try:
            pkg._register_kv_connector()
            cls = KVConnectorFactory.get_connector_class_by_name(
                "InMemorySpyreConnector"
            )
            assert cls.__name__ == "InMemorySpyreConnector"
        finally:
            pkg._connector_registered = old_flag


@pytest.mark.cpu
class TestMetadataSchema:
    def test_spyre_connector_meta_inherits_kv_connector_metadata(self):
        meta = SpyreConnectorMeta()
        assert isinstance(meta, KVConnectorMetadata)

    def test_duplicate_dest_blocks_are_rejected(self):
        meta = SpyreConnectorMeta()
        meta.add_load_request(
            "req-1",
            [3, 4],
            source_req_id="req-src",
            block_mapping=[(0, 4)],
        )
        meta.add_load_request(
            "req-2",
            [5, 6],
            source_req_id="req-src",
            block_mapping=[(1, 4)],
        )
        with pytest.raises(ValueError, match="Duplicate destination block ID"):
            meta.validate()

    def test_host_memory_store_backend_loads_into_destination_tensor(self):
        store = HostMemoryKVStoreBackend()
        key = StoreKey(req_id="req-1", layer_idx=0, block_id=0, kv_kind=KVKind.K)
        source = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)

        version, was_overwrite = store.put(key, source, source_req="req-1")
        assert version == 1
        assert was_overwrite is False

        dest = torch.zeros_like(source)
        assert store.load_into(key, dest) is True
        assert torch.equal(dest, source)

    @pytest.mark.parametrize(
        "backend_cls",
        [
            SerializedHostMemoryKVStoreBackend,
            SerializedSharedMemoryKVStoreBackend,
            SerializedUDSProcessKVStoreBackend,
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_serialized_store_backends_load_into_destination_tensor(self, backend_cls, dtype):
        store = backend_cls()
        key = StoreKey(req_id="req-1", layer_idx=0, block_id=0, kv_kind=KVKind.K)
        source = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4).to(dtype)
        try:
            version, was_overwrite = store.put(key, source, source_req="req-1")
            assert version == 1
            assert was_overwrite is False

            dest = torch.zeros_like(source)
            assert store.load_into(key, dest) is True
            assert torch.equal(dest, source)
            assert store.stats()["backend_name"] in {
                "serialized_host_memory",
                "serialized_shared_memory",
                "serialized_uds_process_store",
            }
        finally:
            store.shutdown()

    def test_serialized_shared_memory_backend_unlinks_segments_on_remove(self):
        store = SerializedSharedMemoryKVStoreBackend()
        key = StoreKey(req_id="req-1", layer_idx=0, block_id=0, kv_kind=KVKind.K)
        source = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)

        try:
            store.put(key, source, source_req="req-1")
            entry = store._store[key]

            shm = shared_memory.SharedMemory(name=entry.shm_name, create=False)
            shm.close()

            removed = store.remove_by_req("req-1")
            assert removed == 1

            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=entry.shm_name, create=False)
        finally:
            store.shutdown()

    def test_build_spyre_kv_store_backend_rejects_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown Spyre KV store backend"):
            build_spyre_kv_store_backend("not-a-real-backend")

    @pytest.mark.parametrize(
        "backend_cls",
        [
            HostMemoryKVStoreBackend,
            SerializedHostMemoryKVStoreBackend,
            SerializedSharedMemoryKVStoreBackend,
            SerializedUDSProcessKVStoreBackend,
        ],
    )
    def test_store_max_bytes_evicts_oldest_request_as_whole_unit(self, backend_cls):
        store = backend_cls(max_bytes=128)
        try:
            _store_complete_block(store, "req-A", 0)
            assert store.available_prefix_blocks("req-A", [0]) == 1

            _store_complete_block(store, "req-B", 0)

            assert store.available_prefix_blocks("req-A", [0]) == 0
            assert store.available_prefix_blocks("req-B", [0]) == 1
            assert store.stats()["unique_requests"] == 1
            assert store.stats()["evictions"] == 1
        finally:
            store.shutdown()

    def test_reset_global_store_uses_selected_backend(self, monkeypatch: pytest.MonkeyPatch):
        from vllm_spyre.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
            get_global_store,
            reset_global_store,
        )

        monkeypatch.setenv("VLLM_SPYRE_KV_STORE_BACKEND", "serialized_host_memory")
        envs_spyre.clear_env_cache()
        reset_global_store()

        store = get_global_store()
        assert isinstance(store, SerializedHostMemoryKVStoreBackend)

        monkeypatch.setenv("VLLM_SPYRE_KV_STORE_BACKEND", "host_memory")
        envs_spyre.clear_env_cache()
        reset_global_store()


@pytest.mark.cpu
class TestInMemorySpyreConnector:
    def test_save_then_load_same_request(self):
        store = InMemoryKVStore()
        connector = _make_connector(store=store, block_size=2)
        staging = _make_staging_caches(
            num_layers=2, num_blocks=4, block_size=2, num_kv_heads=2, head_dim=4
        )
        connector.register_kv_caches(staging)

        for layer_name in staging:
            staging[layer_name][0][0].fill_(1.0)
            staging[layer_name][1][0].fill_(2.0)
            staging[layer_name][0][1].fill_(3.0)
            staging[layer_name][1][1].fill_(4.0)

        save_meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-1",
                    block_ids=[0, 1],
                    is_store=True,
                    token_count=128,
                ),
            ],
            layer_names=sorted(staging.keys()),
            block_size=2,
        )
        connector.bind_connector_metadata(save_meta)
        connector.wait_for_save()
        connector.clear_connector_metadata()

        for layer_name in staging:
            staging[layer_name].zero_()

        load_meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-1",
                    block_ids=[0, 1],
                    is_store=False,
                    source_req_id="req-1",
                    block_mapping=[(0, 0), (1, 1)],
                ),
            ],
            layer_names=sorted(staging.keys()),
            block_size=2,
        )
        connector.bind_connector_metadata(load_meta)
        connector.start_load_kv(MagicMock())

        for layer_name in staging:
            assert torch.equal(staging[layer_name][0][0], torch.full((2, 2, 4), 1.0))
            assert torch.equal(staging[layer_name][1][0], torch.full((2, 2, 4), 2.0))
            assert torch.equal(staging[layer_name][0][1], torch.full((2, 2, 4), 3.0))
            assert torch.equal(staging[layer_name][1][1], torch.full((2, 2, 4), 4.0))

    def test_scheduler_driven_exact_prefix_reuse(self):
        block_size = 4
        store = InMemoryKVStore()
        sched = _make_connector(store=store, role=KVConnectorRole.SCHEDULER, block_size=block_size)
        worker = _make_connector(store=store, role=KVConnectorRole.WORKER, block_size=block_size)
        staging = _make_staging_caches(
            num_layers=2,
            num_blocks=8,
            block_size=block_size,
            num_kv_heads=1,
            head_dim=2,
        )
        worker.register_kv_caches(staging)

        prompt_a = [10, 20, 30, 40, 50, 60, 70, 80]
        req_a = _make_request("req-A", prompt_a)
        blocks_a = _make_blocks([0, 1])

        matched, is_async = sched.get_num_new_matched_tokens(req_a, 0)
        assert matched == 0
        assert is_async is False

        sched.update_state_after_alloc(req_a, blocks_a, num_external_tokens=0)
        meta_a = sched.build_connector_meta(_make_fake_scheduler_output())

        for layer_name in staging:
            staging[layer_name][0][0].fill_(100.0)
            staging[layer_name][1][0].fill_(200.0)
            staging[layer_name][0][1].fill_(300.0)
            staging[layer_name][1][1].fill_(400.0)

        worker.bind_connector_metadata(meta_a)
        worker.wait_for_save()
        worker.clear_connector_metadata()

        sched.request_finished(req_a, [0, 1])

        req_b = _make_request("req-B", prompt_a)
        matched, is_async = sched.get_num_new_matched_tokens(req_b, 0)
        assert matched == 8
        assert is_async is False

        blocks_b = _make_blocks([4, 5])
        sched.update_state_after_alloc(req_b, blocks_b, num_external_tokens=8)
        meta_b = sched.build_connector_meta(_make_fake_scheduler_output())
        load_req = meta_b.requests[0]
        assert load_req.is_store is False
        assert load_req.source_req_id == "req-A"
        assert load_req.block_mapping == [(0, 4), (1, 5)]

        for layer_name in staging:
            staging[layer_name].zero_()

        worker.bind_connector_metadata(meta_b)
        worker.start_load_kv(MagicMock())

        for layer_name in staging:
            assert torch.equal(staging[layer_name][0][4], torch.full((block_size, 1, 2), 100.0))
            assert torch.equal(staging[layer_name][1][4], torch.full((block_size, 1, 2), 200.0))
            assert torch.equal(staging[layer_name][0][5], torch.full((block_size, 1, 2), 300.0))
            assert torch.equal(staging[layer_name][1][5], torch.full((block_size, 1, 2), 400.0))

        assert worker.get_block_ids_with_load_errors() == set()
        _, recving = worker.get_finished({"req-B"})
        assert recving == {"req-B"}

    def test_scheduler_driven_partial_prefix_reuse(self):
        block_size = 4
        store = InMemoryKVStore()
        sched = _make_connector(store=store, role=KVConnectorRole.SCHEDULER, block_size=block_size)

        prompt_a = [10, 20, 30, 40, 50, 60, 70, 80]
        prompt_b = [10, 20, 30, 40, 50, 99, 98]

        _store_complete_block(store, "req-A", 0)
        _store_complete_block(store, "req-A", 1)
        req_a = _make_request("req-A", prompt_a)
        sched.request_finished(req_a, [0, 1])

        req_b = _make_request("req-B", prompt_b)
        matched, is_async = sched.get_num_new_matched_tokens(req_b, 0)

        assert matched == 4
        assert is_async is False

    def test_request_finished_requires_backing_store_entries(self):
        store = InMemoryKVStore()
        sched = _make_connector(store=store, role=KVConnectorRole.SCHEDULER, block_size=4)

        req = _make_request("req-A", [10, 20, 30, 40])
        sched.request_finished(req, [0])

        assert sched.get_cumulative_metrics()["saved_requests_count"] == 0
        matched, is_async = sched.get_num_new_matched_tokens(req, 0)
        assert matched == 0
        assert is_async is False

    def test_registry_cap_eviction_removes_store_entries(self):
        store = InMemoryKVStore()
        sched = _make_connector(store=store, role=KVConnectorRole.SCHEDULER, block_size=4)
        sched._saved_requests_max_size = 1

        _store_complete_block(store, "req-A", 0)
        _store_complete_block(store, "req-B", 0)

        sched.request_finished(_make_request("req-A", [10, 20, 30, 40]), [0])
        sched.request_finished(_make_request("req-B", [50, 60, 70, 80]), [0])

        assert sched.get_cumulative_metrics()["saved_requests_count"] == 1
        assert store.available_prefix_blocks("req-A", [0]) == 0
        assert store.available_prefix_blocks("req-B", [0]) == 1

    def test_matcher_prunes_stale_saved_request_after_store_eviction(self):
        store = InMemoryKVStore(max_bytes=128)
        sched = _make_connector(store=store, role=KVConnectorRole.SCHEDULER, block_size=4)

        _store_complete_block(store, "req-A", 0)
        sched.request_finished(_make_request("req-A", [10, 20, 30, 40]), [0])
        assert sched.get_cumulative_metrics()["saved_requests_count"] == 1

        _store_complete_block(store, "req-B", 0)
        assert store.available_prefix_blocks("req-A", [0]) == 0

        matched, is_async = sched.get_num_new_matched_tokens(
            _make_request("req-C", [10, 20, 30, 40]),
            0,
        )

        assert matched == 0
        assert is_async is False
        assert sched.get_cumulative_metrics()["saved_requests_count"] == 0

    def test_reset_probe_state_clears_registry_store_and_metrics(self):
        block_size = 4
        store = InMemoryKVStore()
        connector = _make_connector(
            store=store,
            role=KVConnectorRole.SCHEDULER,
            block_size=block_size,
        )

        req = _make_request("req-A", [10, 20, 30, 40])
        connector.request_finished(req, [0])
        connector._blocks_saved = 2
        connector._blocks_loaded = 3
        connector._blocks_missing = 1
        connector._stats.record("matched_tokens", 4)
        store.put(
            StoreKey(req_id="req-A", layer_idx=0, block_id=0, kv_kind=KVKind.K),
            torch.ones((block_size, 1, 2)),
            source_req="req-A",
        )

        connector.reset_probe_state()

        assert connector.get_cumulative_metrics() == {
            "blocks_saved": 0,
            "blocks_loaded": 0,
            "blocks_missing": 0,
            "saved_requests_count": 0,
        }
        assert store.stats()["total_entries"] == 0
        assert connector.get_kv_connector_stats() is None

        matched, _ = connector.get_num_new_matched_tokens(req, 0)
        assert matched == 0

    def test_load_miss_reports_error_blocks(self):
        worker = _make_connector(store=InMemoryKVStore(), role=KVConnectorRole.WORKER, block_size=4)
        staging = _make_staging_caches(
            num_layers=1, num_blocks=8, block_size=4, num_kv_heads=1, head_dim=2
        )
        worker.register_kv_caches(staging)

        meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-B",
                    block_ids=[4, 5],
                    is_store=False,
                    source_req_id="req-A",
                    block_mapping=[(0, 4), (1, 5)],
                ),
            ],
            layer_names=sorted(staging.keys()),
        )
        worker.bind_connector_metadata(meta)
        worker.start_load_kv(MagicMock())

        errors = worker.get_block_ids_with_load_errors()
        assert 4 in errors
        assert 5 in errors


@pytest.mark.cpu
class TestSpyreKVConnectorBridge:
    def _make_bridge_and_connector(self):
        from vllm_spyre.v1.worker.spyre_kv_connector_bridge import (
            SpyreKVConnectorBridge,
        )

        store = InMemoryKVStore()
        connector = _make_connector(store=store)
        staging = _make_staging_caches(
            num_layers=1, num_blocks=4, block_size=2, num_kv_heads=2, head_dim=4
        )
        connector.register_kv_caches(staging)

        bridge = SpyreKVConnectorBridge(_make_vllm_config_mock())
        bridge._kv_connector = connector
        return bridge, connector, store, staging

    def test_full_lifecycle_save(self):
        bridge, _connector, store, staging = self._make_bridge_and_connector()
        layer_name = sorted(staging.keys())[0]
        staging[layer_name][0][0].fill_(42.0)
        staging[layer_name][1][0].fill_(43.0)

        meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-1",
                    block_ids=[0],
                    is_store=True,
                ),
            ],
            layer_names=sorted(staging.keys()),
        )
        so = _make_fake_scheduler_output(
            kv_connector_metadata=meta,
            finished_req_ids={"req-1"},
        )

        assert bridge.begin_step(so) is True
        bridge.before_forward(so)
        bridge.after_forward(so)
        output = bridge.finish_step()

        assert output is not None
        assert isinstance(output, KVConnectorOutput)
        entry = store.get(StoreKey("req-1", 0, 0, KVKind.K))
        assert entry is not None
        assert torch.equal(entry.data, torch.full((2, 2, 4), 42.0))

    def test_no_forward_clears_metadata(self):
        bridge, connector, _store, staging = self._make_bridge_and_connector()
        meta = SpyreConnectorMeta(
            requests=[],
            layer_names=sorted(staging.keys()),
        )
        so = _make_fake_scheduler_output(kv_connector_metadata=meta)
        output = bridge.no_forward(so)
        assert output is not None
        assert connector.has_connector_metadata() is False
