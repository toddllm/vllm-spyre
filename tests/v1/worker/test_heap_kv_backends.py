import os
import threading
import uuid
from multiprocessing import get_context
from unittest.mock import MagicMock

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from vllm_spyre.distributed.kv_transfer.kv_connector.v1.metadata import (
    InMemoryKVStore,
    KVKind,
    SerializedSharedMemoryServiceKVStoreBackend,
    StoreKey,
    build_spyre_kv_store_backend,
)
from vllm_spyre.distributed.kv_transfer.kv_connector.v1.persistent_kv_service import (
    PersistentKVServiceClient,
    run_persistent_kv_service,
)


def _make_vllm_config_mock(block_size: int = 4):
    cfg = MagicMock()
    cfg.cache_config.block_size = block_size
    cfg.kv_transfer_config = MagicMock()
    cfg.kv_transfer_config.is_kv_transfer_instance = True
    cfg.kv_transfer_config.kv_connector = "InMemorySpyreConnector"
    cfg.parallel_config.data_parallel_size = 1
    cfg.compilation_config.fast_moe_cold_start = False
    cfg.compilation_config.static_forward_context = {}
    cfg.speculative_config = None
    return cfg


def _make_connector(store=None, role=KVConnectorRole.WORKER, block_size=4):
    from vllm_spyre.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector import (
        InMemorySpyreConnector,
    )

    return InMemorySpyreConnector(
        vllm_config=_make_vllm_config_mock(block_size=block_size),
        role=role,
        kv_cache_config=None,
        store=store or InMemoryKVStore(),
    )


def _make_request(req_id, prompt_token_ids, all_token_ids=None):
    req = MagicMock()
    req.request_id = req_id
    req.prompt_token_ids = list(prompt_token_ids)
    req.all_token_ids = list(all_token_ids or prompt_token_ids)
    req.num_computed_tokens = 0
    req.num_external_computed_tokens = 0
    return req


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


def _start_persistent_service(socket_path: str):
    service = threading.Thread(
        target=run_persistent_kv_service,
        args=(socket_path,),
        daemon=True,
    )
    service.start()
    client = PersistentKVServiceClient(socket_path, connect_timeout_s=5.0)
    client.close()
    return service


def _short_socket_path(prefix: str) -> str:
    return f"/tmp/{prefix}-{uuid.uuid4().hex[:8]}.sock"


def _stop_persistent_service(service, socket_path: str) -> None:
    try:
        PersistentKVServiceClient(socket_path).shutdown_service()
    except Exception:
        pass
    service.join(timeout=2.0)


def _cross_process_prefill(socket_path: str, result_queue) -> None:
    store = SerializedSharedMemoryServiceKVStoreBackend(socket_path=socket_path)
    sched = _make_connector(
        store=store,
        role=KVConnectorRole.SCHEDULER,
        block_size=4,
    )
    try:
        _store_complete_block(store, "req-A", 0)
        _store_complete_block(store, "req-A", 1)
        sched.request_finished(
            _make_request("req-A", [10, 20, 30, 40, 50, 60, 70, 80]),
            [0, 1],
        )
        result_queue.put(
            {
                "saved_requests_count": sched.get_cumulative_metrics()[
                    "saved_requests_count"
                ],
                "available_blocks": store.available_prefix_blocks("req-A", [0, 1]),
            }
        )
    finally:
        sched.shutdown()


def _cross_process_decode(socket_path: str, result_queue) -> None:
    store = SerializedSharedMemoryServiceKVStoreBackend(socket_path=socket_path)
    sched = _make_connector(
        store=store,
        role=KVConnectorRole.SCHEDULER,
        block_size=4,
    )
    try:
        matched, is_async = sched.get_num_new_matched_tokens(
            _make_request("req-B", [10, 20, 30, 40, 50, 60, 70, 80]),
            0,
        )
        result_queue.put(
            {
                "matched": matched,
                "is_async": is_async,
                "saved_requests_count": sched.get_cumulative_metrics()[
                    "saved_requests_count"
                ],
            }
        )
    finally:
        sched.shutdown()


@pytest.mark.cpu
class TestHeapKVBackends:
    def test_build_spyre_kv_store_backend_rejects_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown Spyre KV store backend"):
            build_spyre_kv_store_backend("not-a-real-backend")

    def test_service_store_backend_loads_across_store_instances(self):
        socket_path = _short_socket_path("spyre-kv-service")
        service = _start_persistent_service(socket_path)
        store_a = build_spyre_kv_store_backend(
            "serialized_shared_memory_service",
            service_socket=socket_path,
        )
        store_b = build_spyre_kv_store_backend(
            "serialized_shared_memory_service",
            service_socket=socket_path,
        )
        key = StoreKey(req_id="req-1", layer_idx=0, block_id=0, kv_kind=KVKind.K)
        source = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)

        try:
            version, was_overwrite = store_a.put(key, source, source_req="req-1")
            assert version == 1
            assert was_overwrite is False

            dest = torch.zeros_like(source)
            assert store_b.load_into(key, dest) is True
            assert torch.equal(dest, source)
            assert store_b.stats()["backend_name"] == "serialized_shared_memory_service"
        finally:
            store_a.shutdown()
            store_b.shutdown()
            _stop_persistent_service(service, socket_path)

    def test_scheduler_prunes_stale_saved_request_after_store_eviction(self):
        store = InMemoryKVStore(max_bytes=128)
        sched = _make_connector(
            store=store,
            role=KVConnectorRole.SCHEDULER,
            block_size=4,
        )

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

    def test_scheduler_reuses_saved_request_across_service_backed_connectors(self):
        socket_path = _short_socket_path("spyre-kv-service")
        service = _start_persistent_service(socket_path)
        store_a = SerializedSharedMemoryServiceKVStoreBackend(socket_path=socket_path)
        store_b = SerializedSharedMemoryServiceKVStoreBackend(socket_path=socket_path)
        sched_a = _make_connector(
            store=store_a,
            role=KVConnectorRole.SCHEDULER,
            block_size=4,
        )
        sched_b = _make_connector(
            store=store_b,
            role=KVConnectorRole.SCHEDULER,
            block_size=4,
        )

        prompt = [10, 20, 30, 40, 50, 60, 70, 80]

        try:
            _store_complete_block(store_a, "req-A", 0)
            _store_complete_block(store_a, "req-A", 1)
            sched_a.request_finished(_make_request("req-A", prompt), [0, 1])

            matched, is_async = sched_b.get_num_new_matched_tokens(
                _make_request("req-B", prompt),
                0,
            )

            assert matched == 7
            assert is_async is False
            assert sched_b.get_cumulative_metrics()["saved_requests_count"] == 1
        finally:
            sched_a.shutdown()
            sched_b.shutdown()
            _stop_persistent_service(service, socket_path)

    def test_scheduler_reuses_saved_request_across_processes_with_service_backend(self):
        socket_path = _short_socket_path("spyre-kv-service")
        service = _start_persistent_service(socket_path)
        ctx = get_context("spawn")
        result_queue = ctx.Queue()

        producer = ctx.Process(
            target=_cross_process_prefill,
            args=(socket_path, result_queue),
        )
        consumer = ctx.Process(
            target=_cross_process_decode,
            args=(socket_path, result_queue),
        )

        try:
            producer.start()
            producer.join(timeout=10.0)
            assert producer.exitcode == 0
            prefill_result = result_queue.get(timeout=2.0)
            assert prefill_result == {
                "saved_requests_count": 1,
                "available_blocks": 2,
            }

            consumer.start()
            consumer.join(timeout=10.0)
            assert consumer.exitcode == 0
            decode_result = result_queue.get(timeout=2.0)
            assert decode_result == {
                "matched": 7,
                "is_async": False,
                "saved_requests_count": 1,
            }
        finally:
            if producer.is_alive():
                producer.terminate()
                producer.join(timeout=2.0)
            if consumer.is_alive():
                consumer.terminate()
                consumer.join(timeout=2.0)
            _stop_persistent_service(service, socket_path)
