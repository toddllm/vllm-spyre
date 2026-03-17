from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

import vllm_spyre.envs as envs_spyre
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats,
)
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_spyre.distributed.kv_transfer.kv_connector.v1.metadata import (
    KVKind,
    SpyreConnectorMeta,
    SpyreConnectorRequestMeta,
    SpyreConnectorStats,
    SpyreKVStoreBackend,
    StoreKey,
    build_spyre_kv_store_backend,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)

_GLOBAL_STORE: SpyreKVStoreBackend | None = None


def _build_configured_store(store_backend_name: str | None = None) -> SpyreKVStoreBackend:
    backend_name = store_backend_name or envs_spyre.VLLM_SPYRE_KV_STORE_BACKEND
    return build_spyre_kv_store_backend(
        backend_name,
        max_bytes=envs_spyre.VLLM_SPYRE_KV_STORE_MAX_BYTES,
    )


def get_global_store(store_backend_name: str | None = None) -> SpyreKVStoreBackend:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = _build_configured_store(store_backend_name)
    return _GLOBAL_STORE


def reset_global_store(store_backend_name: str | None = None) -> None:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is not None:
        _GLOBAL_STORE.clear()
    _GLOBAL_STORE = _build_configured_store(store_backend_name)


@dataclass(frozen=True)
class _SavedRequest:
    req_id: str
    prompt_token_ids: tuple[int, ...]
    block_ids: list[int]
    num_tokens: int


@dataclass(frozen=True)
class _PendingLoadSource:
    source: _SavedRequest
    matched_tokens_total: int
    num_local_computed_tokens: int


class InMemorySpyreConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
        store: SpyreKVStoreBackend | None = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._role_name = "scheduler" if role == KVConnectorRole.SCHEDULER else "worker"
        self._block_size = vllm_config.cache_config.block_size
        self._store = store if store is not None else get_global_store()

        self._pending_requests: list[SpyreConnectorRequestMeta] = []
        self._saved_requests: OrderedDict[str, _SavedRequest] = OrderedDict()
        self._saved_requests_max_size = envs_spyre.VLLM_SPYRE_KV_REUSE_REGISTRY_MAX_SIZE
        self._pending_load_sources: dict[str, _PendingLoadSource] = {}

        self._kv_caches: dict[str, torch.Tensor] = {}
        self._num_layers = 0
        self._num_kv_heads = 0
        self._head_dim = 0
        self._dtype_str = ""
        self._layer_names: list[str] = []

        self._step_stores: set[str] = set()
        self._step_loads: set[str] = set()
        self._load_error_block_ids: set[int] = set()
        self._blocks_saved = 0
        self._blocks_loaded = 0
        self._blocks_missing = 0
        self._stats = SpyreConnectorStats()

    def _prune_saved_request(self, req_id: str, *, remove_store: bool = False) -> bool:
        removed = self._saved_requests.pop(req_id, None) is not None
        if remove_store:
            self._store.remove_by_req(req_id)
        return removed

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches
        if kv_caches:
            self._num_layers = len(kv_caches)
            self._layer_names = sorted(kv_caches.keys())
            first_tensor = next(iter(kv_caches.values()))
            self._dtype_str = str(first_tensor.dtype)
            if first_tensor.dim() >= 4:
                self._num_kv_heads = first_tensor.shape[-2]
                self._head_dim = first_tensor.shape[-1]

    def bind_connector_metadata(
        self, connector_metadata: KVConnectorMetadata
    ) -> None:
        if isinstance(connector_metadata, SpyreConnectorMeta):
            connector_metadata.validate()
        self._step_stores.clear()
        self._step_loads.clear()
        self._load_error_block_ids.clear()
        super().bind_connector_metadata(connector_metadata)

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        if not self.has_connector_metadata():
            return

        meta = self._get_connector_metadata()
        if not isinstance(meta, SpyreConnectorMeta):
            return

        total_load = 0
        total_miss = 0
        self._load_error_block_ids.clear()

        for layer_idx, layer_name in enumerate(self._layer_names):
            layer_load, layer_miss = self._load_layer(meta, layer_idx, layer_name)
            total_load += layer_load
            total_miss += layer_miss

        self._blocks_loaded += total_load
        self._blocks_missing += total_miss
        self._stats.record("loaded_blocks", total_load)
        self._stats.record("load_misses", total_miss)

        for req_meta in meta.requests:
            if not req_meta.is_store:
                self._step_loads.add(req_meta.req_id)

    def _load_layer(
        self,
        meta: SpyreConnectorMeta,
        layer_idx: int,
        layer_name: str,
    ) -> tuple[int, int]:
        staging = self._kv_caches.get(layer_name)
        if staging is None:
            return 0, 0

        load_count = 0
        miss_count = 0

        for req_meta in meta.requests:
            if req_meta.is_store:
                continue

            source_req = req_meta.source_req_id or req_meta.req_id
            mapping = (
                list(req_meta.block_mapping)
                if req_meta.block_mapping
                else [(block_id, block_id) for block_id in req_meta.block_ids]
            )

            for src_block_id, dest_bid in mapping:
                if dest_bid < 0 or dest_bid >= staging.shape[1]:
                    miss_count += 1
                    self._load_error_block_ids.add(dest_bid)
                    continue

                for kv_kind, kv_dim in ((KVKind.K, 0), (KVKind.V, 1)):
                    store_key = StoreKey(
                        req_id=source_req,
                        layer_idx=layer_idx,
                        block_id=src_block_id,
                        kv_kind=kv_kind,
                    )
                    if self._store.load_into(store_key, staging[kv_dim][dest_bid]):
                        load_count += 1
                    else:
                        miss_count += 1
                        self._load_error_block_ids.add(dest_bid)

        return load_count, miss_count

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        return

    def _save_kv_bulk(self) -> None:
        if not self.has_connector_metadata():
            return

        meta = self._get_connector_metadata()
        if not isinstance(meta, SpyreConnectorMeta):
            return

        save_count = 0
        for req_meta in meta.requests:
            if not req_meta.is_store:
                continue

            for layer_idx, layer_name in enumerate(self._layer_names):
                staging = self._kv_caches.get(layer_name)
                if staging is None:
                    continue

                for block_id in req_meta.block_ids:
                    for kv_kind, kv_dim in ((KVKind.K, 0), (KVKind.V, 1)):
                        store_key = StoreKey(
                            req_id=req_meta.req_id,
                            layer_idx=layer_idx,
                            block_id=block_id,
                            kv_kind=kv_kind,
                        )
                        self._store.put(
                            store_key,
                            staging[kv_dim][block_id],
                            source_req=req_meta.req_id,
                        )
                        save_count += 1

            self._step_stores.add(req_meta.req_id)

        self._blocks_saved += save_count
        self._stats.record("saved_blocks", save_count)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def wait_for_save(self) -> None:
        self._save_kv_bulk()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        finished_sending = self._step_stores & finished_req_ids
        finished_recving = self._step_loads & finished_req_ids
        return (
            finished_sending if finished_sending else None,
            finished_recving if finished_recving else None,
        )

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set(self._load_error_block_ids)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        prompt = request.prompt_token_ids
        if not prompt or not self._saved_requests:
            self._pending_load_sources.pop(request.request_id, None)
            self._stats.record("match_attempts")
            return 0, False

        prompt_tuple = tuple(prompt)
        best_match: _SavedRequest | None = None
        best_tokens_total = 0
        stale_request_ids: list[str] = []

        for saved in self._saved_requests.values():
            saved_len = len(saved.prompt_token_ids)
            if saved_len == 0:
                continue

            available_blocks = self._store.available_prefix_blocks(
                saved.req_id,
                saved.block_ids,
            )
            if available_blocks < len(saved.block_ids):
                stale_request_ids.append(saved.req_id)
                continue

            common_len = 0
            for prompt_token, saved_token in zip(prompt_tuple, saved.prompt_token_ids):
                if prompt_token != saved_token:
                    break
                common_len += 1

            aligned = (common_len // self._block_size) * self._block_size
            if aligned > best_tokens_total:
                best_tokens_total = aligned
                best_match = saved

        for req_id in stale_request_ids:
            self._prune_saved_request(req_id)

        if best_match is None or best_tokens_total == 0:
            self._pending_load_sources.pop(request.request_id, None)
            self._stats.record("match_attempts")
            return 0, False

        num_local = max(0, num_computed_tokens)
        num_external = max(0, best_tokens_total - num_local)
        num_external = (num_external // self._block_size) * self._block_size
        if num_external == 0:
            self._pending_load_sources.pop(request.request_id, None)
            self._stats.record("match_attempts")
            return 0, False

        self._pending_load_sources[request.request_id] = _PendingLoadSource(
            source=best_match,
            matched_tokens_total=best_tokens_total,
            num_local_computed_tokens=num_local,
        )
        self._stats.record("match_attempts")
        self._stats.record("matched_tokens", num_external)
        return num_external, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        block_id_lists = blocks.get_block_ids() if blocks is not None else ()
        if block_id_lists:
            assert len(block_id_lists) == 1, (
                "InMemorySpyreConnector assumes a single KV cache group"
            )

        flat_block_ids: list[int] = []
        for group in block_id_lists:
            flat_block_ids.extend(group)

        if num_external_tokens > 0:
            pending = self._pending_load_sources.get(request.request_id)
            if pending is None:
                req_meta = SpyreConnectorRequestMeta(
                    req_id=request.request_id,
                    block_ids=flat_block_ids,
                    is_store=True,
                    token_count=len(request.all_token_ids),
                )
            else:
                num_external_blocks = num_external_tokens // self._block_size
                local_blocks = pending.num_local_computed_tokens // self._block_size
                source = pending.source
                block_mapping: list[tuple[int, int]] = []

                for i in range(num_external_blocks):
                    src_idx = local_blocks + i
                    dest_idx = local_blocks + i
                    if src_idx >= len(source.block_ids) or dest_idx >= len(flat_block_ids):
                        break
                    block_mapping.append((source.block_ids[src_idx], flat_block_ids[dest_idx]))

                if len(block_mapping) != num_external_blocks:
                    req_meta = SpyreConnectorRequestMeta(
                        req_id=request.request_id,
                        block_ids=flat_block_ids,
                        is_store=True,
                        token_count=len(request.all_token_ids),
                    )
                else:
                    req_meta = SpyreConnectorRequestMeta(
                        req_id=request.request_id,
                        block_ids=flat_block_ids,
                        is_store=False,
                        token_count=num_external_tokens,
                        source_req_id=source.req_id,
                        block_mapping=block_mapping,
                    )
        else:
            req_meta = SpyreConnectorRequestMeta(
                req_id=request.request_id,
                block_ids=flat_block_ids,
                is_store=True,
                token_count=len(request.all_token_ids),
            )

        self._pending_requests.append(req_meta)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = SpyreConnectorMeta(
            requests=list(self._pending_requests),
            layer_names=list(self._layer_names),
            block_size=self._block_size,
            dtype=self._dtype_str,
            layout="NHD",
            num_layers=self._num_layers,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
        )
        self._pending_requests.clear()
        self._pending_load_sources.clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        prompt = request.prompt_token_ids
        if prompt and block_ids:
            available_blocks = self._store.available_prefix_blocks(
                request.request_id,
                block_ids,
            )
            if available_blocks < len(block_ids):
                self._store.remove_by_req(request.request_id)
                return False, None

            saved = _SavedRequest(
                req_id=request.request_id,
                prompt_token_ids=tuple(prompt),
                block_ids=list(block_ids),
                num_tokens=len(prompt),
            )
            if request.request_id in self._saved_requests:
                self._saved_requests.move_to_end(request.request_id)
            self._saved_requests[request.request_id] = saved

            if self._saved_requests_max_size > 0:
                while len(self._saved_requests) > self._saved_requests_max_size:
                    oldest_req_id, _ = self._saved_requests.popitem(last=False)
                    self._store.remove_by_req(oldest_req_id)
                    self._stats.record("evictions")

        return False, None

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self._stats.is_empty():
            return None
        snapshot = SpyreConnectorStats(data=dict(self._stats.data))
        self._stats.reset()
        return snapshot

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        if data is not None:
            return SpyreConnectorStats(data=data)
        return SpyreConnectorStats()

    def get_cumulative_metrics(self) -> dict[str, int]:
        return {
            "blocks_saved": self._blocks_saved,
            "blocks_loaded": self._blocks_loaded,
            "blocks_missing": self._blocks_missing,
            "saved_requests_count": len(self._saved_requests),
        }

    def get_store(self) -> SpyreKVStoreBackend:
        return self._store

    def reset_probe_state(
        self,
        *,
        clear_store: bool = True,
        clear_saved_requests: bool = True,
        clear_metrics: bool = True,
    ) -> None:
        self._pending_requests.clear()
        self._pending_load_sources.clear()
        self._step_stores.clear()
        self._step_loads.clear()
        self._load_error_block_ids.clear()

        if clear_saved_requests:
            self._saved_requests.clear()

        if clear_metrics:
            self._blocks_saved = 0
            self._blocks_loaded = 0
            self._blocks_missing = 0
            self._stats.reset()

        if clear_store:
            self._store.clear()

    def shutdown(self) -> None:
        logger.info("[InMemorySpyreConnector] Shutdown. Store stats: %s", self._store.stats())
