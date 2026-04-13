"""In-memory KV connector for Spyre.

Implements KVConnectorBase_V1 for the Spyre execution model where:
  - FMS manages its own KV cache via past_key_value_states
  - save_kv_layer() is never called inline (FMS attention is opaque)
  - All operations are synchronous (no async DMA)
  - The bridge (spyre_kv_connector_bridge.py) drives the lifecycle

Scheduler-side methods produce SpyreConnectorMeta with per-request
store/load directives. Worker-side methods execute actual data movement
using the injectable InMemoryKVStore.

The connector reads/writes ONLY staging tensors registered via
register_kv_caches(). The model runner owns FMS<->staging sync.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

import vllm_spyre.envs as envs_spyre

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)

from vllm_spyre.distributed.kv_transfer.kv_connector.v1.metadata import (
    InMemoryKVStore,
    KVKind,
    SavedRequestRecord,
    SpyreConnectorMeta,
    SpyreConnectorRequestMeta,
    SpyreConnectorStats,
    SpyreKVStoreBackend,
    StoreKey,
    build_spyre_kv_store_backend,
)
from vllm_spyre.distributed.kv_transfer.kv_connector.v1.heap_kv_accessor import (
    resolve_heap_kv_paths,
)
from vllm_spyre.distributed.kv_transfer.kv_connector.v1.heap_kv_inprocess_client import (
    InProcessHeapKVClient,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global store (can be replaced for testing)
# ---------------------------------------------------------------------------

_GLOBAL_STORE: SpyreKVStoreBackend | None = None


def _build_configured_store(
    store_backend_name: str | None = None,
) -> SpyreKVStoreBackend:
    backend_name = store_backend_name or envs_spyre.VLLM_SPYRE_KV_STORE_BACKEND
    return build_spyre_kv_store_backend(
        backend_name,
        max_bytes=envs_spyre.VLLM_SPYRE_KV_STORE_MAX_BYTES,
        max_saved_requests=envs_spyre.VLLM_SPYRE_KV_REUSE_REGISTRY_MAX_SIZE,
        service_socket=envs_spyre.VLLM_SPYRE_KV_SERVICE_SOCKET,
    )


def get_global_store(
    store_backend_name: str | None = None,
) -> SpyreKVStoreBackend:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = _build_configured_store(store_backend_name)
    return _GLOBAL_STORE


def reset_global_store(store_backend_name: str | None = None) -> None:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is not None:
        _GLOBAL_STORE.shutdown()
    _GLOBAL_STORE = _build_configured_store(store_backend_name)


# ---------------------------------------------------------------------------
# Scheduler-side saved-request registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SavedRequest:
    """Record of a request whose KV was saved for potential reuse.

    Stored on the scheduler-side connector instance so that
    get_num_new_matched_tokens can find prefix matches.
    """
    req_id: str
    prompt_token_ids: tuple[int, ...]
    block_ids: list[int]
    num_tokens: int


@dataclass(frozen=True)
class _PendingLoadSource:
    """Per-step match context captured by get_num_new_matched_tokens()."""

    source: _SavedRequest
    matched_tokens_total: int
    num_local_computed_tokens: int


# ---------------------------------------------------------------------------
# Prometheus metrics adapter
# ---------------------------------------------------------------------------

class SpyreConnectorPromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics for the InMemorySpyreConnector.

    Registers per-engine counters for KV connector operations.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(
            vllm_config, metric_types, labelnames, per_engine_labelvalues,
        )

        # Counters — monotonically increasing totals.
        counter_matched_tokens = self._counter_cls(
            name="vllm:spyre_kv_matched_tokens_total",
            documentation=(
                "Total tokens matched for KV reuse by the Spyre connector."
            ),
            labelnames=labelnames,
        )
        self.counter_matched_tokens = self.make_per_engine(
            counter_matched_tokens,
        )

        counter_loaded_blocks = self._counter_cls(
            name="vllm:spyre_kv_loaded_blocks_total",
            documentation=(
                "Total KV blocks loaded from the store by the Spyre connector."
            ),
            labelnames=labelnames,
        )
        self.counter_loaded_blocks = self.make_per_engine(
            counter_loaded_blocks,
        )

        counter_saved_blocks = self._counter_cls(
            name="vllm:spyre_kv_saved_blocks_total",
            documentation=(
                "Total KV blocks saved to the store by the Spyre connector."
            ),
            labelnames=labelnames,
        )
        self.counter_saved_blocks = self.make_per_engine(
            counter_saved_blocks,
        )

        counter_load_misses = self._counter_cls(
            name="vllm:spyre_kv_load_misses_total",
            documentation=(
                "Total KV block load misses (missing data in store)."
            ),
            labelnames=labelnames,
        )
        self.counter_load_misses = self.make_per_engine(counter_load_misses)

        counter_evictions = self._counter_cls(
            name="vllm:spyre_kv_evictions_total",
            documentation=(
                "Total saved-request registry evictions (LRU)."
            ),
            labelnames=labelnames,
        )
        self.counter_evictions = self.make_per_engine(counter_evictions)

        counter_match_attempts = self._counter_cls(
            name="vllm:spyre_kv_match_attempts_total",
            documentation=(
                "Total prefix match attempts by the Spyre connector."
            ),
            labelnames=labelnames,
        )
        self.counter_match_attempts = self.make_per_engine(
            counter_match_attempts,
        )

    def observe(
        self,
        transfer_stats_data: dict[str, Any],
        engine_idx: int = 0,
    ) -> None:
        """Map SpyreConnectorStats data dict to Prometheus counters."""
        for prom_counter, key in [
            (self.counter_matched_tokens, "matched_tokens"),
            (self.counter_loaded_blocks, "loaded_blocks"),
            (self.counter_saved_blocks, "saved_blocks"),
            (self.counter_load_misses, "load_misses"),
            (self.counter_evictions, "evictions"),
            (self.counter_match_attempts, "match_attempts"),
        ]:
            value = transfer_stats_data.get(key, 0)
            if value > 0:
                prom_counter[engine_idx].inc(value)


# ---------------------------------------------------------------------------
# InMemorySpyreConnector
# ---------------------------------------------------------------------------

class InMemorySpyreConnector(KVConnectorBase_V1):
    """In-memory KV connector for Spyre.

    Key design decisions:
      1. save_kv_layer() is a no-op. FMS never calls it inline.
         Saving is done via wait_for_save() -> save_kv_bulk().
      2. start_load_kv() performs synchronous bulk load.
      3. The connector reads/writes ONLY staging tensors. The model
         runner owns the staging<->FMS sync.
      4. get_num_new_matched_tokens() is conservative: returns 0 unless
         a full block-aligned prefix match is verified.
      5. get_finished() only marks requests finished when actual work
         was completed this step.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
        store: SpyreKVStoreBackend | None = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._role_name = (
            "scheduler" if role == KVConnectorRole.SCHEDULER else "worker"
        )
        self._block_size = vllm_config.cache_config.block_size
        self._store = store if store is not None else get_global_store()

        # Scheduler-side per-step state (reset in build_connector_meta)
        self._pending_requests: list[SpyreConnectorRequestMeta] = []

        # Scheduler-side saved-request registry for prefix reuse.
        # OrderedDict for LRU eviction — oldest entries evicted first.
        self._saved_requests: OrderedDict[str, _SavedRequest] = OrderedDict()
        self._saved_requests_max_size: int = (
            envs_spyre.VLLM_SPYRE_KV_REUSE_REGISTRY_MAX_SIZE
        )

        # Per-step: maps request_id -> match context for load requests
        # that were matched in get_num_new_matched_tokens this step.
        self._pending_load_sources: dict[str, _PendingLoadSource] = {}

        # Worker-side: registered staging KV caches
        # These are the [2, ...] staging tensors keyed by layer name.
        self._kv_caches: dict[str, torch.Tensor] = {}

        # Model config extracted for metadata
        self._num_layers: int = 0
        self._num_kv_heads: int = 0
        self._head_dim: int = 0
        self._dtype_str: str = ""
        self._layer_names: list[str] = []

        # Track which requests had work done this step (for get_finished)
        self._step_stores: set[str] = set()
        self._step_loads: set[str] = set()
        self._load_error_block_ids: set[int] = set()

        # Cumulative metrics (lifetime of connector)
        self._blocks_saved: int = 0
        self._blocks_loaded: int = 0
        self._blocks_missing: int = 0

        # Per-interval stats (reset each time stats are collected)
        self._stats = SpyreConnectorStats()

        self._use_heap_kv = bool(envs_spyre.VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_ENABLE)
        self._heap_kv_strict = bool(
            envs_spyre.VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_STRICT
        )
        self._heap_kv_client: InProcessHeapKVClient | None = None
        self._heap_kv_init_error: str | None = None

        # Async layer pipeline.
        # When _async_load_enabled is True, per-layer loads are submitted
        # to a thread pool and wait_for_layer_load() blocks on futures.
        # When False (default), loads are synchronous as before.
        self._async_load_workers: int = max(
            0, envs_spyre.VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS
        )
        self._async_load_enabled: bool = self._async_load_workers > 0
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=self._async_load_workers)
            if self._async_load_enabled
            else None
        )
        self._layer_load_futures: dict[str, Future[tuple[int, int]]] = {}
        self._layer_load_done: dict[str, bool] = {}

        logger.info(
            "[InMemorySpyreConnector] Initialized role=%s, block_size=%d, "
            "async_load=%s (workers=%d)",
            self._role_name, self._block_size,
            self._async_load_enabled, self._async_load_workers,
        )

    def _prune_saved_request(self, req_id: str, *, remove_store: bool = False) -> bool:
        if self._store.has_persistent_saved_requests():
            removed = self._store.remove_saved_request(req_id)
        else:
            removed = self._saved_requests.pop(req_id, None) is not None
        if remove_store:
            self._store.remove_by_req(req_id)
        return removed

    def _load_saved_requests(self) -> list[_SavedRequest]:
        if not self._store.has_persistent_saved_requests():
            return list(self._saved_requests.values())

        records = [
            _SavedRequest(
                req_id=str(record.req_id),
                prompt_token_ids=tuple(record.prompt_token_ids),
                block_ids=list(record.block_ids),
                num_tokens=int(record.num_tokens),
            )
            for record in self._store.get_saved_requests()
        ]
        self._saved_requests = OrderedDict((record.req_id, record) for record in records)
        return records

    def _save_request_record(self, record: _SavedRequest) -> None:
        if self._store.has_persistent_saved_requests():
            self._store.save_request_record(
                SavedRequestRecord(
                    req_id=record.req_id,
                    prompt_token_ids=tuple(record.prompt_token_ids),
                    block_ids=list(record.block_ids),
                    num_tokens=record.num_tokens,
                )
            )
            self._load_saved_requests()
            return

        if record.req_id in self._saved_requests:
            self._saved_requests.move_to_end(record.req_id)
        self._saved_requests[record.req_id] = record

        if self._saved_requests_max_size > 0:
            while len(self._saved_requests) > self._saved_requests_max_size:
                oldest_req_id, _ = self._saved_requests.popitem(last=False)
                self._store.remove_by_req(oldest_req_id)
                self._stats.record("evictions")

    def _saved_request_count(self) -> int:
        if self._store.has_persistent_saved_requests():
            return self._store.saved_request_count()
        return len(self._saved_requests)

    @property
    def uses_heap_kv(self) -> bool:
        return self._use_heap_kv

    def heap_kv_active(self) -> bool:
        return self._ensure_heap_kv_client() is not None

    def get_heap_kv_status(self) -> dict[str, Any]:
        active = self._heap_kv_client is not None
        return {
            "requested": bool(self._use_heap_kv),
            "strict": bool(self._heap_kv_strict),
            "active": active,
            "mode": "in_process" if active else None,
            "init_error": self._heap_kv_init_error,
        }

    def _ensure_heap_kv_client(self) -> InProcessHeapKVClient | None:
        if not self._use_heap_kv:
            return None
        if self._heap_kv_client is not None:
            return self._heap_kv_client

        try:
            if not self._kv_caches:
                raise RuntimeError(
                    "KV cache geometry is unavailable before register_kv_caches"
                )
            first_tensor = next(iter(self._kv_caches.values()))
            perfdsc_dir, export_dir = resolve_heap_kv_paths(
                perfdsc_dir=envs_spyre.VLLM_SPYRE_HEAP_KV_PERFDSC_DIR or None,
                export_dir=envs_spyre.VLLM_SPYRE_HEAP_KV_EXPORT_DIR or None,
            )
            client = InProcessHeapKVClient(
                kv_heads=self._num_kv_heads,
                block_size=self._block_size,
                head_dim=self._head_dim,
                dtype=first_tensor.dtype,
                perfdsc_dir=perfdsc_dir,
                export_dir=export_dir,
            )
            self._heap_kv_client = client
            return self._heap_kv_client
        except Exception as exc:
            self._heap_kv_init_error = f"{type(exc).__name__}: {exc}"
            if self._heap_kv_strict:
                raise RuntimeError(
                    "Experimental in-process heap KV strict mode is enabled and "
                    f"initialization failed: {self._heap_kv_init_error}"
                ) from exc
            logger.warning(
                "[InMemorySpyreConnector] Experimental in-process heap KV unavailable, "
                "falling back to staging path: %s",
                self._heap_kv_init_error,
            )
            return None

    # ------------------------------------------------------------------
    # Worker-side: KV cache registration
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register staging KV cache tensors.

        These are the [2, ...] staging tensors created by the worker.
        The connector reads/writes only these. The model runner owns
        the sync between staging and FMS live tensors.
        """
        self._kv_caches = kv_caches

        if kv_caches:
            self._num_layers = len(kv_caches)
            self._layer_names = sorted(kv_caches.keys())

            first_tensor = next(iter(kv_caches.values()))
            self._dtype_str = str(first_tensor.dtype)
            # Staging tensor shape: [2, num_blocks, block_size, num_kv_heads, head_dim]
            # or [2, <whatever shape FMS uses>]
            if first_tensor.dim() >= 4:
                self._num_kv_heads = first_tensor.shape[-2]
                self._head_dim = first_tensor.shape[-1]

        logger.info(
            "[InMemorySpyreConnector] Registered %d staging KV caches, "
            "dtype=%s, num_kv_heads=%d, head_dim=%d",
            len(kv_caches), self._dtype_str,
            self._num_kv_heads, self._head_dim,
        )

    # ------------------------------------------------------------------
    # Worker-side: metadata binding (with type check logging)
    # ------------------------------------------------------------------

    def bind_connector_metadata(
        self, connector_metadata: KVConnectorMetadata
    ) -> None:
        if isinstance(connector_metadata, SpyreConnectorMeta):
            connector_metadata.validate()
        else:
            logger.warning(
                "[InMemorySpyreConnector] Expected SpyreConnectorMeta, "
                "got %s. Load/save operations will be skipped.",
                type(connector_metadata).__name__,
            )
        # Reset per-step tracking
        self._step_stores.clear()
        self._step_loads.clear()
        self._load_error_block_ids.clear()
        super().bind_connector_metadata(connector_metadata)

    # ------------------------------------------------------------------
    # Worker-side: KV load (synchronous bulk load before forward)
    # ------------------------------------------------------------------

    def start_load_kv(
        self, forward_context: ForwardContext, **kwargs: Any
    ) -> None:
        """Load KV cache blocks from the store into staging tensors.

        When async loading is enabled (VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS > 0),
        per-layer loads are submitted to a thread pool and
        wait_for_layer_load() blocks on the individual layer's future.

        When async loading is disabled (default), loads are synchronous
        and every layer is marked done immediately.
        """
        if not self.has_connector_metadata():
            return

        meta = self._get_connector_metadata()
        if not isinstance(meta, SpyreConnectorMeta):
            return

        # Reset per-layer tracking for this step.
        self._layer_load_done = {ln: False for ln in self._layer_names}
        self._layer_load_futures.clear()
        self._load_error_block_ids.clear()

        total_load = 0
        total_miss = 0
        has_load_requests = any(not req_meta.is_store for req_meta in meta.requests)
        heap_client = self._ensure_heap_kv_client() if has_load_requests else None

        if heap_client is not None:
            total_load, total_miss = self._load_via_heap_helper(meta, heap_client)
            for layer_name in self._layer_names:
                self._layer_load_done[layer_name] = True
            self._blocks_loaded += total_load
            self._blocks_missing += total_miss
            self._stats.record("loaded_blocks", total_load)
            self._stats.record("load_misses", total_miss)
        elif self._async_load_enabled and self._executor is not None:
            # Async path: submit per-layer loads to thread pool.
            for layer_idx, layer_name in enumerate(self._layer_names):
                future = self._executor.submit(
                    self._load_layer, meta, layer_idx, layer_name,
                )
                self._layer_load_futures[layer_name] = future
        else:
            # Sync path: load all layers sequentially.
            for layer_idx, layer_name in enumerate(self._layer_names):
                layer_load, layer_miss = self._load_layer(
                    meta, layer_idx, layer_name,
                )
                total_load += layer_load
                total_miss += layer_miss
                self._layer_load_done[layer_name] = True

            self._blocks_loaded += total_load
            self._blocks_missing += total_miss
            self._stats.record("loaded_blocks", total_load)
            self._stats.record("load_misses", total_miss)

            if total_load > 0 or total_miss > 0:
                logger.debug(
                    "[InMemorySpyreConnector] start_load_kv: loaded=%d, missed=%d",
                    total_load, total_miss,
                )

        # Track request IDs that had load work.
        for req_meta in meta.requests:
            if not req_meta.is_store:
                self._step_loads.add(req_meta.req_id)

    def _load_layer(
        self,
        meta: SpyreConnectorMeta,
        layer_idx: int,
        layer_name: str,
    ) -> tuple[int, int]:
        """Load all blocks for one layer. Returns (loaded, missed) counts.

        Factored out so a future async backend can issue per-layer loads
        as independent operations.
        """
        staging = self._kv_caches.get(layer_name)
        if staging is None:
            return 0, 0

        load_count = 0
        miss_count = 0

        for req_meta in meta.requests:
            if req_meta.is_store:
                continue

            source_req = req_meta.source_req_id or req_meta.req_id

            if req_meta.block_mapping:
                mapping = list(req_meta.block_mapping)
            else:
                # Identity mapping fallback: source and destination block IDs
                # are the same when no explicit remap is provided.
                mapping = [
                    (block_id, block_id) for block_id in req_meta.block_ids
                ]

            for src_block_id, dest_bid in mapping:
                if dest_bid < 0 or dest_bid >= staging.shape[1]:
                    miss_count += 1
                    self._load_error_block_ids.add(dest_bid)
                    continue

                for kv_kind, kv_dim in [(KVKind.K, 0), (KVKind.V, 1)]:
                    store_key = StoreKey(
                        req_id=source_req,
                        layer_idx=layer_idx,
                        block_id=src_block_id,
                        kv_kind=kv_kind,
                    )
                    try:
                        loaded = self._store.load_into(
                            store_key, staging[kv_dim][dest_bid]
                        )
                    except RuntimeError:
                        loaded = False

                    if loaded:
                        load_count += 1
                    else:
                        miss_count += 1
                        self._load_error_block_ids.add(dest_bid)

        return load_count, miss_count

    def _load_via_heap_helper(
        self,
        meta: SpyreConnectorMeta,
        heap_client: InProcessHeapKVClient,
    ) -> tuple[int, int]:
        block_values: dict[tuple[int, str, int], torch.Tensor] = {}
        load_count = 0
        miss_count = 0
        dtype = next(iter(self._kv_caches.values())).dtype

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
                for layer_idx in range(self._num_layers):
                    for kv_kind in (KVKind.K, KVKind.V):
                        store_key = StoreKey(
                            req_id=source_req,
                            layer_idx=layer_idx,
                            block_id=src_block_id,
                            kv_kind=kv_kind,
                        )
                        cpu_block = torch.empty(
                            (self._block_size, self._num_kv_heads, self._head_dim),
                            dtype=dtype,
                            device="cpu",
                        )
                        if self._store.load_into(store_key, cpu_block):
                            block_values[
                                (layer_idx, kv_kind.value.lower(), dest_bid)
                            ] = cpu_block
                            load_count += 1
                        else:
                            miss_count += 1
                            self._load_error_block_ids.add(dest_bid)

        if block_values:
            heap_client.write_blocks(block_values)
        return load_count, miss_count

    # ------------------------------------------------------------------
    # Worker-side: KV save
    # ------------------------------------------------------------------

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Per-layer save hook. No-op for Spyre (FMS never calls this)."""
        pass

    def _save_kv_bulk(self) -> None:
        """Bulk save of KV cache blocks from staging tensors into the store.

        For each store-request in the metadata, for each layer, for each
        block_id: extract the block from staging and store it.
        """
        if not self.has_connector_metadata():
            return

        meta = self._get_connector_metadata()
        if not isinstance(meta, SpyreConnectorMeta):
            return

        save_count = 0
        heap_client = self._ensure_heap_kv_client()

        for req_meta in meta.requests:
            if not req_meta.is_store:
                continue

            heap_blocks: dict[tuple[int, str, int], torch.Tensor] = {}
            if heap_client is not None:
                block_refs = [
                    (layer_idx, kv_kind.value.lower(), block_id)
                    for layer_idx in range(self._num_layers)
                    for block_id in req_meta.block_ids
                    for kv_kind in (KVKind.K, KVKind.V)
                ]
                if block_refs:
                    heap_blocks = heap_client.read_blocks(block_refs)

            for layer_idx, layer_name in enumerate(self._layer_names):
                staging = self._kv_caches.get(layer_name)
                if staging is None and not heap_blocks:
                    continue

                for block_id in req_meta.block_ids:
                    for kv_kind, kv_dim in [(KVKind.K, 0), (KVKind.V, 1)]:
                        store_key = StoreKey(
                            req_id=req_meta.req_id,
                            layer_idx=layer_idx,
                            block_id=block_id,
                            kv_kind=kv_kind,
                        )
                        if heap_blocks:
                            cpu_block = heap_blocks[
                                (layer_idx, kv_kind.value.lower(), block_id)
                            ]
                        else:
                            assert staging is not None
                            cpu_block = staging[kv_dim][block_id]
                        self._store.put(
                            store_key,
                            cpu_block,
                            source_req=req_meta.req_id,
                        )
                        save_count += 1

            self._step_stores.add(req_meta.req_id)

        self._blocks_saved += save_count
        self._stats.record("saved_blocks", save_count)

        if save_count > 0:
            logger.debug(
                "[InMemorySpyreConnector] _save_kv_bulk: saved=%d blocks, "
                "store_size=%d",
                save_count, self._store.size,
            )

    # ------------------------------------------------------------------
    # Worker-side: wait/finish lifecycle
    # ------------------------------------------------------------------

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Wait for a specific layer's load to complete.

        With async loading enabled, blocks until the layer's thread pool
        future resolves. With sync loading, this is a no-op (the layer
        is already marked done in start_load_kv).
        """
        if self._layer_load_done.get(layer_name, True):
            return

        future = self._layer_load_futures.get(layer_name)
        if future is not None:
            layer_load, layer_miss = future.result()
            self._layer_load_done[layer_name] = True
            self._blocks_loaded += layer_load
            self._blocks_missing += layer_miss
            self._stats.record("loaded_blocks", layer_load)
            self._stats.record("load_misses", layer_miss)
            del self._layer_load_futures[layer_name]

    def _collect_all_async_loads(self) -> None:
        """Block until all outstanding async layer loads complete.

        Called internally before operations that need all loads done
        (e.g., get_finished, get_block_ids_with_load_errors).
        """
        for layer_name in list(self._layer_load_futures):
            self.wait_for_layer_load(layer_name)

    def wait_for_save(self) -> None:
        """Trigger bulk save. In upstream protocol, this blocks until
        async saves complete. Since FMS never calls save_kv_layer,
        we use this as the trigger for bulk save."""
        self._save_kv_bulk()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Report finished requests.

        Only marks request IDs finished when actual work was completed
        this step. Does NOT over-report.
        """
        self._collect_all_async_loads()
        finished_sending = self._step_stores & finished_req_ids
        finished_recving = self._step_loads & finished_req_ids

        return (
            finished_sending if finished_sending else None,
            finished_recving if finished_recving else None,
        )

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return destination block IDs that failed to load this step."""
        self._collect_all_async_loads()
        return set(self._load_error_block_ids)

    # ------------------------------------------------------------------
    # Scheduler-side: token matching
    # ------------------------------------------------------------------

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Exact-prefix, block-aligned token matching.

        Searches the saved-request registry for a request whose prompt
        is an exact prefix of this request's prompt. Returns the
        block-aligned token count of the longest such prefix.

        Side-effect free: may be called multiple times for the same
        request. The match is recorded in _pending_load_sources only
        if this is the first call for this request_id.

        Returns:
            (num_matched_tokens, is_async):
              - num_matched_tokens: block-aligned count, or 0 if no match
              - is_async: always False (synchronous loading)
        """
        prompt = request.prompt_token_ids
        saved_requests = self._load_saved_requests()
        if not prompt or not saved_requests:
            self._pending_load_sources.pop(request.request_id, None)
            self._stats.record("match_attempts")
            return 0, False

        prompt_tuple = tuple(prompt)
        best_match: _SavedRequest | None = None
        best_tokens_total = 0
        stale_request_ids: list[str] = []

        for saved in saved_requests:
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
            for prompt_token, saved_token in zip(
                prompt_tuple, saved.prompt_token_ids
            ):
                if prompt_token != saved_token:
                    break
                common_len += 1

            aligned = (common_len // self._block_size) * self._block_size
            if aligned > best_tokens_total:
                best_tokens_total = aligned
                best_match = saved

        for req_id in stale_request_ids:
            self._prune_saved_request(req_id, remove_store=True)

        if best_match is not None and best_tokens_total > 0:
            num_local = max(0, num_computed_tokens)
            if best_tokens_total >= len(prompt_tuple):
                target_total_computed = max(0, len(prompt_tuple) - 1)
                num_external = max(0, target_total_computed - num_local)
            else:
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
            logger.debug(
                "[InMemorySpyreConnector] get_num_new_matched_tokens: "
                "req=%s matched source=%s, total=%d, local=%d, external=%d",
                request.request_id,
                best_match.req_id,
                best_tokens_total,
                num_local,
                num_external,
            )
            return num_external, False

        self._pending_load_sources.pop(request.request_id, None)
        self._stats.record("match_attempts")
        return 0, False

    # ------------------------------------------------------------------
    # Scheduler-side: state updates and metadata production
    # ------------------------------------------------------------------

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        """Record block allocation for this request.

        When num_external_tokens > 0, the scheduler matched a prefix via
        get_num_new_matched_tokens. We produce a load request with the
        source_req_id and block_mapping from source blocks to dest blocks.

        When num_external_tokens == 0, this is a normal prefill: produce
        a store request.
        """
        block_id_lists = blocks.get_block_ids() if blocks is not None else ()

        # Single KV cache group assertion
        if block_id_lists:
            assert len(block_id_lists) == 1, (
                f"InMemorySpyreConnector assumes single KV cache group, "
                f"got {len(block_id_lists)}"
            )

        flat_block_ids: list[int] = []
        for group in block_id_lists:
            flat_block_ids.extend(group)

        if num_external_tokens > 0:
            # Load path: produce load request with block mapping
            pending = self._pending_load_sources.get(request.request_id)
            if pending is None:
                logger.warning(
                    "[InMemorySpyreConnector] update_state_after_alloc: "
                    "num_external_tokens=%d but no source found for req=%s. "
                    "Falling back to store request.",
                    num_external_tokens, request.request_id,
                )
                req_meta = SpyreConnectorRequestMeta(
                    req_id=request.request_id,
                    block_ids=flat_block_ids,
                    is_store=True,
                    token_count=len(request.all_token_ids),
                )
            else:
                # Compute how many blocks the external tokens cover
                num_external_blocks = num_external_tokens // self._block_size
                local_blocks = (
                    pending.num_local_computed_tokens // self._block_size
                )
                source = pending.source

                # Build block mapping: source_block_id -> dest_block_id.
                # External blocks start after local-computed blocks in both
                # source and destination block lists.
                block_mapping: list[tuple[int, int]] = []
                for i in range(num_external_blocks):
                    src_idx = local_blocks + i
                    dest_idx = local_blocks + i
                    if src_idx >= len(source.block_ids):
                        break
                    if dest_idx >= len(flat_block_ids):
                        break
                    src_block_id = source.block_ids[src_idx]
                    dest_block_id = flat_block_ids[dest_idx]
                    block_mapping.append((src_block_id, dest_block_id))

                if len(block_mapping) != num_external_blocks:
                    logger.warning(
                        "[InMemorySpyreConnector] update_state_after_alloc: "
                        "incomplete block mapping for req=%s "
                        "(expected %d, built %d). Falling back to store.",
                        request.request_id, num_external_blocks,
                        len(block_mapping),
                    )
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
            # Store path: normal prefill
            req_meta = SpyreConnectorRequestMeta(
                req_id=request.request_id,
                block_ids=flat_block_ids,
                is_store=True,
                token_count=len(request.all_token_ids),
            )

        self._pending_requests.append(req_meta)

        logger.debug(
            "[InMemorySpyreConnector] update_state_after_alloc: "
            "req=%s, blocks=%d, external_tokens=%d, is_store=%s",
            request.request_id, len(flat_block_ids),
            num_external_tokens, req_meta.is_store,
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build SpyreConnectorMeta for this scheduling step.

        Packages pending requests into metadata and resets per-step state.
        """
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

        # Reset per-step state
        self._pending_requests.clear()
        self._pending_load_sources.clear()

        return meta

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request finishes generating.

        Records the request's prompt and block info in the saved-request
        registry so that future requests with matching prefixes can reuse
        the KV data. Returns (False, None) — blocks can be freed
        immediately (the KV data lives in the in-memory store, not in
        the block manager's live blocks).
        """
        prompt = request.prompt_token_ids
        if prompt and block_ids:
            num_prompt_blocks = (len(prompt) + self._block_size - 1) // self._block_size
            prompt_block_ids = list(block_ids[:num_prompt_blocks])
            if not prompt_block_ids:
                logger.info(
                    "[InMemorySpyreConnector] request_finished req=%s prompt_tokens=%d "
                    "received no prompt block ids from %s",
                    request.request_id,
                    len(prompt),
                    block_ids,
                )
                return False, None

            available_blocks = self._store.available_prefix_blocks(
                request.request_id,
                prompt_block_ids,
            )
            if available_blocks < len(prompt_block_ids):
                logger.info(
                    "[InMemorySpyreConnector] request_finished prune req=%s "
                    "prompt_tokens=%d block_ids=%s prompt_block_ids=%s "
                    "available_blocks=%d store_stats=%s",
                    request.request_id,
                    len(prompt),
                    block_ids,
                    prompt_block_ids,
                    available_blocks,
                    self._store.stats(),
                )
                self._store.remove_by_req(request.request_id)
                return False, None

            saved = _SavedRequest(
                req_id=request.request_id,
                prompt_token_ids=tuple(prompt),
                block_ids=prompt_block_ids,
                num_tokens=len(prompt),
            )
            self._save_request_record(saved)
            logger.info(
                "[InMemorySpyreConnector] request_finished saved req=%s "
                "prompt_tokens=%d prompt_block_ids=%s store_stats=%s",
                request.request_id,
                len(prompt),
                prompt_block_ids,
                self._store.stats(),
            )

        return False, None

    # ------------------------------------------------------------------
    # Stats and metrics
    # ------------------------------------------------------------------

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Return accumulated stats for the current interval.

        The bridge calls this in after_forward() and attaches the result
        to KVConnectorOutput. This method resets the internal interval
        counters after returning a snapshot.
        """
        if self._stats.is_empty():
            return None
        snapshot = SpyreConnectorStats(data=dict(self._stats.data))
        self._stats.reset()
        return snapshot

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        """Factory for deserialization on the logger side."""
        if data is not None:
            return SpyreConnectorStats(data=data)
        return SpyreConnectorStats()

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        """Create Prometheus metrics for this connector."""
        return SpyreConnectorPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues,
        )

    def get_cumulative_metrics(self) -> dict[str, int]:
        """Return lifetime cumulative metrics (for testing/inspection)."""
        return {
            "blocks_saved": self._blocks_saved,
            "blocks_loaded": self._blocks_loaded,
            "blocks_missing": self._blocks_missing,
            "saved_requests_count": self._saved_request_count(),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_store(self) -> SpyreKVStoreBackend:
        """Return the underlying store (for testing/inspection)."""
        return self._store

    def shutdown(self) -> None:
        """Clean up on worker shutdown."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._store.shutdown()
        logger.info(
            "[InMemorySpyreConnector] Shutdown. Store stats: %s",
            self._store.stats(),
        )
