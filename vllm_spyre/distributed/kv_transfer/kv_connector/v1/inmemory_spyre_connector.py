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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_spyre.distributed.kv_transfer.kv_connector.v1.metadata import (
    InMemoryKVStore,
    KVKind,
    SpyreConnectorMeta,
    SpyreConnectorRequestMeta,
    StoreKey,
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

_GLOBAL_STORE: InMemoryKVStore | None = None


def get_global_store() -> InMemoryKVStore:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = InMemoryKVStore()
    return _GLOBAL_STORE


def reset_global_store() -> None:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is not None:
        _GLOBAL_STORE.clear()
    else:
        _GLOBAL_STORE = InMemoryKVStore()


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
        store: InMemoryKVStore | None = None,
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
        # Maps req_id -> _SavedRequest for requests whose KV was saved.
        self._saved_requests: dict[str, _SavedRequest] = {}

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

        # Metrics
        self._blocks_saved: int = 0
        self._blocks_loaded: int = 0
        self._blocks_missing: int = 0

        logger.info(
            "[InMemorySpyreConnector] Initialized role=%s, block_size=%d",
            self._role_name, self._block_size,
        )

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
            connector_metadata.validate_block_mapping()
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

        For each load-request in the metadata, for each layer, for each
        block_id: look up the source data and copy into the staging
        tensor at staging[0][block_id] (K) and staging[1][block_id] (V).
        """
        if not self.has_connector_metadata():
            return

        meta = self._get_connector_metadata()
        if not isinstance(meta, SpyreConnectorMeta):
            return

        load_count = 0
        miss_count = 0
        self._load_error_block_ids.clear()

        for req_meta in meta.requests:
            if req_meta.is_store:
                continue

            source_req = req_meta.source_req_id or req_meta.req_id

            # Build block mapping: source_block_id -> dest_block_id
            if req_meta.block_mapping:
                mapping = list(req_meta.block_mapping)
            else:
                # Identity mapping fallback: source and destination block IDs
                # are the same when no explicit remap is provided.
                mapping = [
                    (block_id, block_id) for block_id in req_meta.block_ids
                ]

            for layer_idx, layer_name in enumerate(self._layer_names):
                staging = self._kv_caches.get(layer_name)
                if staging is None:
                    continue

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
                        entry = self._store.get(store_key)
                        if entry is None:
                            miss_count += 1
                            self._load_error_block_ids.add(dest_bid)
                            continue

                        # Copy into staging tensor at the dest block position
                        try:
                            staging[kv_dim][dest_bid].copy_(entry.data)
                            load_count += 1
                        except RuntimeError:
                            miss_count += 1
                            self._load_error_block_ids.add(dest_bid)

            self._step_loads.add(req_meta.req_id)

        self._blocks_loaded += load_count
        self._blocks_missing += miss_count

        if load_count > 0 or miss_count > 0:
            logger.debug(
                "[InMemorySpyreConnector] start_load_kv: loaded=%d, missed=%d",
                load_count, miss_count,
            )

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

        for req_meta in meta.requests:
            if not req_meta.is_store:
                continue

            for layer_idx, layer_name in enumerate(self._layer_names):
                staging = self._kv_caches.get(layer_name)
                if staging is None:
                    continue

                for block_id in req_meta.block_ids:
                    for kv_kind, kv_dim in [(KVKind.K, 0), (KVKind.V, 1)]:
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
        """No-op: all loading is done synchronously in start_load_kv."""
        pass

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
        finished_sending = self._step_stores & finished_req_ids
        finished_recving = self._step_loads & finished_req_ids

        return (
            finished_sending if finished_sending else None,
            finished_recving if finished_recving else None,
        )

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return destination block IDs that failed to load this step."""
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
        if not prompt or not self._saved_requests:
            self._pending_load_sources.pop(request.request_id, None)
            return 0, False

        prompt_tuple = tuple(prompt)
        best_match: _SavedRequest | None = None
        best_tokens_total = 0

        for saved in self._saved_requests.values():
            saved_len = len(saved.prompt_token_ids)
            if saved_len == 0:
                continue

            # Compute common prefix length between saved and new prompts.
            # Either saved is a prefix of new, or new is a prefix of saved,
            # or they share a common prefix that's shorter than both.
            common_len = min(saved_len, len(prompt_tuple))
            if prompt_tuple[:common_len] != saved.prompt_token_ids[:common_len]:
                continue

            # Block-align: round down to block_size boundary
            aligned = (common_len // self._block_size) * self._block_size
            if aligned == 0:
                continue

            if aligned > best_tokens_total:
                best_tokens_total = aligned
                best_match = saved

        if best_match is not None and best_tokens_total > 0:
            # Return only the additional tokens beyond what is already local.
            # Keep this conservative and block-aligned to avoid over-reporting.
            num_local = max(0, num_computed_tokens)
            num_external = max(0, best_tokens_total - num_local)
            num_external = (num_external // self._block_size) * self._block_size

            if num_external == 0:
                self._pending_load_sources.pop(request.request_id, None)
                return 0, False

            # Record match context for update_state_after_alloc.
            self._pending_load_sources[request.request_id] = _PendingLoadSource(
                source=best_match,
                matched_tokens_total=best_tokens_total,
                num_local_computed_tokens=num_local,
            )
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
            saved = _SavedRequest(
                req_id=request.request_id,
                prompt_token_ids=tuple(prompt),
                block_ids=list(block_ids),
                num_tokens=len(prompt),
            )
            self._saved_requests[request.request_id] = saved
            logger.debug(
                "[InMemorySpyreConnector] request_finished: req=%s saved "
                "(%d tokens, %d blocks) for reuse",
                request.request_id, len(prompt), len(block_ids),
            )
        else:
            logger.debug(
                "[InMemorySpyreConnector] request_finished: req=%s, "
                "%d blocks (not saved — no prompt or blocks)",
                request.request_id, len(block_ids),
            )
        return False, None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_store(self) -> InMemoryKVStore:
        """Return the underlying store (for testing/inspection)."""
        return self._store

    def shutdown(self) -> None:
        """Clean up on worker shutdown."""
        logger.info(
            "[InMemorySpyreConnector] Shutdown. Store stats: %s",
            self._store.stats(),
        )
