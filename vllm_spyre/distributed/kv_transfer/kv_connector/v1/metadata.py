"""Metadata schema for the Spyre KV connector.

SpyreConnectorMeta inherits from the upstream KVConnectorMetadata ABC,
which is required by the build_connector_meta() contract. Per-request
metadata uses structured dataclasses with explicit store/load
classification and optional cross-request block mapping.

Key design decisions:
  1. SpyreConnectorMeta inherits KVConnectorMetadata (upstream contract).
  2. Layout uses upstream naming "NHD" (not "BSHD").
  3. Layer names use vLLM convention "model.layers.{i}.self_attn".
  4. block_mapping enables cross-request prefix reuse.
  5. Single KV cache group assumed (block_ids is a flat list).
  6. schema_version field for forward compatibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats,
)


# ---------------------------------------------------------------------------
# Store Key (for in-memory key-value store)
# ---------------------------------------------------------------------------

class KVKind(StrEnum):
    """Discriminator for key vs value tensors.

    FMS stores K and V as separate tensors per layer. For Spyre-internal
    storage we use separate entries keyed by KVKind.
    """
    K = "K"
    V = "V"


@dataclass(frozen=True)
class StoreKey:
    """Composite key for addressing a single KV block in the store.

    Fields:
        req_id:     The request ID that produced this KV data.
        layer_idx:  The 0-based layer index.
        block_id:   The scheduler-assigned block ID.
        kv_kind:    Whether this entry is a key (K) or value (V) tensor.
    """
    req_id: str
    layer_idx: int
    block_id: int
    kv_kind: KVKind

    @property
    def layer_name(self) -> str:
        """Return the vLLM-convention layer name string."""
        return f"model.layers.{self.layer_idx}.self_attn"


# ---------------------------------------------------------------------------
# Store Value
# ---------------------------------------------------------------------------

@dataclass
class InMemoryKVEntry:
    """A single KV block payload stored in the key-value store.

    Fields:
        data:       CPU tensor clone. Shape: [block_size, num_kv_heads, head_dim]
        dtype:      String representation of the tensor dtype.
        shape:      Tuple of ints for validation.
        version:    Monotonically increasing version counter.
        source_req: The request ID that originally computed this data.
    """
    data: torch.Tensor
    dtype: str
    shape: tuple[int, ...]
    version: int = 1
    source_req: str = ""

    def matches_shape_and_dtype(
        self, expected_shape: tuple[int, ...], expected_dtype: str
    ) -> bool:
        return self.shape == expected_shape and self.dtype == expected_dtype


# ---------------------------------------------------------------------------
# Per-Request Connector Metadata
# ---------------------------------------------------------------------------

@dataclass
class SpyreConnectorRequestMeta:
    """Per-request metadata produced by the scheduler-side connector.

    Fields:
        req_id:         The request identifier.
        block_ids:      Flat list of block IDs for this request's single
                        KV cache group.
        is_store:       True if KV should be saved (prefill).
        token_count:    Number of tokens covered by the block_ids.
        source_req_id:  For load operations, the source request ID.
        block_mapping:  For cross-request reuse: (src_block_idx, dest_block_id)
                        pairs. Empty if same-request operation.
    """
    req_id: str
    block_ids: list[int] = field(default_factory=list)
    is_store: bool = True
    token_count: int = 0
    source_req_id: str = ""
    block_mapping: list[tuple[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Connector Metadata (scheduler -> worker via SchedulerOutput)
# ---------------------------------------------------------------------------

@dataclass
class SpyreConnectorMeta(KVConnectorMetadata):
    """Metadata container flowing from scheduler-side to worker-side connector.

    Inherits KVConnectorMetadata as required by the upstream
    build_connector_meta() contract.

    Fields:
        schema_version: Integer version for forward compatibility.
        requests:       Per-request metadata for this scheduling step.
        layer_names:    Ordered vLLM-convention layer name strings.
        block_size:     Tokens per block.
        dtype:          String dtype of KV cache tensors.
        layout:         KV cache layout descriptor ("NHD").
        num_layers:     Total attention layers.
        num_kv_heads:   KV attention heads per layer.
        head_dim:       Head dimension.
    """
    schema_version: int = 1
    requests: list[SpyreConnectorRequestMeta] = field(default_factory=list)
    layer_names: list[str] = field(default_factory=list)
    block_size: int = 0
    dtype: str = ""
    layout: str = "NHD"
    num_layers: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0

    def add_store_request(
        self,
        req_id: str,
        block_ids: list[int],
        token_count: int = 0,
    ) -> None:
        """Add a request that should have its KV data saved after forward."""
        self.requests.append(
            SpyreConnectorRequestMeta(
                req_id=req_id,
                block_ids=block_ids,
                is_store=True,
                token_count=token_count,
            )
        )

    def add_load_request(
        self,
        req_id: str,
        block_ids: list[int],
        source_req_id: str,
        token_count: int = 0,
        block_mapping: list[tuple[int, int]] | None = None,
    ) -> None:
        """Add a request that should load KV data from a previous save."""
        self.requests.append(
            SpyreConnectorRequestMeta(
                req_id=req_id,
                block_ids=block_ids,
                is_store=False,
                token_count=token_count,
                source_req_id=source_req_id,
                block_mapping=block_mapping or [],
            )
        )

    def validate_block_mapping(self) -> None:
        """Check that no two load requests map to the same destination block.

        Raises:
            ValueError: If duplicate destination block IDs are found.
        """
        all_dest_blocks: list[int] = []
        for req in self.requests:
            if not req.is_store and req.block_mapping:
                for _, dest_block in req.block_mapping:
                    all_dest_blocks.append(dest_block)
            elif not req.is_store:
                all_dest_blocks.extend(req.block_ids)

        seen: set[int] = set()
        for block_id in all_dest_blocks:
            if block_id in seen:
                raise ValueError(
                    f"Duplicate destination block ID {block_id} in "
                    f"load requests. This would cause data corruption."
                )
            seen.add(block_id)

    # Supported schema versions for forward compatibility.
    _SUPPORTED_VERSIONS: frozenset[int] = frozenset({1})

    def validate(self) -> None:
        """Strict validation of metadata fields.

        Always checks schema_version and block mapping duplicates.
        Checks dimensional fields (block_size, num_layers, etc.) only
        when requests are present, since empty metadata may have defaults.

        Raises:
            ValueError: If any field is invalid or unsupported.
        """
        if self.schema_version not in self._SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported schema_version {self.schema_version}. "
                f"Supported: {sorted(self._SUPPORTED_VERSIONS)}"
            )

        # Validate per-request metadata.
        for req in self.requests:
            if not req.req_id:
                raise ValueError("Request with empty req_id in metadata")
            if not req.is_store and not req.source_req_id and req.block_mapping:
                raise ValueError(
                    f"Load request {req.req_id} has block_mapping but "
                    f"no source_req_id"
                )

        # Dimensional fields: reject only clearly invalid negative values.
        # block_size=0 is allowed (informational; connector reads from config).
        if self.num_layers < 0:
            raise ValueError(
                f"num_layers must be >= 0, got {self.num_layers}"
            )
        if self.num_kv_heads < 0:
            raise ValueError(
                f"num_kv_heads must be >= 0, got {self.num_kv_heads}"
            )
        if self.head_dim < 0:
            raise ValueError(
                f"head_dim must be >= 0, got {self.head_dim}"
            )
        if self.block_size < 0:
            raise ValueError(
                f"block_size must be >= 0, got {self.block_size}"
            )

        # Block mapping duplicate check (defers to existing method).
        self.validate_block_mapping()

    @staticmethod
    def make_layer_names(num_layers: int) -> list[str]:
        """Generate vLLM-convention layer names."""
        return [f"model.layers.{i}.self_attn" for i in range(num_layers)]


# ---------------------------------------------------------------------------
# In-Memory KV Store
# ---------------------------------------------------------------------------

class InMemoryKVStore:
    """Dictionary-backed KV store with metrics tracking.

    Injectable: tests can create a fresh store and pass it to the connector.
    NOT thread-safe (Spyre uses single-worker model).
    """

    def __init__(self) -> None:
        self._store: dict[StoreKey, InMemoryKVEntry] = {}
        self._version_counter: int = 0

    @property
    def size(self) -> int:
        return len(self._store)

    def put(
        self,
        key: StoreKey,
        data: torch.Tensor,
        source_req: str = "",
    ) -> tuple[int, bool]:
        """Store a KV block tensor. Returns (version, was_overwrite)."""
        self._version_counter += 1
        version = self._version_counter
        was_overwrite = key in self._store

        self._store[key] = InMemoryKVEntry(
            data=data.detach().clone().cpu(),
            dtype=str(data.dtype),
            shape=tuple(data.shape),
            version=version,
            source_req=source_req,
        )
        return version, was_overwrite

    def get(self, key: StoreKey) -> InMemoryKVEntry | None:
        return self._store.get(key)

    def contains(self, key: StoreKey) -> bool:
        return key in self._store

    def remove_by_req(self, req_id: str) -> int:
        """Remove all entries for a request ID. Returns count removed."""
        keys = [k for k in self._store if k.req_id == req_id]
        for k in keys:
            del self._store[k]
        return len(keys)

    def clear(self) -> None:
        self._store.clear()
        self._version_counter = 0

    def stats(self) -> dict[str, Any]:
        req_ids = {k.req_id for k in self._store}
        return {
            "total_entries": len(self._store),
            "unique_requests": len(req_ids),
            "version_counter": self._version_counter,
            "memory_estimate_bytes": sum(
                e.data.nelement() * e.data.element_size()
                for e in self._store.values()
            ),
        }

    def export_to_dir(self, path: str) -> int:
        """Serialize all entries to a directory for cross-process transport.

        Each entry is stored as a .pt file with the tensor and metadata.
        Returns the number of entries exported.
        """
        os.makedirs(path, exist_ok=True)
        count = 0
        for key, entry in self._store.items():
            fname = (
                f"{key.req_id}__L{key.layer_idx}__B{key.block_id}"
                f"__{key.kv_kind}.pt"
            )
            payload = {
                "data": entry.data,
                "dtype": entry.dtype,
                "shape": entry.shape,
                "version": entry.version,
                "source_req": entry.source_req,
                "key_req_id": key.req_id,
                "key_layer_idx": key.layer_idx,
                "key_block_id": key.block_id,
                "key_kv_kind": str(key.kv_kind),
            }
            torch.save(payload, os.path.join(path, fname))
            count += 1
        return count

    def import_from_dir(self, path: str) -> int:
        """Deserialize entries from a directory into this store.

        Returns the number of entries imported.
        """
        count = 0
        for fname in os.listdir(path):
            if not fname.endswith(".pt"):
                continue
            payload = torch.load(
                os.path.join(path, fname), weights_only=False,
            )
            key = StoreKey(
                req_id=payload["key_req_id"],
                layer_idx=payload["key_layer_idx"],
                block_id=payload["key_block_id"],
                kv_kind=KVKind(payload["key_kv_kind"]),
            )
            self._version_counter += 1
            self._store[key] = InMemoryKVEntry(
                data=payload["data"],
                dtype=payload["dtype"],
                shape=payload["shape"],
                version=self._version_counter,
                source_req=payload["source_req"],
            )
            count += 1
        return count


# ---------------------------------------------------------------------------
# Connector Stats (upstream KVConnectorStats subclass)
# ---------------------------------------------------------------------------

_STATS_KEYS = (
    "matched_tokens",
    "loaded_blocks",
    "saved_blocks",
    "load_misses",
    "evictions",
    "match_attempts",
)


@dataclass
class SpyreConnectorStats(KVConnectorStats):
    """Per-interval telemetry for the InMemorySpyreConnector.

    Each counter accumulates observations during a logging interval.
    reset() zeroes the counters for the next interval.
    """

    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        self.data = {k: 0 for k in _STATS_KEYS}

    def record(self, key: str, value: int = 1) -> None:
        self.data[key] = self.data.get(key, 0) + value

    def aggregate(
        self, other: KVConnectorStats
    ) -> SpyreConnectorStats:
        for k in _STATS_KEYS:
            self.data[k] = self.data.get(k, 0) + other.data.get(k, 0)
        return self

    def reduce(self) -> dict[str, int | float]:
        return {k: self.data.get(k, 0) for k in _STATS_KEYS}

    def is_empty(self) -> bool:
        return all(self.data.get(k, 0) == 0 for k in _STATS_KEYS)
