from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats,
)


class KVKind(str, Enum):
    K = "K"
    V = "V"


@dataclass(frozen=True)
class StoreKey:
    req_id: str
    layer_idx: int
    block_id: int
    kv_kind: KVKind

    @property
    def layer_name(self) -> str:
        return f"model.layers.{self.layer_idx}.self_attn"


@dataclass
class HostMemoryKVEntry:
    data: torch.Tensor
    dtype: str
    shape: tuple[int, ...]
    version: int = 1
    source_req: str = ""

    def matches_shape_and_dtype(
        self, expected_shape: tuple[int, ...], expected_dtype: str
    ) -> bool:
        return self.shape == expected_shape and self.dtype == expected_dtype


class SpyreKVStoreBackend(ABC):
    @property
    @abstractmethod
    def size(self) -> int: ...

    @property
    @abstractmethod
    def current_bytes(self) -> int: ...

    @property
    @abstractmethod
    def max_bytes(self) -> int: ...

    @property
    @abstractmethod
    def evictions(self) -> int: ...

    @abstractmethod
    def put(
        self,
        key: StoreKey,
        data: torch.Tensor,
        source_req: str = "",
    ) -> tuple[int, bool]: ...

    @abstractmethod
    def get(self, key: StoreKey) -> HostMemoryKVEntry | None: ...

    @abstractmethod
    def load_into(self, key: StoreKey, dest: torch.Tensor) -> bool: ...

    @abstractmethod
    def contains(self, key: StoreKey) -> bool: ...

    @abstractmethod
    def remove_by_req(self, req_id: str) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def stats(self) -> dict[str, Any]: ...


@dataclass
class SpyreConnectorRequestMeta:
    req_id: str
    block_ids: list[int] = field(default_factory=list)
    is_store: bool = True
    token_count: int = 0
    source_req_id: str = ""
    block_mapping: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class SpyreConnectorMeta(KVConnectorMetadata):
    schema_version: int = 1
    requests: list[SpyreConnectorRequestMeta] = field(default_factory=list)
    layer_names: list[str] = field(default_factory=list)
    block_size: int = 0
    dtype: str = ""
    layout: str = "NHD"
    num_layers: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0

    _SUPPORTED_VERSIONS: frozenset[int] = frozenset({1})
    _KNOWN_LAYOUTS: frozenset[str] = frozenset({"NHD"})

    def add_store_request(
        self,
        req_id: str,
        block_ids: list[int],
        token_count: int = 0,
    ) -> None:
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
                    f"Duplicate destination block ID {block_id} in load requests. "
                    "This would cause data corruption."
                )
            seen.add(block_id)

    def validate(self) -> None:
        if self.schema_version not in self._SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported schema_version {self.schema_version}. "
                f"Supported: {sorted(self._SUPPORTED_VERSIONS)}"
            )

        if self.layout and self.layout not in self._KNOWN_LAYOUTS:
            raise ValueError(f"Unknown layout '{self.layout}'")

        for req in self.requests:
            if not req.req_id:
                raise ValueError("Request with empty req_id in metadata")
            if not req.is_store and not req.source_req_id and req.block_mapping:
                raise ValueError(
                    f"Load request {req.req_id} has block_mapping but no source_req_id"
                )
            for bid in req.block_ids:
                if bid < 0:
                    raise ValueError(
                        f"Request {req.req_id} has negative block_id {bid}"
                    )
            for src, dest in req.block_mapping:
                if src < 0 or dest < 0:
                    raise ValueError(
                        f"Request {req.req_id} block_mapping contains negative IDs"
                    )

        if self.num_layers < 0 or self.num_kv_heads < 0 or self.head_dim < 0:
            raise ValueError("Connector metadata dimensions must be non-negative")
        if self.block_size < 0:
            raise ValueError("block_size must be non-negative")

        self.validate_block_mapping()

    @staticmethod
    def make_layer_names(num_layers: int) -> list[str]:
        return [f"model.layers.{i}.self_attn" for i in range(num_layers)]


class HostMemoryKVStoreBackend(SpyreKVStoreBackend):
    def __init__(self, max_bytes: int = 0) -> None:
        self._store: dict[StoreKey, HostMemoryKVEntry] = {}
        self._version_counter = 0
        self._max_bytes = max(0, max_bytes)
        self._current_bytes = 0
        self._evictions = 0

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def current_bytes(self) -> int:
        return self._current_bytes

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def evictions(self) -> int:
        return self._evictions

    @staticmethod
    def _entry_bytes(entry: HostMemoryKVEntry) -> int:
        return entry.data.nelement() * entry.data.element_size()

    def put(
        self,
        key: StoreKey,
        data: torch.Tensor,
        source_req: str = "",
    ) -> tuple[int, bool]:
        self._version_counter += 1
        version = self._version_counter
        was_overwrite = key in self._store

        if was_overwrite:
            old = self._store[key]
            self._current_bytes -= self._entry_bytes(old)

        entry = HostMemoryKVEntry(
            data=data.detach().clone().cpu(),
            dtype=str(data.dtype),
            shape=tuple(data.shape),
            version=version,
            source_req=source_req,
        )
        entry_size = self._entry_bytes(entry)

        if self._max_bytes > 0:
            while self._store and self._current_bytes + entry_size > self._max_bytes:
                self._evict_oldest()

        self._store[key] = entry
        self._current_bytes += entry_size
        return version, was_overwrite

    def _evict_oldest(self) -> StoreKey | None:
        if not self._store:
            return None
        oldest_key = next(iter(self._store))
        oldest_entry = self._store.pop(oldest_key)
        self._current_bytes -= self._entry_bytes(oldest_entry)
        self._evictions += 1
        return oldest_key

    def get(self, key: StoreKey) -> HostMemoryKVEntry | None:
        return self._store.get(key)

    def load_into(self, key: StoreKey, dest: torch.Tensor) -> bool:
        entry = self.get(key)
        if entry is None:
            return False

        try:
            dest.copy_(entry.data)
        except RuntimeError:
            return False
        return True

    def contains(self, key: StoreKey) -> bool:
        return key in self._store

    def remove_by_req(self, req_id: str) -> int:
        keys = [k for k in self._store if k.req_id == req_id]
        for key in keys:
            entry = self._store.pop(key)
            self._current_bytes -= self._entry_bytes(entry)
        return len(keys)

    def clear(self) -> None:
        self._store.clear()
        self._version_counter = 0
        self._current_bytes = 0

    def stats(self) -> dict[str, Any]:
        req_ids = {k.req_id for k in self._store}
        return {
            "total_entries": len(self._store),
            "unique_requests": len(req_ids),
            "version_counter": self._version_counter,
            "memory_estimate_bytes": self._current_bytes,
            "max_bytes": self._max_bytes,
            "evictions": self._evictions,
        }


# Backward-compatible aliases for the current slice, tests, and examples.
InMemoryKVEntry = HostMemoryKVEntry
InMemoryKVStore = HostMemoryKVStoreBackend


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
    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        self.data = {key: 0 for key in _STATS_KEYS}

    def record(self, key: str, value: int = 1) -> None:
        self.data[key] = self.data.get(key, 0) + value

    def aggregate(self, other: KVConnectorStats) -> "SpyreConnectorStats":
        for key in _STATS_KEYS:
            self.data[key] = self.data.get(key, 0) + other.data.get(key, 0)
        return self

    def reduce(self) -> dict[str, int | float]:
        return {key: self.data.get(key, 0) for key in _STATS_KEYS}

    def is_empty(self) -> bool:
        return all(self.data.get(key, 0) == 0 for key in _STATS_KEYS)
