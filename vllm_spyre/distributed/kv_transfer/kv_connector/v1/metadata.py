from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
import io
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


@dataclass
class SerializedHostMemoryKVEntry:
    payload: bytes
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
    def backend_name(self) -> str: ...

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
    def available_prefix_blocks(self, req_id: str, block_ids: list[int]) -> int: ...

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
        self._request_keys: dict[str, set[StoreKey]] = {}
        self._request_order: OrderedDict[str, None] = OrderedDict()
        self._version_counter = 0
        self._max_bytes = max(0, max_bytes)
        self._current_bytes = 0
        self._evictions = 0

    @property
    def backend_name(self) -> str:
        return "host_memory"

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

    @staticmethod
    def _request_id_for(key: StoreKey, source_req: str) -> str:
        return source_req or key.req_id

    def _track_key(self, req_id: str, key: StoreKey) -> None:
        keys = self._request_keys.setdefault(req_id, set())
        keys.add(key)
        self._request_order.pop(req_id, None)
        self._request_order[req_id] = None

    def put(
        self,
        key: StoreKey,
        data: torch.Tensor,
        source_req: str = "",
    ) -> tuple[int, bool]:
        req_id = self._request_id_for(key, source_req)
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
                if self._evict_oldest_request(exclude_req_id=req_id) is None:
                    break

        self._store[key] = entry
        self._track_key(req_id, key)
        self._current_bytes += entry_size
        return version, was_overwrite

    def _evict_oldest_request(self, exclude_req_id: str | None = None) -> str | None:
        for req_id in list(self._request_order.keys()):
            if req_id == exclude_req_id:
                continue
            if self.remove_by_req(req_id) > 0:
                self._evictions += 1
                return req_id
        return None

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

    def available_prefix_blocks(self, req_id: str, block_ids: list[int]) -> int:
        req_keys = self._request_keys.get(req_id)
        if not req_keys:
            return 0

        layer_ids = sorted({key.layer_idx for key in req_keys})
        if not layer_ids:
            return 0

        available = 0
        for block_id in block_ids:
            block_complete = True
            for layer_idx in layer_ids:
                for kv_kind in (KVKind.K, KVKind.V):
                    if StoreKey(req_id, layer_idx, block_id, kv_kind) not in self._store:
                        block_complete = False
                        break
                if not block_complete:
                    break
            if not block_complete:
                break
            available += 1
        return available

    def remove_by_req(self, req_id: str) -> int:
        keys = list(self._request_keys.pop(req_id, ()))
        for key in keys:
            entry = self._store.pop(key, None)
            if entry is not None:
                self._current_bytes -= self._entry_bytes(entry)
        self._request_order.pop(req_id, None)
        return len(keys)

    def clear(self) -> None:
        self._store.clear()
        self._request_keys.clear()
        self._request_order.clear()
        self._version_counter = 0
        self._current_bytes = 0
        self._evictions = 0

    def stats(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "total_entries": len(self._store),
            "unique_requests": len(self._request_keys),
            "version_counter": self._version_counter,
            "memory_estimate_bytes": self._current_bytes,
            "max_bytes": self._max_bytes,
            "evictions": self._evictions,
        }


class SerializedHostMemoryKVStoreBackend(SpyreKVStoreBackend):
    def __init__(self, max_bytes: int = 0) -> None:
        self._store: dict[StoreKey, SerializedHostMemoryKVEntry] = {}
        self._request_keys: dict[str, set[StoreKey]] = {}
        self._request_order: OrderedDict[str, None] = OrderedDict()
        self._version_counter = 0
        self._max_bytes = max(0, max_bytes)
        self._current_bytes = 0
        self._evictions = 0

    @property
    def backend_name(self) -> str:
        return "serialized_host_memory"

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
    def _entry_bytes(entry: SerializedHostMemoryKVEntry) -> int:
        return len(entry.payload)

    @staticmethod
    def _serialize_tensor(data: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        torch.save(data.detach().clone().cpu(), buffer)
        return buffer.getvalue()

    @staticmethod
    def _materialize_tensor(entry: SerializedHostMemoryKVEntry) -> torch.Tensor:
        buffer = io.BytesIO(entry.payload)
        return torch.load(buffer, map_location="cpu", weights_only=True)

    @staticmethod
    def _request_id_for(key: StoreKey, source_req: str) -> str:
        return source_req or key.req_id

    def _track_key(self, req_id: str, key: StoreKey) -> None:
        keys = self._request_keys.setdefault(req_id, set())
        keys.add(key)
        self._request_order.pop(req_id, None)
        self._request_order[req_id] = None

    def put(
        self,
        key: StoreKey,
        data: torch.Tensor,
        source_req: str = "",
    ) -> tuple[int, bool]:
        req_id = self._request_id_for(key, source_req)
        self._version_counter += 1
        version = self._version_counter
        was_overwrite = key in self._store

        if was_overwrite:
            old = self._store[key]
            self._current_bytes -= self._entry_bytes(old)

        entry = SerializedHostMemoryKVEntry(
            payload=self._serialize_tensor(data),
            dtype=str(data.dtype),
            shape=tuple(data.shape),
            version=version,
            source_req=source_req,
        )
        entry_size = self._entry_bytes(entry)

        if self._max_bytes > 0:
            while self._store and self._current_bytes + entry_size > self._max_bytes:
                if self._evict_oldest_request(exclude_req_id=req_id) is None:
                    break

        self._store[key] = entry
        self._track_key(req_id, key)
        self._current_bytes += entry_size
        return version, was_overwrite

    def _evict_oldest_request(self, exclude_req_id: str | None = None) -> str | None:
        for req_id in list(self._request_order.keys()):
            if req_id == exclude_req_id:
                continue
            if self.remove_by_req(req_id) > 0:
                self._evictions += 1
                return req_id
        return None

    def get(self, key: StoreKey) -> HostMemoryKVEntry | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        return HostMemoryKVEntry(
            data=self._materialize_tensor(entry),
            dtype=entry.dtype,
            shape=entry.shape,
            version=entry.version,
            source_req=entry.source_req,
        )

    def load_into(self, key: StoreKey, dest: torch.Tensor) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return False

        try:
            dest.copy_(self._materialize_tensor(entry))
        except RuntimeError:
            return False
        return True

    def contains(self, key: StoreKey) -> bool:
        return key in self._store

    def available_prefix_blocks(self, req_id: str, block_ids: list[int]) -> int:
        req_keys = self._request_keys.get(req_id)
        if not req_keys:
            return 0

        layer_ids = sorted({key.layer_idx for key in req_keys})
        if not layer_ids:
            return 0

        available = 0
        for block_id in block_ids:
            block_complete = True
            for layer_idx in layer_ids:
                for kv_kind in (KVKind.K, KVKind.V):
                    if StoreKey(req_id, layer_idx, block_id, kv_kind) not in self._store:
                        block_complete = False
                        break
                if not block_complete:
                    break
            if not block_complete:
                break
            available += 1
        return available

    def remove_by_req(self, req_id: str) -> int:
        keys = list(self._request_keys.pop(req_id, ()))
        for key in keys:
            entry = self._store.pop(key, None)
            if entry is not None:
                self._current_bytes -= self._entry_bytes(entry)
        self._request_order.pop(req_id, None)
        return len(keys)

    def clear(self) -> None:
        self._store.clear()
        self._request_keys.clear()
        self._request_order.clear()
        self._version_counter = 0
        self._current_bytes = 0
        self._evictions = 0

    def stats(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "total_entries": len(self._store),
            "unique_requests": len(self._request_keys),
            "version_counter": self._version_counter,
            "memory_estimate_bytes": self._current_bytes,
            "max_bytes": self._max_bytes,
            "evictions": self._evictions,
        }


_STORE_BACKEND_TYPES: dict[str, type[SpyreKVStoreBackend]] = {
    "host_memory": HostMemoryKVStoreBackend,
    "serialized_host_memory": SerializedHostMemoryKVStoreBackend,
}


def build_spyre_kv_store_backend(
    backend_name: str,
    *,
    max_bytes: int = 0,
) -> SpyreKVStoreBackend:
    backend_key = backend_name.strip().lower()
    backend_type = _STORE_BACKEND_TYPES.get(backend_key)
    if backend_type is None:
        supported = ", ".join(sorted(_STORE_BACKEND_TYPES))
        raise ValueError(
            f"Unknown Spyre KV store backend '{backend_name}'. Supported backends: {supported}"
        )
    return backend_type(max_bytes=max_bytes)


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
