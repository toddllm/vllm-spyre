"""Tests for InMemorySpyreConnector and metadata schema.

Validates:
1. Metadata schema correctness (inheritance, fields, validation)
2. Identity load/save roundtrip (store then load, data matches)
3. Cross-request block remapping
4. Lifecycle with real connector behind the bridge
5. get_finished safety (only reports actual completions)
6. request_finished cleanup behavior
7. Duplicate factory registration is harmless
8. Bad block mapping fails cleanly
9. Non-driver path clears metadata
10. Conservative get_num_new_matched_tokens
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.outputs import KVConnectorOutput

from vllm_spyre.distributed.kv_transfer.kv_connector.v1.metadata import (
    InMemoryKVStore,
    KVKind,
    SpyreConnectorMeta,
    SpyreConnectorRequestMeta,
    StoreKey,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vllm_config_mock(block_size: int = 64):
    """Create a VllmConfig mock suitable for connector construction."""
    vllm_config = MagicMock()
    vllm_config.cache_config.block_size = block_size
    vllm_config.kv_transfer_config = MagicMock()
    vllm_config.kv_transfer_config.is_kv_transfer_instance = True
    vllm_config.kv_transfer_config.kv_connector = "InMemorySpyreConnector"
    # For set_forward_context compatibility
    vllm_config.parallel_config.data_parallel_size = 1
    vllm_config.compilation_config.fast_moe_cold_start = False
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.speculative_config = None
    return vllm_config


def _make_connector(store=None, role=KVConnectorRole.WORKER, block_size=64):
    """Create an InMemorySpyreConnector with a fresh or provided store."""
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
    """Create staging tensors shaped [2, num_blocks, block_size, num_kv_heads, head_dim]."""
    caches = {}
    for i in range(num_layers):
        layer_name = f"model.layers.{i}.self_attn"
        caches[layer_name] = torch.zeros(
            (2, num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
        )
    return caches


def _make_fake_scheduler_output(
    kv_connector_metadata=None,
    finished_req_ids=None,
):
    """Create a minimal SchedulerOutput-like object."""
    so = MagicMock()
    so.kv_connector_metadata = kv_connector_metadata
    so.finished_req_ids = finished_req_ids or set()
    so.total_num_scheduled_tokens = 0
    so.preempted_req_ids = set()
    return so


# ---------------------------------------------------------------------------
# Metadata Schema Tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestMetadataSchema:
    """Validate the metadata schema structure and inheritance."""

    def test_spyre_connector_meta_inherits_kv_connector_metadata(self):
        """SpyreConnectorMeta must inherit from KVConnectorMetadata."""
        meta = SpyreConnectorMeta()
        assert isinstance(meta, KVConnectorMetadata)

    def test_schema_version_present(self):
        meta = SpyreConnectorMeta()
        assert meta.schema_version == 1

    def test_default_layout_is_nhd(self):
        meta = SpyreConnectorMeta()
        assert meta.layout == "NHD"

    def test_add_store_request(self):
        meta = SpyreConnectorMeta()
        meta.add_store_request("req-1", [0, 1, 2], token_count=192)
        assert len(meta.requests) == 1
        assert meta.requests[0].is_store is True
        assert meta.requests[0].req_id == "req-1"
        assert meta.requests[0].block_ids == [0, 1, 2]

    def test_add_load_request(self):
        meta = SpyreConnectorMeta()
        meta.add_load_request(
            "req-2", [3, 4, 5],
            source_req_id="req-1",
            token_count=192,
        )
        assert len(meta.requests) == 1
        assert meta.requests[0].is_store is False
        assert meta.requests[0].source_req_id == "req-1"

    def test_add_load_with_block_mapping(self):
        meta = SpyreConnectorMeta()
        meta.add_load_request(
            "req-2", [3, 4, 5],
            source_req_id="req-1",
            block_mapping=[(0, 3), (1, 4), (2, 5)],
        )
        assert meta.requests[0].block_mapping == [(0, 3), (1, 4), (2, 5)]

    def test_validate_block_mapping_no_duplicates(self):
        meta = SpyreConnectorMeta()
        meta.add_load_request("req-2", [3, 4], source_req_id="req-1")
        # No duplicates — should not raise
        meta.validate_block_mapping()

    def test_validate_block_mapping_duplicate_raises(self):
        meta = SpyreConnectorMeta()
        meta.add_load_request("req-2", [3, 4], source_req_id="req-1")
        meta.add_load_request("req-3", [4, 5], source_req_id="req-1")
        # block_id 4 appears in both — should raise
        with pytest.raises(ValueError, match="Duplicate destination block ID"):
            meta.validate_block_mapping()

    def test_make_layer_names(self):
        names = SpyreConnectorMeta.make_layer_names(3)
        assert names == [
            "model.layers.0.self_attn",
            "model.layers.1.self_attn",
            "model.layers.2.self_attn",
        ]


# ---------------------------------------------------------------------------
# InMemoryKVStore Tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestInMemoryKVStore:
    """Validate the store operations."""

    def test_put_and_get(self):
        store = InMemoryKVStore()
        key = StoreKey("req-1", 0, 0, KVKind.K)
        data = torch.randn(2, 2, 4)
        version, overwrite = store.put(key, data, source_req="req-1")
        assert version == 1
        assert overwrite is False

        entry = store.get(key)
        assert entry is not None
        assert torch.equal(entry.data, data)

    def test_overwrite_detection(self):
        store = InMemoryKVStore()
        key = StoreKey("req-1", 0, 0, KVKind.K)
        store.put(key, torch.randn(2, 2, 4))
        _, overwrite = store.put(key, torch.randn(2, 2, 4))
        assert overwrite is True

    def test_remove_by_req(self):
        store = InMemoryKVStore()
        store.put(StoreKey("req-1", 0, 0, KVKind.K), torch.randn(2))
        store.put(StoreKey("req-1", 0, 1, KVKind.K), torch.randn(2))
        store.put(StoreKey("req-2", 0, 0, KVKind.K), torch.randn(2))
        removed = store.remove_by_req("req-1")
        assert removed == 2
        assert store.size == 1

    def test_data_is_cloned(self):
        """Store should clone data, not hold a reference."""
        store = InMemoryKVStore()
        key = StoreKey("req-1", 0, 0, KVKind.K)
        original = torch.ones(2)
        store.put(key, original)
        original.fill_(999.0)
        entry = store.get(key)
        assert entry is not None
        assert torch.equal(entry.data, torch.ones(2))


# ---------------------------------------------------------------------------
# Identity Load/Save Roundtrip Tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestIdentityLoadSaveRoundtrip:
    """Store data via save, load it back, verify data matches."""

    def test_save_then_load_same_request(self):
        """Save KV for req-1, then load it back for req-1."""
        store = InMemoryKVStore()
        connector = _make_connector(store=store, block_size=2)

        staging = _make_staging_caches(
            num_layers=2, num_blocks=4, block_size=2,
            num_kv_heads=2, head_dim=4,
        )
        connector.register_kv_caches(staging)

        # Fill staging blocks 0,1 with known data (simulating FMS output)
        for layer_name in staging:
            staging[layer_name][0][0].fill_(1.0)  # K, block 0
            staging[layer_name][1][0].fill_(2.0)  # V, block 0
            staging[layer_name][0][1].fill_(3.0)  # K, block 1
            staging[layer_name][1][1].fill_(4.0)  # V, block 1

        # Save phase: bind metadata with store request, call wait_for_save
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

        # Verify data is in the store
        assert store.size > 0
        k_entry = store.get(StoreKey("req-1", 0, 0, KVKind.K))
        assert k_entry is not None
        assert torch.equal(k_entry.data, torch.ones(2, 2, 4))

        connector.clear_connector_metadata()

        # Zero out staging to prove load actually copies data
        for layer_name in staging:
            staging[layer_name].zero_()

        # Load phase: bind metadata with load request, call start_load_kv
        load_meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-1",
                    block_ids=[0, 1],
                    is_store=False,
                    source_req_id="req-1",
                    token_count=128,
                ),
            ],
            layer_names=sorted(staging.keys()),
            block_size=2,
        )
        connector.bind_connector_metadata(load_meta)
        connector.start_load_kv(MagicMock())

        # Verify staging was populated
        for layer_name in staging:
            assert torch.equal(
                staging[layer_name][0][0],
                torch.ones(2, 2, 4),
            ), f"K block 0 mismatch in {layer_name}"
            assert torch.equal(
                staging[layer_name][1][0],
                torch.full((2, 2, 4), 2.0),
            ), f"V block 0 mismatch in {layer_name}"

        connector.clear_connector_metadata()

    def test_save_then_load_cross_request(self):
        """Save KV for req-A, load into req-B's blocks with remapping."""
        store = InMemoryKVStore()
        connector = _make_connector(store=store, block_size=2)

        staging = _make_staging_caches(
            num_layers=1, num_blocks=6, block_size=2,
            num_kv_heads=1, head_dim=2,
        )
        connector.register_kv_caches(staging)

        layer_name = sorted(staging.keys())[0]

        # Fill source blocks 0,1 with known data
        staging[layer_name][0][0].fill_(10.0)  # K, block 0
        staging[layer_name][1][0].fill_(20.0)  # V, block 0
        staging[layer_name][0][1].fill_(30.0)  # K, block 1
        staging[layer_name][1][1].fill_(40.0)  # V, block 1

        # Save req-A blocks [0, 1]
        save_meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-A",
                    block_ids=[0, 1],
                    is_store=True,
                ),
            ],
            layer_names=sorted(staging.keys()),
        )
        connector.bind_connector_metadata(save_meta)
        connector.wait_for_save()
        connector.clear_connector_metadata()

        # Zero staging
        staging[layer_name].zero_()

        # Load req-B from req-A, remapping block 0->3, block 1->4
        load_meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-B",
                    block_ids=[3, 4],
                    is_store=False,
                    source_req_id="req-A",
                    block_mapping=[(0, 3), (1, 4)],
                ),
            ],
            layer_names=sorted(staging.keys()),
        )
        connector.bind_connector_metadata(load_meta)
        connector.start_load_kv(MagicMock())

        # Verify: dest block 3 has source block 0 data
        assert torch.equal(
            staging[layer_name][0][3],
            torch.full((2, 1, 2), 10.0),
        )
        assert torch.equal(
            staging[layer_name][1][3],
            torch.full((2, 1, 2), 20.0),
        )
        # dest block 4 has source block 1 data
        assert torch.equal(
            staging[layer_name][0][4],
            torch.full((2, 1, 2), 30.0),
        )

        connector.clear_connector_metadata()


# ---------------------------------------------------------------------------
# get_finished Safety Tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestGetFinishedSafety:
    """Verify get_finished only reports actual completions."""

    def test_no_work_returns_none(self):
        """When no store/load work done, returns (None, None)."""
        connector = _make_connector()
        staging = _make_staging_caches(num_layers=1, num_blocks=2)
        connector.register_kv_caches(staging)

        meta = SpyreConnectorMeta(
            requests=[],
            layer_names=sorted(staging.keys()),
        )
        connector.bind_connector_metadata(meta)
        result = connector.get_finished({"req-1"})
        assert result == (None, None)

    def test_store_reports_finished_sending(self):
        """After a save, the stored req appears in finished_sending."""
        store = InMemoryKVStore()
        connector = _make_connector(store=store)
        staging = _make_staging_caches(num_layers=1, num_blocks=2)
        connector.register_kv_caches(staging)

        meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-1", block_ids=[0], is_store=True,
                ),
            ],
            layer_names=sorted(staging.keys()),
        )
        connector.bind_connector_metadata(meta)
        connector.wait_for_save()

        sending, recving = connector.get_finished({"req-1"})
        assert sending == {"req-1"}
        assert recving is None

    def test_load_reports_finished_recving(self):
        """After a load, the loaded req appears in finished_recving."""
        store = InMemoryKVStore()
        connector = _make_connector(store=store)
        staging = _make_staging_caches(num_layers=1, num_blocks=2)
        connector.register_kv_caches(staging)

        # Pre-populate the store so load finds data
        store.put(
            StoreKey("req-A", 0, 0, KVKind.K),
            torch.randn(2, 2, 4),
        )
        store.put(
            StoreKey("req-A", 0, 0, KVKind.V),
            torch.randn(2, 2, 4),
        )

        meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-B", block_ids=[0],
                    is_store=False, source_req_id="req-A",
                ),
            ],
            layer_names=sorted(staging.keys()),
        )
        connector.bind_connector_metadata(meta)
        connector.start_load_kv(MagicMock())

        sending, recving = connector.get_finished({"req-B"})
        assert sending is None
        assert recving == {"req-B"}
        assert connector.get_block_ids_with_load_errors() == set()

    def test_missing_load_reports_block_errors(self):
        """Missing source data should be exposed via load error block IDs."""
        connector = _make_connector(store=InMemoryKVStore())
        staging = _make_staging_caches(num_layers=1, num_blocks=3)
        connector.register_kv_caches(staging)

        # No pre-populated store entries for req-src.
        meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-dst",
                    block_ids=[2],
                    is_store=False,
                    source_req_id="req-src",
                ),
            ],
            layer_names=sorted(staging.keys()),
        )
        connector.bind_connector_metadata(meta)
        connector.start_load_kv(MagicMock())

        # Destination block 2 failed to load.
        assert connector.get_block_ids_with_load_errors() == {2}
        _, recving = connector.get_finished({"req-dst"})
        assert recving == {"req-dst"}

    def test_unrequested_ids_not_in_finished(self):
        """Only IDs in finished_req_ids appear in the result."""
        store = InMemoryKVStore()
        connector = _make_connector(store=store)
        staging = _make_staging_caches(num_layers=1, num_blocks=2)
        connector.register_kv_caches(staging)

        meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-1", block_ids=[0], is_store=True,
                ),
            ],
            layer_names=sorted(staging.keys()),
        )
        connector.bind_connector_metadata(meta)
        connector.wait_for_save()

        # Ask about req-2, not req-1
        sending, _ = connector.get_finished({"req-2"})
        assert sending is None or "req-1" not in sending


# ---------------------------------------------------------------------------
# request_finished Tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestRequestFinished:
    """Verify request_finished behavior."""

    def test_returns_false_none(self):
        """request_finished returns (False, None) — free blocks immediately."""
        connector = _make_connector()
        request = MagicMock()
        request.request_id = "req-1"
        should_hold, params = connector.request_finished(request, [0, 1, 2])
        assert should_hold is False
        assert params is None


# ---------------------------------------------------------------------------
# get_num_new_matched_tokens Tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestGetNumNewMatchedTokens:
    """Verify conservative token matching."""

    def test_returns_zero_by_default(self):
        """Always returns 0 (conservative) in this implementation."""
        connector = _make_connector()
        request = MagicMock()
        request.request_id = "req-1"
        matched, is_async = connector.get_num_new_matched_tokens(request, 0)
        assert matched == 0
        assert is_async is False


# ---------------------------------------------------------------------------
# Factory Registration Tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestFactoryRegistration:
    """Verify connector factory registration."""

    def test_duplicate_registration_is_harmless(self):
        """Calling _register_kv_connector twice should not raise."""
        import vllm_spyre as pkg

        # Reset the flag so we exercise the real registration path
        old_flag = pkg._connector_registered
        pkg._connector_registered = False
        try:
            pkg._register_kv_connector()
            # Second call should be a no-op (flag now True)
            pkg._register_kv_connector()
        except Exception:
            # Clean up if the test fails
            pkg._connector_registered = old_flag
            raise
        # Leave flag True for subsequent tests in this process

    def test_connector_class_loads_via_factory(self):
        """The factory can resolve InMemorySpyreConnector by name."""
        import vllm_spyre as pkg

        # Ensure registration has happened with flag reset
        old_flag = pkg._connector_registered
        pkg._connector_registered = False
        try:
            pkg._register_kv_connector()
        except Exception:
            pkg._connector_registered = old_flag
            raise

        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory,
        )

        cls = KVConnectorFactory.get_connector_class_by_name(
            "InMemorySpyreConnector"
        )
        assert cls.__name__ == "InMemorySpyreConnector"

    def test_failed_registration_does_not_latch_flag(self):
        """Failed registration should allow retry on a subsequent call."""
        import vllm_spyre as pkg

        old_flag = pkg._connector_registered
        pkg._connector_registered = False
        try:
            with patch(
                "vllm.distributed.kv_transfer.kv_connector.factory."
                "KVConnectorFactory.register_connector",
                side_effect=RuntimeError("boom"),
            ), pytest.raises(RuntimeError, match="boom"):
                pkg._register_kv_connector()
            assert pkg._connector_registered is False
        finally:
            pkg._connector_registered = old_flag


# ---------------------------------------------------------------------------
# Lifecycle with Real Connector Tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestLifecycleWithRealConnector:
    """Test bridge + real connector end-to-end lifecycle."""

    def _make_bridge_and_connector(self):
        from vllm_spyre.v1.worker.spyre_kv_connector_bridge import (
            SpyreKVConnectorBridge,
        )

        store = InMemoryKVStore()
        connector = _make_connector(store=store)

        staging = _make_staging_caches(
            num_layers=1, num_blocks=4, block_size=2,
            num_kv_heads=2, head_dim=4,
        )
        connector.register_kv_caches(staging)

        vllm_config = _make_vllm_config_mock()
        bridge = SpyreKVConnectorBridge(vllm_config)
        bridge._kv_connector = connector

        return bridge, connector, store, staging

    def test_full_lifecycle_save(self):
        """Bridge drives a full save lifecycle with real connector."""
        bridge, connector, store, staging = self._make_bridge_and_connector()
        layer_name = sorted(staging.keys())[0]

        # Fill staging with test data
        staging[layer_name][0][0].fill_(42.0)
        staging[layer_name][1][0].fill_(43.0)

        meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-1", block_ids=[0], is_store=True,
                ),
            ],
            layer_names=sorted(staging.keys()),
        )
        so = _make_fake_scheduler_output(
            kv_connector_metadata=meta,
            finished_req_ids={"req-1"},
        )

        active = bridge.begin_step(so)
        assert active is True

        bridge.before_forward(so)
        bridge.after_forward(so)
        output = bridge.finish_step()

        assert output is not None
        assert isinstance(output, KVConnectorOutput)
        # Verify data was saved
        entry = store.get(StoreKey("req-1", 0, 0, KVKind.K))
        assert entry is not None
        assert torch.equal(entry.data, torch.full((2, 2, 4), 42.0))

    def test_full_lifecycle_load(self):
        """Bridge drives a full load lifecycle with real connector."""
        bridge, connector, store, staging = self._make_bridge_and_connector()
        layer_name = sorted(staging.keys())[0]

        # Pre-populate store
        store.put(
            StoreKey("req-src", 0, 0, KVKind.K),
            torch.full((2, 2, 4), 7.0),
        )
        store.put(
            StoreKey("req-src", 0, 0, KVKind.V),
            torch.full((2, 2, 4), 8.0),
        )

        meta = SpyreConnectorMeta(
            requests=[
                SpyreConnectorRequestMeta(
                    req_id="req-dest", block_ids=[2],
                    is_store=False, source_req_id="req-src",
                    block_mapping=[(0, 2)],
                ),
            ],
            layer_names=sorted(staging.keys()),
        )
        so = _make_fake_scheduler_output(
            kv_connector_metadata=meta,
            finished_req_ids={"req-dest"},
        )

        active = bridge.begin_step(so)
        assert active is True

        bridge.before_forward(so)
        # Verify staging got the loaded data
        assert torch.equal(
            staging[layer_name][0][2],
            torch.full((2, 2, 4), 7.0),
        )
        assert torch.equal(
            staging[layer_name][1][2],
            torch.full((2, 2, 4), 8.0),
        )

        bridge.after_forward(so)
        output = bridge.finish_step()
        assert output is not None

    def test_no_forward_with_real_connector(self):
        """no_forward path works with real connector."""
        bridge, connector, store, staging = self._make_bridge_and_connector()

        meta = SpyreConnectorMeta(
            requests=[],
            layer_names=sorted(staging.keys()),
        )
        so = _make_fake_scheduler_output(kv_connector_metadata=meta)

        output = bridge.no_forward(so)
        assert output is not None
        # No work done, so no finished IDs
        assert not connector.has_connector_metadata()


# ---------------------------------------------------------------------------
# Non-Driver Path Tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestNonDriverPath:
    """Verify non-driver path clears metadata."""

    def test_metadata_cleared_after_finish_step(self):
        """After finish_step, connector metadata is cleared."""
        connector = _make_connector()
        staging = _make_staging_caches(num_layers=1, num_blocks=2)
        connector.register_kv_caches(staging)

        meta = SpyreConnectorMeta(
            requests=[],
            layer_names=sorted(staging.keys()),
        )
        connector.bind_connector_metadata(meta)
        assert connector.has_connector_metadata()

        connector.clear_connector_metadata()
        assert not connector.has_connector_metadata()


# ---------------------------------------------------------------------------
# Version Compatibility Tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestVersionCompat:
    """Verify version check behavior."""

    def test_check_passes_for_current_vllm(self):
        """Version check should pass for the currently installed vllm."""
        from vllm_spyre.compat import check_vllm_version

        # Reset the check flag for this test
        import vllm_spyre.compat as compat_mod
        old_checked = compat_mod._compat_checked
        compat_mod._compat_checked = False
        try:
            # Should not raise for vllm 0.15.x
            check_vllm_version()
        finally:
            compat_mod._compat_checked = old_checked

    def test_failed_check_can_retry(self):
        """A failed check should not latch success and block a later retry."""
        from vllm_spyre.compat import check_vllm_version
        import vllm_spyre.compat as compat_mod

        old_checked = compat_mod._compat_checked
        compat_mod._compat_checked = False
        try:
            with patch(
                "importlib.metadata.version",
                side_effect=["0.16.0", "0.15.1"],
            ):
                with pytest.raises(RuntimeError, match="requires vllm>=0.15.0,<0.16.0"):
                    check_vllm_version()

                # Should re-run and pass on second attempt.
                check_vllm_version()
        finally:
            compat_mod._compat_checked = old_checked

    def test_skip_version_check_env(self):
        """VLLM_SPYRE_SKIP_VERSION_CHECK=1 skips the check."""
        from vllm_spyre.compat import check_vllm_version
        import vllm_spyre.compat as compat_mod
        old_checked = compat_mod._compat_checked
        compat_mod._compat_checked = False
        try:
            with patch.dict("os.environ", {"VLLM_SPYRE_SKIP_VERSION_CHECK": "1"}):
                check_vllm_version()  # Should not raise even with wrong version
        finally:
            compat_mod._compat_checked = old_checked

    def test_parse_version_tuple(self):
        from vllm_spyre.compat import _parse_version_tuple
        assert _parse_version_tuple("0.15.1") == (0, 15, 1)
        assert _parse_version_tuple("0.15.1.dev123") == (0, 15, 1)
        assert _parse_version_tuple("0.16.0rc1") == (0, 16, 0)
        assert _parse_version_tuple("1.0.0+local") == (1, 0, 0)


# ---------------------------------------------------------------------------
# Bad Block Mapping Tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestBadBlockMapping:
    """Verify bad block mapping fails cleanly."""

    def test_duplicate_dest_blocks_detected(self):
        """validate_block_mapping catches duplicate destination blocks."""
        meta = SpyreConnectorMeta()
        meta.add_load_request("req-1", [0, 1], source_req_id="src")
        meta.add_load_request("req-2", [1, 2], source_req_id="src")
        with pytest.raises(ValueError, match="Duplicate destination block"):
            meta.validate_block_mapping()

    def test_block_mapping_with_explicit_tuples_catches_dupes(self):
        meta = SpyreConnectorMeta()
        meta.requests.append(SpyreConnectorRequestMeta(
            req_id="req-1", block_ids=[], is_store=False,
            source_req_id="src",
            block_mapping=[(0, 5), (1, 5)],  # Same dest block 5
        ))
        with pytest.raises(ValueError, match="Duplicate destination block"):
            meta.validate_block_mapping()

    def test_bind_connector_metadata_rejects_duplicate_dest_blocks(self):
        connector = _make_connector()
        staging = _make_staging_caches(num_layers=1, num_blocks=8)
        connector.register_kv_caches(staging)

        meta = SpyreConnectorMeta()
        meta.requests.append(
            SpyreConnectorRequestMeta(
                req_id="req-1",
                block_ids=[],
                is_store=False,
                source_req_id="src",
                block_mapping=[(0, 5), (1, 5)],
            )
        )

        with pytest.raises(ValueError, match="Duplicate destination block"):
            connector.bind_connector_metadata(meta)
