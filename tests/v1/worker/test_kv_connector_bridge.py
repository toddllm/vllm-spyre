"""Tests for SpyreKVConnectorBridge lifecycle correctness.

Validates:
1. Lifecycle order: begin_step -> before_forward -> after_forward -> finish_step
2. No-forward path handling
3. Metadata always cleared (even on error paths)
4. get_finished safety (no premature block-free signals)
5. Bridge inactive when feature flag is disabled or metadata is None
"""

import copy
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.outputs import KVConnectorOutput


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------

@dataclass
class FakeSchedulerOutput:
    """Minimal mock of SchedulerOutput with the fields the bridge reads."""
    total_num_scheduled_tokens: int = 0
    finished_req_ids: set = field(default_factory=set)
    kv_connector_metadata: Any = None
    preempted_req_ids: set = field(default_factory=set)


class LifecycleTracker:
    """Records the order of connector method calls."""

    def __init__(self):
        self.calls: list[str] = []
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()
        self._metadata_bound = False

    def handle_preemptions(self, preempted_req_ids):
        self.calls.append("handle_preemptions")

    def bind_connector_metadata(self, metadata):
        self.calls.append("bind_connector_metadata")
        self._metadata_bound = True

    def start_load_kv(self, forward_context):
        self.calls.append("start_load_kv")

    def wait_for_save(self):
        self.calls.append("wait_for_save")

    def get_finished(self, finished_req_ids):
        self.calls.append("get_finished")
        return (
            copy.copy(self._finished_sending),
            copy.copy(self._finished_recving),
        )

    def get_block_ids_with_load_errors(self):
        self.calls.append("get_block_ids_with_load_errors")
        return set()

    def get_kv_connector_stats(self):
        return None

    def get_kv_connector_kv_cache_events(self):
        return None

    def clear_connector_metadata(self):
        self.calls.append("clear_connector_metadata")
        self._metadata_bound = False

    def register_kv_caches(self, kv_caches_dict):
        self.calls.append("register_kv_caches")

    def shutdown(self):
        self.calls.append("shutdown")


def _make_vllm_config_mock():
    """Create a VllmConfig mock with sane defaults for set_forward_context."""
    vllm_config = MagicMock()
    # set_forward_context checks these fields
    vllm_config.parallel_config.data_parallel_size = 1
    vllm_config.compilation_config.fast_moe_cold_start = False
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.speculative_config = None
    return vllm_config


def make_bridge_with_tracker():
    """Create a bridge with a LifecycleTracker as the connector."""
    from vllm_spyre.v1.worker.spyre_kv_connector_bridge import (
        SpyreKVConnectorBridge,
    )

    tracker = LifecycleTracker()
    vllm_config = _make_vllm_config_mock()

    bridge = SpyreKVConnectorBridge(vllm_config)
    # Inject the tracker as the connector
    bridge._kv_connector = tracker

    return bridge, tracker


# ---------------------------------------------------------------------------
# Lifecycle order tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestLifecycleOrder:
    """Verify connector methods are called in the correct order."""

    def test_normal_forward_lifecycle(self):
        """Full lifecycle: begin -> before_forward -> after_forward -> finish."""
        bridge, tracker = make_bridge_with_tracker()
        so = FakeSchedulerOutput(kv_connector_metadata={"test": True})

        active = bridge.begin_step(so)
        assert active is True

        bridge.before_forward(so)
        # [simulate forward pass here]
        bridge.after_forward(so)
        output = bridge.finish_step()

        assert output is not None
        assert isinstance(output, KVConnectorOutput)

        expected_order = [
            "bind_connector_metadata",
            "start_load_kv",
            "wait_for_save",
            "get_finished",
            "get_block_ids_with_load_errors",
            "clear_connector_metadata",
        ]
        assert tracker.calls == expected_order

    def test_no_forward_lifecycle(self):
        """No-forward path: begin -> before_forward -> after_forward(no wait) -> finish."""
        bridge, tracker = make_bridge_with_tracker()
        so = FakeSchedulerOutput(kv_connector_metadata={"test": True})

        output = bridge.no_forward(so)
        assert output is not None

        # no_forward should NOT call wait_for_save
        assert "wait_for_save" not in tracker.calls
        assert "bind_connector_metadata" in tracker.calls
        assert "start_load_kv" in tracker.calls
        assert "get_finished" in tracker.calls
        assert "clear_connector_metadata" in tracker.calls

    def test_preemption_handling(self):
        """Preempted requests trigger handle_preemptions before other calls."""
        bridge, tracker = make_bridge_with_tracker()
        so = FakeSchedulerOutput(
            kv_connector_metadata={"test": True},
            preempted_req_ids={"req-1", "req-2"},
        )

        bridge.begin_step(so)
        assert tracker.calls[0] == "handle_preemptions"


# ---------------------------------------------------------------------------
# Metadata clearing tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestMetadataClearing:
    """Verify metadata is always cleared, preventing leaks across steps."""

    def test_metadata_cleared_on_normal_exit(self):
        """Metadata is cleared at the end of a normal step."""
        bridge, tracker = make_bridge_with_tracker()
        so = FakeSchedulerOutput(kv_connector_metadata={"test": True})

        bridge.begin_step(so)
        bridge.before_forward(so)
        bridge.after_forward(so)
        bridge.finish_step()

        assert "clear_connector_metadata" in tracker.calls
        assert tracker._metadata_bound is False

    def test_metadata_cleared_on_inactive_step(self):
        """finish_step is safe to call when bridge was inactive."""
        bridge, tracker = make_bridge_with_tracker()
        so = FakeSchedulerOutput(kv_connector_metadata=None)

        active = bridge.begin_step(so)
        assert active is False

        output = bridge.finish_step()
        assert output is None
        # No connector calls should have been made
        assert len(tracker.calls) == 0

    def test_consecutive_steps_dont_leak(self):
        """Back-to-back steps don't leak metadata from one to the next."""
        bridge, tracker = make_bridge_with_tracker()

        # Step 1: active
        so1 = FakeSchedulerOutput(kv_connector_metadata={"step": 1})
        bridge.begin_step(so1)
        bridge.before_forward(so1)
        bridge.after_forward(so1)
        output1 = bridge.finish_step()
        assert output1 is not None
        assert tracker._metadata_bound is False

        # Step 2: inactive
        tracker.calls.clear()
        so2 = FakeSchedulerOutput(kv_connector_metadata=None)
        active = bridge.begin_step(so2)
        assert active is False
        output2 = bridge.finish_step()
        assert output2 is None


# ---------------------------------------------------------------------------
# get_finished safety tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestGetFinishedSafety:
    """Verify get_finished never reports premature completion."""

    def test_empty_finished_sets_by_default(self):
        """When no transfers completed, finished sets are empty."""
        bridge, tracker = make_bridge_with_tracker()
        so = FakeSchedulerOutput(
            kv_connector_metadata={"test": True},
            finished_req_ids={"req-1"},
        )

        bridge.begin_step(so)
        bridge.before_forward(so)
        bridge.after_forward(so)
        output = bridge.finish_step()

        assert output is not None
        assert output.finished_sending == set()
        assert output.finished_recving == set()

    def test_finished_only_reports_actual_completions(self):
        """Only actually completed request IDs appear in finished sets."""
        bridge, tracker = make_bridge_with_tracker()
        # Simulate one completed send
        tracker._finished_sending = {"req-1"}

        so = FakeSchedulerOutput(
            kv_connector_metadata={"test": True},
            finished_req_ids={"req-1", "req-2"},
        )

        bridge.begin_step(so)
        bridge.before_forward(so)
        bridge.after_forward(so)
        output = bridge.finish_step()

        assert output is not None
        assert output.finished_sending == {"req-1"}
        assert output.finished_recving == set()
        # req-2 should NOT be in finished_sending
        assert "req-2" not in output.finished_sending

    def test_get_finished_not_called_before_bind(self):
        """get_finished is called after bind and load, never before."""
        bridge, tracker = make_bridge_with_tracker()
        so = FakeSchedulerOutput(kv_connector_metadata={"test": True})

        bridge.begin_step(so)
        bridge.before_forward(so)
        bridge.after_forward(so)
        bridge.finish_step()

        bind_idx = tracker.calls.index("bind_connector_metadata")
        get_finished_idx = tracker.calls.index("get_finished")
        assert get_finished_idx > bind_idx


# ---------------------------------------------------------------------------
# Bridge activation tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestBridgeActivation:
    """Verify bridge correctly activates/deactivates based on config."""

    def test_inactive_when_no_metadata(self):
        """Bridge stays inactive when kv_connector_metadata is None."""
        bridge, tracker = make_bridge_with_tracker()
        so = FakeSchedulerOutput(kv_connector_metadata=None)

        active = bridge.begin_step(so)
        assert active is False
        assert len(tracker.calls) == 0

    def test_inactive_when_no_connector(self):
        """Bridge stays inactive when no connector is available."""
        from vllm_spyre.v1.worker.spyre_kv_connector_bridge import (
            SpyreKVConnectorBridge,
        )

        vllm_config = MagicMock()
        bridge = SpyreKVConnectorBridge(vllm_config)
        # Don't inject a connector — it should be None

        so = FakeSchedulerOutput(kv_connector_metadata={"test": True})
        active = bridge.begin_step(so)
        assert active is False

    def test_feature_flag_disabled(self):
        """maybe_create_bridge returns None when flag is disabled."""
        from vllm_spyre.v1.worker.spyre_kv_connector_bridge import (
            maybe_create_bridge,
        )

        vllm_config = MagicMock()
        vllm_config.kv_transfer_config = MagicMock()
        vllm_config.kv_transfer_config.is_kv_transfer_instance = True

        with patch("vllm_spyre.envs._cache", {}), \
             patch.dict("os.environ", {"VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE": "0"}):
            result = maybe_create_bridge(vllm_config)

        assert result is None

    def test_feature_flag_enabled_no_transfer_config(self):
        """Bridge not created when kv_transfer_config is None."""
        from vllm_spyre.v1.worker.spyre_kv_connector_bridge import (
            maybe_create_bridge,
        )

        vllm_config = MagicMock()
        vllm_config.kv_transfer_config = None

        with patch("vllm_spyre.envs._cache", {}), \
             patch.dict("os.environ", {"VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE": "1"}):
            result = maybe_create_bridge(vllm_config)

        assert result is None


# ---------------------------------------------------------------------------
# Output propagation tests
# ---------------------------------------------------------------------------

@pytest.mark.cpu
class TestOutputPropagation:
    """Verify KVConnectorOutput is correctly assembled."""

    def test_output_includes_invalid_block_ids(self):
        """invalid_block_ids from connector are propagated to output."""
        bridge, tracker = make_bridge_with_tracker()
        # Override to return some invalid blocks
        tracker.get_block_ids_with_load_errors = lambda: {42, 99}

        so = FakeSchedulerOutput(kv_connector_metadata={"test": True})

        bridge.begin_step(so)
        bridge.before_forward(so)
        bridge.after_forward(so)
        output = bridge.finish_step()

        assert output is not None
        assert output.invalid_block_ids == {42, 99}

    def test_no_forward_output_propagation(self):
        """no_forward path correctly returns connector output."""
        bridge, tracker = make_bridge_with_tracker()
        tracker._finished_recving = {"req-1"}

        so = FakeSchedulerOutput(kv_connector_metadata={"test": True})
        output = bridge.no_forward(so)

        assert output is not None
        assert output.finished_recving == {"req-1"}
