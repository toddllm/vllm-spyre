"""
Spyre KV Connector Bridge: Lifecycle orchestrator for KV connector
integration in the Spyre model execution path.

This module wraps the upstream KV connector lifecycle (bind, load, save,
finish) around the FMS model forward call. It does NOT modify scheduler
logic — it only drives the worker-side connector contract.

Design mirrors upstream ActiveKVConnector (vllm/v1/worker/gpu/kv_connector.py)
adapted for Spyre where:
  - FMS manages its own KV cache via past_key_value_states
  - save_kv_layer() is never called inline (FMS attention is opaque)
  - All operations are synchronous (no async DMA during forward)

The bridge is feature-flagged via VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE.
When disabled (default), all methods are no-ops.
"""

from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.v1.outputs import KVConnectorOutput

import vllm_spyre.envs as envs_spyre

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
    )
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class SpyreKVConnectorBridge:
    """
    Synchronous bridge that drives the KV connector lifecycle around
    the FMS model forward in Spyre's execution path.

    Lifecycle per step:
        1. begin_step()     -- validate connector, handle preemptions
        2. before_forward() -- bind metadata, start_load_kv
        3. [caller runs model forward]
        4. after_forward()  -- wait_for_save, collect finished/stats
        5. finish_step()    -- clear metadata, return KVConnectorOutput

    Thread safety:
        NOT thread-safe. Called from a single worker thread.
    """

    def __init__(self, vllm_config: "VllmConfig"):
        self._vllm_config = vllm_config
        self._kv_connector: "KVConnectorBase_V1 | None" = None
        self._active = False
        self._output: KVConnectorOutput | None = None

        # Best-effort initial acquisition at bridge construction time.
        # The connector may not be initialized yet depending on worker
        # lifecycle ordering; later begin_step() calls retry acquisition
        # via is_available while _kv_connector is still None.
        self._try_acquire_connector()

    def _try_acquire_connector(self) -> bool:
        """Best-effort acquisition of the global KV connector.

        Called once from __init__ and retried lazily by is_available on
        later steps until a valid connector becomes available.
        """
        if self._kv_connector is not None:
            return True

        if not has_kv_transfer_group():
            return False

        connector = get_kv_transfer_group()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )

        if not isinstance(connector, KVConnectorBase_V1):
            logger.warning(
                "KV connector is not a V1 connector (got %s). "
                "Bridge will be disabled.",
                type(connector).__name__,
            )
            return False

        self._kv_connector = connector
        logger.info(
            "SpyreKVConnectorBridge acquired connector: %s",
            type(connector).__name__,
        )
        return True

    @property
    def is_available(self) -> bool:
        """True if the bridge has or can now acquire a valid connector."""
        return self._kv_connector is not None or self._try_acquire_connector()

    def begin_step(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> bool:
        """
        Phase 1: Validate connector and metadata presence.

        Call at the start of execute_model, before prepare_model_input.

        Returns True if the bridge will be active this step.
        """
        self._active = False
        self._output = None

        # Retry lazy connector acquisition on each step until the global
        # KV connector becomes available.
        if not self.is_available:
            return False

        assert self._kv_connector is not None

        # If scheduler didn't emit connector metadata, stay inactive.
        if scheduler_output.kv_connector_metadata is None:
            return False

        # Handle preemptions if scheduler reports any.
        preempted = getattr(scheduler_output, "preempted_req_ids", None)
        if preempted:
            self._kv_connector.handle_preemptions(preempted)

        self._active = True
        return True

    def before_forward(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        """
        Phase 2: Bind connector metadata and start KV loading.

        Must be called from within a set_forward_context() block when
        possible. Falls back to creating a temporary context if needed
        (matching upstream ActiveKVConnector.pre_forward pattern).
        """
        if not self._active:
            return

        assert self._kv_connector is not None
        assert scheduler_output.kv_connector_metadata is not None

        # Bind metadata from scheduler
        self._kv_connector.bind_connector_metadata(
            scheduler_output.kv_connector_metadata
        )

        # Start KV loading — needs ForwardContext
        if is_forward_context_available():
            self._kv_connector.start_load_kv(get_forward_context())
        else:
            with set_forward_context(None, self._vllm_config):
                self._kv_connector.start_load_kv(get_forward_context())

    def after_forward(
        self,
        scheduler_output: "SchedulerOutput",
        wait_for_save: bool = True,
    ) -> None:
        """
        Phase 3: Collect connector results after forward pass.

        In Spyre, save_kv_layer() is never called inline during FMS forward
        (FMS attention is opaque). wait_for_save() is called for protocol
        correctness.
        """
        if not self._active:
            return

        assert self._kv_connector is not None

        output = KVConnectorOutput()

        if wait_for_save:
            self._kv_connector.wait_for_save()

        output.finished_sending, output.finished_recving = (
            self._kv_connector.get_finished(scheduler_output.finished_req_ids)
        )
        output.invalid_block_ids = (
            self._kv_connector.get_block_ids_with_load_errors()
        )
        output.kv_connector_stats = (
            self._kv_connector.get_kv_connector_stats()
        )
        output.kv_cache_events = (
            self._kv_connector.get_kv_connector_kv_cache_events()
        )

        self._output = output

    def finish_step(self) -> KVConnectorOutput | None:
        """
        Phase 4: Clear connector metadata and return output.

        Always call this — returns None if bridge was inactive.
        """
        if not self._active:
            return None

        assert self._kv_connector is not None

        # Always clear metadata
        self._kv_connector.clear_connector_metadata()

        output = self._output
        self._output = None
        self._active = False
        return output

    def no_forward(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorOutput | None:
        """
        Handle no-work steps where connector still needs to process
        transfers. Mirrors upstream ActiveKVConnector.no_forward().
        """
        if not self.begin_step(scheduler_output):
            return None

        self.before_forward(scheduler_output)
        self.after_forward(scheduler_output, wait_for_save=False)
        return self.finish_step()


def maybe_create_bridge(
    vllm_config: "VllmConfig",
) -> SpyreKVConnectorBridge | None:
    """
    Create the worker-side SpyreKVConnectorBridge if the feature flag is
    enabled and this process is configured for KV transfer.

    This only enables worker lifecycle wiring. The bridge still remains
    inactive on any given step unless the scheduler emits
    kv_connector_metadata for that step.

    Returns None if disabled or not configured.
    """
    if not envs_spyre.VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE:
        return None

    if (
        vllm_config.kv_transfer_config is None
        or not vllm_config.kv_transfer_config.is_kv_transfer_instance
    ):
        logger.debug(
            "KV transfer not configured. Bridge not created."
        )
        return None

    bridge = SpyreKVConnectorBridge(vllm_config)
    logger.info("SpyreKVConnectorBridge created.")
    return bridge
