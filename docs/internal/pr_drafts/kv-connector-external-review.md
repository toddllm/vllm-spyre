# Spyre KV Connector: Implementation Review Packet

## What this document is

A single-file narrative for external reviewers who need to understand what was built, why it was built this way, where to look in the code, and what the remaining risks are. This covers the full branch stack from lifecycle wiring through metrics integration.

**Repository**: [toddllm/vllm-spyre](https://github.com/toddllm/vllm-spyre)
**Branch stack base**: `codex/spyre-kv-base-pr759` (frozen mirror of `origin/pr-759`)
**Combined test branch**: `codex/spyre-kv-combined` ([compare to base](https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-base-pr759...codex/spyre-kv-combined))
**Test results**: 172 passed across `tests/v1/worker/` (CPU, CUDA, and Spyre-card environments)

---

## 1. The problem

vllm-spyre uses FMS (Foundation Model Stack) for model execution. FMS manages its own KV cache internally via `past_key_value_states`, which means:

- vLLM's standard `save_kv_layer()` hook is never called (FMS attention is opaque)
- The upstream `ActiveKVConnector` assumes GPU-side DMA-driven load/save, which doesn't apply
- Scheduler-owned block IDs (introduced by upstream `pr-759`) need a Spyre-side consumer

Without a connector implementation, there is no mechanism for prefix-aware KV reuse, disaggregated prefill/decode, or observability of KV transfer operations on the Spyre path.

## 2. What was built

**1,833 lines of implementation** across 4 production files. **4,815 lines of tests** across 6 test files. **11 commits** organized into 4 reviewable slices.

### Architecture at a glance

```
Scheduler side                    Worker side
─────────────                    ───────────
get_num_new_matched_tokens() ─┐
update_state_after_alloc()    ├─ SpyreConnectorMeta ──→ bind_connector_metadata()
build_connector_meta()       ─┘                         start_load_kv()
request_finished()                                      wait_for_save()
                                                        get_finished()
                              ┌─────────────────────┐
                              │ SpyreKVConnectorBridge│
                              │  begin_step()        │
                              │  before_forward()    │
                              │  [FMS forward]       │
                              │  after_forward()     │
                              │  finish_step()       │
                              └─────────────────────┘
                                        │
                              ┌─────────▼───────────┐
                              │ InMemoryKVStore      │
                              │  put() / get()       │
                              │  byte-capped LRU     │
                              │  export/import (.pt) │
                              └─────────────────────┘
```

### Files

| File | Lines | Purpose |
|------|-------|---------|
| [`spyre_kv_connector_bridge.py`](https://github.com/toddllm/vllm-spyre/blob/codex/spyre-kv-combined/vllm_spyre/v1/worker/spyre_kv_connector_bridge.py) | 259 | Lifecycle orchestrator — sequences bind/load/save/finish around FMS forward |
| [`inmemory_spyre_connector.py`](https://github.com/toddllm/vllm-spyre/blob/codex/spyre-kv-combined/vllm_spyre/distributed/kv_transfer/kv_connector/v1/inmemory_spyre_connector.py) | 959 | Full `KVConnectorBase_V1` implementation — scheduler + worker sides |
| [`metadata.py`](https://github.com/toddllm/vllm-spyre/blob/codex/spyre-kv-combined/vllm_spyre/distributed/kv_transfer/kv_connector/v1/metadata.py) | 521 | Metadata schema, in-memory KV store, connector stats |
| [`compat.py`](https://github.com/toddllm/vllm-spyre/blob/codex/spyre-kv-combined/vllm_spyre/compat.py) | 94 | Version band guard for vLLM compatibility |
| [`envs.py`](https://github.com/toddllm/vllm-spyre/blob/codex/spyre-kv-combined/vllm_spyre/envs.py) (delta) | +31 | 3 new env vars for connector configuration |

---

## 3. The four slices

The implementation is split into 4 stacked PRs. Each slice has a specific proof obligation and can be reviewed independently.

### Slice 1: Bridge Lifecycle Wiring

**Branch**: [`codex/spyre-kv-slice1-bridge`](https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-base-pr759...codex/spyre-kv-slice1-bridge)
**Commits**: 3 | **Diffstat**: +947 / -63

**What it proves**: Spyre worker/model-runner can participate in the upstream KV connector lifecycle without forking scheduler behavior.

**Key code**:

The bridge centralizes the fragile sequencing that would otherwise be scattered across model runner branches:

```python
# spyre_kv_connector_bridge.py:104-133
def begin_step(self, scheduler_output):
    self._active = False
    self._output = None
    if not self.is_available:
        return False
    if scheduler_output.kv_connector_metadata is None:
        return False
    preempted = getattr(scheduler_output, "preempted_req_ids", None)
    if preempted:
        self._kv_connector.handle_preemptions(preempted)
    self._active = True
    return True
```

Feature-flagged via `VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE` (default off). When disabled, all methods are no-ops — existing behavior is unchanged.

**Reviewer focus**: Call ordering in all execute paths. Cleanup on all return paths. Cache registration timing after warmup.

**Risk**: If finalization is missed on any branch, stale metadata leaks to the next step.

---

### Slice 2: Connector Core + Metadata Contract

**Branch**: [`codex/spyre-kv-slice2-core`](https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-slice1-bridge...codex/spyre-kv-slice2-core)
**Commits**: 3 | **Diffstat**: +2,633 / -1

**What it proves**: Scheduler-owned block IDs from `pr-759` can drive a Spyre-side connector without inventing a parallel block manager.

This is the architectural hinge. Five invariants are established here:

1. **Metadata is typed**: `SpyreConnectorMeta` inherits `KVConnectorMetadata` (upstream contract).

    ```python
    # metadata.py:127
    class SpyreConnectorMeta(KVConnectorMetadata):
    ```

2. **Block mappings use real scheduler IDs, not indices**:

    ```python
    # inmemory_spyre_connector.py:458-461
    # Identity mapping fallback: source and destination block IDs
    # are the same when no explicit remap is provided.
    mapping = [(block_id, block_id) for block_id in req_meta.block_ids]
    ```

    This was a real bug caught during review. The original code used `enumerate()` which silently produced index-based lookups (e.g., source block 10 was looked up as 0).

3. **Incomplete mappings fall back to safe recompute**:

    ```python
    # inmemory_spyre_connector.py:778-791
    if len(block_mapping) != num_external_blocks:
        logger.warning(...)
        req_meta = SpyreConnectorRequestMeta(
            req_id=request.request_id, block_ids=flat_block_ids,
            is_store=True,  # Safe recompute path
            token_count=len(request.all_token_ids))
    ```

4. **Matching is conservative**: `get_num_new_matched_tokens()` returns 0 unless a full block-aligned prefix match is verified against the saved-request registry.

5. **Connector only touches staging tensors**: It never owns FMS live tensors. The model runner owns the staging-to-FMS sync boundary.

**Reviewer focus**: `SpyreConnectorMeta` field set. Scheduler-side request classification (`is_store` vs load). Identity mapping semantics. Incomplete mapping fallback.

**Risk**: If matching is too aggressive, requests try to load KV they should recompute. The current implementation is intentionally conservative.

---

### Slice 3: Runtime Hardening

**Branch**: [`codex/spyre-kv-slice3-runtime`](https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-slice2-core...codex/spyre-kv-slice3-runtime)
**Commits**: 4 | **Diffstat**: +3,059 / -61

**What it proves**: The Slice 2 contract survives realistic runtime conditions, not just isolated unit tests.

Key additions:

- **Async-ready layer pipeline**: `ThreadPoolExecutor`-based per-layer load with configurable workers via `VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS`. Defaults to 0 (synchronous). Async and sync paths produce identical results (tested).

    ```python
    # inmemory_spyre_connector.py:295-305
    self._async_load_workers = max(0, envs_spyre.VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS)
    self._async_load_enabled = self._async_load_workers > 0
    self._executor = ThreadPoolExecutor(max_workers=self._async_load_workers) \
        if self._async_load_enabled else None
    ```

- **Byte-capped LRU store**: `InMemoryKVStore(max_bytes=N)` evicts oldest entries when byte budget is exceeded. Configured via `VLLM_SPYRE_KV_STORE_MAX_BYTES`.

    ```python
    # metadata.py:369-374
    while (self._store and
           self._current_bytes + entry_size > self._max_bytes):
        self._evict_oldest()
    ```

- **Strict metadata validation**: Schema version, layout whitelist (`NHD`), non-negative block IDs and mapping IDs, load-needs-source consistency.

    ```python
    # metadata.py:219-293
    def validate(self) -> None:
        if self.schema_version not in self._SUPPORTED_VERSIONS: ...
        if self.layout and self.layout not in self._KNOWN_LAYOUTS: ...
        for req in self.requests:
            if not req.req_id: ...
            for bid in req.block_ids:
                if bid < 0: ...
    ```

- **2-process disaggregated E2E test**: Real `multiprocessing.Process` with file-backed store transport (`export_to_dir` / `import_from_dir`). Prefill process saves KV, decode process loads it, data integrity is verified.

- **Scheduler feedback loop**: `get_block_ids_with_load_errors()` returns destination block IDs that failed to load. Bridge propagates these via `KVConnectorOutput.invalid_block_ids`. Tested through full scheduler → worker → scheduler cycle.

**Reviewer focus**: Async-ready path structure. `get_finished()` correctness under partial work. Whether the 2-process harness actually proves what it claims.

**Risk**: "Async-ready" could be misread as "async complete." The slice is explicit that sync semantics still dominate — async is opt-in and tested for equivalence.

---

### Slice 4: Prometheus Metrics Adapter

**Branch**: [`codex/spyre-kv-slice4-metrics`](https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-slice3-runtime...codex/spyre-kv-slice4-metrics)
**Commits**: 1 | **Diffstat**: +408 / -3

**What it proves**: Connector behavior can be surfaced through the upstream metrics seam without destabilizing connector semantics.

6 Prometheus counters exposed via `SpyreConnectorPromMetrics`:

| Counter | Description |
|---------|-------------|
| `vllm:spyre_kv_matched_tokens_total` | Tokens matched for KV reuse |
| `vllm:spyre_kv_loaded_blocks_total` | KV blocks loaded from store |
| `vllm:spyre_kv_saved_blocks_total` | KV blocks saved to store |
| `vllm:spyre_kv_load_misses_total` | Block load misses (missing data) |
| `vllm:spyre_kv_evictions_total` | Saved-request registry LRU evictions |
| `vllm:spyre_kv_match_attempts_total` | Prefix match attempts |

```python
# inmemory_spyre_connector.py:117-213
class SpyreConnectorPromMetrics(KVConnectorPromMetrics):
    def observe(self, transfer_stats_data, engine_idx=0):
        for prom_counter, key in [
            (self.counter_matched_tokens, "matched_tokens"),
            (self.counter_loaded_blocks, "loaded_blocks"),
            ...
        ]:
            value = transfer_stats_data.get(key, 0)
            if value > 0:
                prom_counter[engine_idx].inc(value)
```

Follows the same pattern as upstream `NixlPromMetrics`. Wired via `build_prom_metrics()` classmethod.

**Reviewer focus**: Counter names and labels. Zero/empty stats handling. Whether this should stay on the connector class or move elsewhere.

**Risk**: Naming churn if upstream conventions shift. Double-counting if stats reset semantics are misunderstood.

---

## 4. Test coverage

172 tests across 6 files, all passing on CPU, CUDA, and Spyre-card environments.

| Test file | Tests | What it validates |
|-----------|-------|-------------------|
| [`test_kv_connector_bridge.py`](https://github.com/toddllm/vllm-spyre/blob/codex/spyre-kv-combined/tests/v1/worker/test_kv_connector_bridge.py) | 18 | Bridge lifecycle ordering, cleanup invariants, no-forward paths |
| [`test_inmemory_spyre_connector.py`](https://github.com/toddllm/vllm-spyre/blob/codex/spyre-kv-combined/tests/v1/worker/test_inmemory_spyre_connector.py) | 37 | Storage semantics, metadata behavior, identity mapping fix |
| [`test_scheduler_driven_kv_reuse.py`](https://github.com/toddllm/vllm-spyre/blob/codex/spyre-kv-combined/tests/v1/worker/test_scheduler_driven_kv_reuse.py) | 30 | Reuse matching, block mapping, incomplete mapping fallback |
| [`test_kv_integration.py`](https://github.com/toddllm/vllm-spyre/blob/codex/spyre-kv-combined/tests/v1/worker/test_kv_integration.py) | 22 | Bridge + connector integration, failure wiring, LRU registry |
| [`test_kv_engine_level.py`](https://github.com/toddllm/vllm-spyre/blob/codex/spyre-kv-combined/tests/v1/worker/test_kv_engine_level.py) | 13 | Engine-like sequencing, stats accumulation, no-premature-finish |
| [`test_kv_phase6.py`](https://github.com/toddllm/vllm-spyre/blob/codex/spyre-kv-combined/tests/v1/worker/test_kv_phase6.py) | 52 | 2-process E2E, async pipeline, schema validation, byte-capped store, Prometheus adapter |

### Bugs caught by tests

1. **Identity load used wrong block IDs** (Slice 2, caught in review): `enumerate()` indices instead of scheduler block IDs. Silent data misses for non-zero blocks.

2. **Incomplete mapping silently under-loaded** (Slice 2, caught in review): When saved registry had fewer blocks than implied by matched tokens, mapping was truncated. No error signal.

3. **Strict `block_size > 0` broke existing tests** (Slice 3, caught during development): Metadata validation rejected `block_size=0`, which is the default for informational metadata. Changed to reject only negative values.

---

## 5. Configuration

Three new environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE` | `0` | Feature flag for bridge lifecycle integration |
| `VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS` | `0` | Thread pool size for per-layer async loads (0 = synchronous) |
| `VLLM_SPYRE_KV_STORE_MAX_BYTES` | `0` | Byte cap for in-memory store LRU eviction (0 = unlimited) |

Pre-existing variables also used:

| Variable | Default | Purpose |
|----------|---------|---------|
| `VLLM_SPYRE_KV_REUSE_REGISTRY_MAX_SIZE` | `1024` | Max saved requests retained for prefix matching |

---

## 6. What is stable vs. what will change

**More stable** (likely to survive compiler migration):
- The need for a bridge boundary between runner and connector
- The metadata contract shape (`SpyreConnectorMeta`, `SpyreConnectorRequestMeta`)
- Scheduler-owned block IDs as the control-plane anchor
- Conservative prefix matching semantics

**Less stable** (implementation details):
- In-memory store internals (dict-backed, file-based export)
- Async thread pool structure
- Exact staging tensor shape assumptions
- Prometheus counter names if upstream conventions shift

---

## 7. What this does NOT solve

- **Production transport**: The in-memory store + file-backed export is for correctness testing, not production disaggregation. Real transport (NIXL, RDMA, shared memory) is a separate workstream.
- **True async DMA overlap**: The `ThreadPoolExecutor` path is async-ready scaffolding, not GPU-side DMA overlap.
- **Spyre compiler integration**: This work targets the current FMS path. When `torch-spyre` / new compiler work lands, some implementation details will change.
- **`/metrics` endpoint integration testing**: Adapter is unit-tested but not validated at the scrape endpoint level (requires running engine instance).

---

## 8. How to validate

### Quick regression

```bash
pytest -q tests/v1/worker
```

### Targeted slice validation

```bash
# Slice 1 — bridge lifecycle
pytest -q tests/v1/worker/test_kv_connector_bridge.py

# Slice 2 — connector core + reuse
pytest -q tests/v1/worker/test_inmemory_spyre_connector.py \
         tests/v1/worker/test_scheduler_driven_kv_reuse.py

# Slice 3 — runtime hardening
pytest -q tests/v1/worker/test_kv_integration.py \
         tests/v1/worker/test_kv_engine_level.py \
         tests/v1/worker/test_kv_phase6.py

# Slice 4 — metrics
pytest -q tests/v1/worker/test_kv_phase6.py -k "prom or metric"
```

### Lint

```bash
ruff check vllm_spyre/distributed/kv_transfer/kv_connector/v1 tests/v1/worker
```

---

## 9. Decision guide for reviewers

| If you care most about... | Start with |
|---------------------------|-----------|
| Lifecycle correctness | [Slice 1 diff](https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-base-pr759...codex/spyre-kv-slice1-bridge) |
| Metadata contract and reuse semantics | [Slice 2 diff](https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-slice1-bridge...codex/spyre-kv-slice2-core) |
| Runtime behavior and failure handling | [Slice 3 diff](https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-slice2-core...codex/spyre-kv-slice3-runtime) |
| Operational observability | [Slice 4 diff](https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-slice3-runtime...codex/spyre-kv-slice4-metrics) |
| Full combined diff | [Combined](https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-base-pr759...codex/spyre-kv-combined) |

### Recommended first PR

Open **Slice 1** first. It is the smallest reviewable unit (3 commits, 947 insertions) and changes production control flow without touching metadata or storage semantics. Once lifecycle wiring is accepted, Slice 2 becomes the natural next review because it defines the contract everything else depends on.

If reviewers prefer fewer PRs, collapse Slices 3 and 4 together — they represent operational refinement rather than architectural boundary changes.

---

## 10. Open questions for reviewers

1. **Connector registration path**: Currently registered in `vllm_spyre/__init__.py`. Should it use a different plugin mechanism?

2. **Metadata field naming**: Fields use `NHD` layout naming (upstream convention). Will this hold for `vllm-spyre-next`?

3. **Version band**: `compat.py` targets vLLM 0.15.x (`>=0.15.0,<0.16.0`). Is this the right band for the current integration target?

4. **First Spyre-card scope**: For the first hardware POC, should connector-aware execution be tested, or should the first call be plain generation with connector disabled?
