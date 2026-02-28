# Slice 2: Connector Core + Metadata Contract

Branch:
- `codex/spyre-kv-slice2-core`

Suggested PR base:
- `codex/spyre-kv-slice1-bridge` (stacked)

Compare:
- <https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-slice1-bridge...codex/spyre-kv-slice2-core>

## Scope

Adds the concrete in-memory connector, metadata schema, factory registration, and scheduler-driven reuse correctness.

In scope:
- `KVConnectorMetadata`-derived schema for Spyre
- in-memory connector implementation (scheduler + worker interfaces)
- connector registration
- conservative version compatibility guard
- reuse mapping correctness fixes
- core connector tests

Out of scope:
- async load worker pool behavior
- 2-process transport and phase6 runtime hardening
- prometheus adapter

## What this slice is trying to prove

This slice is the first place where the connector becomes real.

It is meant to prove:

1. The scheduler-owned block IDs from `pr-759` can drive a Spyre-side connector
   without inventing a parallel block manager.
2. A correctness-first in-memory connector is enough to define the contract:
   metadata, block mapping, load/store direction, and completion signaling.
3. The current FMS-based path can participate in reuse semantics now, even
   before any future compiler migration.

This is the architectural hinge slice. If we cannot defend this contract,
future offload/disaggregation work stays ambiguous.

## Included commits

- `ce4ae06` Add InMemorySpyreConnector, metadata schema, version guard, and factory registration
- `11ee5d4` Add scheduler-driven KV reuse with exact-prefix matching and block mapping
- `5b4069a` Fix identity load block-ID lookup and incomplete mapping fallback

## Files changed

New:
- `vllm_spyre/compat.py`
- `vllm_spyre/distributed/.../v1/metadata.py`
- `vllm_spyre/distributed/.../v1/inmemory_spyre_connector.py`
- package init files under `vllm_spyre/distributed/...`
- `tests/v1/worker/test_inmemory_spyre_connector.py`
- `tests/v1/worker/test_scheduler_driven_kv_reuse.py`

Modified:
- `vllm_spyre/__init__.py`

Diffstat summary:
- 10 files changed
- 2633 insertions
- 1 deletion

## File-by-file intent

`vllm_spyre/distributed/.../v1/metadata.py`
- defines the connector metadata contract
- declares what a request must carry from scheduler to worker
- fixes the earlier "not a `KVConnectorMetadata` subclass" problem

`vllm_spyre/distributed/.../v1/inmemory_spyre_connector.py`
- implements both scheduler-side and worker-side connector interfaces
- stores KV in-process for correctness-first reuse
- encodes current load/store/failure semantics

`vllm_spyre/__init__.py`
- registers the connector with the factory
- adds version guard wiring

`vllm_spyre/compat.py`
- fails fast on incompatible vLLM version bands

`tests/v1/worker/test_inmemory_spyre_connector.py`
- validates storage semantics, metadata behavior, and lifecycle edges

`tests/v1/worker/test_scheduler_driven_kv_reuse.py`
- validates reuse matching and scheduler-driven mapping

## Why this is a separate PR

This slice introduces the contract and the first real data movement semantics.

It deserves its own review because the main questions are different from Slice 1:
- Is the metadata contract defensible?
- Are the scheduler-side hooks conservative and correct?
- Is the in-memory connector a valid correctness baseline?

If this is mixed with Slice 3/4, reviewers end up debating metrics and async
details before agreeing on the underlying contract.

## Core invariants introduced here

1. Metadata must be a `KVConnectorMetadata` subtype.
2. Block mappings refer to real scheduler block IDs, not positional indices.
3. If mapping coverage is incomplete, fall back to recompute-safe store mode.
4. `get_num_new_matched_tokens()` remains conservative rather than optimistic.
5. The connector only operates on staging tensors; it does not own FMS tensors.

## Reviewer focus

Primary review targets:
- `SpyreConnectorMeta` field set and validation assumptions
- scheduler-side request classification (`is_store` vs load)
- identity mapping semantics
- incomplete mapping fallback behavior
- version gate strictness

Secondary review targets:
- whether connector registration belongs in plugin registration path
- naming of metadata fields for future compatibility with `vllm-spyre-next`

## Main risks in this slice

1. Metadata drift:
- if upstream connector expectations change, this is the first place likely to break.

2. False-positive reuse:
- if matching is too aggressive, requests may try to load KV they should recompute.

3. Hidden FMS layout mismatch:
- the in-memory connector is only safe because the model runner owns staging sync.

## What this slice does not solve

- async overlap
- large-scale transport
- offload backend semantics
- production observability

## Relationship to current vs future Spyre compiler work

Current compiler / FMS path:
- directly useful now
- defines the practical contract for a correctness POC

Future `torch-spyre` path:
- still useful
- the exact storage implementation may change, but metadata shape and lifecycle
  should carry forward if the upstream seam is preserved

## Validation checklist

Minimal:
```bash
pytest -q tests/v1/worker/test_inmemory_spyre_connector.py \
         tests/v1/worker/test_scheduler_driven_kv_reuse.py
```

Recommended:
```bash
pytest -q tests/v1/worker
```

Additional targeted checks:
```bash
pytest -q tests/v1/worker/test_inmemory_spyre_connector.py -k "identity or mapping"
pytest -q tests/v1/worker/test_scheduler_driven_kv_reuse.py -k "fallback or prefix"
```

## Suggested smoke environments

Local CPU:
- best place to validate metadata and matching logic quickly

Remote CUDA:
- useful for checking that connector factory and version guard behave under a
  fuller runtime stack

Spyre cards:
- not needed to validate contract correctness
- needed later when connecting this contract to real Spyre execution

## Draft PR title

`[KVConnector][Slice2] Add Spyre in-memory connector, metadata contract, and reuse correctness`

## Draft PR body

```markdown
## Summary
This PR introduces the Spyre in-memory KV connector implementation and metadata contract, plus scheduler-driven reuse and mapping correctness fixes.

## What changed
- Added `SpyreConnectorMeta` and request metadata structures.
- Added `InMemorySpyreConnector` implementing v1 connector interfaces.
- Registered connector in plugin registration path.
- Added strict fallback when external mapping coverage is incomplete.
- Fixed identity mapping behavior to use actual scheduler block IDs.
- Added unit tests for metadata, load/save, reuse behavior, and failure handling.

## Why
This is the first fully usable connector core for correctness-first reuse testing.

## Test plan
- `pytest -q tests/v1/worker/test_inmemory_spyre_connector.py`
- `pytest -q tests/v1/worker/test_scheduler_driven_kv_reuse.py`

## Non-goals
- No async overlap tuning.
- No cross-process production transport.
- No Prometheus integration in this slice.
```
