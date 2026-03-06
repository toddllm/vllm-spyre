# Seam: KV Connector Lifecycle

## Question

What is the actual KV connector lifecycle, and which parts belong to the
connector itself versus the scheduler/runtime above it?

## Decision target

Feasibility decision.

## Current answer

A correctness-first Spyre connector is feasible if it follows the upstream
scheduler and worker lifecycle exactly.

The connector should surface granular success/failure and remain idempotent. The
scheduler/runtime should own semantic policy such as retry, invalidate, or
recompute.

## What this page proves

- Upstream vLLM already provides a concrete scheduler-side and worker-side
  lifecycle for connectors.
- A connector is not the owner of block identity, placement policy, or retry
  policy.
- Invalid-load feedback must flow back to the scheduler/runtime so that the safe
  fallback is invalidation and recompute rather than silent reuse.

## Story snippets

### 1. Scheduler-side hooks already define the control-plane ordering

Upstream scheduler calls the connector in an ordered sequence for:

- token matching
- post-allocation state capture
- connector metadata construction

This means connector implementations should consume scheduler-owned metadata,
not try to re-derive ownership from local state.

Relevant anchors in [../source-map.md](../source-map.md): `UP-SCHED-CONN-STEP`,
`UP-SCHED-KV-META`.

### 2. `KVConnectorBase_V1` defines separate scheduler and worker responsibilities

The connector contract is already split cleanly:

- scheduler-side methods decide match accounting and metadata construction
- worker-side methods bind metadata, load, save, and report completion

This is the right place to preserve a push/pull-neutral architecture. The
connector contract should describe source identity, destination placement, and
movement results, not hard-code one transport direction.

Relevant anchors in [../source-map.md](../source-map.md): `UP-ACTIVE-PRE`,
`UP-ACTIVE-POST`, `UP-ACTIVE-NOFWD` and the `KVConnectorBase_V1` references in
that same source map section.

### 3. The native worker-side lifecycle is explicit and narrow

Upstream `ActiveKVConnector` shows the intended worker lifecycle shape:

- bind metadata
- start load before compute
- finalize after compute
- handle no-forward steps explicitly
- clear metadata

That is the lifecycle old Spyre has to emulate while its data plane is still
bridge-shaped.

Relevant anchors in [../source-map.md](../source-map.md): `UP-ACTIVE-PRE`,
`UP-ACTIVE-POST`, `UP-ACTIVE-NOFWD`.

### 4. The old-stack bridge works because it makes the lifecycle explicit

The experimental bridge is useful because it makes the old-stack phases visible:

- begin step
- before forward
- after forward
- finish step

That does not make it the final architecture. It just provides a safe place to
mirror the upstream lifecycle while the current data plane cannot participate
natively.

Relevant anchors in [../source-map.md](../source-map.md):
`EXP-BRIDGE-BEGIN`, `EXP-BRIDGE-BEFORE`, `EXP-BRIDGE-AFTER`,
`EXP-BRIDGE-FINISH`.

### 5. Failure handling must end in explicit invalidation, not silent reuse

A connector can report granular load failures. It should not own the full retry
policy. The runtime/scheduler above it should decide whether to retry,
invalidate, or recompute.

The key invariant is that uncertain loads degrade to explicit invalidation and
recompute, never silent reuse.

Relevant anchors in [../source-map.md](../source-map.md):
`EXP-CONN-LOADERR`, `EXP-CONN-FINISH`, `UP-SCHED-CONN-STEP`.

## What differs in old stack

Old `vllm-spyre` does not naturally expose the same worker/model-runner and
layer-level hook surfaces that upstream vLLM uses. That is why connector work on
old stack tends to be bridge-shaped and bulk-shaped.

This is a data-plane limitation, not evidence that the scheduler-side connector
model is wrong.

## What should survive into future stack

These parts should survive directly:

- scheduler-owned token matching and connector metadata construction
- connector-side idempotent load/save semantics
- worker-side explicit lifecycle phases
- granular invalidation feedback to the scheduler/runtime
- retry/recompute policy staying above the connector layer

## Relevant anchors in `source-map.md`

- `UP-SCHED-KV-META`
- `UP-SCHED-CONN-STEP`
- `UP-ACTIVE-PRE`
- `UP-ACTIVE-POST`
- `UP-ACTIVE-NOFWD`
- `EXP-BRIDGE-BEGIN`
- `EXP-BRIDGE-BEFORE`
- `EXP-BRIDGE-AFTER`
- `EXP-BRIDGE-FINISH`
- `EXP-CONN-BIND`
- `EXP-CONN-LOAD`
- `EXP-CONN-SAVE`
- `EXP-CONN-FINISH`
- `EXP-CONN-LOADERR`
