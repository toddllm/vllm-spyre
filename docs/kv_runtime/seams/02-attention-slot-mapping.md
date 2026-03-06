# Seam: Attention Backend and Slot Mapping

## Question

What is the explicit attention/backend contract for paged KV, and where must
Spyre conform to upstream placement metadata instead of inventing its own
interpretation?

## Decision target

Interface decision.

## Current answer

Treat slot mapping, block tables, and the attention backend KV
shape/stride/layout contract as one inseparable execution seam.

A correctness-first Spyre backend is feasible if it consumes upstream placement
metadata directly and proves shape/stride correctness before attempting any
kernel-native optimization.

## What this page proves

- Upstream attention backends are given an explicit KV cache
  shape/stride/metadata contract.
- `slot_mapping` and block tables are not optional hints; they are the concrete
  placement metadata the data plane must consume.
- The CUDA paged-attention kernels encode low-level layout assumptions that
  Spyre must either match logically or translate explicitly.

## Story snippets

### 1. The attention backend contract is explicit, not implicit

Upstream `AttentionBackend` defines methods for KV cache shape and stride and
expects backends to consume shared metadata fields.

That means a backend can be slow and still correct, but it cannot be casual
about metadata semantics.

Relevant anchors in [../source-map.md](../source-map.md): `UP-ATTN-BACKEND`,
`UP-ATTN-CPU`, `UP-ATTN-SLOTMAP`.

### 2. Slot mapping is how placement becomes execution

The control plane allocates logical blocks. The data plane then relies on slot
mapping and block tables to translate those logical choices into actual KV read
and write positions.

This is the boundary where many “we know the block IDs” stories fail: block IDs
alone are not enough to execute the attention update correctly.

Relevant anchors in [../source-map.md](../source-map.md): `UP-BT-SLOTMAP`,
`UP-ATTN-SLOTMAP`.

### 3. CPU backend behavior shows the contract can be proved before optimization

The upstream CPU backend is useful because it shows that the contract is not
CUDA-only. The backend can consume the same metadata and still provide a
reference behavior for correctness-first comparisons.

This is the right model for early Spyre work: prove conformance first, then
optimize layout and kernel strategy.

Relevant anchors in [../source-map.md](../source-map.md): `UP-ATTN-CPU`,
`UP-ATTN-BACKEND`.

### 4. CUDA kernels show the physical layout assumptions that matter later

The CUDA paged-attention kernels encode concrete assumptions about:

- KV cache shape
- block-table interpretation
- how block indices become cache pointers

Spyre does not need to match those assumptions physically on day one, but it
does need to match them logically or translate them explicitly.

Relevant anchors in [../source-map.md](../source-map.md): `UP-CUDA-KV-SHAPE`,
`UP-CUDA-PAGED-V1`, `UP-CUDA-PAGED-V2`.

### 5. Old Spyre remains awkward here because the current data plane is still FMS-shaped

Today’s `vllm-spyre` path still routes through FMS model loader and
`past_key_value_states` ownership, so even when it receives upstream-shaped
placement metadata, it is not yet a native vLLM attention backend.

That is why old-stack correctness work tends to look like translation or bridge
logic rather than a clean backend implementation.

Relevant anchors in [../source-map.md](../source-map.md): `OLD-DUMMY-KVSPEC`,
`OLD-FMS-KV-PATH`.

## What differs in old stack

Old stack can consume slot-mapping-related metadata, but it does so through
FMS-shaped attention and model-runner glue rather than a native upstream
attention backend.

That means:

- metadata can be upstream-shaped
- execution semantics are still bridge-shaped
- physical layout and update semantics remain harder to reason about

## What should survive into future stack

These parts should survive directly:

- block table plus slot mapping as the canonical placement contract
- attention backend shape/stride contract as an explicit interface
- correctness-first gather/scatter or translation path before optimization
- exact comparison against upstream backend behavior before performance tuning

## Relevant anchors in `source-map.md`

- `UP-ATTN-BACKEND`
- `UP-ATTN-CPU`
- `UP-BT-SLOTMAP`
- `UP-ATTN-SLOTMAP`
- `UP-CUDA-KV-SHAPE`
- `UP-CUDA-PAGED-V1`
- `UP-CUDA-PAGED-V2`
- `OLD-DUMMY-KVSPEC`
- `OLD-FMS-KV-PATH`
