# Seam: Old-Stack Divergences

## Question

Why is current `vllm-spyre` still bridge-shaped, and what exactly should be
considered old-stack divergence rather than durable architecture?

## Decision target

Design choice decision.

## Current answer

Treat current old-stack `vllm-spyre` as a useful laboratory for validating
runtime contracts, not as the final data-plane design.

The durable lesson is the control-plane and lifecycle contract. The bridge,
dummy KV spec, and FMS-owned data path are transitional mechanisms.

## What this page proves

- Old `vllm-spyre` still overrides more runtime internals than the future stack
  should.
- The FMS path still owns the actual KV tensors and forward behavior.
- Current old-stack connector integration has to compensate for that limitation,
  which is why the bridge exists.

## Story snippets

### 1. The platform forces several transitional config decisions today

Current Spyre platform wiring overrides worker choice, scheduler behavior, and
cache-related config in ways that are useful for the current stack but should
not be mistaken for neutral upstream extension seams.

These overrides explain why today’s stack feels more invasive than the
future-stack direction.

Relevant anchors in [../source-map.md](../source-map.md): old-stack `pr-759`
platform anchors and `OLD-DUMMY-KVSPEC`.

### 2. The custom scheduler is a policy divergence layer, not the architectural baseline

Current old-stack scheduling includes hardware-specific heuristics and chunked
prefill constraints. Those may be necessary in practice, but they should be
understood as policy layered on top of an upstream scheduler model, not as the
contract future stack should preserve.

Relevant anchors in [../source-map.md](../source-map.md): `UP-SCHED-SCHEDULE`,
`UP-SCHED-CONN-STEP` and the old-stack scheduler anchor section.

### 3. The dummy KV spec shows that the current data plane is not yet vLLM-native

The old model runner still returns a dummy FMS-derived KV spec instead of a
native vLLM attention-backed spec.

That is the clearest code-level signal that today’s stack still needs a bridge
rather than naturally fitting the upstream data-plane seam.

Relevant anchors in [../source-map.md](../source-map.md): `OLD-DUMMY-KVSPEC`.

### 4. The FMS model path still owns the actual KV bytes

`SpyreCausalLM` owns `past_key_value_states`, allocates them, passes them into
FMS forward, and receives them back.

This is the central reason old-stack connector work is awkward: upstream block
identity can be made canonical, but the actual KV bytes still live behind an
FMS-shaped execution path.

Relevant anchors in [../source-map.md](../source-map.md): `OLD-FMS-KV-PATH`.

### 5. The bridge is the explicit compensation layer

The experimental bridge makes the old-stack lifecycle explicit around `forward()`.
That is useful because it lets the old stack mirror upstream worker lifecycle
semantics while the underlying data plane is still FMS-owned.

This is the correct interpretation of the bridge: necessary and useful, but not
a durable final seam.

Relevant anchors in [../source-map.md](../source-map.md):
`EXP-BRIDGE-BEGIN`, `EXP-BRIDGE-BEFORE`, `EXP-BRIDGE-AFTER`,
`EXP-BRIDGE-FINISH`.

## What differs in old stack

Old stack currently differs in these durable ways:

- platform wiring overrides more internals
- scheduler policy is more custom
- data plane is FMS-owned rather than vLLM-native
- connector integration is bridge-shaped rather than layer-native
- some visibility/export paths rely on experimental or temporary mechanisms

That is why old-stack work should be framed as validating runtime contracts,
metadata shapes, and failure semantics rather than defining the final API.

## What should survive into future stack

These lessons should survive:

- scheduler-owned block identity and metadata ownership
- explicit lifecycle phases and failure semantics
- the distinction between source identity and destination placement
- the principle that runtime, not compiler-local hacks, should own KV
  residency/movement policy

These parts should not survive as architecture:

- dummy KV spec compatibility paths
- bridge-shaped integration as the primary design
- any old-stack-only visibility hack treated as the final export/import seam

## Relevant anchors in `source-map.md`

- `UP-SCHED-SCHEDULE`
- `UP-SCHED-CONN-STEP`
- `OLD-DUMMY-KVSPEC`
- `OLD-FMS-KV-PATH`
- `EXP-BRIDGE-BEGIN`
- `EXP-BRIDGE-BEFORE`
- `EXP-BRIDGE-AFTER`
- `EXP-BRIDGE-FINISH`
