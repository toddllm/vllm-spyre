# Seam: Next-Stack and Runtime Direction

## Question

What is real today in `vllm-spyre-next` and `torch-spyre`, and what still
belongs to the future-capability story rather than the current old-stack data
plane?

## Decision target

Feasibility decision.

## Current answer

The next-stack direction is real at the platform/runtime seam, but still
partial.

`vllm-spyre-next` already has a concrete platform scaffold and CPU-path entry
point. `torch-spyre` already registers a PrivateUse1 backend and inductor-side
hooks. But the runtime still has important copy, fallback, stream, and event
gaps that keep this in “correctness-first and capability-building” territory
rather than “production-complete data plane.”

## What this page establishes

- `vllm-spyre-next` is not just a placeholder idea; it already has a concrete
  platform scaffold and example path.
- `torch-spyre` is not only a device name; it already wires backend
  registration and inductor integration into PyTorch.
- The remaining runtime gaps are concrete and code-visible, especially around
  copy semantics, stream/event hooks, and fallback pressure.

## Story snippets

### 1. `vllm-spyre-next` already defines a real platform scaffold

The next-stack platform is intentionally conservative today: it inherits the CPU
platform path, uses upstream worker/scheduler defaults, and leaves explicit
comments for where Spyre-specific worker/model-runner/attention backend wiring
will eventually go.

That is exactly what a scaffold should do: provide a real seam without
pretending feature parity exists already.

Relevant anchors in [source-map.md](../source-map.md):
[`NEXT-PLATFORM`](../source-map.md#next-platform),
[`NEXT-EXAMPLE`](../source-map.md#next-example).

### 2. The current next-stack example is still CPU-path baseline, not full Spyre execution

The example script explicitly says the new stack is currently using upstream
vLLM CPU worker/runner classes. That is useful because it sets the maturity
level correctly.

The point of this path today is to validate extension seams and runtime shape,
not to overstate device readiness.

Relevant anchors in [source-map.md](../source-map.md):
[`NEXT-EXAMPLE`](../source-map.md#next-example).

### 3. `torch-spyre` already registers a real backend and PrivateUse1 device module

The backend registration path is concrete:

- `pyproject.toml` exposes a `torch.backends` entrypoint
- `_autoload()` renames the PrivateUse1 backend and registers the device module

That means the future stack is building on real runtime integration points, not
just conceptual placeholders.

Relevant anchors in [source-map.md](../source-map.md):
[`TS-ENTRYPOINT`](../source-map.md#ts-entrypoint),
[`TS-BACKEND-REGISTER`](../source-map.md#ts-backend-register).

### 4. Inductor integration is real, but it is still patch-heavy

`torch-spyre._inductor._autoload()` already:

- registers a device interface
- registers backend codegen and device op overrides
- imports custom ops, decompositions, and lowering
- monkey-patches AOTAutograd / compile-to-module paths

That is evidence of real integration, but it is also evidence that the current
runtime path is still carrying significant custom integration weight.

Relevant anchors in [source-map.md](../source-map.md):
[`TS-INDUCTOR-AUTOLOAD`](../source-map.md#ts-inductor-autoload).

### 5. The remaining runtime gaps are concrete and directly relevant to KV movement

The runtime limitations are not abstract:

- fallbacks explicitly route unsupported ops through CPU
- preload removes decompositions because some paths are not yet reliable
- copy semantics still carry TODO/FIXME comments
- stream/event hooks are stubbed or effectively unimplemented

Those are exactly the kinds of gaps that matter for offload, import/export, and
asynchronous transfer semantics.

Relevant anchors in [source-map.md](../source-map.md):
[`TS-FALLBACKS`](../source-map.md#ts-fallbacks),
[`TS-PRELOAD`](../source-map.md#ts-preload),
[`TS-COPY`](../source-map.md#ts-copy),
[`TS-COPY-FIXME`](../source-map.md#ts-copy-fixme),
[`TS-HOOKS-STREAM`](../source-map.md#ts-hooks-stream).

## What differs from old stack

The future stack differs in the right direction:

- it is trying to converge on upstream platform/plugin seams
- it already has a real runtime/backend registration path
- it shifts the problem toward runtime capability and away from old FMS-shaped
  visibility hacks

But it is still incomplete enough that old-stack experiments remain useful for
learning control-plane, lifecycle, and metadata contracts.

## What should survive into future stack

These parts should survive directly:

- scheduler-owned logical identity and placement semantics
- runtime-visible export/import capability
- region-handle framing rather than raw-address assumptions
- explicit copy/stream/event capability requirements
- the rule that compiler/runtime is a capability provider, not the policy owner

## Relevant anchors in `source-map.md`

- [`NEXT-PLATFORM`](../source-map.md#next-platform)
- [`NEXT-EXAMPLE`](../source-map.md#next-example)
- [`TS-ENTRYPOINT`](../source-map.md#ts-entrypoint)
- [`TS-BACKEND-REGISTER`](../source-map.md#ts-backend-register)
- [`TS-INDUCTOR-AUTOLOAD`](../source-map.md#ts-inductor-autoload)
- [`TS-FALLBACKS`](../source-map.md#ts-fallbacks)
- [`TS-PRELOAD`](../source-map.md#ts-preload)
- [`TS-COPY`](../source-map.md#ts-copy)
- [`TS-COPY-FIXME`](../source-map.md#ts-copy-fixme)
- [`TS-HOOKS-STREAM`](../source-map.md#ts-hooks-stream)
