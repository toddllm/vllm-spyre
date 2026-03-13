# Draft RFC: Spyre KV Offload Path

Status: Draft

Last updated: 2026-03-13

## Summary

This document proposes an incremental path for KV offload and KV reuse support
on Spyre. The proposed direction is to reuse upstream vLLM connector and
metadata abstractions, reuse the existing Spyre worker-side bridge and staging
buffers, and introduce Spyre-specific transport support in phases.

The draft uses the same basic framing as the vLLM KV Offloading Connector
writeup: KV offload is valuable both for request-latency reduction and for
throughput improvement under memory pressure or reuse-heavy workloads.

The first phase is intentionally small: a synchronous in-memory connector that
proves correctness, reuse behavior, and measurable latency impact before any
transport-specific work is attempted.

## Motivation

There are three reasons to pursue a Spyre-specific KV offload path:

- Upstream vLLM already has an evolving connector and offload model, which
  makes it possible to align Spyre work with shared abstractions rather than
  invent a separate path.
- Spyre already has useful worker-side plumbing for connector-facing staging
  buffers, which reduces the amount of new integration required.
- Early prototype results show that even a synchronous in-memory reuse path can
  produce measurable request-latency improvements on Spyre.

This draft focuses first on proving correctness and latency benefit. Throughput
benefit remains an important motivation, but it likely depends on broader
benchmarking and a later transport-backed implementation.

## Background

Current upstream vLLM offload support is centered on connector abstractions plus
native transfer backends that assume CUDA-like devices. That makes direct reuse
of the current transport layer unattractive for Spyre, but it does not prevent
reuse of the higher-level connector lifecycle and metadata model.

Spyre already exposes a connector bridge and staging buffers in its worker path.
That existing seam makes phased development practical:

1. prove reuse and measurement first
2. improve accounting and observability
3. add a Spyre-specific transport path later

## Goals

- Align Spyre KV offload work with upstream vLLM abstractions where practical.
- Keep the first implementation slice small, measurable, and easy to validate.
- Preserve the existing Spyre worker-side staging seam as the main integration
  point.
- Establish whether reuse delivers enough value to justify a transport-backed
  implementation.

## Non-Goals

- Replacing all existing prefix-caching behavior in vLLM
- Defining a final cross-process or cross-engine protocol in the first phase
- Promising async DMA overlap in the initial implementation
- Committing to a finalized public API for Spyre-specific offload behavior

## Proposal

### Phase 1: synchronous in-memory reuse

Implement and validate:

- a synchronous `InMemorySpyreConnector`
- exact-prefix and partial-prefix, block-aligned reuse
- worker-side staging save/load
- connector statistics for saved blocks, loaded blocks, and load misses
- a repeatable offline benchmark for cold vs reuse scenarios

Rationale:

- This is the smallest slice that proves end-to-end behavior.
- It reuses the existing Spyre worker seam.
- It avoids overcommitting to transport details before the reuse path has been
  measured.

### Phase 2: observability and benchmark depth

Extend the prototype with:

- better scheduler-side accounting in offline runs
- broader benchmark coverage across prompt and output shapes
- clearer separation of cold, seed, exact-reuse, and partial-reuse scenarios
- a path toward online or TTFT-oriented measurement

Rationale:

- Current results are already useful, but they rely on worker-side accounting
  plus request latency.
- Better observability reduces ambiguity before transport work begins.

### Phase 3: Spyre-specific transport-backed offload

If the earlier phases remain favorable, add a transport-backed implementation
behind the same connector seam.

This phase would explore:

- a Spyre-specific transfer backend
- device and host transfer constraints
- overlap opportunities, if supported by the runtime
- broader offload policies beyond simple in-memory reuse

## Current Status

The current prototype already validates the Phase 1 direction.

Implemented so far:

- synchronous in-memory connector support
- bridge integration through the existing Spyre worker path
- exact and partial-prefix reuse
- worker-side save/load statistics
- a standalone offline benchmark

Current benchmark result on 2026-03-13 with `repeats=3`:

- exact replay improved mean request latency from `0.1703s` to `0.1413s`
- exact replay delivered `1.205x` speedup and `17.0%` latency reduction
- partial-prefix replay improved mean request latency from `0.1659s` to
  `0.1407s`
- partial-prefix replay delivered `1.179x` speedup and `15.2%` latency
  reduction
- reuse runs loaded `24` entries with `0` misses

Current limitations:

- the benchmark measures request latency, not TTFT
- scheduler-side accounting is not yet surfaced in the offline output
- the prototype is synchronous and single-process

## Alternatives Considered

### Adapt the existing native offload backend directly

This would likely force Spyre support into CUDA-oriented transfer assumptions
too early. The current upstream transport path is the least portable part of
the system.

### Build transport support first

This would front-load risk in the least understood part of the problem. It
would also make it harder to tell whether any observed win came from reuse
itself or from transport-specific behavior.

### Rely only on built-in prefix caching

Built-in prefix caching is useful, but it does not answer the broader question
of how Spyre should participate in the upstream offload connector model.

## Risks And Open Questions

- A future Spyre transport path may still need abstractions that differ from
  current CUDA-oriented assumptions.
- Scheduler-side accounting may require additional plumbing before reuse can be
  observed consistently in all execution modes.
- Request-latency improvements in the offline path may not translate directly
  into TTFT improvements in an online/server path.
- Device constraints around memory registration, alignment, or transfer
  granularity may change the most effective KV block layout.

## Validation Plan

- Keep focused connector and worker tests passing as the connector evolves.
- Continue hardware-backed cold-vs-reuse benchmarking.
- Expand the benchmark matrix before transport work begins.
- Add an online or TTFT-oriented benchmark when the observability path is ready.

## Success Criteria

The proposal is on track if the following remain true:

- exact and partial-prefix reuse continue to work reliably
- reuse remains observable through connector statistics
- measured request-latency improvements remain stable across a broader prompt
  matrix
- the implementation can evolve toward transport-backed offload without
  replacing the existing Spyre worker seam

## References

- [vLLM KV Offloading Connector blog](https://vllm.ai/blog/kv-offloading-connector)
- [vLLM engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)
- [IBM Research: Lifting the cover on the IBM Spyre Accelerator](https://research.ibm.com/blog/lifting-the-cover-on-the-ibm-spyre-accelerator)
- [IBM Research: Building PyTorch-native support for the IBM Spyre Accelerator](https://research.ibm.com/blog/pytorch-support-ibm-spyre)
- [IBM AIU Trace Analyzer](https://github.com/IBM/aiu-trace-analyzer)
