# Spyre KV Offload Research

Status: Exploratory

Last updated: 2026-03-13

## Purpose

This note captures the current state of exploratory work around KV offload and
KV reuse on Spyre. It is intended as a working document for tracking public
references, implementation status, measurements, and open questions. It may
later be replaced by a more formal design document or RFC.

The terminology and high-level framing in this note follow the vLLM
KV Offloading Connector blog, especially its distinction between request
latency benefits and throughput benefits.

## Summary

- vLLM already has a KV offloading connector and related offload abstractions.
- The current native offload backend is oriented around CUDA-like devices and
  is not a direct fit for Spyre.
- Spyre already has a useful integration seam through its worker-side KV
  connector bridge and connector-facing staging buffers.
- The current prototype validates synchronous in-memory KV reuse on Spyre and
  shows measurable request-latency improvements in hardware-backed offline
  runs.
- The current work is strongest on the latency and reuse side of the problem;
  a fuller throughput story will require broader benchmarking and a richer
  transport path.

## Current Prototype Scope

The current prototype provides:

- a synchronous `InMemorySpyreConnector`
- connector registration through `vllm_spyre.register()`
- worker-side bridge integration through the existing Spyre worker path
- staging tensor registration after Spyre warmup
- exact-prefix and partial-prefix, block-aligned reuse
- worker-side save/load statistics
- byte-capped in-memory storage

The current prototype does not yet provide:

- real DMA or transport-backed offload
- async overlap with forward execution
- cross-process or cross-engine transfer
- a finalized `OffloadingSpec`-based Spyre transport backend

## Current Measurements

### Functional validation

Hardware-backed single-process offline runs have validated:

- exact-prefix reuse
- partial-prefix reuse
- zero-miss worker-side loads for aligned reusable prefix blocks

The current validation setup uses:

- backend `sendnn`
- a single-process `LLM` path
- built-in prefix caching disabled
- the model `ibm-ai-platform/micro-g3.3-8b-instruct-1b`

Observed worker-side reuse behavior:

- warm requests saved `48` entries
- reuse requests loaded `24` entries
- reuse requests had `0` load misses

For the measured prompts:

- common prefix length was `199` tokens
- block size was `64`
- aligned reusable prefix was `192` tokens, or `3` blocks
- the model registers `4` KV cache layers
- expected reusable entries were `3 blocks * 4 layers * 2 (K/V) = 24`

The observed load count matched that expectation.

### Latency benchmark

The current benchmark script is:

- `examples/offline_inference/spyre_kv_reuse_benchmark.py`

It compares:

- cold exact prompt latency vs exact replay latency
- cold partial-prefix prompt latency vs partial-prefix replay latency

It currently measures request latency and connector activity, not TTFT.

Latest hardware-backed benchmark result on 2026-03-13 with `repeats=3`:

- exact cold mean latency: `0.1703s`
- exact reuse mean latency: `0.1413s`
- exact reuse speedup: `1.205x`
- exact reuse latency reduction: `17.0%`
- exact cold mean throughput: `47.0 tok/s`
- exact reuse mean throughput: `56.6 tok/s`

- partial cold mean latency: `0.1659s`
- partial reuse mean latency: `0.1407s`
- partial reuse speedup: `1.179x`
- partial reuse latency reduction: `15.2%`
- partial cold mean throughput: `48.2 tok/s`
- partial reuse mean throughput: `56.8 tok/s`

Important caveat:

- scheduler-side `matched_tokens` accounting is not yet reflected in the
  offline benchmark output, so the current proof is based on worker-side
  connector activity plus end-to-end request latency

## Technical Observations

### Upstream fit

The most reusable upstream pieces appear to be:

- `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`
- `vllm/v1/kv_offload/spec.py`

These provide the connector lifecycle, metadata flow, and higher-level offload
abstractions.

The least reusable upstream pieces today are the native transfer backends:

- `vllm/v1/kv_offload/cpu.py`
- `vllm/v1/kv_offload/worker/cpu_gpu.py`

Those paths are built around CUDA-like assumptions such as platform checks,
CUDA streams, CUDA events, and GPU-oriented block movement helpers.

### Spyre fit

Spyre already has the main seam needed for a staged approach:

- `vllm_spyre/v1/worker/spyre_kv_connector_bridge.py`
- `vllm_spyre/v1/worker/spyre_model_runner.py`

That seam is promising because:

- the worker already supports connector lifecycle hooks
- staging buffers already exist
- staging/live KV synchronization already exists around the forward path

That same seam is also the main current limitation:

- the connector path is synchronous today
- the current prototype proves reuse and measurement, not overlap

## Public References

- [vLLM KV Offloading Connector blog](https://vllm.ai/blog/kv-offloading-connector)
- [vLLM engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)
- [IBM Research: Lifting the cover on the IBM Spyre Accelerator](https://research.ibm.com/blog/lifting-the-cover-on-the-ibm-spyre-accelerator)
- [IBM Research: Building PyTorch-native support for the IBM Spyre Accelerator](https://research.ibm.com/blog/pytorch-support-ibm-spyre)
- [IBM AIU Trace Analyzer](https://github.com/IBM/aiu-trace-analyzer)

## Publicly Visible Gaps

At the time of writing, the following gaps remain visible in public material:

- no public Spyre-specific offloading backend for vLLM
- no public Spyre DMA microbenchmark analogous to the vLLM GPU/CPU benchmark
- no public host-to-Spyre KV-shaped transfer benchmark with block-size and
  bandwidth curves

## Open Questions

- What is the right Spyre-specific transport abstraction under the existing
  vLLM connector model?
- Is a synchronous staging-based connector sufficient to justify a richer
  transport implementation?
- What device constraints on pinned memory, alignment, or transfer granularity
  should shape KV block layout?
- What is the right place to surface scheduler-side accounting for reuse in the
  offline path?
- Should TTFT-oriented measurement wait for an online/server-path benchmark, or
  be approximated in a dedicated offline harness first?

## Near-Term Next Steps

- Expand the benchmark matrix across shared-prefix length, prompt length, and
  output length.
- Improve scheduler-side observability so reuse can be measured at both the
  scheduler and worker layers.
- Decide whether the next document should be a transport-focused RFC or a
  broader phased design note.
- If the current latency improvements remain stable, prototype a Spyre-specific
  transport-backed offload path behind the existing connector seam.
