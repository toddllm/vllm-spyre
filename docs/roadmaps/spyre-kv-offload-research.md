# Spyre KV Offload Research

Status: Exploratory

Last updated: 2026-03-16

## Purpose

This note tracks the current state of exploratory work around KV reuse, KV
offload, and related P-D disaggregation questions for Spyre.

It is intentionally broader than the first connector prototype note. The goal
now is to place that prototype in the larger context of:

- the current `vllm_spyre` stack
- the emerging `vllm_spyre_next` stack
- upstream `vllm` KV connector / HMA / P-D disaggregation work
- the torch-spyre runtime/compiler work needed for the next stack

The framing follows the vLLM KV Offloading Connector writeup, especially its
separation between latency benefit, throughput benefit, and connector
abstractions.

## Naming

This note uses the following names consistently.

- `current stack` / `old stack`
  - `vllm_spyre`
  - custom Spyre scheduler / worker / model runner
  - FMS model code and FMS attention path
  - current SendNN / DeepTools-oriented execution path

- `next stack` / `Spyre-Next`
  - `vllm_spyre_next`
  - intended to consume upstream vLLM model code directly
  - intended to rely on torch-spyre for the device/runtime/compiler path

- `current compiler stack`
  - the current torch-spyre compiler/runtime contract
  - publicly represented by `SuperDSC-Bundle` and `SpyreCode` / `JobPlan`

- `future compiler stack`
  - the direction after `SuperDSC-Bundle`
  - publicly represented by `KTIR`

## Summary

- The current prototype validates synchronous in-memory KV reuse on the
  current `vllm_spyre` stack.
- The prototype already shows measurable request-latency benefit on
  hardware-backed offline runs.
- The current upstream vLLM connector work is moving quickly in two related
  areas:
  - local/offloaded KV reuse through HMA and CPU offload
  - P-D disaggregation and connector-based KV transfer
- The `Spyre-Next` direction remains the likely long-term landing zone for a
  cleaner Spyre + vLLM integration, but it is not yet the best place to prove
  the first AIU-backed offload win.
- For the near term, the current-stack worker-side KV seam is still the most
  practical place to prove connector value on AIU.

## Architecture Snapshot: Current Stack

```text
                      CURRENT STACK / OLD STACK
                   (vllm_spyre + FMS + SendNN path)

  User / OpenAI API / LLM.generate
                 |
                 v
         upstream vLLM engine core
                 |
                 v
     +----------------------------------+
     | vllm_spyre plugin                |
     |                                  |
     |  - SpyrePlatform                 |
     |  - custom scheduler              |
     |  - custom worker                 |
     |  - custom model runner           |
     +----------------------------------+
                 |
                 v
     +----------------------------------+
     | execution model                  |
     |                                  |
     |  - FMS model code                |
     |  - FMS attention / KV handling   |
     |  - custom warmup / batching      |
     +----------------------------------+
                 |
                 v
          torch.compile (Dynamo level)
                 |
        +--------+---------+
        |                  |
        v                  v
     sendnn             inductor
        |
        v
   DeepTools / current runtime
        |
        v
       AIU
```

Key observation:

- current KV connector work on this stack has to attach at the worker/model
  runner seam, because attention and KV management are still owned by the FMS
  path rather than upstream vLLM model code

## Architecture Snapshot: Next Stack

```text
                        NEXT STACK / SPYRE-NEXT
          (vllm_spyre_next + torch-spyre + upstream vLLM modeling)

  User / OpenAI API / LLM.generate
                 |
                 v
         upstream vLLM engine core
                 |
                 v
     +----------------------------------+
     | thinner Spyre plugin             |
     |                                  |
     |  - Spyre-specific platform       |
     |  - likely Spyre worker           |
     |  - likely Spyre model runner     |
     |  - no FMS dependency             |
     +----------------------------------+
                 |
                 v
     +----------------------------------+
     | upstream vLLM execution model    |
     |                                  |
     |  - upstream model code           |
     |  - Spyre attention backend       |
     |  - upstream scheduler behavior   |
     +----------------------------------+
                 |
                 v
        torch.compile / Inductor path
                 |
                 v
     +----------------------------------+
     | torch-spyre device/runtime       |
     |                                  |
     |  - device tensors                |
     |  - allocator / layout            |
     |  - stream support                |
     |  - copy path                     |
     |  - distributed backend           |
     +----------------------------------+
                 |
                 v
      current compiler contract today:
        SuperDSC-Bundle -> SpyreCode / JobPlan
                 |
      future compiler contract later:
                   KTIR
                 |
                 v
                AIU
```

Key observation:

- this is the more attractive end-state for long-term maintainability
- but it depends on significantly more torch-spyre maturity than the current
  worker-side connector prototype does

## Scenario Map

### 1. Current-stack local KV reuse / offload path

```text
   current scheduler emits
   kv_connector_metadata
         |
         v
   SpyreKVConnectorBridge
   (worker-side, synchronous)
         |
         v
   staging tensors <-> live FMS KV
         |
         v
   connector medium
     |           |
     |           +--> host-memory reuse store
     |
     +--> future offload / transport target
```

### 2. Current-stack P-D disaggregation direction

```text
 Prefill node (current stack)           Decode node (current stack)
 +---------------------------+          +---------------------------+
 | vllm_spyre scheduler/     |          | vllm_spyre scheduler/     |
 | worker/model runner       |          | worker/model runner       |
 | + FMS attention/KV        |          | + FMS attention/KV        |
 +-------------+-------------+          +-------------+-------------+
               |                                      ^
               |                                      |
               +---- connector transport / KV push ---+
                         via worker-side bridge
```

### 3. Next-stack target shape

```text
            upstream scheduler / connector / HMA / PD logic
                               |
                               v
                 +-------------------------------+
                 | vllm_spyre_next worker/model  |
                 | runner on torch-spyre         |
                 +-------------------------------+
                               |
                               v
                upstream model code + Spyre attn backend
                               |
                               v
                     torch-spyre device tensors
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
           local offload medium         P-D disagg transport
           (host / future medium)       (prefill <-> decode)
```

### 4. Transition path

```text
  Near-term proving ground:
    current stack on AIU
      -> prove reuse/offload value now
      -> validate benchmark and observability

  Long-term convergence target:
    next stack on torch-spyre
      -> remove FMS dependency
      -> reuse upstream vLLM model execution
      -> move toward upstream connector abstractions

  Transition rule:
    do not move the primary offload bet to the next stack until
    paged attention, device tensors, copy, stream, and enough
    multi-Spyre/distributed support are credible there
```

## Current Prototype Scope

The current prototype provides:

- a synchronous `InMemorySpyreConnector`
- worker-side bridge integration through the existing Spyre worker seam
- staging tensor registration after Spyre warmup
- exact-prefix and partial-prefix, block-aligned reuse
- worker-side save/load statistics
- byte-capped in-memory storage

The current prototype does not yet provide:

- real DMA or transport-backed offload
- async overlap with forward execution
- cross-process or cross-engine transfer
- a finalized Spyre-specific upstream offload backend
- a next-stack implementation on top of torch-spyre device tensors

## What Has Been Tested

### Local / CPU

Validated locally:

- focused connector tests
- worker-side integration coverage for the prototype
- offline probe / benchmark harnesses

These runs validate connector behavior and regressions, but they do not prove
real device transport behavior.

### Current stack on AIU

Hardware-backed offline runs have already validated:

- exact-prefix reuse
- partial-prefix reuse
- zero-miss worker-side loads for aligned reusable prefix blocks

Observed worker-side reuse behavior:

- warm requests saved `48` entries
- reuse requests loaded `24` entries
- reuse requests had `0` misses

For the measured prompts:

- common prefix length was `199` tokens
- block size was `64`
- aligned reusable prefix was `192` tokens, or `3` blocks
- the model registered `4` KV cache layers
- expected reusable entries were `3 blocks * 4 layers * 2 (K/V) = 24`

The observed load count matched that expectation.

Latest hardware-backed benchmark result on 2026-03-13 with `repeats=3`:

- exact cold mean latency: `0.1703s`
- exact reuse mean latency: `0.1413s`
- exact reuse speedup: `1.205x`
- exact reuse latency reduction: `17.0%`

- partial cold mean latency: `0.1659s`
- partial reuse mean latency: `0.1407s`
- partial reuse speedup: `1.179x`
- partial reuse latency reduction: `15.2%`

Important caveat:

- this is still a single-process offline current-stack path
- it proves reuse benefit, not yet serving-path TTFT or multi-node P-D behavior
- these AIU measurements should now be treated as a pre-sync baseline, because
  they were gathered before the more recent vLLM sync work on the broader
  branch line; they should be re-run on the target pinned environment before
  becoming the standing benchmark reference

### Next stack on AIU

Not yet validated for end-to-end KV offload or P-D disaggregation.

The public `Spyre-Next` workstream appears to still be in the phase of:

- CPU/dev-test readiness
- wrapped layer bring-up
- custom attention backend scaffolding
- compatibility with newer upstream vLLM versions
- upstream test harness and filtering

## Technical Fit

### Upstream pieces worth reusing

The most reusable upstream vLLM pieces appear to be:

- `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`
- `vllm/v1/kv_offload/spec.py`

These provide:

- connector lifecycle
- metadata flow
- offload abstractions

### Upstream pieces that are still not a direct fit

The least reusable upstream pieces today remain the native transfer backends:

- `vllm/v1/kv_offload/cpu.py`
- `vllm/v1/kv_offload/worker/cpu_gpu.py`

Those paths are still built around CUDA-like assumptions such as:

- CUDA streams
- CUDA events
- CUDA-oriented platform checks
- GPU-centric block movement helpers

### Spyre seam that exists today

Spyre already has a useful worker-side seam:

- `vllm_spyre/v1/worker/spyre_kv_connector_bridge.py`
- `vllm_spyre/v1/worker/spyre_model_runner.py`

That seam is attractive because:

- connector lifecycle is already wired
- staging buffers already exist
- staging/live KV synchronization already exists

That same seam is also the current limitation:

- it is synchronous
- it is worker-side rather than natively integrated into upstream vLLM model
  execution

## Public Work To Watch

### vllm-spyre

- issue `#639` `[RFC] vllm-spyre-next`
- issue `#745` `[Epic] Develop KVCacheConnector for Spyre`
- issue `#648` paged KV-cache attention backend using torch-spyre
- issue `#647` contiguous KV-cache attention backend using torch-spyre
- issue `#689` layer-wise split execution in torch-spyre
- issue `#666` run vLLM modeling code instead of FMS
- PR `#798` custom attention backend for `Spyre-Next`
- PR `#826` update vLLM and torch-spyre for `Spyre-Next`
- PR `#836` wrapped embedding layer for `Spyre-Next`
- PR `#837` upstream tests framework and RMSNorm tests for `Spyre-Next`

### upstream vLLM

- PR `#35264` KV push from prefill to decode node using NIXL connector
- PR `#35760` PD-disagg + speculative decoding acceptance tests
- PR `#36687` support hybrid SSM-FA models in PD disaggregation
- PR `#36957` heterogeneous TP in hybrid-model PD disaggregation
- issue `#36780` RFC for hybrid SSM-FA NIXL connector support
- PR `#37160` simple general CPU KV cache offloading
- PRs `#36642`, `#36644`, `#36645`, `#35223`, `#36549`
  - HMA / multi-group / sliding-window / recovery / multi-connector plumbing

### torch-spyre

- PR `#816` multi-Spyre device support framework
- PR `#918` stream support
- PR `#1007` graph-free copy
- PR `#1049` profiling toolkit RFC
- PR `#1010` SpyreCode / JobPlan alignment
- PR `#1011` tensor memory access analysis
- PR `#868` SuperDSC-Bundle specification
- issue `#682` KTIR
- issue `#183` eager codegen through torch.compile
- issue `#200` allocator work for VF mode

### PyTorch upstream

- `pytorch/pytorch#172154` privateuse1 backend integration with Kineto
- `pytorch/pytorch#176877` distributed support for OpenReg
- `pytorch/pytorch#175954` decomposition-table handling RFC
- PyTorch dev-discuss:
  `IBM Spyre Accelerator: PyTorch Enabling Status and Feature Plan - 1H 2026`
  - important because it makes the next-stack PyTorch-native direction more
    concrete and spells out a staged multi-card plan

## Multi-Spyre Context

Multi-Spyre is not the first proof point for the current connector prototype,
but it is a major dependency for the long-term next-stack story.

Why it matters:

- tensor-parallel inference on the next stack depends on it
- realistic decode-node scaling in P-D disaggregation depends on it
- upstream vLLM model execution on Spyre will eventually need collective and
  transport behavior that fits this substrate

Public work worth treating as directly relevant:

- `torch-spyre` PR `#816`
  - distributed backend / `spyreccl` direction
- `torch-spyre` PR `#918`
  - stream support
- `torch-spyre` PR `#1007`
  - graph-free copy
- PyTorch dev-discuss roadmap thread for Spyre in 1H 2026
  - compiled functional collectives first
  - `torch.distributed` migration second
  - eventual `torch.comms` alignment later

Practical implication:

- the current-stack AIU reuse/offload effort should proceed without waiting for
  full multi-Spyre maturity
- the next-stack offload / P-D disaggregation effort should not be treated as
  fully credible until this multi-card substrate is substantially clearer

## Near-Term Validation Plan

### Track A: current stack on AIU

This remains the best proving ground for offload value right now.

Recommended order:

1. expand the current benchmark matrix across prompt shape and output length
2. improve scheduler-side observability
3. add serving-path / TTFT-oriented measurement
4. decide whether the next experiment should be a transport-backed local
   offload path or a current-stack P-D disaggregation path

### Track B: next stack readiness

This remains the longer-term convergence path.

Recommended order:

1. continue layer and attention backend bring-up
2. continue syncing to current upstream vLLM
3. keep expanding upstream-test coverage
4. track multi-Spyre, stream, copy, and compiler/runtime prerequisites
5. move AIU offload experiments there only when the basic serving path is
   credible

## Open Questions

- What is the smallest AIU serving-path benchmark that can give a credible TTFT
  story for the current stack?
- Which upstream vLLM connector changes should be reflected in Spyre’s local
  architecture language now to avoid churn later?
- What is the minimum next-stack milestone that makes AIU KV-offload
  experiments reasonable there?
- Which torch-spyre prerequisites are strict blockers for next-stack P-D
  disaggregation and multi-Spyre support?

## Public References

- [vLLM KV Offloading Connector blog](https://vllm.ai/blog/kv-offloading-connector)
- [vLLM engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)
- [IBM Research: Lifting the cover on the IBM Spyre Accelerator](https://research.ibm.com/blog/lifting-the-cover-on-the-ibm-spyre-accelerator)
- [IBM Research: Building PyTorch-native support for the IBM Spyre Accelerator](https://research.ibm.com/blog/pytorch-support-ibm-spyre)
- [IBM Spyre Accelerator: PyTorch Enabling Status and Feature Plan - 1H 2026](https://dev-discuss.pytorch.org/t/ibm-spyre-accelerator-pytorch-enabling-status-and-feature-plan-1h-2026/3319)
- [IBM AIU Trace Analyzer](https://github.com/IBM/aiu-trace-analyzer)
- [vllm-spyre issue #639: vllm-spyre-next](https://github.com/vllm-project/vllm-spyre/issues/639)
- [vllm-spyre issue #745: Develop KVCacheConnector for Spyre](https://github.com/vllm-project/vllm-spyre/issues/745)
- [vLLM issue #36780: hybrid SSM-FA NIXL connector RFC](https://github.com/vllm-project/vllm/issues/36780)
- [torch-spyre RFC 0171: Spyre Device](https://github.com/torch-spyre/torch-spyre/tree/main/RFCs/0171-SpyreDevice)
