# Draft RFC: Spyre KV Reuse / Offload Roadmap

Status: Draft

Last updated: 2026-03-16

## Summary

This document proposes a phased path for KV reuse and KV offload support on
Spyre that explicitly distinguishes between:

- the current `vllm_spyre` stack
- the emerging `vllm_spyre_next` stack
- the current torch-spyre compiler/runtime contract
- the future torch-spyre compiler/runtime direction

The proposal is intentionally conservative about where to prove value first.
The immediate proving ground remains the current stack on AIU, because the KV
seam there already exists and has already produced measurable hardware-backed
latency benefit. The next stack remains the likely long-term landing zone, but
it should not carry the first serious offload bet until its serving path is
more mature.

## Motivation

There are three overlapping reasons to pursue this work:

- upstream vLLM already has a fast-moving connector and offload model, and
  Spyre should align with that model where practical
- the current Spyre worker seam already allows meaningful connector
  experimentation on real hardware
- the `Spyre-Next` direction offers a path to a much thinner, more upstream-
  aligned plugin over time

The current question is not simply "how do we implement KV offload?" It is:

- where do we prove reuse/offload value first?
- which pieces belong to the current stack vs the next stack?
- what must be true before the next stack becomes the main place to do this
  work?

## Background

### Current stack

```text
  upstream vLLM engine
         |
         v
  vllm_spyre plugin
    - custom scheduler
    - custom worker
    - custom model runner
         |
         v
  FMS model code + FMS attention/KV
         |
         v
  torch.compile (Dynamo level)
         |
         v
  sendnn / current runtime path
         |
         v
        AIU
```

The important property here is that KV work attaches at the worker/model-runner
seam through staging and bridge logic.

### Next stack

```text
  upstream vLLM engine
         |
         v
  vllm_spyre_next plugin
    - thinner Spyre-specific layer
         |
         v
  upstream vLLM model code
  + Spyre attention backend
         |
         v
  torch.compile / Inductor
         |
         v
  torch-spyre device/runtime/compiler
         |
         v
  current contract today:
    SuperDSC-Bundle -> SpyreCode / JobPlan
         |
  future contract later:
    KTIR
```

The important property here is that offload and P-D disaggregation can
eventually become much more naturally aligned with upstream vLLM connector
abstractions, but only after the next-stack serving path is credible.

## Goals

- Keep the current AIU reuse/offload effort grounded in the smallest working
  seam that already exists.
- Make the relationship between current-stack work and next-stack work
  explicit.
- Reuse upstream vLLM connector/offload abstractions where practical.
- Avoid prematurely locking the design to CUDA-oriented transport assumptions.
- Create a validation path that produces measurable hardware-backed results at
  each phase.

## Non-Goals

- Defining a final Spyre transport implementation in this RFC
- Forcing the first end-to-end offload implementation onto `Spyre-Next`
- Replacing all built-in prefix-caching behavior in vLLM
- Defining the final torch-spyre compiler/runtime contract
- Solving all multi-Spyre/distributed requirements as part of the first reuse
  prototype

## Proposal

### Phase 1: current-stack reuse proof on AIU

Continue building on the current worker-side seam in `vllm_spyre`:

- synchronous connector integration
- exact and partial-prefix reuse
- worker-side save/load accounting
- single-process offline benchmarking

Why this phase exists:

- it already works
- it already has AIU validation
- it proves usefulness without waiting for next-stack maturity

### Phase 2: current-stack offload and serving-path measurement

Extend the current-stack path with:

- broader benchmark matrix coverage
- clearer scheduler-side accounting
- serving-path / TTFT-oriented measurement
- possibly a transport-backed local offload step or a current-stack P-D
  disaggregation step

Why this phase exists:

- the current prototype proves reuse, but not yet the more operational
  serving-path story

### Phase 3: next-stack readiness

In parallel, continue moving `Spyre-Next` toward a usable serving path by
tracking:

- wrapped layer enablement
- paged attention backend work
- upstream vLLM compatibility
- upstream-test coverage
- torch-spyre runtime/compiler prerequisites

This phase is not the place to prove the first AIU offload win. It is the
place to prepare the long-term landing zone.

### Phase 4: next-stack offload convergence

Once the next stack has:

- upstream model execution
- a usable paged attention backend
- device tensors and a stable copy path
- stream support
- enough multi-Spyre/distributed support
- enough compiler/runtime maturity

then the offload / PD-disagg design should move toward that stack and rely
more directly on upstream vLLM connector abstractions.

## Current Status

### Already implemented

- synchronous in-memory connector support
- worker-side bridge integration on the current stack
- exact and partial-prefix reuse
- worker-side save/load statistics
- offline benchmark harnesses

### Already validated

Local / CPU:

- focused connector tests
- worker-side integration coverage
- offline probe / benchmark harnesses

Current stack on AIU:

- exact-prefix reuse
- partial-prefix reuse
- zero-miss worker-side load behavior
- measurable request-latency improvement in the offline path

Latest benchmark result on 2026-03-13 with `repeats=3`:

- exact replay reduced mean request latency from `0.1703s` to `0.1413s`
  (`1.205x`, about `17.0%` reduction)
- partial-prefix replay reduced mean request latency from `0.1659s` to
  `0.1407s` (`1.179x`, about `15.2%` reduction)

Important caveat:

- these AIU measurements should now be treated as a pre-sync baseline, because
  they were gathered before the more recent vLLM sync work on the broader
  branch line; they should be re-run on the target pinned environment before
  becoming the standing benchmark reference

### Not yet validated

- serving-path TTFT benefits
- transport-backed offload
- next-stack AIU offload path
- next-stack P-D disaggregation

## Transition Model

```text
  Track A: current stack on AIU
  -----------------------------
  prove value now
    |
    +--> reuse correctness
    +--> latency benefit
    +--> serving-path measurement
    +--> current-stack offload / PD step if warranted


  Track B: next stack maturation
  ------------------------------
  reduce plugin footprint over time
    |
    +--> layer wrappers
    +--> paged attention backend
    +--> upstream test coverage
    +--> torch-spyre prerequisites
    +--> AIU bring-up


  Convergence criterion
  ---------------------
  when Track B has a credible serving path, move the primary offload
  architecture there and align more directly with upstream vLLM connector work
```

## Dependencies And Related Work

### vllm-spyre

- issue `#639` `[RFC] vllm-spyre-next`
- issue `#745` `[Epic] Develop KVCacheConnector for Spyre`
- issue `#647` contiguous KV-cache attention backend using torch-spyre
- issue `#648` paged KV-cache attention backend using torch-spyre
- issue `#666` run vLLM modeling code instead of FMS
- issue `#689` layer-wise split execution in torch-spyre
- PRs `#798`, `#826`, `#836`, `#837`

### upstream vLLM

- KV push / P-D disaggregation:
  - `#35264`
  - `#35760`
  - `#36687`
  - `#36957`
  - `#36780`

- KV offload / HMA:
  - `#37160`
  - `#36642`
  - `#36644`
  - `#36645`
  - `#35223`
  - `#36549`

### torch-spyre

- multi-Spyre / distributed:
  - `#816`
  - issue `#99`

- runtime / transport prerequisites:
  - `#918` stream support
  - `#1007` graph-free copy
  - issue `#200` allocator work for VF mode

- compiler/runtime contract:
  - `#868` SuperDSC-Bundle
  - issue `#277` / `#1010` SpyreCode / JobPlan
  - issue `#682` KTIR

- tooling:
  - issue `#601` / `#1049` profiling toolkit

### PyTorch upstream

- `pytorch/pytorch#172154`
  - PrivateUse1 backend integration with Kineto

- `pytorch/pytorch#176877`
  - distributed support for OpenReg

- `pytorch/pytorch#175954`
  - decomposition-table handling for out-of-tree backends

- PyTorch dev-discuss:
  `IBM Spyre Accelerator: PyTorch Enabling Status and Feature Plan - 1H 2026`
  - important because it makes the staged multi-card plan explicit:
    compiled functional collectives first, `torch.distributed` migration
    second, eventual `torch.comms` alignment later

## Multi-Spyre Context

Multi-Spyre is not the first milestone for the current-stack reuse proof, but
it is a direct dependency for the longer-term next-stack direction.

In practice, this matters because:

- next-stack tensor parallel inference depends on it
- future P-D disaggregation scaling depends on it
- upstream-model execution on Spyre will eventually need collective and
  transport behavior that fits this substrate

This RFC therefore treats the following as explicit dependencies rather than
background noise:

- `torch-spyre` PR `#816`
- `torch-spyre` PR `#918`
- `torch-spyre` PR `#1007`
- the PyTorch dev-discuss Spyre roadmap thread for 1H 2026

## Risks

### Risk 1: proving too much on the next stack too early

The next stack is attractive architecturally, but it still depends on a large
set of torch-spyre capabilities that are not all mature yet.

### Risk 2: proving only local/offline benefit

The current prototype already proves offline request-latency benefit, but that
does not automatically translate into serving-path TTFT or multi-node
disaggregation benefit.

### Risk 3: transport assumptions drifting away from upstream

Upstream vLLM offload and HMA support is evolving quickly. Spyre-specific work
should keep reusing the connector abstractions while avoiding premature
commitment to a transport design that diverges from upstream concepts.

## Alternatives Considered

### Put the first serious offload effort directly on Spyre-Next

Rejected for now because the next-stack serving path is not mature enough yet.

### Wait for next-stack maturity before doing any offload work

Rejected because it leaves current AIU value unproven even though a working
worker-side seam already exists.

### Build a completely separate Spyre-specific offload framework

Rejected because the direction should remain aligned with upstream vLLM
connector abstractions where practical.

## Validation Plan

### Current stack first

1. keep focused connector and worker tests passing locally
2. keep broadening AIU benchmark coverage
3. add serving-path / TTFT-oriented measurement
4. decide whether the next experiment should be local offload or current-stack
   P-D disaggregation

### Next stack second

1. keep advancing layer and attention backend support
2. keep tracking vLLM compatibility and upstream tests
3. keep tracking torch-spyre distributed/copy/stream/compiler prerequisites
4. start AIU offload experiments only once the next-stack serving path is
   credible

## Success Criteria

This roadmap is on track if:

- current-stack AIU measurements continue to show stable reuse benefit
- serving-path measurements become possible without architectural churn
- next-stack maturity increases without requiring a parallel offload framework
- the long-term landing zone remains compatible with upstream vLLM connector
  direction

## References

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
