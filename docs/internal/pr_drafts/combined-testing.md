# Combined Branch Testing Guide

Combined branch:
- `codex/spyre-kv-combined`

Purpose:
- one branch with all connector slices for integration and exploratory testing.

Compare to base mirror:
- <https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-base-pr759...codex/spyre-kv-combined>

## What this branch includes

- Slice1 bridge lifecycle wiring
- Slice2 connector core + metadata
- Slice3 runtime hardening and async-ready behavior
- Slice4 metrics adapter

## What this branch is for

Use `codex/spyre-kv-combined` when the question is:

- "Does the full current connector stack still work together?"
- "Can I test a realistic integration path without caring about PR boundaries?"
- "Can I run one branch for exploratory validation while the PR slices stay clean?"

Do not use this branch when the question is:

- "What is the smallest thing we should merge?"
- "Which exact code should reviewers look at first?"

For those, use the slice branches and the per-slice docs.

## What this branch is not

- It is not the best review branch.
- It is not the branch with the internal draft docs.
- It is not guaranteed to remain minimal.

It is the test branch.

## Suggested validation order

1. Unit and integration suite:
```bash
pytest -q tests/v1/worker
```

2. Optional lint pass:
```bash
ruff check vllm_spyre/distributed/kv_transfer/kv_connector/v1 \
           tests/v1/worker
```

3. CPU-only local smoke (if no accelerator):
```bash
VLLM_TARGET_DEVICE=cpu pytest -q tests/v1/worker
```

4. Focused quick regression:
```bash
pytest -q tests/v1/worker/test_kv_connector_bridge.py \
         tests/v1/worker/test_inmemory_spyre_connector.py \
         tests/v1/worker/test_kv_phase6.py -k "prom or metric or identity"
```

## Environment tiers

Local CPU:
- best for rapid iteration
- validates logic and most test scaffolding

Remote CUDA:
- best for heavier runtime checks and any upstream CUDA-path sanity
- useful for validating that test harnesses do not rely on CPU-only behavior

Spyre cards:
- use only when testing the current Spyre compiler path or a minimal real card
  POC
- keep the target simple at first: one inference call, one model, no disagg

## What to test first on Spyre

The first Spyre-hardware target should remain intentionally small:

1. one offline inference call
2. one prompt
3. one request in batch
4. no connector reuse expectations
5. no disaggregated prefill/decode

That isolates:
- model load
- warmup
- one generate path
- current compiler/runtime viability

Only after that works should connector-aware testing on Spyre cards be added.

## Minimal Spyre POC target (single inference call)

Use a single offline inference call first before disaggregated flow:

```bash
VLLM_USE_V1=1 \
VLLM_SPYRE_DYNAMO_BACKEND=sendnn \
python examples/offline_inference/text_inference.py \
  --model ibm-ai-platform/micro-g3.3-8b-instruct-1b \
  --num-prompts 1 \
  --max-num-seqs 1 \
  --max-model-len 512 \
  --max-tokens 16 \
  --backend sendnn
```

If compiler/runtime is not ready in that environment, use:
- `--backend eager` and `VLLM_SPYRE_DYNAMO_BACKEND=eager`

## Known caveats

- `codex/spyre-kv-combined` does not contain the PR draft docs; those live on
  `codex/spyre-kv-connector`.
- Full `tests/v1/worker` is still the best single internal signal, but not every
  test is equally important for every experiment.
- Some Slice 3 tests are intentionally broader than what the first Spyre-card
  demo needs.

## Note on PR hygiene

Use the combined branch for integration testing and architecture iteration.
For upstream PRs, prefer the slice branches so scope stays reviewable.
