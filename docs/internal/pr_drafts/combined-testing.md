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

## Note on PR hygiene

Use the combined branch for integration testing and architecture iteration.
For upstream PRs, prefer the slice branches so scope stays reviewable.
