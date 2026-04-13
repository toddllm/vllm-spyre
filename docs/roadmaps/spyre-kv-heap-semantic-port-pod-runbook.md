# Spyre KV Heap Semantic Port Pod Runbook

Status: Draft

Last updated: 2026-04-13

## Purpose

This runbook is the heap-only follow-on to the earlier connector perf-port
pod validation. It assumes the pod is running the forward-ported branch that
includes:

- the restored heap/service backend layer
- the updated async-capable connector
- the worker bridge/model-runner heap bypass
- the restored offline heap demo scripts

The goal is to validate the semantically correct heap path on the pod without
repeating the environment mistakes that showed up in the connector-only run.

## Pod Assumptions

- pod repo path: `/home/tdeshane/vllm-spyre`
- DTI root: `/home/tdeshane/dt-inductor`
- repo venv already exists at `/home/tdeshane/vllm-spyre/.venv`
- do not assume normal shell utilities exist
- do not use `uv sync` as a default pod bring-up step
- do set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for test-only and setup flows

## Pod Bring-Up

```bash
oc exec -it <pod-name> -- bash -l
```

Inside the pod:

```bash
set -uo pipefail

export HOME=/home/tdeshane
export DTI_PROJECT_ROOT=${HOME}/dt-inductor
source "$DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh"

cd "$HOME/vllm-spyre"
pwd
test -d .git || { echo "Missing git repo at $(pwd)"; exit 1; }
test -x .venv/bin/python || { echo "Missing repo venv python"; exit 1; }

source .venv/bin/activate
hash -r

export TORCH_DEVICE_BACKEND_AUTOLOAD=0

python - <<'PY'
import os, sys
print(f"python={sys.executable}")
print(f"pwd={os.getcwd()}")
print(f"TORCH_DEVICE_BACKEND_AUTOLOAD={os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')}")
PY
```

## Branch Check

This runbook assumes the branch is already checked out on the pod.

```bash
git branch --show-current
git rev-parse --short HEAD
git status --short --branch
```

## Local `torch-spyre` Overlay

Use the existing repo venv and reinstall the local `torch-spyre` overlay.

```bash
python -m pip install expecttest
python -m pip uninstall -y torch-spyre torch_spyre || true
python -m pip install --no-deps --no-build-isolation -e "$DTI_PROJECT_ROOT/torch-spyre"
```

## Artifact Setup

```bash
RUN_ID=2026-04-13-aiu-vllm-spyre-kv-heap-semantic-port-r1
BASE=lab-artifacts/spyre-kv-heap-semantic-port
ART_DIR="$BASE/$RUN_ID"
mkdir -p "$ART_DIR"
```

## Heap Env

These match the old-stack heap-backed setup from the published markdown, but
avoid unsafe `uv sync` behavior on the current pod image.

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PERFDSC_DEBUG=isg,sdsc,mtracker
export PERFDSC_DUMP_DIR=perfdsc_dumps
export TORCH_SPYRE_DEBUG=1
export DEM_COMPILE_VERSION=1
export DTCOMPILER_KEEP_EXPORT=true

export VLLM_SPYRE_ENABLE_KV_CONNECTOR_BRIDGE=1
export VLLM_SPYRE_KV_STORE_BACKEND=serialized_shared_memory_service
export VLLM_SPYRE_KV_SERVICE_SOCKET=/tmp/spyre-kv-persistent.sock
export VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_ENABLE=1
export VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_STRICT=1
export VLLM_SPYRE_HEAP_KV_PERFDSC_DIR="$PWD/perfdsc_dumps/execute_itr0"
export VLLM_SPYRE_HEAP_KV_EXPORT_DIR="$PWD/export_dtcompiler/r0_1/export_deeprt"
```

## Smoke Tests

Start with the version guard plus the new backend-specific tests.

```bash
OUT="$ART_DIR/2026-04-13-aiu-vllm-spyre-kv-heap-semantic-port-smoke.raw.txt"
rm -f "$OUT"

{
  echo "ts=$(date -Is)"
  echo "pwd=$(pwd)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse --short HEAD)"
  echo "TORCH_DEVICE_BACKEND_AUTOLOAD=${TORCH_DEVICE_BACKEND_AUTOLOAD:-unset}"
  echo "VLLM_SPYRE_KV_STORE_BACKEND=${VLLM_SPYRE_KV_STORE_BACKEND:-unset}"
  echo "VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_ENABLE=${VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_ENABLE:-unset}"
  time python -m pytest -q \
    tests/v1/worker/test_inmemory_spyre_connector.py::TestVersionCompat \
    tests/v1/worker/test_heap_kv_backends.py
} 2>&1 | tee "$OUT"
```

## Sequential Heap POC

This is the first semantically meaningful pod run. It uses the checked-in POC
script, skips `uv sync`, reinstalls local `torch-spyre`, and drives the
service-backed heap path end-to-end.

```bash
OUT="$ART_DIR/2026-04-13-aiu-vllm-spyre-kv-heap-semantic-port-sequential.raw.txt"
rm -f "$OUT"

{
  echo "ts=$(date -Is)"
  echo "pwd=$(pwd)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse --short HEAD)"
  echo "TORCH_DEVICE_BACKEND_AUTOLOAD=${TORCH_DEVICE_BACKEND_AUTOLOAD:-unset}"
  echo "VLLM_SPYRE_KV_STORE_BACKEND=${VLLM_SPYRE_KV_STORE_BACKEND:-unset}"
  echo "VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_ENABLE=${VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_ENABLE:-unset}"
  time bash examples/offline_inference/spyre_kv_sequential_poc.sh \
    --skip-uv-sync \
    --artifact-dir "$BASE/$RUN_ID" \
    --artifact-tag 2026-04-13-aiu-vllm-spyre-kv-heap-semantic-port-sequential \
    --service-socket "$VLLM_SPYRE_KV_SERVICE_SOCKET" \
    --shutdown-service
} 2>&1 | tee "$OUT"
```

## Async A/B On Heap Path

Once the strict heap semantic run is green, rerun the same script with only
the async-load knob changed.

Sync baseline:

```bash
export VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS=0
```

Async candidate:

```bash
export VLLM_SPYRE_KV_ASYNC_LOAD_WORKERS=4
```

Keep every other env var and prompt shape fixed.

## Copy Back

From the local machine:

```bash
POD=<pod-name>
RUN_ID=2026-04-13-aiu-vllm-spyre-kv-heap-semantic-port-r1
REMOTE_BASE=/home/tdeshane/vllm-spyre/lab-artifacts/spyre-kv-heap-semantic-port/$RUN_ID

oc cp "$POD:$REMOTE_BASE/2026-04-13-aiu-vllm-spyre-kv-heap-semantic-port-smoke.raw.txt" "./2026-04-13-aiu-vllm-spyre-kv-heap-semantic-port-smoke.raw.txt"
oc cp "$POD:$REMOTE_BASE/2026-04-13-aiu-vllm-spyre-kv-heap-semantic-port-sequential.raw.txt" "./2026-04-13-aiu-vllm-spyre-kv-heap-semantic-port-sequential.raw.txt"
```

The sequential script also writes its own JSON summaries under the same
artifact prefix. Copy those back only after confirming the filenames that
actually exist on the pod.
