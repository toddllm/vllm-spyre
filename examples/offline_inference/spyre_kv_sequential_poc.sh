#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: spyre_kv_sequential_poc.sh [options]

Run the strict in-process heap-KV sequential handoff POC end-to-end:
  - ensure the persistent KV service is running
  - install local torch-spyre into the active vllm-spyre env
  - run exact baseline/prefill/decode with semantic comparison
  - run partial baseline/prefill/decode with semantic comparison
  - write raw logs, per-phase JSON artifacts, and one summary JSON

Options:
  --artifact-tag TAG             Prefix for artifact filenames.
  --artifact-dir DIR             Artifact directory (default: lab-artifacts/spyre-kv-inmemory-slice)
  --model PATH                   Model path (default: /mnt/models/tiny-granite-3.3-8b)
  --backend NAME                 Backend name (default: sendnn)
  --service-socket PATH          Persistent KV service socket (default: /tmp/spyre-kv-persistent.sock)
  --max-model-len N              Max model length passed to sequential demo (default: 4096)
  --prompt-pair-json PATH        Real prompt-pair JSON for the exact path.
  --partial-prompt-pair-json PATH
                                 Optional prompt-pair JSON for the partial path.
                                 Defaults to --prompt-pair-json when omitted.
  --demo-template NAME           Prompt template passed to sequential demo
                                 (default: java_char_to_string)
  --demo-scenario TEXT           Optional scenario override for the prompt template.
  --demo-question TEXT           Optional question override for the prompt template.
  --demo-partial-tail TEXT       Optional partial-tail override for the prompt template.
  --shared-prefix-tokens N       Shared prefix tokens (default: 384)
  --partial-tail-tokens N        Partial tail tokens (default: 16)
  --max-new-tokens N             Max new tokens (default: 8)
  --max-num-batched-tokens N     Max batched tokens (default: 128)
  --store-max-bytes N            Store max bytes (default: 0)
  --dti-project-root PATH        dt-inductor root (default: \$DTI_PROJECT_ROOT or \$HOME/dt-inductor)
  --skip-uv-sync                 Skip `uv sync` even when `uv` is installed.
  --skip-torch-spyre-install     Skip reinstalling torch-spyre into the active .venv.
  --skip-partial                 Skip the partial-prefix path and run exact only.
  --shutdown-service             Shut the KV service down before exiting.
  -h, --help                     Show this help.
EOF
}

ARTIFACT_DIR="lab-artifacts/spyre-kv-inmemory-slice"
ARTIFACT_TAG=""
MODEL="/mnt/models/tiny-granite-3.3-8b"
BACKEND="sendnn"
SERVICE_SOCKET="/tmp/spyre-kv-persistent.sock"
MAX_MODEL_LEN=4096
PROMPT_PAIR_JSON=""
PARTIAL_PROMPT_PAIR_JSON=""
DEMO_TEMPLATE="java_char_to_string"
DEMO_SCENARIO=""
DEMO_QUESTION=""
DEMO_PARTIAL_TAIL=""
SHARED_PREFIX_TOKENS=384
PARTIAL_TAIL_TOKENS=16
MAX_NEW_TOKENS=8
MAX_NUM_BATCHED_TOKENS=128
STORE_MAX_BYTES=0
SKIP_UV_SYNC=0
SKIP_TORCH_SPYRE_INSTALL=0
SKIP_PARTIAL=0
SHUTDOWN_SERVICE=0
DTI_PROJECT_ROOT_DEFAULT="${DTI_PROJECT_ROOT:-$HOME/dt-inductor}"
DTI_PROJECT_ROOT="$DTI_PROJECT_ROOT_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-tag)
      ARTIFACT_TAG="$2"
      shift 2
      ;;
    --artifact-dir)
      ARTIFACT_DIR="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --service-socket)
      SERVICE_SOCKET="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --prompt-pair-json)
      PROMPT_PAIR_JSON="$2"
      shift 2
      ;;
    --partial-prompt-pair-json)
      PARTIAL_PROMPT_PAIR_JSON="$2"
      shift 2
      ;;
    --demo-template)
      DEMO_TEMPLATE="$2"
      shift 2
      ;;
    --demo-scenario)
      DEMO_SCENARIO="$2"
      shift 2
      ;;
    --demo-question)
      DEMO_QUESTION="$2"
      shift 2
      ;;
    --demo-partial-tail)
      DEMO_PARTIAL_TAIL="$2"
      shift 2
      ;;
    --shared-prefix-tokens)
      SHARED_PREFIX_TOKENS="$2"
      shift 2
      ;;
    --partial-tail-tokens)
      PARTIAL_TAIL_TOKENS="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --max-num-batched-tokens)
      MAX_NUM_BATCHED_TOKENS="$2"
      shift 2
      ;;
    --store-max-bytes)
      STORE_MAX_BYTES="$2"
      shift 2
      ;;
    --dti-project-root)
      DTI_PROJECT_ROOT="$2"
      shift 2
      ;;
    --skip-uv-sync)
      SKIP_UV_SYNC=1
      shift
      ;;
    --skip-torch-spyre-install)
      SKIP_TORCH_SPYRE_INSTALL=1
      shift
      ;;
    --skip-partial)
      SKIP_PARTIAL=1
      shift
      ;;
    --shutdown-service)
      SHUTDOWN_SERVICE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "$ARTIFACT_TAG" ]]; then
  ARTIFACT_TAG="$(date -u +%Y-%m-%dT%H%M%SZ)-spyre-kv-seq-poc"
fi

ARTIFACT_DIR_ABS="$REPO_ROOT/$ARTIFACT_DIR"
mkdir -p "$ARTIFACT_DIR_ABS"

PREFIX="$ARTIFACT_DIR_ABS/$ARTIFACT_TAG"
SERVICE_LOG="$PREFIX-kv-service.raw.txt"
SERVICE_STATS_JSON="$PREFIX-service-stats.json"
SUMMARY_JSON="$PREFIX-summary.json"

ensure_service() {
  if python examples/offline_inference/spyre_kv_service.py \
    stats \
    --socket-path "$SERVICE_SOCKET" >/dev/null 2>&1; then
    echo "KV service already running at $SERVICE_SOCKET"
    return 0
  fi

  echo "Starting KV service at $SERVICE_SOCKET"
  rm -f "$SERVICE_SOCKET"
  nohup python examples/offline_inference/spyre_kv_service.py \
    serve \
    --socket-path "$SERVICE_SOCKET" >"$SERVICE_LOG" 2>&1 &
  sleep 2
  python examples/offline_inference/spyre_kv_service.py \
    stats \
    --socket-path "$SERVICE_SOCKET" >/dev/null
}

install_local_torch_spyre() {
  python -m pip install expecttest
  python -m pip uninstall -y torch-spyre torch_spyre || true
  python -m pip install --no-deps --no-build-isolation -e "$DTI_PROJECT_ROOT/torch-spyre"
}

run_phase() {
  local variant="$1"
  local role="$2"
  local extra_args=("${@:3}")
  local raw_path="$PREFIX-$variant-$role.raw.txt"
  local json_path="$PREFIX-$variant-$role.json"
  local prompt_pair_json=""
  local cmd=(
    python
    examples/offline_inference/spyre_kv_sequential_handoff_demo.py
    --role "$role"
    --write-json "$json_path"
    --backend "$BACKEND"
    --service-socket "$SERVICE_SOCKET"
    --store-max-bytes "$STORE_MAX_BYTES"
    --model "$MODEL"
    --max-model-len "$MAX_MODEL_LEN"
    --demo-template "$DEMO_TEMPLATE"
    --decode-variant "$variant"
    --shared-prefix-tokens "$SHARED_PREFIX_TOKENS"
    --partial-tail-tokens "$PARTIAL_TAIL_TOKENS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  )

  if [[ "$variant" == "exact" && -n "$PROMPT_PAIR_JSON" ]]; then
    prompt_pair_json="$PROMPT_PAIR_JSON"
  elif [[ "$variant" == "partial" ]]; then
    if [[ -n "$PARTIAL_PROMPT_PAIR_JSON" ]]; then
      prompt_pair_json="$PARTIAL_PROMPT_PAIR_JSON"
    elif [[ -n "$PROMPT_PAIR_JSON" ]]; then
      prompt_pair_json="$PROMPT_PAIR_JSON"
    fi
  fi

  if [[ -n "$prompt_pair_json" ]]; then
    cmd+=(--prompt-pair-json "$prompt_pair_json")
  else
    if [[ -n "$DEMO_SCENARIO" ]]; then
      cmd+=(--demo-scenario "$DEMO_SCENARIO")
    fi
    if [[ -n "$DEMO_QUESTION" ]]; then
      cmd+=(--demo-question "$DEMO_QUESTION")
    fi
    if [[ -n "$DEMO_PARTIAL_TAIL" ]]; then
      cmd+=(--demo-partial-tail "$DEMO_PARTIAL_TAIL")
    fi
  fi
  cmd+=("${extra_args[@]}")

  echo
  echo "=== Running $variant $role ==="
  "${cmd[@]}" 2>&1 | tee "$raw_path"
}

build_summary() {
  python3 - <<'PY' "$SUMMARY_JSON" "$SERVICE_STATS_JSON" "$PREFIX" "$SKIP_PARTIAL"
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
service_stats_path = pathlib.Path(sys.argv[2])
prefix = pathlib.Path(sys.argv[3])
skip_partial = bool(int(sys.argv[4]))

def load_json(path: pathlib.Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start == -1:
            raise
        decoder = json.JSONDecoder()
        idx = start
        last_obj = None
        while idx < len(text):
            next_start = text.find("{", idx)
            if next_start == -1:
                break
            try:
                obj, end = decoder.raw_decode(text[next_start:])
            except json.JSONDecodeError:
                idx = next_start + 1
                continue
            last_obj = obj
            idx = next_start + end
        if last_obj is None:
            raise
        return last_obj

exact_baseline = load_json(pathlib.Path(f"{prefix}-exact-baseline.json"))
exact_prefill = load_json(pathlib.Path(f"{prefix}-exact-prefill.json"))
exact_decode = load_json(pathlib.Path(f"{prefix}-exact-decode.json"))
service_stats = load_json(service_stats_path)

summary = {
    "artifact_prefix": str(prefix),
    "git": exact_decode["run_metadata"]["git"],
    "backend": exact_decode["backend"],
    "model": exact_decode["model"],
    "store_backend": exact_decode["store_backend"],
    "service_socket": exact_decode["service_socket"],
    "heap_kv": exact_decode["heap_kv"],
    "demo_template": exact_decode.get("demo_template"),
    "demo_template_display_name": exact_decode.get("demo_template_display_name"),
    "exact": {
        "baseline": exact_baseline["result"],
        "prefill": exact_prefill["result"],
        "decode": exact_decode["result"],
        "semantic_check": exact_decode["semantic_check"],
    },
    "service_stats": service_stats,
}

if not skip_partial:
    partial_baseline = load_json(pathlib.Path(f"{prefix}-partial-baseline.json"))
    partial_prefill = load_json(pathlib.Path(f"{prefix}-partial-prefill.json"))
    partial_decode = load_json(pathlib.Path(f"{prefix}-partial-decode.json"))
    summary["partial"] = {
        "baseline": partial_baseline["result"],
        "prefill": partial_prefill["result"],
        "decode": partial_decode["result"],
        "semantic_check": partial_decode["semantic_check"],
    }

summary["all_semantic_checks_pass"] = bool(summary["exact"]["semantic_check"]["match"])
if not skip_partial:
    summary["all_semantic_checks_pass"] = bool(
        summary["all_semantic_checks_pass"]
        and summary["partial"]["semantic_check"]["match"]
    )

summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(summary_path)
PY
}

cleanup() {
  if [[ "$SHUTDOWN_SERVICE" -eq 1 ]]; then
    python examples/offline_inference/spyre_kv_service.py \
      shutdown \
      --socket-path "$SERVICE_SOCKET" >/dev/null 2>&1 || true
    rm -f "$SERVICE_SOCKET"
  fi
}
trap cleanup EXIT

source "$DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh"
source "$REPO_ROOT/.venv/bin/activate"
if [[ "$SKIP_UV_SYNC" -eq 1 ]]; then
  echo "Skipping uv sync (--skip-uv-sync)"
elif command -v uv >/dev/null 2>&1; then
  uv sync --frozen --active --inexact
else
  echo "uv not found; skipping uv sync"
fi

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PERFDSC_DEBUG=isg,sdsc,mtracker
export PERFDSC_DUMP_DIR=perfdsc_dumps
export TORCH_SPYRE_DEBUG=1
export DEM_COMPILE_VERSION=1
export DTCOMPILER_KEEP_EXPORT=true
export VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_ENABLE=1
export VLLM_SPYRE_EXPERIMENTAL_HEAP_KV_STRICT=1
export VLLM_SPYRE_HEAP_KV_PERFDSC_DIR="$REPO_ROOT/perfdsc_dumps/execute_itr0"
export VLLM_SPYRE_HEAP_KV_EXPORT_DIR="$REPO_ROOT/export_dtcompiler/r0_1/export_deeprt"

rm -rf "$REPO_ROOT/perfdsc_dumps" "$REPO_ROOT/export_dtcompiler"
ensure_service
python examples/offline_inference/spyre_kv_service.py \
  stats \
  --socket-path "$SERVICE_SOCKET" >"$SERVICE_STATS_JSON"

if [[ "$SKIP_TORCH_SPYRE_INSTALL" -eq 0 ]]; then
  install_local_torch_spyre
fi

run_phase exact baseline --clear-service
run_phase exact prefill --clear-service
run_phase exact decode --compare-output-json "$PREFIX-exact-baseline.json"

if [[ "$SKIP_PARTIAL" -eq 0 ]]; then
  run_phase partial baseline --clear-service
  run_phase partial prefill --clear-service
  run_phase partial decode --compare-output-json "$PREFIX-partial-baseline.json"
fi

python examples/offline_inference/spyre_kv_service.py \
  stats \
  --socket-path "$SERVICE_SOCKET" >"$SERVICE_STATS_JSON"
build_summary

echo
echo "Sequential POC complete."
echo "Summary: $SUMMARY_JSON"
