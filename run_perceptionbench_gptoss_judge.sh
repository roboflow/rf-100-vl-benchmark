#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/matveipopov/vlms-rf100-vl/rf-100-vl-benchmark}
PYTHON_BIN=${PYTHON_BIN:-/home/matveipopov/.pyenv/versions/3.12.0/bin/python}
DATA_DIR=${DATA_DIR:-PerceptionBench}
RUN_ROOT=${RUN_ROOT:-perceptionbench-runs}
CONCURRENCY=${CONCURRENCY:-16}
JUDGE_MODEL=${JUDGE_MODEL:-gpt-oss-120b}
JUDGE_MODEL_REVISION=${JUDGE_MODEL_REVISION:-b5c939de8f754692c1647ca79fbf85e8c1e70f8a}
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai@sha256:14ea8b431aaaf75eb873c46c8ebfbad2b4b0790d30c66126d789d8cb9bd0aab9}
JUDGE_STATE_DIR=${JUDGE_STATE_DIR:-$RUN_ROOT/gpt-oss-120b-judge-v1}
QUEUE_LOG=${QUEUE_LOG:-$JUDGE_STATE_DIR/judge-queue.log}
RUNPOD_LAUNCH_ATTEMPTS=${RUNPOD_LAUNCH_ATTEMPTS:-120}
JUDGE_READY_ATTEMPTS=${JUDGE_READY_ATTEMPTS:-360}
MODELS_CSV=${MODELS_CSV:-qwen3.8-flash,qwen3.8-max}
POD_NAME=${POD_NAME:-perceptionbench-gpt-oss-120b-judge}
COMPLETION_MARKER=${COMPLETION_MARKER:-_QWEN38_MAX_FLASH_EVALUATION_COMPLETE}

IFS=',' read -r -a MODELS <<<"$MODELS_CSV"
if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "MODELS_CSV must contain at least one model." >&2
  exit 2
fi
for model in "${MODELS[@]}"; do
  case "$model" in
    qwen3.8-flash|qwen3.8-max) ;;
    *)
      echo "Unsupported model in MODELS_CSV: $model" >&2
      exit 2
      ;;
  esac
done

cd "$REPO_DIR"
mkdir -p "$JUDGE_STATE_DIR"
exec > >(tee -a "$QUEUE_LOG") 2>&1

if [[ -f infra/runpod.env ]]; then
  source infra/runpod.env
fi
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is required through gitignored infra/runpod.env}"

umask 077
runpod_curl_config="$JUDGE_STATE_DIR/runpod-curl.conf"
printf 'header = "Authorization: Bearer %s"\n' "$RUNPOD_API_KEY" >"$runpod_curl_config"

API=https://rest.runpod.io/v1
rp() {
  local method=$1 path=$2 body=${3:-}
  if [[ -n "$body" ]]; then
    curl -fsS --config "$runpod_curl_config" -X "$method" "$API$path" \
      -H 'Content-Type: application/json' \
      -d "$body"
  else
    curl -fsS --config "$runpod_curl_config" -X "$method" "$API$path"
  fi
}

token_path="$JUDGE_STATE_DIR/api-key"
if [[ ! -s "$token_path" ]]; then
  umask 077
  openssl rand -hex 32 >"$token_path"
fi
judge_api_key=$(<"$token_path")
judge_curl_config="$JUDGE_STATE_DIR/judge-curl.conf"
printf 'header = "Authorization: Bearer %s"\n' "$judge_api_key" >"$judge_curl_config"

pod_id=""
if [[ -s "$JUDGE_STATE_DIR/pod-id" ]]; then
  pod_id=$(<"$JUDGE_STATE_DIR/pod-id")
  if ! rp GET "/pods/$pod_id" \
    | jq -e '.desiredStatus == "RUNNING"' >/dev/null 2>&1; then
    pod_id=""
  fi
fi

terminate_own_pod() {
  if [[ -n "$pod_id" ]]; then
    echo "[$(date -u +%FT%TZ)] terminating judge pod $pod_id"
    rp DELETE "/pods/$pod_id" >/dev/null 2>&1 || true
  fi
}
trap terminate_own_pod EXIT
trap 'exit 130' INT TERM

if [[ -z "$pod_id" ]]; then
  body=$(JUDGE_API_KEY="$judge_api_key" VLLM_IMAGE="$VLLM_IMAGE" \
    POD_NAME="$POD_NAME" \
    JUDGE_MODEL_REVISION="$JUDGE_MODEL_REVISION" "$PYTHON_BIN" - <<'PY'
import json
import os

print(json.dumps({
    "name": os.environ["POD_NAME"],
    "imageName": os.environ["VLLM_IMAGE"],
    "cloudType": "SECURE",
    "computeType": "GPU",
    "gpuTypeIds": ["NVIDIA H100 80GB HBM3"],
    "gpuCount": 1,
    "containerDiskInGb": 30,
    "volumeInGb": 200,
    "volumeMountPath": "/workspace",
    "interruptible": False,
    "allowedCudaVersions": ["12.8", "12.9", "13.0"],
    "ports": ["8000/http"],
    "supportPublicIp": True,
    "env": {
        "HF_HOME": "/workspace/huggingface",
        "HF_HUB_CACHE": "/workspace/huggingface/hub",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_TOKEN": "{{ RUNPOD_SECRET_HF_TOKEN }}",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    },
    "dockerStartCmd": [
        "--model", "openai/gpt-oss-120b",
        "--revision", os.environ["JUDGE_MODEL_REVISION"],
        "--served-model-name", "gpt-oss-120b",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--api-key", os.environ["JUDGE_API_KEY"],
        "--max-model-len", "32768",
        "--max-num-seqs", "16",
        "--max-num-batched-tokens", "1024",
        "--gpu-memory-utilization", "0.95",
        "--async-scheduling",
        "--no-enable-prefix-caching",
    ],
}))
PY
  )
  echo "[$(date -u +%FT%TZ)] launching one dedicated H100 judge pod"
  response=""
  for launch_attempt in $(seq 1 "$RUNPOD_LAUNCH_ATTEMPTS"); do
    if response=$(rp POST /pods "$body" 2>/dev/null); then
      pod_id=$(printf '%s' "$response" | jq -r '.id // empty')
      if [[ -n "$pod_id" ]]; then
        break
      fi
    fi
    echo "[$(date -u +%FT%TZ)] RunPod launch attempt $launch_attempt/$RUNPOD_LAUNCH_ATTEMPTS did not allocate an H100; retrying in 30 seconds"
    sleep 30
  done
  if [[ -z "$pod_id" ]]; then
    echo "RunPod did not return a pod ID after $RUNPOD_LAUNCH_ATTEMPTS attempts." >&2
    exit 2
  fi
  printf '%s' "$pod_id" >"$JUDGE_STATE_DIR/pod-id"
  printf '%s' "$response" | jq '{id,name,costPerHr,adjustedCostPerHr,image,gpuCount,volumeInGb}' \
    >"$JUDGE_STATE_DIR/pod-launch.json"
fi

judge_base_url="https://$pod_id-8000.proxy.runpod.net/v1"
echo "[$(date -u +%FT%TZ)] waiting for the pinned judge server"
ready=0
for _ in $(seq 1 "$JUDGE_READY_ATTEMPTS"); do
  if curl -fsS --config "$judge_curl_config" --max-time 15 \
    "$judge_base_url/models" \
    | jq -e --arg model "$JUDGE_MODEL" '.data[] | select(.id == $model)' >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 10
done
if [[ "$ready" != "1" ]]; then
  ready_minutes=$((JUDGE_READY_ATTEMPTS * 10 / 60))
  echo "Pinned gpt-oss-120b judge did not become ready within $ready_minutes minutes." >&2
  exit 3
fi

export PERCEPTIONBENCH_JUDGE_API_KEY="$judge_api_key"
export PERCEPTIONBENCH_JUDGE_BASE_URL="$judge_base_url"

for model in "${MODELS[@]}"; do
  run_dir="$RUN_ROOT/$model-xhigh-v1"
  echo "[$(date -u +%FT%TZ)] judging $model with the paper's exact judge"
  "$PYTHON_BIN" evaluate_qwen38_perceptionbench.py judge \
    --data-dir "$DATA_DIR" \
    --run-dir "$run_dir" \
    --judge-model "$JUDGE_MODEL" \
    --concurrency "$CONCURRENCY"
  for retry_round in 1 2 3; do
    failure_count=$(find "$run_dir/judgments" -type f -name '*.json' -print0 \
      | xargs -0 -r jq -s 'map(select(.status == "judge_failure")) | length')
    failure_count=${failure_count:-0}
    if [[ "$failure_count" == "0" ]]; then
      break
    fi
    echo "[$(date -u +%FT%TZ)] retrying $failure_count failed judgments, round $retry_round"
    "$PYTHON_BIN" evaluate_qwen38_perceptionbench.py judge \
      --data-dir "$DATA_DIR" \
      --run-dir "$run_dir" \
      --judge-model "$JUDGE_MODEL" \
      --concurrency "$CONCURRENCY" \
      --retry-failures
  done
  "$PYTHON_BIN" evaluate_qwen38_perceptionbench.py score \
    --data-dir "$DATA_DIR" --run-dir "$run_dir"
done

touch "$RUN_ROOT/$COMPLETION_MARKER"
echo "[$(date -u +%FT%TZ)] paper-matched evaluations are complete for: $MODELS_CSV"
