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

cd "$REPO_DIR"
mkdir -p "$JUDGE_STATE_DIR"
exec > >(tee -a "$QUEUE_LOG") 2>&1

if [[ -f infra/runpod.env ]]; then
  source infra/runpod.env
fi
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is required through gitignored infra/runpod.env}"

API=https://rest.runpod.io/v1
rp() {
  local method=$1 path=$2 body=${3:-}
  if [[ -n "$body" ]]; then
    curl -fsS -X "$method" "$API$path" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -H 'Content-Type: application/json' \
      -d "$body"
  else
    curl -fsS -X "$method" "$API$path" \
      -H "Authorization: Bearer $RUNPOD_API_KEY"
  fi
}

token_path="$JUDGE_STATE_DIR/api-key"
if [[ ! -s "$token_path" ]]; then
  umask 077
  openssl rand -hex 32 >"$token_path"
fi
judge_api_key=$(<"$token_path")

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
    JUDGE_MODEL_REVISION="$JUDGE_MODEL_REVISION" "$PYTHON_BIN" - <<'PY'
import json
import os

print(json.dumps({
    "name": "perceptionbench-gpt-oss-120b-judge",
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
    },
    "dockerStartCmd": [
        "--model", "openai/gpt-oss-120b",
        "--revision", os.environ["JUDGE_MODEL_REVISION"],
        "--served-model-name", "gpt-oss-120b",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--api-key", os.environ["JUDGE_API_KEY"],
        "--max-model-len", "32768",
        "--max-num-seqs", "32",
        "--gpu-memory-utilization", "0.90",
        "--async-scheduling",
        "--no-enable-prefix-caching",
    ],
}))
PY
  )
  echo "[$(date -u +%FT%TZ)] launching one dedicated H100 judge pod"
  response=$(rp POST /pods "$body")
  pod_id=$(printf '%s' "$response" | jq -r '.id // empty')
  if [[ -z "$pod_id" ]]; then
    echo "RunPod did not return a pod ID." >&2
    exit 2
  fi
  printf '%s' "$pod_id" >"$JUDGE_STATE_DIR/pod-id"
  printf '%s' "$response" | jq '{id,name,costPerHr,adjustedCostPerHr,image,gpuCount,volumeInGb}' \
    >"$JUDGE_STATE_DIR/pod-launch.json"
fi

judge_base_url="https://$pod_id-8000.proxy.runpod.net/v1"
echo "[$(date -u +%FT%TZ)] waiting for the pinned judge server"
ready=0
for _ in $(seq 1 180); do
  if curl -fsS --max-time 15 \
    -H "Authorization: Bearer $judge_api_key" \
    "$judge_base_url/models" \
    | jq -e --arg model "$JUDGE_MODEL" '.data[] | select(.id == $model)' >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 10
done
if [[ "$ready" != "1" ]]; then
  echo "Pinned gpt-oss-120b judge did not become ready within 30 minutes." >&2
  exit 3
fi

export PERCEPTIONBENCH_JUDGE_API_KEY="$judge_api_key"
export PERCEPTIONBENCH_JUDGE_BASE_URL="$judge_base_url"

for model in qwen3.8-flash qwen3.8-max; do
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

touch "$RUN_ROOT/_QWEN38_MAX_FLASH_EVALUATION_COMPLETE"
echo "[$(date -u +%FT%TZ)] both paper-matched evaluations are complete"
