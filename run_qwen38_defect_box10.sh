#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DASHSCOPE_API_KEY:-}" && "${PREPARE_ONLY:-0}" != "1" ]]; then
  echo "DASHSCOPE_API_KEY is required." >&2
  exit 1
fi

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
DATASET_DIR=${DATASET_DIR:-RF100VL/rf20-vl-fsod-fresh-20260813/defect-detection}
RUN_DIR=${RUN_DIR:-qwen38-fsod-runs/defect-detection-box10-v1}
CONDITIONS=${CONDITIONS:-qwen38-fsod-configs/defect-detection-box10-three-repeat.json}
MAX_PASSES=${MAX_PASSES:-6}
PREPARE_ONLY=${PREPARE_ONLY:-0}
COMMON=(
  --concurrency 256
  --requests-per-minute 13500
  --tokens-per-minute 1800000
  --timeout-seconds 180
  --max-completion-tokens 8192
  --max-retries 3
  --temperature 0
  --allow-shared-reference-images
)

run_resumable() {
  local attempt status
  status=1
  for ((attempt = 1; attempt <= MAX_PASSES; attempt++)); do
    if "$@"; then
      return 0
    else
      status=$?
    fi
    echo "Resumable pass ${attempt}/${MAX_PASSES} exited ${status}." >&2
    sleep 10
  done
  return "$status"
}

mkdir -p "$RUN_DIR"

"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python evaluate_qwen38_recipe.py \
  --dataset-dir "$DATASET_DIR" \
  --conditions "$CONDITIONS" \
  --output-dir "$RUN_DIR" \
  --prepare-only \
  "${COMMON[@]}"

if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "Defect Detection 10-reference preflight completed; inference intentionally skipped."
  exit 0
fi

run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python evaluate_qwen38_recipe.py \
  --dataset-dir "$DATASET_DIR" \
  --conditions "$CONDITIONS" \
  --output-dir "$RUN_DIR" \
  "${COMMON[@]}"

echo "Defect Detection 10-reference experiment completed successfully."
