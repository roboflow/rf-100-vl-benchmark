#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DASHSCOPE_API_KEY:-}" && "${PREPARE_ONLY:-0}" != "1" ]]; then
  echo "DASHSCOPE_API_KEY is required." >&2
  exit 1
fi

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
DATASET=${DATASET:-RF100VL/rf20-vl-fsod-fresh-20260813/defect-detection}
RUN_ROOT=${RUN_ROOT:-qwen38-fsod-runs/defect-five-shot-anchor-random-v1}
CONDITIONS=${CONDITIONS:-qwen38-fsod-configs/defect-five-shot-anchor-random-explicit.json}
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
  --reference-first-strategy largest-then-seeded-random
  --reference-random-seed 1234
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

mkdir -p "$RUN_ROOT"
"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python evaluate_qwen38_recipe.py \
  --dataset-dir "$DATASET" \
  --conditions "$CONDITIONS" \
  --output-dir "$RUN_ROOT" \
  --prepare-only \
  "${COMMON[@]}"

"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python preflight_qwen38_defect_anchor_random.py \
  --dataset "$DATASET" \
  --run-root "$RUN_ROOT" \
  --one-shot-run-root qwen38-fsod-runs/rf20-three-way-matched-v1/defect-detection \
  --report "$RUN_ROOT/preflight.json"

if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "Defect anchor-plus-random five-shot preflight completed."
  exit 0
fi

run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python evaluate_qwen38_recipe.py \
  --dataset-dir "$DATASET" \
  --conditions "$CONDITIONS" \
  --output-dir "$RUN_ROOT" \
  "${COMMON[@]}"

echo "Defect anchor-plus-random five-shot evaluation completed successfully."
