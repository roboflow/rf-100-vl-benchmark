#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DASHSCOPE_API_KEY:-}" && "${PREPARE_ONLY:-0}" != "1" ]]; then
  echo "DASHSCOPE_API_KEY is required." >&2
  exit 1
fi

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
DATASET_ROOT=${DATASET_ROOT:-RF100VL/rf20-vl-fsod-fresh-20260813}
RUN_ROOT=${RUN_ROOT:-qwen38-fsod-runs/scale-selector-median-v1}
CONDITIONS=${CONDITIONS:-qwen38-fsod-configs/scale-selector-median-one-shot.json}
MAX_PASSES=${MAX_PASSES:-6}
PREPARE_ONLY=${PREPARE_ONLY:-0}
DATASETS=(paper-parts actions defect-detection)
COMMON=(
  --concurrency 256
  --requests-per-minute 13500
  --tokens-per-minute 1800000
  --timeout-seconds 180
  --max-completion-tokens 8192
  --max-retries 3
  --temperature 0
  --allow-shared-reference-images
  --reference-first-strategy median-relative-area
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
for dataset in "${DATASETS[@]}"; do
  "$UV_BIN" run --with-requirements requirements-cosmos.txt \
    python evaluate_qwen38_recipe.py \
    --dataset-dir "$DATASET_ROOT/$dataset" \
    --conditions "$CONDITIONS" \
    --output-dir "$RUN_ROOT/$dataset" \
    --prepare-only \
    "${COMMON[@]}"
done

"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python preflight_qwen38_scale_selector.py \
  --dataset-root "$DATASET_ROOT" \
  --run-root "$RUN_ROOT" \
  --largest-run-root qwen38-fsod-runs/rf20-three-way-matched-v1 \
  --report "$RUN_ROOT/preflight.json"

if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "Scale-selector A/B preflight completed; inference intentionally skipped."
  exit 0
fi

for dataset in "${DATASETS[@]}"; do
  run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
    python evaluate_qwen38_recipe.py \
    --dataset-dir "$DATASET_ROOT/$dataset" \
    --conditions "$CONDITIONS" \
    --output-dir "$RUN_ROOT/$dataset" \
    "${COMMON[@]}" || exit $?
done

echo "Scale-selector median one-shot inference completed successfully."
