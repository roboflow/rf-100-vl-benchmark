#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DASHSCOPE_API_KEY:-}" && "${PREPARE_ONLY:-0}" != "1" ]]; then
  echo "DASHSCOPE_API_KEY is required." >&2
  exit 1
fi

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
DATASET_ROOT=${DATASET_ROOT:-RF100VL/rf20-vl-fsod-fresh-20260813}
RUN_ROOT=${RUN_ROOT:-qwen38-fsod-runs/large-dataset-box-count-v1}
CONDITIONS=${CONDITIONS:-qwen38-fsod-configs/large-dataset-box-count-five-repeat.json}
REPEAT_COUNT=${REPEAT_COUNT:-3}
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
)

if [[ "$REPEAT_COUNT" != "3" && "$REPEAT_COUNT" != "5" ]]; then
  echo "REPEAT_COUNT must be 3 or 5." >&2
  exit 1
fi

MODES=()
for ((repeat = 1; repeat <= REPEAT_COUNT; repeat++)); do
  for count in 01 02 05; do
    MODES+=("box_b${count}_repeat_$(printf '%02d' "$repeat")")
  done
done

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

if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "Large-dataset box-count preflight completed; inference intentionally skipped."
  exit 0
fi

for dataset in "${DATASETS[@]}"; do
  run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
    python evaluate_qwen38_recipe.py \
    --dataset-dir "$DATASET_ROOT/$dataset" \
    --conditions "$CONDITIONS" \
    --output-dir "$RUN_ROOT/$dataset" \
    --modes "${MODES[@]}" \
    "${COMMON[@]}" || exit $?
done

"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python analyze_qwen38_large_dataset_box_count.py \
  --repeat-count "$REPEAT_COUNT" \
  --run "paper-parts=$RUN_ROOT/paper-parts" \
  --run "actions=$RUN_ROOT/actions" \
  --run "defect-detection=$RUN_ROOT/defect-detection" \
  --output "$RUN_ROOT/box_count_summary_${REPEAT_COUNT}_repeats.json" \
  --csv "$RUN_ROOT/box_count_summary_${REPEAT_COUNT}_repeats.csv"

echo "Large-dataset box-count stage completed successfully."
