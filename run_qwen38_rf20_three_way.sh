#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is required." >&2
  exit 1
fi

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
DATASET_ROOT=${DATASET_ROOT:-RF100VL/rf20-vl-fsod}
RUN_ROOT=${RUN_ROOT:-qwen38-fsod-runs/rf20-three-way-v1}
CONDITIONS=${CONDITIONS:-qwen38-fsod-configs/rf20-three-way.json}
MAX_PASSES=${MAX_PASSES:-6}
COMMON=(
  --concurrency 256
  --requests-per-minute 13500
  --tokens-per-minute 1800000
  --timeout-seconds 180
  --max-completion-tokens 8192
  --max-retries 3
  --temperature 0
)

mapfile -t datasets < <(find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
if [[ ${#datasets[@]} -ne 20 ]]; then
  echo "RF20 contract requires exactly 20 dataset directories; found ${#datasets[@]}." >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"
for dataset in "${datasets[@]}"; do
  status=1
  for ((attempt = 1; attempt <= MAX_PASSES; attempt++)); do
    if "$UV_BIN" run --with-requirements requirements-cosmos.txt \
      python evaluate_qwen38_recipe.py \
      --dataset-dir "$DATASET_ROOT/$dataset" \
      --conditions "$CONDITIONS" \
      --output-dir "$RUN_ROOT/$dataset" \
      "${COMMON[@]}"; then
      status=0
      break
    else
      status=$?
    fi
    echo "$dataset resumable pass $attempt/$MAX_PASSES exited $status." >&2
    sleep 10
  done
  if [[ $status -ne 0 ]]; then
    echo "Unable to complete $dataset after $MAX_PASSES passes." >&2
    exit "$status"
  fi
done

"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python aggregate_qwen38_rf20.py \
  --dataset-root "$DATASET_ROOT" \
  --run-root "$RUN_ROOT" \
  --conditions "$CONDITIONS"

echo "Complete RF20 three-way benchmark and aggregate validation succeeded."
