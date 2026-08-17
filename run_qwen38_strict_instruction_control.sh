#!/usr/bin/env bash
set -euo pipefail

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
DATASET_ROOT=${DATASET_ROOT:-RF100VL/rf20-vl-fsod-fresh-20260813}
RUN_ROOT=${RUN_ROOT:-qwen38-fsod-runs/instruction-study-v2/strict-permuted-six-dataset}
CONDITIONS=${CONDITIONS:-qwen38-fsod-configs/instruction-study-strict-permuted-control.json}
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
DATASETS=(
  actions
  all-elements
  defect-detection
  flir-camera-objects
  paper-parts
  water-meter
)

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is required for inference." >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"
"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python preflight_qwen38_strict_instruction_control.py \
  --dataset-root "$DATASET_ROOT" \
  --conditions "$CONDITIONS" \
  --report "$RUN_ROOT/preflight.json"

for dataset in "${DATASETS[@]}"; do
  readme="$DATASET_ROOT/$dataset/README.dataset.txt"
  [[ -s "$readme" ]] || { echo "Missing README: $readme" >&2; exit 1; }
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
    echo "$dataset strict-control resumable pass $attempt/$MAX_PASSES exited $status." >&2
    sleep 10
  done
  if [[ $status -ne 0 ]]; then
    echo "Unable to complete strict control for $dataset." >&2
    exit "$status"
  fi
done

"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python aggregate_qwen38_collection.py \
  --dataset-root "$DATASET_ROOT" \
  --run-root "$RUN_ROOT" \
  --conditions "$CONDITIONS" \
  --datasets "${DATASETS[@]}"

echo "Strict semantic instruction control succeeded."
