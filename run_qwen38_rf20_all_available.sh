#!/usr/bin/env bash
set -euo pipefail

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
DATASET_ROOT=${DATASET_ROOT:-RF100VL/rf20-vl-fsod-fresh-20260813}
RUN_ROOT=${RUN_ROOT:-qwen38-fsod-runs/rf20-all-available-explicit-sparse-v1}
CONDITIONS=${CONDITIONS:-qwen38-fsod-configs/rf20-all-available-explicit-sparse.json}
MAX_PASSES=${MAX_PASSES:-6}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
SMOKE_ONLY=${SMOKE_ONLY:-0}
PRIORITY_DATASET=${PRIORITY_DATASET:-}
SMOKE_ROOT=$RUN_ROOT/_smoke-paper-parts-image-0
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

mapfile -t datasets < <(find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
if [[ ${#datasets[@]} -ne 20 ]]; then
  echo "RF20-VL-FSOD requires exactly 20 dataset directories; found ${#datasets[@]}." >&2
  exit 1
fi
if [[ -n "$PRIORITY_DATASET" ]]; then
  found_priority=0
  reordered=("$PRIORITY_DATASET")
  for dataset in "${datasets[@]}"; do
    if [[ "$dataset" == "$PRIORITY_DATASET" ]]; then
      found_priority=1
    else
      reordered+=("$dataset")
    fi
  done
  if [[ $found_priority -ne 1 ]]; then
    echo "Unknown priority dataset: $PRIORITY_DATASET" >&2
    exit 1
  fi
  datasets=("${reordered[@]}")
fi
echo "Dataset queue: ${datasets[*]}"

mkdir -p "$RUN_ROOT"
"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python preflight_qwen38_rf20_all_available.py \
  --dataset-root "$DATASET_ROOT" \
  --conditions "$CONDITIONS" \
  --report "$RUN_ROOT/preflight.json"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "RF20-VL-FSOD all-available preflight completed; API calls skipped."
  exit 0
fi
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is required for inference." >&2
  exit 1
fi

if [[ ! -f "$SMOKE_ROOT/_SUCCESS.json" ]]; then
  "$UV_BIN" run --with-requirements requirements-cosmos.txt \
    python evaluate_qwen38_recipe.py \
    --dataset-dir "$DATASET_ROOT/paper-parts" \
    --conditions "$CONDITIONS" \
    --output-dir "$SMOKE_ROOT" \
    --image-ids 0 \
    "${COMMON[@]}"
fi
"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python validate_qwen38_all_available_smoke.py \
  --run-root "$SMOKE_ROOT" | tee "$SMOKE_ROOT/smoke_validation.json"

if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "Maximum-context Paper Parts smoke test passed; full inference skipped."
  exit 0
fi

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

echo "Complete RF20-VL-FSOD all-available benchmark and aggregate validation succeeded."
