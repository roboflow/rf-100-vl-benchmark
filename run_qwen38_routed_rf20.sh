#!/usr/bin/env bash
set -euo pipefail

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
DATASET_ROOT=${DATASET_ROOT:-RF100VL/rf20-vl-fsod-fresh-20260813}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
ROUTE_SUMMARY=${ROUTE_SUMMARY:?ROUTE_SUMMARY is required}
MAX_PASSES=${MAX_PASSES:-6}
REQUESTS_PER_MINUTE=${REQUESTS_PER_MINUTE:-6750}
TOKENS_PER_MINUTE=${TOKENS_PER_MINUTE:-900000}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

mkdir -p "$RUN_ROOT"
"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python preflight_qwen38_routed_rf20.py \
  --dataset-root "$DATASET_ROOT" \
  --route "$ROUTE_SUMMARY" \
  --report "$RUN_ROOT/preflight.json"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "Routed RF20 preflight completed; API calls skipped."
  exit 0
fi
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is required for inference." >&2
  exit 1
fi

mapfile -t datasets < <(find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
if [[ ${#datasets[@]} -ne 20 ]]; then
  echo "RF20 requires exactly 20 datasets; found ${#datasets[@]}." >&2
  exit 1
fi

for dataset in "${datasets[@]}"; do
  count=$(jq -er --arg dataset "$dataset" '.rows[] | select(.dataset == $dataset) | .selected_count' "$ROUTE_SUMMARY")
  config="qwen38-fsod-configs/calibrated-count-b$(printf '%02d' "$count").json"
  status=1
  for ((attempt = 1; attempt <= MAX_PASSES; attempt++)); do
    if "$UV_BIN" run --with-requirements requirements-cosmos.txt \
      python evaluate_qwen38_recipe.py \
      --dataset-dir "$DATASET_ROOT/$dataset" \
      --conditions "$config" \
      --output-dir "$RUN_ROOT/$dataset" \
      --concurrency 256 \
      --requests-per-minute "$REQUESTS_PER_MINUTE" \
      --tokens-per-minute "$TOKENS_PER_MINUTE" \
      --timeout-seconds 180 \
      --max-completion-tokens 8192 \
      --max-retries 3 \
      --temperature 0 \
      --allow-shared-reference-images; then
      status=0
      break
    else
      status=$?
    fi
    echo "$dataset routed pass $attempt/$MAX_PASSES exited $status; resuming." >&2
    sleep 10
  done
  if [[ $status -ne 0 ]]; then
    echo "Unable to complete $dataset after $MAX_PASSES passes." >&2
    exit "$status"
  fi
done

"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python aggregate_qwen38_routed_rf20.py \
  --dataset-root "$DATASET_ROOT" \
  --run-root "$RUN_ROOT" \
  --route "$ROUTE_SUMMARY"

echo "Complete routed RF20 benchmark succeeded."
