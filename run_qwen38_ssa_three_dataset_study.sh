#!/usr/bin/env bash
set -euo pipefail

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
DATA_ROOT=${DATA_ROOT:-RF100VL/rf20-vl-fsod-fresh-20260813}
RUN_ROOT=${RUN_ROOT:-qwen38-fsod-runs/ssa-three-dataset-validation-v1}
BUDGET_USD=${BUDGET_USD:-20}
INFERENCE_SEED=${INFERENCE_SEED:-1234}
ORDER_SEEDS=(1234 4321 2026)
DATASETS=(the-dreidel-project orionproducts lacrosse-object-detection)

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is required for inference." >&2
  exit 1
fi

spent_usd() {
  find "$RUN_ROOT" -type f -name summary.json -print0 2>/dev/null \
    | xargs -0 -r jq -r '.invocation_usage.estimated_usd // 0' \
    | awk '{total += $1} END {printf "%.6f", total + 0}'
}

assert_budget() {
  local spent
  spent=$(spent_usd)
  if ! awk -v spent="$spent" -v cap="$BUDGET_USD" 'BEGIN {exit !(spent < cap)}'; then
    echo "Study budget reached: \$$spent spent; cap is \$$BUDGET_USD." >&2
    exit 2
  fi
  echo "research spend checkpoint: \$$spent / \$$BUDGET_USD"
}

run_dataset_seed() {
  local dataset=$1
  local seed=$2
  local adaptation_only=$3
  local dataset_dir="$DATA_ROOT/$dataset"
  local output_dir="$RUN_ROOT/$dataset/seed-$seed"
  local zero_cache="$RUN_ROOT/$dataset/zero-cache"
  local support_images
  support_images=$(jq '[.annotations[].image_id] | unique | length' \
    "$dataset_dir/train/_annotations.coco.json")
  local args=(
    --dataset-dir "$dataset_dir"
    --output-dir "$output_dir"
    --seed "$seed"
    --inference-seed "$INFERENCE_SEED"
    --zero-cache-dir "$zero_cache"
    --test-prefixes 0 1 2 4 8 "$support_images"
    --concurrency 64
    --requests-per-minute 6750
    --tokens-per-minute 900000
    --timeout-seconds 180
    --max-completion-tokens 8192
    --max-retries 3
  )
  if [[ "$adaptation_only" == "1" ]]; then
    args+=(--adaptation-only)
  fi
  "$UV_BIN" run --with-requirements requirements-cosmos.txt \
    python evaluate_qwen38_ssa.py "${args[@]}"
  assert_budget
}

mkdir -p "$RUN_ROOT"

# The canonical order supplies the one permitted analysis grid per dataset and
# warms every order-independent zero-shot support probe. Later order seeds only
# collect adaptation curves, so no test request is duplicated.
for dataset in "${DATASETS[@]}"; do
  assert_budget
  run_dataset_seed "$dataset" 1234 0
done

for seed in "${ORDER_SEEDS[@]:1}"; do
  for dataset in "${DATASETS[@]}"; do
    assert_budget
    run_dataset_seed "$dataset" "$seed" 1
  done
done

touch "$RUN_ROOT/_COLLECTION_COMPLETE"
echo "Three-dataset SSA collection complete. Actual API spend: \$(spent_usd)"
