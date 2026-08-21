#!/usr/bin/env bash
set -euo pipefail

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
RUN_ROOT=${RUN_ROOT:-qwen38-fsod-runs/ssa-three-dataset-validation-v1}
BUDGET_USD=${BUDGET_USD:-20}
DATASETS=(the-dreidel-project orionproducts lacrosse-object-detection)

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is required for a potentially missing locked prefix." >&2
  exit 1
fi

while [[ ! -f "$RUN_ROOT/_COLLECTION_COMPLETE" ]]; do
  if ! tmux has-session -t qwen38-ssa-3ds 2>/dev/null; then
    echo "Collection exited without its completion marker." >&2
    exit 2
  fi
  sleep 10
done

"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python analyze_qwen38_ssa_study.py --run-root "$RUN_ROOT" \
  > "$RUN_ROOT/stopping-validation.log" 2>&1

for dataset in "${DATASETS[@]}"; do
  selected=$(jq -r --arg dataset "$dataset" \
    '.datasets[$dataset].decisions["1234"].selected_prefix' \
    "$RUN_ROOT/stopping_validation.json")
  already_in_grid=$(jq -r --arg dataset "$dataset" \
    '.datasets[$dataset].canonical_selected_grid_result != null' \
    "$RUN_ROOT/stopping_validation.json")
  if [[ "$already_in_grid" == "true" ]]; then
    echo "$dataset selected prefix $selected is already in the locked analysis grid."
    continue
  fi
  spent=$(find "$RUN_ROOT" -type f -name summary.json -print0 \
    | xargs -0 -r jq -r '.invocation_usage.estimated_usd // 0' \
    | awk '{total += $1} END {printf "%.6f", total + 0}')
  if ! awk -v spent="$spent" -v cap="$BUDGET_USD" 'BEGIN {exit !(spent < cap)}'; then
    echo "Skipping $dataset locked prefix: \$$spent has reached the \$$BUDGET_USD cap." >&2
    continue
  fi
  "$UV_BIN" run --with-requirements requirements-cosmos.txt \
    python evaluate_qwen38_ssa_selected_prefix.py \
      --run-root "$RUN_ROOT" --dataset "$dataset" --seed 1234
done

touch "$RUN_ROOT/_VALIDATION_COMPLETE"
