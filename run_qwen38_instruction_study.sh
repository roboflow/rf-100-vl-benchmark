#!/usr/bin/env bash
set -euo pipefail

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
DATASET_ROOT=${DATASET_ROOT:-RF100VL/rf20-vl-fsod-fresh-20260813}
RUN_ROOT=${RUN_ROOT:-qwen38-fsod-runs/instruction-study-v1}
FULL_CONDITIONS=${FULL_CONDITIONS:-qwen38-fsod-configs/rf20-instructions-only.json}
SUBSET_CONDITIONS=${SUBSET_CONDITIONS:-qwen38-fsod-configs/instruction-study-six-dataset-five-arm.json}
RATINGS=${RATINGS:-qwen38-fsod-configs/rf20-label-sufficiency-ratings.json}
MAX_PASSES=${MAX_PASSES:-6}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
FULL_ROOT=$RUN_ROOT/full-rf20
SUBSET_ROOT=$RUN_ROOT/matched-six-dataset
SMOKE_ROOT=$RUN_ROOT/_smoke-actions-image-0
COMMON=(
  --concurrency 256
  --requests-per-minute 13500
  --tokens-per-minute 1800000
  --timeout-seconds 180
  --max-completion-tokens 8192
  --max-retries 3
  --temperature 0
)
SUBSET_DATASETS=(
  actions
  all-elements
  defect-detection
  flir-camera-objects
  paper-parts
  water-meter
)

mapfile -t all_datasets < <(find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
if [[ ${#all_datasets[@]} -ne 20 ]]; then
  echo "RF20-VL-FSOD requires exactly 20 dataset directories; found ${#all_datasets[@]}." >&2
  exit 1
fi

mkdir -p "$RUN_ROOT" "$FULL_ROOT" "$SUBSET_ROOT"
"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python preflight_qwen38_instruction_study.py \
  --dataset-root "$DATASET_ROOT" \
  --full-conditions "$FULL_CONDITIONS" \
  --subset-conditions "$SUBSET_CONDITIONS" \
  --ratings "$RATINGS" \
  --report "$RUN_ROOT/preflight.json"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "Instruction-study preflight completed; API calls skipped."
  exit 0
fi
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is required for inference." >&2
  exit 1
fi

if [[ ! -f "$SMOKE_ROOT/_SUCCESS.json" ]]; then
  "$UV_BIN" run --with-requirements requirements-cosmos.txt \
    python evaluate_qwen38_recipe.py \
    --dataset-dir "$DATASET_ROOT/actions" \
    --conditions "$SUBSET_CONDITIONS" \
    --output-dir "$SMOKE_ROOT" \
    --image-ids 0 \
    "${COMMON[@]}"
fi
"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python - "$SMOKE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
progress = json.loads((root / "progress.json").read_text())
summary = json.loads((root / "comparison_summary.json").read_text())
if progress["total"] != {"total": 5, "success": 5, "model_failure": 0, "error": 0, "pending": 0}:
    raise SystemExit(f"Smoke progress failed: {progress['total']}")
if not all(row["complete"] and row["task_count"] == 1 for row in summary["rows"]):
    raise SystemExit("Smoke scoring contract failed.")
print("Five-arm live API smoke test passed.")
PY

for dataset in "${all_datasets[@]}"; do
  status=1
  for ((attempt = 1; attempt <= MAX_PASSES; attempt++)); do
    if "$UV_BIN" run --with-requirements requirements-cosmos.txt \
      python evaluate_qwen38_recipe.py \
      --dataset-dir "$DATASET_ROOT/$dataset" \
      --conditions "$FULL_CONDITIONS" \
      --output-dir "$FULL_ROOT/$dataset" \
      "${COMMON[@]}"; then
      status=0
      break
    else
      status=$?
    fi
    echo "$dataset instructions-only resumable pass $attempt/$MAX_PASSES exited $status." >&2
    sleep 10
  done
  if [[ $status -ne 0 ]]; then
    echo "Unable to complete instructions-only $dataset after $MAX_PASSES passes." >&2
    exit "$status"
  fi
done

"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python aggregate_qwen38_rf20.py \
  --dataset-root "$DATASET_ROOT" \
  --run-root "$FULL_ROOT" \
  --conditions "$FULL_CONDITIONS"

for dataset in "${SUBSET_DATASETS[@]}"; do
  status=1
  for ((attempt = 1; attempt <= MAX_PASSES; attempt++)); do
    if "$UV_BIN" run --with-requirements requirements-cosmos.txt \
      python evaluate_qwen38_recipe.py \
      --dataset-dir "$DATASET_ROOT/$dataset" \
      --conditions "$SUBSET_CONDITIONS" \
      --output-dir "$SUBSET_ROOT/$dataset" \
      "${COMMON[@]}"; then
      status=0
      break
    else
      status=$?
    fi
    echo "$dataset five-arm resumable pass $attempt/$MAX_PASSES exited $status." >&2
    sleep 10
  done
  if [[ $status -ne 0 ]]; then
    echo "Unable to complete five-arm $dataset after $MAX_PASSES passes." >&2
    exit "$status"
  fi
done

"$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python aggregate_qwen38_collection.py \
  --dataset-root "$DATASET_ROOT" \
  --run-root "$SUBSET_ROOT" \
  --conditions "$SUBSET_CONDITIONS" \
  --datasets "${SUBSET_DATASETS[@]}"

echo "Complete RF20 instruction study and matched six-dataset study succeeded."
