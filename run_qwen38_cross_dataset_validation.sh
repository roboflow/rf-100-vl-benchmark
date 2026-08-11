#!/usr/bin/env bash
set -uo pipefail

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is required." >&2
  exit 1
fi

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}

run_resumable() {
  local attempt status
  for attempt in 1 2 3 4; do
    "$@" && return 0
    status=$?
    echo "Invocation failed with status ${status}; resumable pass ${attempt}/4." >&2
    sleep 5
  done
  return "$status"
}

run_dataset() {
  local dataset_name=$1
  local pair_file=$2
  local run_name=$3
  local dataset_dir="RF100VL/rf20-vl-fsod/${dataset_name}"
  local base_dir="qwen38-fsod-runs/${run_name}-selected-base-v1"
  local combined_dir="qwen38-fsod-runs/${run_name}-selected-combined-v1"

  run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
    python evaluate_qwen38_orion.py \
    --dataset-dir "$dataset_dir" \
    --negative-pairs-file "$pair_file" \
    --output-dir "$base_dir" \
    --modes multi_class_names positive_numeric positive_drawn \
      positive_negative_numeric \
    --reasoning-effort none \
    --concurrency 256 \
    --requests-per-minute 570 \
    --tokens-per-minute 900000 \
    --timeout-seconds 180 \
    --max-completion-tokens 8192 \
    --max-retries 3 || return

  run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
    python evaluate_qwen38_orion_subset.py \
    --subset full \
    --dataset-dir "$dataset_dir" \
    --negative-pairs-file "$pair_file" \
    --existing-full-run-none "$base_dir" \
    --output-dir "$combined_dir" \
    --new-modes multi_class_positive_numeric \
    --reasoning-efforts none low \
    --concurrency 100 \
    --timeout-seconds 180 \
    --max-completion-tokens 8192 \
    --max-retries 3
}

run_dataset \
  lacrosse-object-detection \
  qwen38-fsod-configs/lacrosse-object-detection-negative-pairs.json \
  lacrosse || exit $?

run_dataset \
  the-dreidel-project \
  qwen38-fsod-configs/the-dreidel-project-negative-pairs.json \
  dreidel || exit $?
