#!/usr/bin/env bash
set -euo pipefail

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
DATASET_DIR=${DATASET_DIR:-RF100VL/rf20-vl-fsod-fresh-20260813/the-dreidel-project}
RUN_ROOT=${RUN_ROOT:-qwen38-fsod-runs/ssa-dreidel-pilot-v1}
PREPARE_ONLY=${PREPARE_ONLY:-0}

args=(
  --dataset-dir "$DATASET_DIR"
  --output-dir "$RUN_ROOT"
  --seed 1234
  --max-support-turns 12
  --test-prefixes 0 1 2 4 8 12
  --test-image-limit 20
  --concurrency 64
  --requests-per-minute 6750
  --tokens-per-minute 900000
  --timeout-seconds 180
  --max-completion-tokens 8192
  --max-retries 3
)

if [[ "$PREPARE_ONLY" == "1" ]]; then
  args+=(--prepare-only)
elif [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is required for inference." >&2
  exit 1
fi

exec "$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python evaluate_qwen38_ssa.py "${args[@]}"
