#!/usr/bin/env bash
set -euo pipefail

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
DATASET_ROOT=${DATASET_ROOT:-RF100VL/rf20-vl-fsod-fresh-20260813}
CALIBRATION_ROOT=${CALIBRATION_ROOT:-qwen38-fsod-runs/reference-count-calibration-v1}
RUN_ROOT=${RUN_ROOT:-qwen38-fsod-runs/reference-count-calibrated-test-v1}
MAX_PASSES=${MAX_PASSES:-6}
REQUESTS_PER_MINUTE=${REQUESTS_PER_MINUTE:-6750}
TOKENS_PER_MINUTE=${TOKENS_PER_MINUTE:-900000}

mkdir -p "$CALIBRATION_ROOT" "$RUN_ROOT"
status=1
for ((attempt = 1; attempt <= MAX_PASSES; attempt++)); do
  if "$UV_BIN" run --with-requirements requirements-cosmos.txt \
    python evaluate_qwen38_reference_count_calibration.py \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$CALIBRATION_ROOT" \
    --minimum-gain-points 2 \
    --concurrency 256 \
    --requests-per-minute "$REQUESTS_PER_MINUTE" \
    --tokens-per-minute "$TOKENS_PER_MINUTE" \
    --timeout-seconds 180 \
    --max-completion-tokens 8192 \
    --max-retries 3; then
    status=0
    break
  else
    status=$?
  fi
  echo "Count calibration pass $attempt/$MAX_PASSES exited $status; resuming." >&2
  sleep 10
done
if [[ $status -ne 0 ]]; then
  echo "Unable to complete count calibration after $MAX_PASSES passes." >&2
  exit "$status"
fi

RUN_ROOT="$RUN_ROOT" ROUTE_SUMMARY="$CALIBRATION_ROOT/summary.json" \
DATASET_ROOT="$DATASET_ROOT" REQUESTS_PER_MINUTE="$REQUESTS_PER_MINUTE" \
TOKENS_PER_MINUTE="$TOKENS_PER_MINUTE" MAX_PASSES="$MAX_PASSES" \
  bash run_qwen38_routed_rf20.sh
