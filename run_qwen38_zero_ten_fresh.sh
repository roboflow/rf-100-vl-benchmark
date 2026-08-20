#!/usr/bin/env bash
set -euo pipefail

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
RUN_ROOT=${RUN_ROOT:-qwen38-fsod-runs/support-calibrated-zero-ten-fresh-v1}
ROUTE_SUMMARY=$RUN_ROOT/route_summary.json

mkdir -p "$RUN_ROOT"
if [[ ! -f "$ROUTE_SUMMARY" ]]; then
  "$UV_BIN" run --with-requirements requirements-cosmos.txt \
    python prepare_qwen38_zero_ten_route.py \
    --pilot qwen38-fsod-runs/support-calibrated-router-pilot-v2/summary.json \
    --heldout qwen38-fsod-runs/support-calibrated-router-heldout14-v1/summary.json \
    --output "$ROUTE_SUMMARY"
fi

RUN_ROOT="$RUN_ROOT" ROUTE_SUMMARY="$ROUTE_SUMMARY" \
  bash run_qwen38_routed_rf20.sh
