#!/usr/bin/env bash
set -uo pipefail

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is required." >&2
  exit 1
fi

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
OUTPUT_DIR=${OUTPUT_DIR:-qwen38-fsod-runs/dreidel-box-count-ablation-v1}

for attempt in 1 2 3 4 5 6; do
  "$UV_BIN" run --with-requirements requirements-cosmos.txt \
    python evaluate_qwen38_box_count_ablation.py \
    --dataset-dir RF100VL/rf20-vl-fsod/the-dreidel-project \
    --output-dir "$OUTPUT_DIR" \
    --concurrency 256 \
    --requests-per-minute 570 \
    --tokens-per-minute 900000 \
    --timeout-seconds 180 \
    --max-completion-tokens 8192 \
    --max-retries 3 && exit 0
  status=$?
  echo "Ablation pass ${attempt}/6 exited ${status}; resuming unresolved records." >&2
  sleep 10
done

exit "$status"
