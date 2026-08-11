#!/usr/bin/env bash
set -uo pipefail

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is required." >&2
  exit 1
fi

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
OUTPUT_DIR=${OUTPUT_DIR:-qwen38-fsod-runs/dreidel-exemplar-only-box-b07-b10-v1}

status=1
for attempt in 1 2 3 4 5 6; do
  if "$UV_BIN" run --with-requirements requirements-cosmos.txt \
    python evaluate_qwen38_exemplar_only_ablation.py \
    --dataset-dir RF100VL/rf20-vl-fsod/the-dreidel-project \
    --output-dir "$OUTPUT_DIR" \
    --box-counts 7 10 \
    --concurrency 256 \
    --requests-per-minute 570 \
    --tokens-per-minute 900000 \
    --timeout-seconds 180 \
    --max-completion-tokens 8192 \
    --max-retries 3; then
    status=0
    break
  else
    status=$?
  fi
  echo "Extension pass ${attempt}/6 exited ${status}; resuming unresolved records." >&2
  sleep 10
done

if [[ $status -ne 0 ]]; then
  exit "$status"
fi

"$UV_BIN" run python combine_qwen38_exemplar_results.py \
  --base-dir qwen38-fsod-runs/dreidel-exemplar-only-box-v1 \
  --extension-dir "$OUTPUT_DIR" \
  --output-dir qwen38-fsod-runs/dreidel-exemplar-only-box-combined-v1
