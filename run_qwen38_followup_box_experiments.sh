#!/usr/bin/env bash
set -uo pipefail

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is required." >&2
  exit 1
fi

UV_BIN=${UV_BIN:-/home/matveipopov/.local/bin/uv}
EXEMPLAR_OUTPUT=${EXEMPLAR_OUTPUT:-qwen38-fsod-runs/dreidel-exemplar-only-box-v1}
ORION_OUTPUT=${ORION_OUTPUT:-qwen38-fsod-runs/orion-box-count-ablation-v1}

run_resumable() {
  local attempt status
  for attempt in 1 2 3 4 5 6; do
    "$@" && return 0
    status=$?
    echo "Invocation exited ${status}; resumable pass ${attempt}/6." >&2
    sleep 10
  done
  return "$status"
}

# Run the smaller, novel semantic-label-free experiment first so its result is
# available quickly. The second command then receives the full account quota.
run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python evaluate_qwen38_exemplar_only_ablation.py \
  --dataset-dir RF100VL/rf20-vl-fsod/the-dreidel-project \
  --output-dir "$EXEMPLAR_OUTPUT" \
  --concurrency 256 \
  --requests-per-minute 570 \
  --tokens-per-minute 900000 \
  --timeout-seconds 180 \
  --max-completion-tokens 8192 \
  --max-retries 3 || exit $?

run_resumable "$UV_BIN" run --with-requirements requirements-cosmos.txt \
  python evaluate_qwen38_box_count_ablation.py \
  --dataset-dir RF100VL/rf20-vl-fsod/orionproducts \
  --output-dir "$ORION_OUTPUT" \
  --allow-shared-reference-images \
  --concurrency 256 \
  --requests-per-minute 570 \
  --tokens-per-minute 900000 \
  --timeout-seconds 180 \
  --max-completion-tokens 8192 \
  --max-retries 3
