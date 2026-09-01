#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/matveipopov/vlms-rf100-vl/rf-100-vl-benchmark}
PYTHON_BIN=${PYTHON_BIN:-/home/matveipopov/.pyenv/versions/3.12.0/bin/python}
DATA_DIR=${DATA_DIR:-PerceptionBench}
RUN_ROOT=${RUN_ROOT:-perceptionbench-runs}
CONCURRENCY=${CONCURRENCY:-16}
SMOKE_INDICES=${SMOKE_INDICES:-0,10,2999}
QUEUE_LOG=${QUEUE_LOG:-$RUN_ROOT/qwen38-max-flash-xhigh-v1.queue.log}

cd "$REPO_DIR"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$QUEUE_LOG") 2>&1

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  if [[ ! -f /tmp/qwen38_extract_current_key.py ]]; then
    echo "DASHSCOPE_API_KEY is required and the secure local loader is absent." >&2
    exit 2
  fi
  DASHSCOPE_API_KEY=$(/usr/bin/python3 /tmp/qwen38_extract_current_key.py)
  export DASHSCOPE_API_KEY
fi

echo "[$(date -u +%FT%TZ)] validating code and pinned benchmark data"
"$PYTHON_BIN" -m pytest -q test_evaluate_qwen38_perceptionbench.py
"$PYTHON_BIN" prepare_perceptionbench.py --output-dir "$DATA_DIR"

models=(qwen3.8-flash qwen3.8-max)
for model in "${models[@]}"; do
  run_dir="$RUN_ROOT/$model-xhigh-v1"
  echo "[$(date -u +%FT%TZ)] API smoke for $model on records $SMOKE_INDICES"
  "$PYTHON_BIN" evaluate_qwen38_perceptionbench.py predict \
    --data-dir "$DATA_DIR" \
    --run-dir "$run_dir" \
    --model "$model" \
    --indices "$SMOKE_INDICES" \
    --concurrency 3
  for index in ${SMOKE_INDICES//,/ }; do
    checkpoint=$(printf '%s/predictions/%04d.json' "$run_dir" "$index")
    jq -e '
      .status == "complete"
      and ((.prediction // "") | length > 0)
      and (
        (.reasoning_characters_observed // 0) > 0
        or (.usage.completion_tokens_details.reasoning_tokens // 0) > 0
      )
    ' "$checkpoint" >/dev/null
  done
done

echo "[$(date -u +%FT%TZ)] both model/API smoke tests passed"

for model in "${models[@]}"; do
  run_dir="$RUN_ROOT/$model-xhigh-v1"
  echo "[$(date -u +%FT%TZ)] running/resuming all PerceptionBench predictions for $model"
  "$PYTHON_BIN" evaluate_qwen38_perceptionbench.py predict \
    --data-dir "$DATA_DIR" \
    --run-dir "$run_dir" \
    --model "$model" \
    --concurrency "$CONCURRENCY"

  for retry_round in 1 2 3; do
    "$PYTHON_BIN" evaluate_qwen38_perceptionbench.py status \
      --data-dir "$DATA_DIR" --run-dir "$run_dir" --model "$model" >/dev/null
    failure_count=$(jq -r '.prediction_status.model_failure // 0' "$run_dir/status.json")
    if [[ "$failure_count" == "0" ]]; then
      break
    fi
    echo "[$(date -u +%FT%TZ)] retrying $failure_count failed $model predictions, round $retry_round"
    "$PYTHON_BIN" evaluate_qwen38_perceptionbench.py predict \
      --data-dir "$DATA_DIR" \
      --run-dir "$run_dir" \
      --model "$model" \
      --concurrency "$CONCURRENCY" \
      --retry-failures
  done

  "$PYTHON_BIN" evaluate_qwen38_perceptionbench.py status \
    --data-dir "$DATA_DIR" --run-dir "$run_dir" --model "$model"
  jq -e '
    ((.prediction_status.complete // 0) + (.prediction_status.model_failure // 0)) == 3000
    and (.prediction_status.missing // 0) == 0
  ' "$run_dir/status.json" >/dev/null
done

touch "$RUN_ROOT/_QWEN38_MAX_FLASH_PREDICTIONS_COMPLETE"
echo "[$(date -u +%FT%TZ)] all Max and Flash predictions are checkpointed"

if [[ -n "${PERCEPTIONBENCH_JUDGE_API_KEY:-${OPENAI_API_KEY:-}}" \
      && -n "${PERCEPTIONBENCH_JUDGE_BASE_URL:-}" ]]; then
  echo "[$(date -u +%FT%TZ)] paper-matched judge credentials found; judging both runs"
  for model in "${models[@]}"; do
    run_dir="$RUN_ROOT/$model-xhigh-v1"
    "$PYTHON_BIN" evaluate_qwen38_perceptionbench.py judge \
      --data-dir "$DATA_DIR" --run-dir "$run_dir" --concurrency "$CONCURRENCY"
    "$PYTHON_BIN" evaluate_qwen38_perceptionbench.py score \
      --data-dir "$DATA_DIR" --run-dir "$run_dir"
  done
  touch "$RUN_ROOT/_QWEN38_MAX_FLASH_EVALUATION_COMPLETE"
else
  echo "[$(date -u +%FT%TZ)] predictions complete; exact gpt-oss-120b judge endpoint is not configured"
fi
