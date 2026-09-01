#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/matveipopov/vlms-rf100-vl/rf-100-vl-benchmark}
PREDICTION_UNIT=${PREDICTION_UNIT:-qwen38-perceptionbench-max-flash-xhigh-v1.service}
PREDICTION_MARKER=${PREDICTION_MARKER:-perceptionbench-runs/_QWEN38_MAX_FLASH_PREDICTIONS_COMPLETE}
PREDICTION_RESTART_LIMIT=${PREDICTION_RESTART_LIMIT:-5}

cd "$REPO_DIR"
echo "Waiting for the Max/Flash prediction queue."
restart_count=0
while [[ ! -f "$PREDICTION_MARKER" ]]; do
  if systemctl --user is-active --quiet "$PREDICTION_UNIT"; then
    sleep 30
    continue
  fi
  restart_count=$((restart_count + 1))
  if (( restart_count > PREDICTION_RESTART_LIMIT )); then
    echo "Prediction queue stopped without its completion marker after $PREDICTION_RESTART_LIMIT resumptions." >&2
    exit 2
  fi
  echo "[$(date -u +%FT%TZ)] prediction queue stopped early; resuming checkpointed work ($restart_count/$PREDICTION_RESTART_LIMIT)"
  systemctl --user reset-failed "$PREDICTION_UNIT" >/dev/null 2>&1 || true
  systemctl --user restart "$PREDICTION_UNIT"
  sleep 30
done

bash run_perceptionbench_gptoss_judge.sh
