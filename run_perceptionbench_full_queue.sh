#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/matveipopov/vlms-rf100-vl/rf-100-vl-benchmark}
PREDICTION_UNIT=${PREDICTION_UNIT:-qwen38-perceptionbench-max-flash-xhigh-v1.service}
PREDICTION_MARKER=${PREDICTION_MARKER:-perceptionbench-runs/_QWEN38_MAX_FLASH_PREDICTIONS_COMPLETE}

cd "$REPO_DIR"
echo "Waiting for the Max/Flash prediction queue."
while systemctl --user is-active --quiet "$PREDICTION_UNIT"; do
  sleep 30
done
if [[ ! -f "$PREDICTION_MARKER" ]]; then
  echo "Prediction queue stopped without its completion marker." >&2
  exit 2
fi

bash run_perceptionbench_gptoss_judge.sh
