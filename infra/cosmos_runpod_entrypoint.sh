#!/usr/bin/env bash
set -euo pipefail
cd "${COSMOS_BENCHMARK_ROOT:-/benchmark}"

WORK_DIR=${COSMOS_WORK_DIR:-/workspace/cosmos-runpod-work}
EVAL_PYTHON=${COSMOS_EVAL_PYTHON:-/opt/cosmos-eval/bin/python}
mkdir -p "${WORK_DIR}"
JOB_LOG="${WORK_DIR}/job.log"

# Account-level RunPod secrets arrive as environment variables. Materialize
# only the GCP service-account JSON, with owner-only permissions, for ADC.
if [ -n "${GCP_SA_JSON_B64:-}" ]; then
    umask 077
    printf '%s' "${GCP_SA_JSON_B64}" | base64 -d > /tmp/cosmos-gcp-sa.json
    chmod 600 /tmp/cosmos-gcp-sa.json
    export GOOGLE_APPLICATION_CREDENTIALS=/tmp/cosmos-gcp-sa.json
elif [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    echo "[entrypoint] ERROR: GCS credentials were not injected." >&2
    exit 1
fi

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    set +e

    local exit_record="${WORK_DIR}/job_exit.json"
    "${EVAL_PYTHON}" infra/write_job_exit.py \
        --path "${exit_record}" \
        --stage "${COSMOS_STAGE:-unknown}" \
        --exit-code "${rc}" \
        --git-sha "${BENCHMARK_GIT_SHA:-unknown}" \
        --image-ref "${COSMOS_IMAGE_REF:-unknown}"

    "${EVAL_PYTHON}" infra/gcs_io.py upload-if-possible \
        --root-uri "${COSMOS_GCS_RUN_URI:-}" \
        --source "${JOB_LOG}" \
        --relative-path "control/${COSMOS_STAGE:-unknown}/job.log"
    "${EVAL_PYTHON}" infra/gcs_io.py upload-if-possible \
        --root-uri "${COSMOS_GCS_RUN_URI:-}" \
        --source "${exit_record}" \
        --relative-path "control/${COSMOS_STAGE:-unknown}/job_exit.json"

    if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
        # Preserve the volume after preflight (for the human gate and full-run
        # reuse) and after every failure. Only a verified successful full run
        # may delete its pod after all durable artifacts have reached GCS.
        if [ "${rc}" -eq 0 ] && [ "${COSMOS_STAGE:-}" = "full" ] && \
           [ "${RUNPOD_TERMINATE_ON_EXIT:-1}" = "1" ]; then
            echo "[entrypoint] verified full workload succeeded; terminating pod"
            "${EVAL_PYTHON}" infra/runpod_self_terminate.py \
                "${RUNPOD_POD_ID}" terminate
        elif [ "${RUNPOD_STOP_ON_EXIT:-1}" = "1" ]; then
            echo "[entrypoint] workload exited rc=${rc}; stopping pod to preserve volume"
            "${EVAL_PYTHON}" infra/runpod_self_terminate.py \
                "${RUNPOD_POD_ID}" stop
        fi
    fi
    exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set +e
"${EVAL_PYTHON}" infra/run_cosmos_job.py 2>&1 | tee "${JOB_LOG}"
rc=${PIPESTATUS[0]}
set -e
exit "${rc}"
