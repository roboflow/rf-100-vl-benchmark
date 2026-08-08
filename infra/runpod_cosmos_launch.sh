#!/usr/bin/env bash
# Submit and inspect staged Cosmos3-Edge RF100VL RunPod jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ -f infra/runpod.env ] && source infra/runpod.env

API=https://rest.runpod.io/v1

rp() {
    local method=$1 path=$2 body=${3:-}
    : "${RUNPOD_API_KEY:?export RUNPOD_API_KEY or place it in gitignored infra/runpod.env}"
    if [ -n "${body}" ]; then
        curl -fsS -X "${method}" "${API}${path}" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
            -H 'Content-Type: application/json' \
            -d "${body}"
    else
        curl -fsS -X "${method}" "${API}${path}" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}"
    fi
}

pretty() { python3 -m json.tool; }

command=${1:-help}
[ $# -gt 0 ] && shift

case "${command}" in
launch)
    NAME= IMAGE= STAGE= GCS_RUN_URI= DATASET_GCS_URI= SMOKE_DATASET=
    GPU_TYPE="NVIDIA H100 80GB HBM3" GPU_COUNT=1 DISK_GB=100 VOLUME_GB=200
    REGISTRY_AUTH=${RUNPOD_REGISTRY_AUTH_ID:-}
    MODEL_REVISION=2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2
    CUDA_VERSIONS=${RUNPOD_CUDA_VERSIONS:-12.8,12.9,13.0}
    SPOT=false DRY_RUN=0 PREFLIGHT_APPROVED=0 ALLOW_ADDITIONAL_POD=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --name) NAME=$2; shift 2 ;;
            --image) IMAGE=$2; shift 2 ;;
            --stage) STAGE=$2; shift 2 ;;
            --gcs-run-uri) GCS_RUN_URI=$2; shift 2 ;;
            --dataset-gcs-uri) DATASET_GCS_URI=$2; shift 2 ;;
            --smoke-dataset) SMOKE_DATASET=$2; shift 2 ;;
            --gpu-type) GPU_TYPE=$2; shift 2 ;;
            --gpus) GPU_COUNT=$2; shift 2 ;;
            --disk) DISK_GB=$2; shift 2 ;;
            --volume-size) VOLUME_GB=$2; shift 2 ;;
            --registry-auth) REGISTRY_AUTH=$2; shift 2 ;;
            --model-revision) MODEL_REVISION=$2; shift 2 ;;
            --cuda) CUDA_VERSIONS=$2; shift 2 ;;
            --spot) SPOT=true; shift ;;
            --preflight-approved) PREFLIGHT_APPROVED=1; shift ;;
            --allow-additional-pod) ALLOW_ADDITIONAL_POD=1; shift ;;
            --dry-run) DRY_RUN=1; shift ;;
            *) echo "ERROR: unknown flag $1" >&2; exit 1 ;;
        esac
    done
    : "${NAME:?--name is required}"
    : "${IMAGE:?--image is required}"
    : "${STAGE:?--stage preflight|full is required}"
    : "${GCS_RUN_URI:?--gcs-run-uri is required}"
    : "${REGISTRY_AUTH:?--registry-auth or RUNPOD_REGISTRY_AUTH_ID is required}"
    if [ "${STAGE}" != "preflight" ] && [ "${STAGE}" != "full" ]; then
        echo "ERROR: --stage must be preflight or full." >&2
        exit 1
    fi
    if [ "${GPU_COUNT}" != "1" ]; then
        echo "ERROR: the canonical benchmark is single-GPU; use --gpus 1." >&2
        exit 1
    fi
    if ! [[ "${GCS_RUN_URI}" =~ ^gs://[^/]+/.+ ]]; then
        echo "ERROR: --gcs-run-uri must be a run-specific gs://bucket/prefix." >&2
        exit 1
    fi
    if ! [[ "${MODEL_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
        echo "ERROR: --model-revision must be a full 40-character commit SHA." >&2
        exit 1
    fi
    if [ "${STAGE}" = "full" ]; then
        if [ "${PREFLIGHT_APPROVED}" != "1" ]; then
            echo "ERROR: full launch requires --preflight-approved after visual review." >&2
            exit 1
        fi
        if ! [[ "${IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
            echo "ERROR: full launch requires an immutable --image ...@sha256:DIGEST." >&2
            exit 1
        fi
    elif [ "${PREFLIGHT_APPROVED}" = "1" ]; then
        echo "ERROR: --preflight-approved is meaningful only for --stage full." >&2
        exit 1
    fi

    BODY=$(NAME="${NAME}" IMAGE="${IMAGE}" STAGE="${STAGE}" \
        GCS_RUN_URI="${GCS_RUN_URI}" DATASET_GCS_URI="${DATASET_GCS_URI}" \
        SMOKE_DATASET="${SMOKE_DATASET}" GPU_TYPE="${GPU_TYPE}" \
        GPU_COUNT="${GPU_COUNT}" DISK_GB="${DISK_GB}" VOLUME_GB="${VOLUME_GB}" \
        REGISTRY_AUTH="${REGISTRY_AUTH}" MODEL_REVISION="${MODEL_REVISION}" \
        CUDA_VERSIONS="${CUDA_VERSIONS}" SPOT="${SPOT}" \
        PREFLIGHT_APPROVED="${PREFLIGHT_APPROVED}" python3 - <<'PY'
import json
import os

env = {
    "COSMOS_STAGE": os.environ["STAGE"],
    "COSMOS_GCS_RUN_URI": os.environ["GCS_RUN_URI"],
    "COSMOS_MODEL_ID": "nvidia/Cosmos3-Edge",
    "COSMOS_MODEL_REVISION": os.environ["MODEL_REVISION"],
    "COSMOS_EXPECTED_DATASETS": "100",
    "COSMOS_WORKERS": "1",
    "COSMOS_IMAGE_REF": os.environ["IMAGE"],
    "COSMOS_PREFLIGHT_APPROVED": os.environ["PREFLIGHT_APPROVED"],
    "COSMOS_WORK_DIR": "/workspace/cosmos-runpod-work",
    "RF100VL_DIR": "/workspace/rf100-vl",
    "HF_HOME": "/workspace/huggingface",
    "HF_HUB_CACHE": "/workspace/huggingface/hub",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "RUNPOD_TERMINATE_ON_EXIT": "1",
    "HF_TOKEN": "{{ RUNPOD_SECRET_HF_TOKEN }}",
    "GCP_SA_JSON_B64": "{{ RUNPOD_SECRET_GCP_SA_JSON_B64 }}",
    "RUNPOD_API_KEY": "{{ RUNPOD_SECRET_POD_API_KEY }}",
}
if os.environ.get("DATASET_GCS_URI"):
    env["RF100VL_GCS_URI"] = os.environ["DATASET_GCS_URI"]
else:
    env["ROBOFLOW_API_KEY"] = "{{ RUNPOD_SECRET_ROBOFLOW_API_KEY }}"
if os.environ.get("SMOKE_DATASET"):
    env["COSMOS_SMOKE_DATASET"] = os.environ["SMOKE_DATASET"]

body = {
    "name": os.environ["NAME"],
    "imageName": os.environ["IMAGE"],
    "containerRegistryAuthId": os.environ["REGISTRY_AUTH"],
    "cloudType": "SECURE",
    "computeType": "GPU",
    "gpuTypeIds": os.environ["GPU_TYPE"].split(","),
    "gpuCount": int(os.environ["GPU_COUNT"]),
    "containerDiskInGb": int(os.environ["DISK_GB"]),
    "volumeInGb": int(os.environ["VOLUME_GB"]),
    "volumeMountPath": "/workspace",
    "interruptible": os.environ["SPOT"] == "true",
    "allowedCudaVersions": os.environ["CUDA_VERSIONS"].split(","),
    "dockerStartCmd": ["bash", "infra/cosmos_runpod_entrypoint.sh"],
    "env": env,
}
print(json.dumps(body, indent=2))
PY
)

    echo "[runpod] validated pod specification:"
    echo "${BODY}"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[runpod] dry run only; no pod submitted"
        exit 0
    fi

    if [ "${ALLOW_ADDITIONAL_POD}" != "1" ]; then
        ACTIVE=$(rp GET /pods | python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d) if isinstance(d,list) else 0)')
        if [ "${ACTIVE}" -ge 2 ]; then
            echo "ERROR: ${ACTIVE} pods are already active; refusing another without --allow-additional-pod." >&2
            exit 1
        fi
    fi
    RESPONSE=$(rp POST /pods "${BODY}")
    echo "${RESPONSE}" | pretty
    POD_ID=$(echo "${RESPONSE}" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("id", ""))')
    if [ -z "${POD_ID}" ]; then
        echo "ERROR: RunPod did not return a pod ID." >&2
        exit 1
    fi
    echo "[runpod] submitted pod ${POD_ID}"
    ;;
list) rp GET /pods | pretty ;;
status) rp GET "/pods/${1:?usage: status POD_ID}" | pretty ;;
terminate) rp DELETE "/pods/${1:?usage: terminate POD_ID}" | pretty ;;
registry-auths) rp GET /containerregistryauth | pretty ;;
help|*)
    sed -n '1,34p' "${BASH_SOURCE[0]}"
    ;;
esac
