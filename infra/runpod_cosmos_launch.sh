#!/usr/bin/env bash
# Submit and inspect staged Cosmos3 RF100VL RunPod jobs.
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
    SHARD_MANIFEST_URI= SHARD_MANIFEST_SHA256= SHARD_ID=
    GPU_TYPE="NVIDIA H100 80GB HBM3" GPU_COUNT= DISK_GB= VOLUME_GB=
    REGISTRY_AUTH=${RUNPOD_REGISTRY_AUTH_ID:-}
    MODEL_ID=nvidia/Cosmos3-Edge MODEL_REVISION= TENSOR_PARALLEL_SIZE=
    CUDA_VERSIONS=${RUNPOD_CUDA_VERSIONS:-12.8,12.9,13.0}
    SPOT=false DRY_RUN=0 PREFLIGHT_APPROVED=0 ALLOW_INCOMPLETE_PREFLIGHT=0
    ALLOW_ADDITIONAL_POD=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --name) NAME=$2; shift 2 ;;
            --image) IMAGE=$2; shift 2 ;;
            --stage) STAGE=$2; shift 2 ;;
            --gcs-run-uri) GCS_RUN_URI=$2; shift 2 ;;
            --dataset-gcs-uri) DATASET_GCS_URI=$2; shift 2 ;;
            --smoke-dataset) SMOKE_DATASET=$2; shift 2 ;;
            --shard-manifest-uri) SHARD_MANIFEST_URI=$2; shift 2 ;;
            --shard-manifest-sha256) SHARD_MANIFEST_SHA256=$2; shift 2 ;;
            --shard-id) SHARD_ID=$2; shift 2 ;;
            --gpu-type) GPU_TYPE=$2; shift 2 ;;
            --gpus) GPU_COUNT=$2; shift 2 ;;
            --disk) DISK_GB=$2; shift 2 ;;
            --volume-size) VOLUME_GB=$2; shift 2 ;;
            --registry-auth) REGISTRY_AUTH=$2; shift 2 ;;
            --model-id) MODEL_ID=$2; shift 2 ;;
            --model-revision) MODEL_REVISION=$2; shift 2 ;;
            --tensor-parallel-size) TENSOR_PARALLEL_SIZE=$2; shift 2 ;;
            --cuda) CUDA_VERSIONS=$2; shift 2 ;;
            --spot) SPOT=true; shift ;;
            --preflight-approved) PREFLIGHT_APPROVED=1; shift ;;
            --allow-incomplete-preflight) ALLOW_INCOMPLETE_PREFLIGHT=1; shift ;;
            --allow-additional-pod) ALLOW_ADDITIONAL_POD=1; shift ;;
            --dry-run) DRY_RUN=1; shift ;;
            *) echo "ERROR: unknown flag $1" >&2; exit 1 ;;
        esac
    done
    : "${NAME:?--name is required}"
    : "${IMAGE:?--image is required}"
    : "${STAGE:?--stage preflight|full|shard is required}"
    : "${GCS_RUN_URI:?--gcs-run-uri is required}"
    : "${REGISTRY_AUTH:?--registry-auth or RUNPOD_REGISTRY_AUTH_ID is required}"
    case "${MODEL_ID}" in
        nvidia/Cosmos3-Edge)
            DEFAULT_MODEL_REVISION=2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2
            DEFAULT_GPU_COUNT=1
            DEFAULT_TENSOR_PARALLEL_SIZE=1
            DEFAULT_DISK_GB=100
            DEFAULT_VOLUME_GB=200
            MIN_DISK_GB=100
            MIN_VOLUME_GB=200
            ;;
        nvidia/Cosmos3-Super)
            DEFAULT_MODEL_REVISION=e0262be9d8f7586bc24c069a2aed2b665bdff266
            DEFAULT_GPU_COUNT=4
            DEFAULT_TENSOR_PARALLEL_SIZE=4
            DEFAULT_DISK_GB=120
            DEFAULT_VOLUME_GB=400
            MIN_DISK_GB=100
            MIN_VOLUME_GB=300
            ;;
        *)
            echo "ERROR: unsupported --model-id ${MODEL_ID}." >&2
            exit 1
            ;;
    esac
    MODEL_REVISION=${MODEL_REVISION:-${DEFAULT_MODEL_REVISION}}
    GPU_COUNT=${GPU_COUNT:-${DEFAULT_GPU_COUNT}}
    TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-${DEFAULT_TENSOR_PARALLEL_SIZE}}
    DISK_GB=${DISK_GB:-${DEFAULT_DISK_GB}}
    VOLUME_GB=${VOLUME_GB:-${DEFAULT_VOLUME_GB}}
    if [ "${STAGE}" != "preflight" ] && [ "${STAGE}" != "full" ] && \
       [ "${STAGE}" != "shard" ]; then
        echo "ERROR: --stage must be preflight, full, or shard." >&2
        exit 1
    fi
    if ! [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] || \
       ! [[ "${TENSOR_PARALLEL_SIZE}" =~ ^[0-9]+$ ]] || \
       [ "${GPU_COUNT}" != "${TENSOR_PARALLEL_SIZE}" ]; then
        echo "ERROR: --gpus must equal --tensor-parallel-size for the canonical benchmark." >&2
        exit 1
    fi
    if [ "${MODEL_ID}" = "nvidia/Cosmos3-Edge" ] && \
       [ "${TENSOR_PARALLEL_SIZE}" != "1" ]; then
        echo "ERROR: Cosmos3-Edge uses exactly one GPU in this benchmark." >&2
        exit 1
    fi
    if [ "${MODEL_ID}" = "nvidia/Cosmos3-Super" ] && \
       [ "${TENSOR_PARALLEL_SIZE}" != "4" ] && \
       [ "${TENSOR_PARALLEL_SIZE}" != "8" ]; then
        echo "ERROR: Cosmos3-Super requires tensor parallel size 4 or 8." >&2
        exit 1
    fi
    if ! [[ "${DISK_GB}" =~ ^[0-9]+$ ]] || [ "${DISK_GB}" -lt "${MIN_DISK_GB}" ]; then
        echo "ERROR: the Cosmos image requires --disk of at least ${MIN_DISK_GB} GB." >&2
        exit 1
    fi
    if ! [[ "${VOLUME_GB}" =~ ^[0-9]+$ ]] || [ "${VOLUME_GB}" -lt "${MIN_VOLUME_GB}" ]; then
        echo "ERROR: ${MODEL_ID}, RF100VL, and results require --volume-size of at least ${MIN_VOLUME_GB} GB." >&2
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
    if [ "${STAGE}" = "full" ] || [ "${STAGE}" = "shard" ]; then
        if [ "${PREFLIGHT_APPROVED}" != "1" ]; then
            echo "ERROR: ${STAGE} launch requires --preflight-approved after visual review." >&2
        exit 1
        fi
        if ! [[ "${IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
            echo "ERROR: ${STAGE} launch requires an immutable --image ...@sha256:DIGEST." >&2
            exit 1
        fi
    elif [ "${PREFLIGHT_APPROVED}" = "1" ]; then
        echo "ERROR: --preflight-approved is meaningful only for --stage full." >&2
        exit 1
    fi
    if [ "${ALLOW_INCOMPLETE_PREFLIGHT}" = "1" ] && \
       { { [ "${STAGE}" != "full" ] && [ "${STAGE}" != "shard" ]; } || \
         [ "${PREFLIGHT_APPROVED}" != "1" ]; }; then
        echo "ERROR: --allow-incomplete-preflight requires an approved full or shard run." >&2
        exit 1
    fi
    if [ "${STAGE}" = "shard" ]; then
        : "${SHARD_MANIFEST_URI:?--shard-manifest-uri is required for shard stage}"
        : "${SHARD_MANIFEST_SHA256:?--shard-manifest-sha256 is required for shard stage}"
        : "${SHARD_ID:?--shard-id is required for shard stage}"
        if ! [[ "${SHARD_MANIFEST_URI}" =~ ^gs://[^/]+/.+ ]]; then
            echo "ERROR: --shard-manifest-uri must be an exact gs:// object URI." >&2
            exit 1
        fi
        if ! [[ "${SHARD_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
            echo "ERROR: --shard-id contains unsafe characters." >&2
            exit 1
        fi
        if ! [[ "${SHARD_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
            echo "ERROR: --shard-manifest-sha256 must be 64 lowercase hex characters." >&2
            exit 1
        fi
    elif [ -n "${SHARD_MANIFEST_URI}" ] || [ -n "${SHARD_MANIFEST_SHA256}" ] || \
         [ -n "${SHARD_ID}" ]; then
        echo "ERROR: shard flags are valid only with --stage shard." >&2
        exit 1
    fi

    BODY=$(NAME="${NAME}" IMAGE="${IMAGE}" STAGE="${STAGE}" \
        GCS_RUN_URI="${GCS_RUN_URI}" DATASET_GCS_URI="${DATASET_GCS_URI}" \
        SMOKE_DATASET="${SMOKE_DATASET}" GPU_TYPE="${GPU_TYPE}" \
        SHARD_MANIFEST_URI="${SHARD_MANIFEST_URI}" \
        SHARD_MANIFEST_SHA256="${SHARD_MANIFEST_SHA256}" SHARD_ID="${SHARD_ID}" \
        GPU_COUNT="${GPU_COUNT}" DISK_GB="${DISK_GB}" VOLUME_GB="${VOLUME_GB}" \
        REGISTRY_AUTH="${REGISTRY_AUTH}" MODEL_REVISION="${MODEL_REVISION}" \
        MODEL_ID="${MODEL_ID}" TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
        CUDA_VERSIONS="${CUDA_VERSIONS}" SPOT="${SPOT}" \
        PREFLIGHT_APPROVED="${PREFLIGHT_APPROVED}" \
        ALLOW_INCOMPLETE_PREFLIGHT="${ALLOW_INCOMPLETE_PREFLIGHT}" python3 - <<'PY'
import json
import os

env = {
    "COSMOS_STAGE": os.environ["STAGE"],
    "COSMOS_GCS_RUN_URI": os.environ["GCS_RUN_URI"],
    "COSMOS_MODEL_ID": os.environ["MODEL_ID"],
    "COSMOS_MODEL_REVISION": os.environ["MODEL_REVISION"],
    "COSMOS_TENSOR_PARALLEL_SIZE": os.environ["TENSOR_PARALLEL_SIZE"],
    "COSMOS_EXPECTED_DATASETS": "100",
    "COSMOS_WORKERS": "1",
    "COSMOS_IMAGE_REF": os.environ["IMAGE"],
    "COSMOS_PREFLIGHT_APPROVED": os.environ["PREFLIGHT_APPROVED"],
    "COSMOS_ALLOW_INCOMPLETE_PREFLIGHT": os.environ["ALLOW_INCOMPLETE_PREFLIGHT"],
    "COSMOS_WORK_DIR": "/workspace/cosmos-runpod-work",
    "RF100VL_DIR": "/workspace/rf100-vl",
    "HF_HOME": "/workspace/huggingface",
    "HF_HUB_CACHE": "/workspace/huggingface/hub",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "RUNPOD_TERMINATE_ON_EXIT": "1",
    "RUNPOD_STOP_ON_EXIT": "1",
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
if os.environ.get("SHARD_MANIFEST_URI"):
    env["COSMOS_SHARD_MANIFEST_URI"] = os.environ["SHARD_MANIFEST_URI"]
    env["COSMOS_SHARD_MANIFEST_SHA256"] = os.environ["SHARD_MANIFEST_SHA256"]
    env["COSMOS_SHARD_ID"] = os.environ["SHARD_ID"]

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
