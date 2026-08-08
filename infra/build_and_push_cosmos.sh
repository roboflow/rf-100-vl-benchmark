#!/usr/bin/env bash
# Build the exact benchmark image and push it to the configured registry.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TAG=${1:?usage: COSMOS_IMAGE_REPO=REGISTRY/REPO/IMAGE bash infra/build_and_push_cosmos.sh TAG}
: "${COSMOS_IMAGE_REPO:?export COSMOS_IMAGE_REPO as the full registry/repository/image path}"
if ! [[ "${TAG}" =~ ^[a-z0-9][a-z0-9._-]{0,127}$ ]]; then
    echo "ERROR: invalid lowercase image tag: ${TAG}" >&2
    exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet || \
   [ -n "$(git ls-files --others --exclude-standard)" ]; then
    echo "ERROR: build only from a clean, committed worktree." >&2
    exit 1
fi

GIT_SHA=$(git rev-parse HEAD)
IMAGE="${COSMOS_IMAGE_REPO}:${TAG}"
REGISTRY=${COSMOS_IMAGE_REPO%%/*}

echo "[build] configuring registry authentication"
gcloud auth configure-docker "${REGISTRY}" --quiet

echo "[build] building ${IMAGE} from ${GIT_SHA}"
docker build --pull \
    -f infra/Dockerfile.cosmos3-edge \
    --build-arg "BENCHMARK_GIT_SHA=${GIT_SHA}" \
    -t "${IMAGE}" .

echo "[build] pushing ${IMAGE}"
docker push "${IMAGE}"

REPO_DIGEST=$(docker image inspect "${IMAGE}" --format '{{join .RepoDigests "\n"}}' | head -n 1)
if [ -z "${REPO_DIGEST}" ]; then
    echo "ERROR: pushed image has no locally recorded repository digest." >&2
    exit 1
fi
echo "[build] immutable image: ${REPO_DIGEST}"
