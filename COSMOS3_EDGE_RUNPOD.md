# Cosmos3-Edge RF100VL on RunPod

The canonical benchmark runs entirely on RunPod. The local machine is only the
control plane used to commit code, build/push the image, review a dry-run pod
specification, and submit or inspect pods.

The same immutable image has two stages:

1. `preflight`: starts the pinned BF16 vLLM server, obtains RF100VL, runs the
   offline and real-GCS tests, validates all 100 test splits, runs the 10-image
   inference and GCS-only resume checks, scores one complete dataset, uploads
   all evidence, and self-terminates.
2. `full`: only after human review of the ten overlays/raw responses, restores
   the preflight evidence, revalidates its exact data/model/prompt contract,
   evaluates all 100 test splits, verifies 100/100 scored plus the durable GCS
   success marker, uploads final verification, and self-terminates.

Splitting these into two pods is intentional. It preserves the mandatory human
visual gate without billing for an idle GPU between preflight and approval.

## Account assumptions

The existing RunPod account already supplies account-level secret references
for Hugging Face, GCS ADC, dataset download, and pod self-termination. Do not
put their values in Git, a Docker layer, a launch command, or a pod's plain
environment configuration.

The local gitignored `infra/runpod.env` contains only:

```bash
export RUNPOD_API_KEY=...
export RUNPOD_REGISTRY_AUTH_ID=...
```

The launcher references these existing RunPod secrets by name:

- `HF_TOKEN`
- `GCP_SA_JSON_B64`
- `ROBOFLOW_API_KEY` (omitted when `--dataset-gcs-uri` is used)
- `POD_API_KEY` for self-termination

## Build the image

Build only from a clean committed worktree. Set the already-authorized image
repository path and use a commit-derived tag:

```bash
export COSMOS_IMAGE_REPO=REGISTRY/REPOSITORY/IMAGE
bash infra/build_and_push_cosmos.sh runpod-$(git rev-parse --short HEAD)
```

The build runs the offline evaluator tests inside the release-tested
`vllm/vllm-openai:cosmos3` base image. Evaluator and RF100VL downloader
dependencies live in a separate virtual environment so their OpenCV constraint
cannot modify the vLLM runtime. Record the immutable image reference
printed at the end (`...@sha256:...`); the full launcher rejects a mutable tag.

## Choose one durable run root

Both stages use the same run-specific root:

```bash
export COSMOS_GCS_RUN_URI=gs://BUCKET/rf100vl/cosmos3-edge/RUN_ID
export COSMOS_IMAGE=REGISTRY/REPOSITORY/IMAGE@sha256:DIGEST
```

It contains independent prefixes:

```text
control/preflight/             manifests, job/vLLM logs, gate summary
preflight/storage/             preflight report and disposable GCS probes
preflight/live-smoke/          ten-image raw records, overlays, resume evidence,
                               and one complete dataset score
control/full/                  full job manifest, logs, verification
full/                          all 100 datasets, records, predictions, summaries,
                               aggregate score, and final _SUCCESS.json
```

## Preflight pod

First inspect the exact pod specification:

```bash
source infra/runpod.env
bash infra/runpod_cosmos_launch.sh launch \
  --name cosmos3-edge-rf100vl-preflight \
  --image "${COSMOS_IMAGE}" \
  --stage preflight \
  --gcs-run-uri "${COSMOS_GCS_RUN_URI}" \
  --gpus 1 \
  --gpu-type "NVIDIA H100 80GB HBM3" \
  --dry-run
```

After the scientific intent is approved, submit the same command without
`--dry-run`. Save the returned pod ID. The pod self-terminates after success or
failure; if it remains listed after the job log ends, terminate it manually.

The automated gate is successful only when
`control/preflight/gate_summary.json` says
`awaiting_human_visual_review`. Review all ten referenced overlays and raw
responses. Confirm boxes align, axes and scaling are correct, labels come only
from the supplied class list, output is not truncated, and thinking text is
absent.

## Full 100-dataset pod

After that review, prepare and review a new exact launch contract. The full dry
run must include the immutable image and explicit approval flag:

```bash
bash infra/runpod_cosmos_launch.sh launch \
  --name cosmos3-edge-rf100vl-full \
  --image "${COSMOS_IMAGE}" \
  --stage full \
  --gcs-run-uri "${COSMOS_GCS_RUN_URI}" \
  --gpus 1 \
  --gpu-type "NVIDIA H100 80GB HBM3" \
  --preflight-approved \
  --dry-run
```

Obtain the separate exact-launch approval, then execute the emitted approved
command without `--dry-run`. Do not reconstruct it manually after approval.

## Completion contract

The full run is complete only when all of the following agree:

- pod workload exited with code zero and self-terminated;
- `full/aggregate_summary.json` has `status: complete`;
- selected, processed, and scored dataset counts are all exactly 100;
- all 100 embedded dataset summaries are complete and contain COCO metrics;
- `full/_SUCCESS.json` exists in GCS;
- `control/full/verification.json` independently records the same 100/100
  result.

No hard time cutoff is used. Per-image records are uploaded immediately, so a
replacement pod using the same GCS run root resumes durable progress.
