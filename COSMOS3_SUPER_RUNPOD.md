# Cosmos3-Super RF100VL on RunPod

This run measures the highest-capacity general Cosmos 3 checkpoint,
`nvidia/Cosmos3-Super`, in the same RF100VL class-names-only object-detection
setting used for Cosmos3-Edge. It uses only each test image and that dataset's
complete class list. It does not read training images, few-shot examples,
dataset instructions, or README text.

## Frozen model and inference contract

- model: `nvidia/Cosmos3-Super` (64B unified checkpoint)
- revision: `e0262be9d8f7586bc24c069a2aed2b665bdff266`
- serving surface: Cosmos3 Reasoner through vLLM's OpenAI-compatible API
- weights: BF16, with no quantization
- topology: four H100 80GB GPUs per server, tensor parallel size 4
- collective: PyNCCL (`--disable-custom-all-reduce`) with the incompatible
  FlashInfer fused all-reduce/RMS pass disabled
- evaluator concurrency: one request at a time per server
- request limits: 8,192 generated tokens and 180 seconds per image
- scoring: the unchanged RF100VL COCO evaluator with `maxDets=500`

The detection prompt, media-first message layout, robust parser, normalized
0–1000 xyxy-to-pixel COCO xywh conversion, output artifacts, and scoring code
are identical to the completed Edge run. The inherited prompt identifier still
contains `edge` because preserving it proves the prompt itself did not change;
the run and every record identify the actual model as Cosmos3-Super.
Recoverable individual detections (for example, a null box for an absent class)
are excluded from predictions while their diagnostics and the complete raw
response remain durable. They do not abort an otherwise valid image or dataset.

The standard collective is required because the vLLM Cosmos image otherwise
auto-selects FlashInfer MNNVL fused/custom all-reduce on this RunPod H100
topology and crashes during CUDA-graph capture. This setting changes only the
multi-GPU communication implementation; it does not quantize or otherwise
change the BF16 model, prompt, decoding settings, responses, or scoring.

NVIDIA's maintained Reasoner recipe uses tensor parallel size 4 for Super.
Eight independent one-H100 pods are therefore not a supported BF16 layout.
To use eight H100s, create two disjoint dataset shards and run one four-H100
server for each shard. Each shard writes to its own GCS prefix; the finalizer
verifies exactly one owner for every dataset and writes the canonical success
marker only after all 100 datasets and 14,237 image records are present.

## Storage and lifecycle

Super's Hugging Face repository is about 135 GB. Each pod therefore uses a
120 GB container disk and a 400 GB persistent `/workspace` volume. Model cache,
RF100VL, local checkpoints, and logs stay on the volume; every inference record,
prediction file, dataset score, log, manifest, and aggregate is uploaded to a
fresh run-specific GCS root as it is produced.

Use the same preflight and completion gates documented in
[`COSMOS3_EDGE_RUNPOD.md`](COSMOS3_EDGE_RUNPOD.md), substituting the Super model
profile:

```bash
bash infra/runpod_cosmos_launch.sh launch \
  --name cosmos3-super-rf100vl-preflight \
  --image "$COSMOS_IMAGE" \
  --stage preflight \
  --gcs-run-uri "$COSMOS_GCS_RUN_URI" \
  --model-id nvidia/Cosmos3-Super \
  --dry-run
```

The launcher fills in the pinned revision, four GPUs, tensor parallel size 4,
and the larger disk allocations. A full or shard launch additionally requires
the immutable image digest and the successful preflight's visual approval.
