# Cosmos3-Edge on RF100-VL

`evaluate_cosmos.py` evaluates the basic zero-shot object-detection setting only.
Every request contains one test image and the complete class list from that
dataset's test COCO file. It does not read train images, few-shot examples,
`README.dataset.txt`, or any other dataset instructions.

The official downloader creates the directory layout expected by the evaluator:

```bash
python -m pip install "rf100vl[cli]"
export ROBOFLOW_API_KEY=YOUR_KEY
rf100vl download rf100vl ./rf100-vl/
```

Run the evaluator from the repository root and leave the downloaded directory at
`./rf100-vl/` to use the default path. A differently named directory, including
`./RF100VL/`, also works when passed explicitly as `--dataset-dir ./RF100VL`.
Each immediate child must be one RF100-VL dataset containing a `test/` directory
with one COCO annotation file and its test images.

Cosmos3-Edge is expected to run behind its OpenAI-compatible vLLM server. One
example, when the local RF100-VL directory is mounted at `/data` in the server:

```bash
docker pull vllm/vllm-openai:cosmos3

docker run --rm --gpus all \
  -p 8000:8000 \
  -v /absolute/path/to/rf100-vl:/data:ro \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:cosmos3 \
  --model nvidia/Cosmos3-Edge \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --seed 0 \
  --max-model-len 131072 \
  --allowed-local-media-path /data \
  --mm-processor-kwargs '{"do_resize":true,"min_pixels":4096,"max_pixels":16777216}'
```

BF16 is the checkpoint's published tensor type and the only precision NVIDIA
currently tests for Cosmos3-Edge. Do not add `--quantization`, use FP8/FP4 KV
cache, or force FP16 for the primary benchmark. `--kv-cache-dtype auto` keeps the
KV cache at the model dtype. For the cleanest primary result, use one GPU and do
not enable tensor parallelism or layer offload.

Install the evaluation-side dependencies:

```bash
# Python 3.10 or newer
python -m pip install -r requirements-cosmos.txt
```

Start with a small, visualized smoke test using data URLs. Data URLs work even
when the evaluator and server do not share filesystem paths:

```bash
python evaluate_cosmos.py \
  --dataset-dir /absolute/path/to/rf100-vl \
  --dataset one-rf100-dataset-name \
  --max-images 10 \
  --workers 1 \
  --visualize-limit 10 \
  --save-dir ./cosmos-smoke
```

An intentionally incomplete smoke test writes predictions and visualizations,
but does not report COCO AP. Continue the same run without `--max-images`; its
per-image checkpoints will be reused:

```bash
python evaluate_cosmos.py \
  --dataset-dir /absolute/path/to/rf100-vl \
  --dataset one-rf100-dataset-name \
  --workers 1 \
  --save-dir ./cosmos-smoke
```

For the conservative, most reproducible complete benchmark:

```bash
python evaluate_cosmos.py \
  --dataset-dir /absolute/path/to/rf100-vl \
  --workers 1 \
  --expected-datasets 100 \
  --save-dir ./cosmos-rf100-results
```

## Durable GCS results and resume

For an ephemeral GPU machine, give the run its own GCS prefix. The evaluator
verifies create/read/list access before inference, restores existing artifacts
from that exact prefix, uploads every successful image record immediately, and
refreshes errors, per-dataset outputs, aggregate scores, and the log as the run
progresses:

```bash
export COSMOS_GCS_RESULTS_URI="gs://YOUR_BUCKET/rf100vl/cosmos3-edge/YOUR_RUN_ID"

python evaluate_cosmos.py \
  --dataset-dir /absolute/path/to/rf100-vl \
  --workers 1 \
  --expected-datasets 100 \
  --save-dir ./cosmos-rf100-results \
  --gcs-results-uri "${COSMOS_GCS_RESULTS_URI}"
```

The Python client uses Google Application Default Credentials (ADC). On the
existing RunPod account, reuse the account-level `GCP_SA_JSON_B64` secret:
inject it into the container, decode it to a mode-600 temporary JSON file, and
set `GOOGLE_APPLICATION_CREDENTIALS` to that file before starting the evaluator.
Do not put the JSON, its base64 value, or a signed URL in Git, the image, or the
launch command. A replacement pod resumes by using the same GCS results URI.

Resume requires create, read, update, and list access to objects below the run
prefix. At bucket scope, `roles/storage.objectUser` provides those object
operations without granting bucket administration. `roles/storage.objectCreator`
alone is insufficient because it cannot list or read checkpoints.

The root `aggregate_summary.json` is refreshed after every dataset. A root
`_SUCCESS.json` is written only when every selected dataset completed and was
scored. For the canonical run, the combination of `--expected-datasets 100`, a
zero process exit, `scored_dataset_count: 100`, and `_SUCCESS.json` verifies that
all 100 test splits were scored before the pod is terminated. If a GCS operation
fails, the evaluator stops instead of continuing to spend GPU time without
durable checkpoints.

vLLM can serve concurrent requests, but online scheduling is not guaranteed to
be bitwise invariant. Before increasing `--workers`, compare a fixed smoke set
at `1` and the intended concurrency using separate save directories. If the raw
answers and boxes agree, concurrency is a reasonable throughput optimization.
Worker count is included in the checkpoint configuration hash, so results from
different concurrency settings are never silently mixed.

To avoid base64 transfer overhead, use file URLs. In this example the local
dataset root is the directory mounted to `/data` in the vLLM container:

```bash
python evaluate_cosmos.py \
  --dataset-dir /absolute/path/to/rf100-vl \
  --image-transport file-url \
  --server-media-root /data \
  --workers 1 \
  --save-dir ./cosmos-rf100-results-file-url
```

Use file URLs only after confirming every selected image is already RGB/RGBA.
RF100-VL includes specialized imaging domains that may contain grayscale files.
The default data-URL path safely converts those inputs to RGB before inference;
file-URL mode deliberately rejects them because it cannot rewrite files inside
the read-only server mount. Data URLs are therefore the recommended primary
benchmark transport unless the entire download has been checked.

Outputs include:

- `cosmos_detection_results.json`: COCO prediction list for each dataset;
- `summary.json`: per-dataset completion state and COCO metrics;
- `aggregate_summary.json`: macro AP and AP50 over completed datasets;
- `records/`: atomic per-image checkpoints including raw model responses;
- `errors_*.jsonl`: failed requests or unparseable responses;
- `visualizations/`: optional detection previews.

The primary metric is COCO `AP@[.50:.95]`. Evaluation uses
`maxDets=[1, 10, 500]`. Since Cosmos does not return calibrated detector
confidences, every accepted prediction receives `score=1.0`, matching the
existing generative-VLM evaluators in this repository.
