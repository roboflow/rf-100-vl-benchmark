# Cosmos3-Edge preflight gates

The full RF100VL run must not start until all five gates below pass using the
same commit, container image, vLLM arguments, model ID, BF16 precision, prompt
version, worker count, dataset copy, and GCS credentials intended for the full
benchmark.

On RunPod these gates are automated by the image's `preflight` stage. It
uploads a gate summary and stops while preserving `/workspace` before the
mandatory human visual review. The separately approved `full` stage consumes
that evidence; see
[`COSMOS3_EDGE_RUNPOD.md`](COSMOS3_EDGE_RUNPOD.md).

## Gate 1: offline contracts

Install the benchmark dependencies and run the complete offline suite:

```bash
python -m pip install -r requirements-cosmos.txt
python -m unittest -v \
  test_evaluate_cosmos.py \
  test_preflight_cosmos.py \
  test_runpod_cosmos.py
```

This suite must have zero failures and zero skips. It covers:

- the exact detection prompt text and complete JSON-encoded class list;
- NVIDIA's published one-user-message, image-before-text layout;
- no added system prompt, `temperature=0`, seed 0, and thinking disabled;
- JSON parsing, empty output, malformed output, and label filtering;
- 0–1000 normalized xyxy to pixel COCO xywh conversion, including a non-square
  image that proves x and y are scaled independently;
- duplicate, invalid, out-of-range, reversed, and ambiguous-label handling;
- COCO AP scoring, including perfect and empty predictions;
- atomic per-image records, aggregate outputs, and success-marker behavior;
- simulated total local-disk loss followed by GCS-only resume;
- immediate termination of evaluation when a durable record upload fails.
- preservation and immediate failure of a token-capped response without four
  identical retries;
- a simulated generation timeout proving an expensive request is not repeated;
- a complete 100-dummy-dataset inference, COCO scoring, checkpoint, aggregate,
  and success-marker run through a fake OpenAI endpoint and local GCS double;
- the 100,000-token output cap's input reserve and the explicit 0.80 vLLM
  memory-reservation contract.

The separate real-GCS test is intentionally skipped here because it requires
live credentials; Gate 2 runs it explicitly.

## Gate 2: real GCS round trip

Use a disposable parent prefix in the same bucket and with the same pod-side
ADC credentials as the benchmark:

```bash
export COSMOS_TEST_GCS_URI="gs://YOUR_BUCKET/rf100vl/cosmos3-edge/preflight"

python -m unittest -v test_gcs_live.py
```

The test creates a random child prefix, exercises create, overwrite, list,
read, subtree restore, and delete, and removes only the exact test objects it
created. A skip is a failure for this gate.

## Gate 3: data, endpoint, and storage preflight

Start the exact vLLM server intended for the benchmark, then run:

```bash
python preflight_cosmos.py \
  --dataset-dir /absolute/path/to/RF100VL \
  --expected-datasets 100 \
  --base-url http://localhost:8000/v1 \
  --model-id nvidia/Cosmos3-Edge \
  --gcs-test-uri "${COSMOS_TEST_GCS_URI}" \
  --report ./cosmos_preflight_report.json
```

This opens and verifies every test image, checks its actual pixel dimensions
against COCO metadata, validates every category/image/annotation reference and
ground-truth bbox, and reports any zero-area boxes without mutating or removing
them so scoring uses the same canonical COCO ground truth as other models. It
confirms exactly 100 test datasets, confirms that the live
endpoint advertises `nvidia/Cosmos3-Edge`, repeats the real GCS storage
contract, and uploads `preflight_report.json` to the supplied GCS prefix. The
report must say `status: passed`.

## Gate 4: live inference, coordinates, visualization, and resume

Choose one representative dataset and a fresh GCS smoke prefix:

```bash
export SMOKE_DATASET="ONE_DATASET_DIRECTORY_NAME"
export SMOKE_GCS_URI="gs://YOUR_BUCKET/rf100vl/cosmos3-edge/preflight/live-smoke"

python evaluate_cosmos.py \
  --dataset-dir /absolute/path/to/RF100VL \
  --dataset "${SMOKE_DATASET}" \
  --max-images 10 \
  --workers 1 \
  --visualize-limit 10 \
  --save-dir ./cosmos-preflight-first \
  --gcs-results-uri "${SMOKE_GCS_URI}"
```

Before continuing, inspect all ten raw records and visualizations. Required
observations:

- no request, parse, or image errors;
- responses contain only the requested class names;
- raw `bbox_2d` values behave like 0–1000 xyxy values;
- converted overlays align with the objects rather than appearing scaled,
  transposed, or offset;
- responses are not truncated and thinking text is absent.

This visual gate is mandatory because NVIDIA publishes support for bounding-box
reasoning but does not currently publish a canonical object-detection prompt or
coordinate normalization contract.

Now use a different empty local directory with the same GCS prefix:

```bash
python evaluate_cosmos.py \
  --dataset-dir /absolute/path/to/RF100VL \
  --dataset "${SMOKE_DATASET}" \
  --max-images 10 \
  --workers 1 \
  --visualize-limit 10 \
  --save-dir ./cosmos-preflight-restored \
  --gcs-results-uri "${SMOKE_GCS_URI}"
```

The log must report ten resumed images and zero pending images. The restored
raw record files must match the first run. Then rerun with `--max-images 20`;
the log must report ten resumed and ten pending. This proves real inference and
real GCS resume work together, not merely in the mock test.

## Gate 5: one complete scored dataset

Remove `--max-images`, keep the same prompt settings and GCS prefix, and finish
the chosen dataset. Require:

- process exit code 0;
- `complete: true` and `completed_image_count == image_count` in `summary.json`;
- zero image errors and zero clamped boxes, reordered axes, or ignored labels in
  the diagnostic totals (or an explicit investigation before approval);
- COCO metrics present and finite;
- `_SUCCESS.json` uploaded for this one-dataset run;
- raw-response and visualization spot checks still look correct.

Only after these gates pass should the 100-dataset command be prepared. Its
final approval card should record the preflight report URI, live-smoke prefix,
one-dataset summary URI, exact Git commit and image digest, BF16 vLLM command,
`workers=1`, and the fresh GCS results prefix. The full run must include
`--expected-datasets 100`, `--gcs-results-uri`, and
`--preflight-report ./cosmos_preflight_report.json`. The evaluator refuses to
start a detected 100-dataset scored run without these fields and rechecks every
annotation hash, the endpoint/model/prompt identity, and GCS bucket against the
report. Completion requires exit code 0,
`scored_dataset_count: 100`, and a final `_SUCCESS.json`.
