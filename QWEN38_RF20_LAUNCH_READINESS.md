# Qwen3.8-Max RF20-VL-FSOD launch readiness

## Locked comparison

The full benchmark contains exactly three conditions, all using one multi-class
request per test image:

1. Class names only.
2. Class names plus one positive numeric train reference per class.
3. Class names plus two positive numeric train references per class.

All conditions use temperature 0, seed 1234, reasoning disabled through both
`reasoning_effort="none"` and `enable_thinking=false`, a three-minute generation
deadline, and an 8,192-token completion limit. Conditions are interleaved by
target image. The planned total is 11,910 requests across 3,970 test images and
20 datasets.

## Prompt and few-shot contract

The two visual conditions use only train-split reference images. Every reference
annotation and requested prediction is serialized by the same code path as:

```json
[{"bbox_2d":[x1,y1,x2,y2],"label":"exact class name"}]
```

Coordinates are integer XYXY values independently normalized to `[0,1000]`
relative to the immediately associated image. Reference labels exactly match
the allowed output labels. References are grouped by class, and the target
image is always last. The launcher rejects any conditions file that differs
from this locked contract, including the historical box-only A/B format.

References are selected deterministically without test labels or predictions:
rank one maximizes relative object area, and rank two maximizes crop-appearance
diversity. The two references for every class come from distinct train images.
No README instructions, validation images, test annotations, or interactive
feedback enter a prompt.

## Inference and post-processing

The production recipe evaluator uses the established Qwen3.8-Max request and
post-processing implementation in `evaluate_qwen38_orion.py`:

- OpenAI-compatible DashScope streaming endpoint;
- original image data URLs with provider-managed vision preprocessing;
- explicit temperature, seed, thinking controls, timeout, and token limit;
- resumable retry handling for provider/rate-limit/network failures;
- terminal and explicitly typed truncation, malformed-response, and timeout
  records;
- raw response, usage, latency, diagnostics, prompt summary, and request
  fingerprint retained for every call;
- JSON fence/wrapper/alias/trailing-comma and complete-object recovery;
- exact case/whitespace-normalized class matching, without substring matching;
- finite-coordinate validation, `[0,1000]` clamping, reversed-axis repair,
  degenerate-box rejection, and exact duplicate removal;
- independent x/y conversion from normalized XYXY to original-image COCO XYWH;
- score `1.0`, matching the repository's generative-VLM convention;
- atomic per-request checkpoints and deterministic resume without repeating a
  matching terminal request.

Per-dataset prediction JSON, raw records, metrics, progress, logs, and usage are
retained. Final scoring uses `pycocotools` with `maxDets=[1,10,500]`; mAP50–95
and mAP50 are computed directly from the accumulated tensors at maxDets 500.
The aggregate stage refuses to write its RF20 success marker unless all 20
datasets and all three conditions are complete and token accounting matches the
individual records.

Compared with the older Qwen2.5 evaluator, the only omitted convenience is
rendered prediction-image visualization, which is not used for scoring. The
coordinate preprocessing intentionally differs: Qwen2.5 used resized pixel
coordinates, while Qwen3.8-Max's documented/pretested contract is normalized
XYXY `[0,1000]` with provider-managed image preprocessing.

## Fresh dataset verification

On 2026-08-13, `rf100vl.download_rf20vl_fsod(..., model_format="coco")` was used
to download a fresh copy into
`RF100VL/rf20-vl-fsod-fresh-20260813/`. Validation found:

- exactly the expected 20 datasets;
- 5,556 train/validation/test images, all decodable and dimensionally
  consistent with COCO metadata;
- 1,401,393,229 image bytes compared;
- every image byte-identical to the existing copy;
- all 60 COCO split annotation files exactly equal to the existing copy;
- no train/validation/test filename leakage;
- no non-finite or out-of-bounds boxes.

The official data contains two zero-width, area-zero test annotations in
`soda-bottles` (annotation IDs 642 and 2567). Both copies contain the same two
annotations. They are preserved because changing official ground truth would
break benchmark comparability; the same ground truth applies to every method.

The exact launcher's preflight-only mode passed against the fresh copy with
3,970 test images, 110 classes, 57,285 test annotations, and 11,910 requests.

## Residual limitations, not implementation errors

- DashScope has shown non-zero output variance despite temperature 0 and fixed
  seed. Interleaving conditions reduces temporal confounding, and macro
  averaging over 20 datasets should be more stable than the earlier small
  screens, but a single pass is not a deterministic measurement.
- Generative detections have no calibrated confidence scores, so all predictions
  receive score 1.0 consistently across this benchmark family.
- The historical unmatched numeric format remains available solely to reproduce
  the completed A/B experiment. The RF20 launch preflight rejects it.

These limitations do not block the planned three-way comparison. No inference
is launched by this readiness review; final billable launch remains subject to
explicit approval.
