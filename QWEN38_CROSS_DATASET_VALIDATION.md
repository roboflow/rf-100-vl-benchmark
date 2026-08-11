# Qwen3.8-Max cross-dataset prompt validation

Status: complete (2026-08-11)

This validation tests the six selected Orion recipes on two additional
RF20-VL-FSOD test sets:

| Dataset | Domain | Test images | Classes | Ground-truth objects |
|---|---|---:|---:|---:|
| `lacrosse-object-detection` | Sports | 50 | 4 | 355 |
| `the-dreidel-project` | Fine-grained objects and symbols | 54 | 6 | 171 |

Both test splits contain ground truth for every declared class. References are
selected deterministically from each dataset's train split. Test images and
class names are the only target-side information supplied to the model.

The evaluated recipes are multi-class class names, per-class positive numeric
boxes, per-class positive drawn boxes, per-class positive and negative numeric
boxes, and one-call multi-class positive numeric boxes with reasoning `none`
and `low`. Per-class predictions are merged before COCO scoring. All metrics use
pycocotools with `maxDets=[1, 10, 500]`.

The complete matrix requires 750 API calls for lacrosse and 1,134 for dreidel.
Every response is atomically checkpointed, and the launcher retries resumable
invocations without repeating terminal fingerprint-matching records.

## Results

Scores are `mAP50-95 / mAP50` percentages. The macro average weights each
dataset equally.

| Strategy | Calls/image | Orion | Lacrosse | Dreidel | Macro average |
|---|---:|---:|---:|---:|---:|
| Multi-class names, no reasoning | 1 | 14.44 / 26.56 | **33.58 / 53.57** | 42.33 / 57.02 | 30.12 / 45.71 |
| Positive numeric, per class, no reasoning | Classes | **23.46 / 49.50** | 24.92 / 39.24 | 32.00 / 45.12 | 26.80 / 44.62 |
| Positive drawn, per class, no reasoning | Classes | 21.56 / 47.18 | 24.68 / 39.33 | 31.12 / 42.86 | 25.79 / 43.13 |
| Positive + negative numeric, per class, no reasoning | Classes | 22.21 / 47.33 | 24.54 / 38.16 | 34.98 / 50.58 | 27.24 / 45.36 |
| Multi-class positive numeric, no reasoning | 1 | 14.89 / 36.93 | 31.29 / 52.08 | **48.56 / 71.43** | **31.58 / 53.48** |
| Multi-class positive numeric, low reasoning | 1 | 16.72 / 39.26 | 25.23 / 43.13 | 36.62 / 53.64 | 26.19 / 45.34 |

The Orion accuracy-first result does not generalize exactly. Per-class positive
numeric prompting wins Orion, class names win lacrosse, and a single combined
numeric-reference prompt wins dreidel. Across all three datasets, the combined
numeric prompt with reasoning off has the strongest macro average while using
one call per image. Numeric references consistently outperform drawn references
on mAP50-95. Positive/negative references are dataset-dependent. Low reasoning
does not improve the combined prompt overall and produced five terminal model
failures on dreidel, versus none for the no-reasoning combined condition.

Run both datasets with:

```bash
bash run_qwen38_cross_dataset_validation.sh
```
