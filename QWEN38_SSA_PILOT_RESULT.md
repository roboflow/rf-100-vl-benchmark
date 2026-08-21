# Qwen3.8-Max sequential support acquisition pilot

## Question

Can a clean multi-turn support conversation provide useful object-detection
context, and can we measure its benefit prequentially without exposing support
ground truth before prediction?

## Pilot

- Dataset: RF20-VL-FSOD Dreidel
- Support order: deterministic seed 1234
- Acquisition cap: 12 support images, containing 12 annotated objects total
- Test set: deterministic first 20 untouched test images
- Test prefixes: 0, 1, 2, 4, 8, and 12 support images
- Final metric: COCO mAP50-95, with mAP50 secondary and `maxDets=500`
- Support routing signal: known-object class-macro recall50-95, with recall50
  as a guard
- No reasoning, temperature 0, normalized 0-1000 XYXY boxes

The persisted trunk contained only official support images and their canonical
gold JSON annotations. Every support and test prediction was made in a
discarded branch. Test images and annotations were never used during
adaptation.

## Test-grid result

| Support images | Annotated objects | mAP50-95 | mAP50 | Delta vs names |
|---:|---:|---:|---:|---:|
| 0 | 0 | 50.90 | 63.10 | baseline |
| 1 | 1 | 54.81 | 68.57 | +3.91 / +5.47 |
| **2** | **2** | **63.30** | 83.09 | **+12.40 / +19.98** |
| 4 | 4 | 59.82 | **83.58** | +8.92 / **+20.48** |
| 8 | 8 | 56.52 | 78.66 | +5.62 / +15.56 |
| 12 | 12 | 59.20 | 78.49 | +8.29 / +15.38 |

All 143 API requests succeeded. Total research cost was approximately $1.26,
including adaptation and all six test-grid branches.

## Interpretation

The core mechanism works and warrants further study. A clean multi-turn visual
context produced a large gain over the exactly matched zero-prefix prompt, and
the curve was non-monotone as expected. Two support images were the test-grid
oracle on the primary metric.

The preliminary support-only smoothing preview selected one image. That still
improved over zero-shot, but it missed the two-image test oracle. This is not a
failure of the planned method because the preview used one order and an
untuned placeholder rule. It confirms why the full method needs three support
orders, pooled noise estimation, patience, and leave-one-dataset-out policy
selection before reporting RF20 performance.

This pilot is implementation validation, not a benchmark result. The subset is
small, Dreidel was chosen because visual context had helped previously, and
the prefix was inspected through an analysis-only test grid.
