# Qwen3.8-Max Orion prompt-mode subset

Status: complete (2026-08-11)

The experiment includes nested five- and 20-image Orion Products subsets. The
20-image subset retains the original five images and adds 15 selected without
model predictions to closely match the full test set's class-frequency
distribution. It contains 188 annotations across all eight classes.

The six original prompt modes are rescored from completed full-test records.
The 20-image subset corresponds to 820 original-mode requests: 20 multi-class
calls plus 800 single-class calls. The two combined visual-reference modes
required 80 records total; 20 were fingerprint-verified records reused from
the nested five-image run and 60 were new API calls.

All metrics use pycocotools with `maxDets=[1, 10, 500]`. Values below are
percentages.

## 20-image stratified subset

| Prompt mode | Reasoning | mAP50-95 | mAP50 |
|---|---|---:|---:|
| Multi-class names | None | **24.38** | 40.99 |
| Multi-class names | Low | 24.22 | 41.89 |
| Single-class names, merged | None | 21.84 | 34.92 |
| Single-class names, merged | Low | 21.20 | 39.09 |
| Positive numeric boxes, per class | None | 23.84 | 48.86 |
| Positive numeric boxes, per class | Low | 23.65 | 45.26 |
| Positive drawn boxes, per class | None | 21.57 | 44.48 |
| Positive drawn boxes, per class | Low | 21.66 | 41.97 |
| Positive + negative numeric boxes, per class | None | 23.42 | **49.57** |
| Positive + negative numeric boxes, per class | Low | 22.76 | 47.40 |
| Positive + negative drawn boxes, per class | None | 19.77 | 48.29 |
| Positive + negative drawn boxes, per class | Low | 19.58 | 39.60 |
| Multi-class positive numeric boxes, one call | None | 18.32 | 41.59 |
| Multi-class positive numeric boxes, one call | Low | 20.47 | 42.22 |
| Multi-class positive drawn boxes, one call | None | 18.84 | 43.25 |
| Multi-class positive drawn boxes, one call | Low | 14.56 | 30.32 |

## Five-image subset

| Prompt mode | Reasoning | mAP50-95 | mAP50 |
|---|---|---:|---:|
| Multi-class names | None | 21.25 | 42.89 |
| Multi-class names | Low | 20.22 | 37.00 |
| Single-class names, merged | None | 22.10 | 37.06 |
| Single-class names, merged | Low | **32.69** | 54.20 |
| Positive numeric boxes, per class | None | 30.70 | 60.81 |
| Positive numeric boxes, per class | Low | 29.73 | 60.71 |
| Positive drawn boxes, per class | None | 23.45 | 53.93 |
| Positive drawn boxes, per class | Low | 27.33 | 47.65 |
| Positive + negative numeric boxes, per class | None | 30.75 | **72.16** |
| Positive + negative numeric boxes, per class | Low | 25.88 | 55.82 |
| Positive + negative drawn boxes, per class | None | 31.43 | 70.61 |
| Positive + negative drawn boxes, per class | Low | 29.18 | 58.29 |
| Multi-class positive numeric boxes, one call | None | 19.98 | 47.48 |
| Multi-class positive numeric boxes, one call | Low | 28.32 | 54.15 |
| Multi-class positive drawn boxes, one call | None | 16.24 | 50.38 |
| Multi-class positive drawn boxes, one call | Low | 15.96 | 35.40 |

## Interpretation

- Explicit `reasoning_effort=none` produced zero reasoning tokens. Across the
  40 combined-prompt records in the 20-image run, no reasoning averaged 14.40
  seconds. Low reasoning used 46,090 reasoning tokens and averaged 30.47
  seconds.
- The five-image result overstated the combined numeric prompt. On 20 images,
  low-reasoning combined numeric boxes score 20.47/42.22 versus 23.65/45.26
  for the per-class equivalent. The one-call recipe remains a cost/throughput
  option, but it is not the accuracy-first choice.
- Positive numeric boxes per class with reasoning off remain the recommended
  cross-image recipe: 23.84/48.86 here and 23.46/49.50 on all 59 images.
- Positive + negative numeric boxes slightly improve mAP50 on this subset but
  reduce mAP50-95, and they do not beat positive-only numeric boxes on the
  complete 59-image evaluation.
- Reasoning is mode-dependent, but reasoning off is generally as good or
  better for the per-class cross-image recipes while being substantially
  faster and cheaper.
- Combining drawn reference boxes into one prompt performs poorly in both
  reasoning conditions.
- Positive/negative prompting is kept per class. In a single all-class prompt,
  each paired "negative" is itself another requested positive class, so a
  global include/exclude instruction would be contradictory rather than a
  controlled comparison.
- The completed 59-image Orion results remain the strongest evidence for the
  original six modes. The 20-image result supplies substantially better
  evidence for the two combined modes than the original five-image pilot.

## Artifacts

The resumable local artifact root is
`qwen38-orion-runs/orion-five-image-single-prompt-v1/`. It contains:

- `comparison_summary.csv` and `comparison_summary.json`
- per-reasoning aggregate and per-mode metric JSON
- filtered per-mode predictions
- raw response and usage records for the 20 new calls
- reference images, subset ground truth, manifest, and logs

The equivalent 20-image artifacts are under
`qwen38-orion-runs/orion-twenty-image-single-prompt-v1/`.

The original complete runs, including raw responses for the six rescored
modes, are in `qwen38-orion-runs/orion-prompt-modes-v1/` (low reasoning) and
`qwen38-orion-runs/orion-prompt-modes-v1-no-thinking/` (no reasoning).
