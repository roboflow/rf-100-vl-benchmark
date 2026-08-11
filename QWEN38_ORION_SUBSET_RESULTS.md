# Qwen3.8-Max Orion prompt-mode subset

Status: complete (2026-08-11)

This directional ablation uses five fixed Orion Products test images (IDs 36,
2, 27, 30, and 0) containing all eight dataset classes. The six original
prompt modes are rescored from their completed full-test records. This subset
corresponds to 205 original-mode requests: five multi-class calls plus 200
single-class calls. The two combined visual-reference modes required 20 new
calls total (five images x two modes x two reasoning settings).

All metrics use pycocotools with `maxDets=[1, 10, 500]`. Values below are
percentages.

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

- Explicit `reasoning_effort=none` produced zero reasoning tokens. The ten new
  no-reasoning requests averaged 10.61 seconds; the low-reasoning requests
  used 9,180 reasoning tokens and averaged 25.13 seconds.
- Reasoning is not uniformly beneficial. It strongly improves merged
  single-class text prompts and the combined numeric-reference prompt, while
  no reasoning is stronger for several per-class positive/negative modes.
- The combined numeric-reference prompt with low reasoning is promising:
  28.32 mAP versus 29.73 for per-class positive numeric prompts, while making
  one target request instead of eight. Its mAP50 is 6.56 points lower.
- Combining drawn reference boxes into one prompt performs poorly in both
  reasoning conditions.
- Positive/negative prompting is kept per class. In a single all-class prompt,
  each paired "negative" is itself another requested positive class, so a
  global include/exclude instruction would be contradictory rather than a
  controlled comparison.
- Five images are too few for a final model claim. These results select which
  modes merit a larger follow-up; the completed 59-image Orion results remain
  the stronger evidence for the original six modes.

## Artifacts

The resumable local artifact root is
`qwen38-orion-runs/orion-five-image-single-prompt-v1/`. It contains:

- `comparison_summary.csv` and `comparison_summary.json`
- per-reasoning aggregate and per-mode metric JSON
- filtered per-mode predictions
- raw response and usage records for the 20 new calls
- reference images, subset ground truth, manifest, and logs

The original complete runs, including raw responses for the six rescored
modes, are in `qwen38-orion-runs/orion-prompt-modes-v1/` (low reasoning) and
`qwen38-orion-runs/orion-prompt-modes-v1-no-thinking/` (no reasoning).
