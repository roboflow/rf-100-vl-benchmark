# Qwen3.8-Max strict binary routing result

## Question

Can routing be separated from detection so a zero-shot decision exactly
preserves the established class-names baseline, while visually under-specified
datasets use the established clean one-reference detector?

## Defensive architecture

The router makes one score-blind, dataset-level decision:

- `class_names_only`: use the exact saved class-names-only prediction branch;
- `visual_references`: use the exact saved one-positive-numeric-reference branch.

The routing transcript is never included in detection. Therefore, selecting
`class_names_only` cannot introduce the degradation observed when detection was
performed inside the adaptive conversation.

## Live Paper Parts test

Paper Parts was selected because the earlier per-image adaptive router requested
references for only 6 of 500 images, despite fixed one-shot prompting producing
one of the largest RF20 gains.

The strict dataset-level router returned `visual_references` with confidence
0.95. It identified document-layout roles and annotation boundaries such as
`equation number`, `figure caption`, `section`, `subsection`, and `paragraph` as
requiring dataset-specific visual context.

| Selected detector branch | mAP50-95 | mAP50 | Delta vs. class names |
|---|---:|---:|---:|
| Class names only | 14.30 | 34.95 | baseline |
| One positive numeric reference/class | **27.90** | **54.20** | **+13.60 / +19.25** |

The router used 317 prompt tokens and 135 completion tokens with reasoning off,
temperature 0, and seed 1234. At the same list prices used in the RF20 report,
the routing call cost approximately $0.0014.

## Existing-decision clean-branch replay

As a separate no-API diagnostic, the decisions from the completed adaptive run
were reduced to a binary gate and replayed using the exact clean saved detector
branches on all 20 datasets.

| Method | Macro mAP50-95 | Macro mAP50 | Delta vs. class names |
|---|---:|---:|---:|
| Class names only | 24.37 | 43.54 | baseline |
| Fixed one-shot | 25.35 | 46.73 | +0.98 / +3.19 |
| Binary replay with isolated detectors | **25.48** | 46.20 | **+1.11 / +2.66** |

This replay is diagnostic rather than a new prospective inference result, but it
shows that isolating the detector removes the principal degradation. The Paper
Parts live test additionally shows that conservative dataset-level wording can
correct an important false zero-shot decision. Broader score-blind routing
validation is still required before treating the router as a locked recipe.

Complete local artifact:
`qwen38-fsod-runs/paper-parts-strict-binary-router-v1/result.json`.

## Full RF20 routing result

The locked router was then called three times sequentially for every RF20
dataset using identical temperature-0, seed-1234 settings. Majority vote chose
between the same two clean saved detector branches. The router received only
class names; it never received images, predictions, scores, or ground truth.

| Method | Macro mAP50-95 | Macro mAP50 | Estimated total cost |
|---|---:|---:|---:|
| Class names only | 24.37 | 43.54 | $22.28 |
| Fixed one-shot | 25.35 | 46.73 | $34.20 |
| **Strict binary routing** | **25.55** | **46.91** | **$32.64** |

Strict routing improved over class names by **+1.18 / +3.37**. It exceeded
fixed one-shot by only +0.20 / +0.18, which is within the measured API noise and
should be treated as a tie. The routing calls themselves cost $0.064; the
selected detector branches cost $32.57.

The router selected visual references for 15 datasets and class names for five.
All three calls agreed on 19/20 datasets. Soda Bottles was the only unstable
case, with a 2-to-1 majority for class names.

Paper Parts was the development dataset used to refine the conservative prompt.
Across the other 19 datasets, routing scored 25.42 / 46.53 and improved over
their class-names baselines by +0.52 / +2.53. Paper Parts therefore accounts for
a substantial portion of the full RF20 gain.

The router correctly preserved class names on Aerial Airport, Aquarium, FLIR,
Soda Bottles, and Trail Camera. It captured large reference gains on Defect
Detection, Orion, Paper Parts, Dreidel, and WB Prova. Important false-positive
reference decisions remain: All Elements lost 7.59 / 0.65, Global Wheat Head
lost 2.69 / 3.64, and Water Meter lost 16.55 / 29.58 relative to class names.

The result establishes that strict isolated routing can recover an RF20 uplift
without the conversational degradation of the earlier adaptive method. It does
not yet establish a meaningful advantage over always using one reference,
because the accuracy difference is within noise and the router selected the
more expensive branch for most datasets.

Complete local artifacts:
`qwen38-fsod-runs/rf20-strict-binary-router-v1/summary.json` and
`qwen38-fsod-runs/rf20-strict-binary-router-v1/per_dataset.csv`.

## Train-support calibration follow-up

The class-name-only semantic router was subsequently replaced with a direct
train-only A/B gate using the official RF20-VL-FSOD support objects. It
corrected all three major false-reference decisions in a six-dataset diagnostic
pilot. On the other 14 prospectively held-out datasets, it tied fixed one-shot
within 0.03 mAP50-95 while selecting references only 5/14 times. The descriptive
all-20 score was 26.67 / 47.79. See
[QWEN38_SUPPORT_CALIBRATED_ROUTER_RESULT.md](QWEN38_SUPPORT_CALIBRATED_ROUTER_RESULT.md).
