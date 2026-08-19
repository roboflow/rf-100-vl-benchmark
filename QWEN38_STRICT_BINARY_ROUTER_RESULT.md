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
