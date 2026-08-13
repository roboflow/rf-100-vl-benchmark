# Qwen3.8 numeric-reference format A/B

## Question

Does making each numeric reference annotation exactly match the requested
prediction JSON shape improve detection?

- Existing grouped form: `{"bbox_2d":[x1,y1,x2,y2]}` under
  `REFERENCE GROUP <class>`.
- Prediction-shaped form:
  `[{"bbox_2d":[x1,y1,x2,y2],"label":"<class>"}]` under the same group.

Both forms use integer XYXY coordinates independently normalized to `[0,1000]`.
All other request content and settings were held fixed.

## Experiment

- Dataset: complete Dreidel RF20-VL-FSOD test split, 54 images and 6 classes.
- Recipe: one positive train reference per class, one multi-class call per target.
- Model/settings: Qwen3.8-Max, temperature 0, seed 1234, reasoning disabled.
- Execution: the two conditions were interleaved by target image.
- Completion: 108/108 successful requests; zero failures or errors.
- Scoring: `pycocotools`, `maxDets=[1,10,500]`.

| Reference annotation | mAP50–95 | mAP50 | Mean latency | Estimated cost |
|---|---:|---:|---:|---:|
| Grouped box only | 50.37 | 72.44 | 8.55 s | $0.57 |
| Prediction-shaped box + label | 50.98 | 74.42 | 8.34 s | $0.55 |
| Delta, prediction-shaped minus grouped | +0.61 | +1.98 | -0.20 s | -$0.03 |

A 500-resample paired-image bootstrap gave a 95% interval of
`[-2.46,+4.04]` for the mAP50–95 delta and `[-3.97,+5.46]` for the mAP50
delta. Both include zero, and the point differences are below Dreidel's
previously measured API-variance floor.

## Decision

The formats are practically tied on this test. There is no evidence that the
extra label/list wrapper materially improves accuracy, but it also caused no
degradation or cost/latency penalty. Use the prediction-shaped form in the
RF20 scale-up because it exactly matches the requested output schema and is
therefore the cleaner format contract—not because this A/B establishes an
accuracy improvement.

Raw responses, predictions, metrics, usage, and the paired bootstrap are under
`qwen38-fsod-runs/reference-format-ab-dreidel-v1/` in the experiment workspace.
