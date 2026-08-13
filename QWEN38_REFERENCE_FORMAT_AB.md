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

- Datasets: complete Dreidel, Orion, and Lacrosse RF20-VL-FSOD test splits;
  163 images total.
- Recipe: one positive train reference per class, one multi-class call per target.
- Model/settings: Qwen3.8-Max, temperature 0, seed 1234, reasoning disabled.
- Execution: the two conditions were interleaved by target image.
- Completion: 326/326 successful requests; zero failures or errors.
- Scoring: `pycocotools`, `maxDets=[1,10,500]`.

| Dataset | Grouped mAP50–95 / mAP50 | Prediction-shaped mAP50–95 / mAP50 | Delta |
|---|---:|---:|---:|
| Dreidel | 50.37 / 72.44 | 50.98 / 74.42 | +0.61 / +1.98 |
| Orion | 15.09 / 36.40 | 18.39 / 43.19 | +3.30 / +6.79 |
| Lacrosse | 33.63 / 56.11 | 33.69 / 55.83 | +0.06 / -0.27 |
| Three-dataset macro | 33.03 / 54.98 | 34.35 / 57.82 | **+1.32 / +2.83** |

The 500-resample paired-image mAP50–95 intervals were `[-2.46,+4.04]` on
Dreidel, `[+1.08,+5.93]` on Orion, and `[-1.82,+2.47]` on Lacrosse. Orion's
paired result resolves in favor of prediction-shaped references; Dreidel and
Lacrosse are ties. Lacrosse's small negative mAP50 delta is well within noise.
The complete three-dataset test cost an estimated $2.61.

## Decision

Use the prediction-shaped form in the RF20 scale-up. It exactly matches the
requested output schema, improves the three-dataset macro, materially helps
Orion, and has no resolved regression on Dreidel or Lacrosse. The evidence does
not establish a universal accuracy gain—the per-dataset effects include two
ties—but it removes a needless input/output schema discrepancy without an
observed downside.

Raw responses, predictions, metrics, usage, and paired bootstraps are under the
three `qwen38-fsod-runs/reference-format-ab-*-v1/` directories in the
experiment workspace.
