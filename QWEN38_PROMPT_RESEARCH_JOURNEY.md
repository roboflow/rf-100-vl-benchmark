# Qwen3.8-Max RF20-VL-FSOD prompt research journey

## Current conclusion

Use one multi-class request per target image, real class names, positive numeric
`bbox_2d` references in the model's normalized 0–1000 XYXY prediction format,
reasoning off, `temperature=0`, and seed `1234`.

- **Default/cost-first:** class names only.
- **Optional visual-prompt mode:** positive numeric references with explicit
  sparse-label wording. Across RF20, one reference improved macro mAP50-95 / mAP50
  by only **+0.98 / +3.19**, so the extra cost was not justified as the default.
- **Current final experiment:** use all official 10-shot train references on all
  20 RF20-VL-FSOD datasets. This removes reference sub-sampling as a confound and
  tests whether more examples help particular datasets even when the RF20 macro
  gain is small.

All results use test targets, train-only references, robust response parsing,
normalized-box conversion to original-image COCO coordinates, and pycocotools
with `maxDets=[1,10,500]`. Values below are mAP50-95 / mAP50.

## How the study progressed

### 1. Broad prompt screen on Orion

We first compared class names, numeric and drawn references, positive and
positive-plus-negative references, single-class calls with merging, one
multi-class call, and no versus low reasoning.

| Best representative modes | 20-image Orion |
|---|---:|
| Class names, multi-class, no reasoning | **24.38 / 40.99** |
| Positive numeric, single-class merged, no reasoning | 23.84 / 48.86 |
| Positive + negative numeric, single-class merged, no reasoning | 23.42 / **49.57** |
| Positive numeric, multi-class, no reasoning | 18.32 / 41.59 |

The complete 59-image Orion screen gave **23.46 / 49.50** for positive numeric
single-class merged. This established that cross-image box prompting works, but
also showed that the best formulation may depend on the dataset.

### 2. Cross-dataset formulation and reference-count tests

Dreidel, Orion, and Lacrosse reversed several Orion-only rankings. Multi-class
strongly beat single-class on Dreidel, while single-class won on Orion.
Multi-class became the shared recipe because its macro result was stronger and
it needs one call rather than one call per class.

Counts from 1 through 10 were non-monotonic:

| Positive numeric references/class | Dreidel | Orion |
|---:|---:|---:|
| 1 | 49.61 / 70.66 | **18.19 / 43.03** |
| 2 | **54.78 / 77.12** | 13.85 / 36.91 |
| 3 | 50.96 / 75.22 | 14.52 / 33.25 |
| 5 | 52.02 / 78.28 | 12.11 / 32.13 |
| 10 | 51.29 / 76.67 | 12.60 / 34.07 |

Two examples were initially selected as a compromise, not as a universal
optimum. One remained attractive for simplicity.

### 3. Eliminate weaker prompt branches

- Numeric coordinates tied drawn boxes within noise on Dreidel and clearly won
  on Orion. In the locked two-dataset comparison, numeric gained **+3.86 / +4.89**
  macro over drawn.
- Low reasoning was more than 2x slower, hurt Dreidel by roughly 6–8 mAP, and
  produced no Orion gain beyond noise. Medium and higher reasoning were skipped.
- Positive-plus-negative references did not beat positive-only references.
- Anonymous box-only prompts worked but were substantially weaker than real
  class names plus boxes. Minimal box-only wording was worse than explicit
  same-kind wording.
- Model-generated class names did not recover the loss from withholding the
  ground-truth names.

### 4. Measure provider variance

Ten otherwise identical full-test repeats used `temperature=0`, fixed seed
`1234`, and reasoning off. Outputs still varied materially:

| Dataset | mAP50-95 range | mAP50 range |
|---|---:|---:|
| Dreidel, 54 images | 3.31 | 5.44 |
| Orion, 59 images | 4.92 | 9.81 |

Strictly sequential repeats on Dreidel, Orion, and Lacrosse confirmed that
interleaving was not the explanation. Later 188–500-image experiments had much
smaller baseline noise, so small test sets were an important contributor.
Differences inside the measured repeatability threshold are treated as ties.

### 5. Match reference and prediction formats

We changed numeric references from coordinate-only JSON to the model's own
prediction-shaped format: `[{"bbox_2d":[x1,y1,x2,y2],"label":"class"}]`.
The normalized coordinate convention was unchanged. On Dreidel, Orion, and
Lacrosse this gained **+1.32 / +2.83** macro; Orion improved clearly, and the
other two datasets tied within noise. All later reference prompts use this
matched format.

### 6. Scale to all RF20-VL-FSOD datasets

| Recipe | RF20 macro mAP50-95 | RF20 macro mAP50 | Delta vs names | Estimated cost |
|---|---:|---:|---:|---:|
| Class names only | 24.37 | 43.54 | baseline | $22.28 |
| One positive numeric reference/class | **25.35** | **46.73** | +0.98 / +3.19 | $34.20 |
| Two positive numeric references/class | 25.14 | 46.55 | +0.77 / +3.01 | $44.43 |

One and two examples improved 14/20 datasets on mAP50-95, but the RF20 macro
gain was modest relative to added cost. This changed the default recommendation
from mandatory examples to class names only, with visual prompting optional.

### 7. Recheck uplift and noise on larger datasets

Six matched runs showed that one reference had clear, repeatable gains on Paper
Parts and Actions. Defect Detection was mixed.

| Dataset | Images | One-reference uplift | Baseline run-to-run noise |
|---|---:|---:|---:|
| Paper Parts | 500 | +13.43 / +20.07 | 0.95 / 3.87 |
| Actions | 409 | +2.22 / +4.98 | 0.49 / 1.19 |
| Defect Detection | 188 | +2.03 / +3.60 | 2.17 / 2.89 |

The value of box prompting is therefore real but highly dataset-dependent.

### 8. Revisit example count and selection bias

On Defect Detection, 1, 2, 5, and 10 numeric references scored 31.26 / 49.98,
37.02 / 56.74, 38.70 / 60.98, and 38.72 / 61.97. Five and ten were effectively
tied, showing a plateau after the large 1-to-2 gain.

The original nested selector chose the largest relative-area object first and
then maximized crop diversity. A largest-versus-median one-shot A/B changed
scores beyond noise on Actions and Defect Detection, with opposite directions.
Reference selection was therefore a real, dataset-dependent confound. Keeping
the largest anchor and choosing four fixed random objects also underperformed
the diversity-selected five-shot Defect recipe by **-6.02 / -5.51**.

Explicit wording now tells the model that marked boxes are sparse positive
exemplars and that unmarked objects or regions are unlabeled, not negatives.
On Defect five-shot, this wording changed scores by only **+0.25 / -0.17**, well
inside noise, but it is semantically safer and is retained.

## Why the all-10 experiment is next

Sub-sampling examples introduces a selection policy whose effect can be as
large as the prompt-method effect. Using all official RF20-VL-FSOD 10-shot
train references avoids that choice, preserves one multi-class call per target,
and directly tests whether the Defect Detection plateau generalizes across all
20 datasets. It does not replace the class-names-only production default unless
its RF20 macro gain and per-dataset consistency justify the additional input
cost.

## Reproducibility artifacts

- Full detailed chronology: [`QWEN38_RF100VL_RESEARCH_LOG.md`](QWEN38_RF100VL_RESEARCH_LOG.md)
- RF20 0/1/2 result: [`QWEN38_RF20_FSOD_RESULT.md`](QWEN38_RF20_FSOD_RESULT.md)
- Final two-dataset decision: [`QWEN38_FINAL_RECIPE_DECISION.md`](QWEN38_FINAL_RECIPE_DECISION.md)
- Large-dataset noise study: [`QWEN38_LARGE_DATASET_NOISE_RESULT.md`](QWEN38_LARGE_DATASET_NOISE_RESULT.md)
- Reference-format A/B: [`QWEN38_REFERENCE_FORMAT_AB.md`](QWEN38_REFERENCE_FORMAT_AB.md)
