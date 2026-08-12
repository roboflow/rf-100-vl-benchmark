# Qwen3.8-Max RF20-VL-FSOD recipe decision

Status: final two-dataset decision record
Decision date: 2026-08-12
Model/API: `qwen3.8-max` through DashScope International's OpenAI-compatible endpoint

This document explains, step by step, why the study selected a particular
Qwen3.8-Max few-shot object-detection recipe. It covers every requested
decision axis, distinguishes screening results from locked final results, and
states the limits of the conclusion.

Machine-readable artifacts remain authoritative for exact values. The
chronological experiment history is in `QWEN38_RF100VL_RESEARCH_LOG.md`.

## Decision

For the best shared accuracy recipe across the locked Dreidel and Orion test
sets, use:

> **One multi-class request per target image, real class names, two positive
> numeric-box reference examples per class, reasoning disabled. Merge nothing
> because all classes are returned by the same request.**

Here, a numeric-box reference is not a text-only example. The request contains
the original train image and the annotated object's normalized XYXY
coordinates. The same two train-only examples per class are selected
deterministically for every target image.

For a materially cheaper baseline, use:

> **One multi-class class-names-only request per target image, reasoning
> disabled.**

The box recipe is the accuracy-first choice. Class names only are the cost-first
choice because they use about one tenth as many tokens.

## What was optimized

The primary objective was equal-weight macro mAP50-95 across complete Dreidel
and Orion test sets. Secondary evidence was:

- per-dataset mAP50-95 and mAP50;
- API calls per image;
- token use and latency;
- terminal model failures;
- consistency across datasets;
- measured inference/API stochasticity; and
- paired-image bootstrap intervals for close locked finalists.

Scores in this document are **mAP50-95 / mAP50**, in percentage points. COCO
scoring uses pycocotools with `maxDets=[1, 10, 500]`. A terminal model failure
contributes no detections for that request rather than disappearing from the
evaluation.

## Evaluation contract

- Test images come only from each RF20-VL-FSOD `test` split.
- Reference examples come only from `train`.
- No test annotation, object count, validation example, or correction is put
  in a prompt.
- Reference selection is deterministic: largest relative object first, then
  greedy object-crop diversity.
- Numeric and drawn conditions use the same underlying examples.
- Numeric coordinates and predicted boxes use normalized XYXY coordinates in
  `[0, 1000]`, independently scaled by target width and height.
- Locked finalists use `temperature=0`, seed `1234`, reasoning `none`, 8,192
  maximum completion tokens, and a 180-second deadline.
- Every request stores its prompt/settings fingerprint, raw output, parsing
  diagnostics, prediction, latency, token use, and terminal status.

Dataset sizes:

| Dataset | Test images | Classes | Ground-truth objects |
|---|---:|---:|---:|
| Dreidel | 54 | 6 | 171 |
| Orion Products | 59 | 8 | 555 |
| Lacrosse, earlier external screen | 50 | 4 | 355 |

## Step 1: Measure residual stochasticity before declaring winners

Temperature zero and a fixed seed did not make the API deterministic. We ran
the identical class-names-only request over each complete test set ten times.
No image produced identical raw output or identical predictions across all ten
repeats.

| Dataset | Mean mAP | mAP SD | Observed mAP range | mAP tie threshold | mAP50 tie threshold |
|---|---:|---:|---:|---:|---:|
| Dreidel | 40.41 | 0.95 | 3.31 | **3.31** | **5.44** |
| Orion | 15.46 | 1.60 | 4.92 | **4.92** | **9.81** |

The tie threshold is the larger of the observed ten-run range and the 95%
repeatability limit. Differences below it are not treated as reliable wins on
that dataset. This prevented the study from selecting a more expensive prompt
for a small, likely stochastic gain.

The floor is a dataset-level operational threshold measured with one recipe.
We did not run ten repeats of every finalist, so recipe-specific variance may
differ. Paired-image bootstrap intervals quantify finite-image uncertainty,
not API stochasticity; both forms of uncertainty are considered separately.

## Step 2: Decide whether reasoning should be enabled

Historical screens showed occasional reasoning gains on tiny subsets, but
they did not reproduce consistently. The locked stopping rule compared no
reasoning with low reasoning on the same stratified 20 images from each
dataset. Low reasoning had to improve mAP50-95 by more than the dataset's
noise threshold on **both** datasets before medium reasoning would be tested.

| Arm | Dataset | No reasoning | Low reasoning | Low minus none |
|---|---|---:|---:|---:|
| Class names + two drawn boxes | Dreidel | 45.67 / 73.80 | 39.69 / 65.30 | **-5.98 / -8.50** |
| Class names + two drawn boxes | Orion | 10.44 / 31.22 | 14.58 / 38.06 | +4.14 / +6.84 |
| Class names only | Dreidel | 42.15 / 56.72 | 34.59 / 46.09 | **-7.56 / -10.63** |
| Class names only | Orion | 18.64 / 34.34 | 17.72 / 37.32 | -0.92 / +2.98 |

Low reasoning failed both gates:

- it materially hurt both arms on Dreidel;
- its apparent +4.14 Orion mAP gain for drawn boxes was still below Orion's
  4.92-point noise threshold; and
- it more than doubled mean latency in the gate.

Medium and higher reasoning were therefore skipped according to the
predeclared stopping rule. **Decision: reasoning off.**

## Step 3: Establish whether positive box references help

The complete named box-count screens compared class names only with positive
train-image references. These screens used provider-default temperature, so
their exact scores are not mixed causally with the locked final scores. The
within-screen comparisons are still controlled because prompt version,
reference selection, target set, and settings match.

### Multi-class numeric references

| Boxes/class | Dreidel | Orion |
|---:|---:|---:|
| 0, class names only | 40.93 / 54.86 | 10.47 / 19.90 |
| 1 | 49.61 / 70.66 | **18.19 / 43.03** |
| 2 | **54.78 / 77.12** | 13.85 / 36.91 |
| 3 | 50.96 / 75.22 | 14.52 / 33.25 |
| 5 | 52.02 / 78.28 | 12.11 / 32.13 |
| 10 | 51.29 / 76.67 | 12.60 / 34.07 |

Positive boxes clearly helped both datasets, but the optimal count was not
monotonic. Dreidel favored two, while Orion's point estimate favored one.

The earlier three-dataset screen supported the same general pattern. A
one-call positive numeric prompt without reasoning scored:

| Orion | Lacrosse | Dreidel | Three-dataset macro |
|---:|---:|---:|---:|
| 14.89 / 36.93 | 31.29 / 52.08 | 48.56 / 71.43 | **31.58 / 53.48** |

That was the strongest macro result in that earlier six-recipe screen, though
it used an earlier prompt/reference version and provider-default temperature.
It is supporting generalization evidence, not part of the locked final score.

**Decision: include positive box references in the accuracy recipe.**

## Step 4: Decide how many positive examples to use

Two examples were selected by a noise-aware efficiency rule:

1. On Dreidel, two numeric examples beat one by 5.17 mAP, exceeding the 3.31
   mAP noise floor.
2. On Orion, one beat two by 4.34 mAP, which is inside Orion's 4.92 floor.
3. Counts of three, five, and ten did not reliably beat two on either dataset.
4. More examples sharply increased image tokens and latency. On Dreidel, the
   multi-class drawn prompt grew from 10.30 seconds at one box to 35.48 seconds
   at ten boxes without an accuracy gain.

Thus two is the smallest count that captures Dreidel's material gain without a
reliable Orion loss. **Decision: two positive references per class.**

This is a shared-recipe decision. If optimizing Orion alone, one example is a
reasonable candidate; it was not significantly better under the measured
noise floor.

## Step 5: Compare numeric coordinates with boxes drawn on the image

Both encodings include the reference image:

- **Numeric:** original image plus normalized XYXY coordinates in text.
- **Drawn:** the same box rendered visibly on the image.

At two examples per class in the initial multi-class screens:

| Encoding | Dreidel | Orion |
|---|---:|---:|
| Numeric | **54.78 / 77.12** | **13.85 / 36.91** |
| Drawn | 54.06 / **78.08** | 12.49 / 34.24 |

Those differences were within the measured noise floor. The locked final
comparison resolved the ambiguity:

| Encoding | Dreidel | Orion | Macro |
|---|---:|---:|---:|
| Numeric | **53.11 / 76.06** | **16.89 / 41.10** | **35.00 / 58.58** |
| Drawn | 51.53 / 74.30 | 10.75 / 33.07 | 31.14 / 53.69 |

Paired-image bootstrap comparison of drawn relative to numeric:

- Dreidel mAP difference: -1.64, 95% CI `[-4.96, +1.80]`; tied.
- Orion mAP difference: -6.82, 95% CI `[-10.72, -3.28]`; numeric wins.

Numeric also averaged 0.82 seconds less per image in the locked final while
using only 291 more tokens per image. **Decision: numeric coordinates.**

## Step 6: Compare one multi-class call with per-class calls and merging

The matched full-set box screens revealed a real dataset interaction.

### Two examples per class

| Dataset | Encoding | Multi-class | Single-class merged | Direction |
|---|---|---:|---:|---|
| Dreidel | Numeric | **54.78 / 77.12** | 31.63 / 47.88 | Multi +23.15 mAP |
| Dreidel | Drawn | **54.06 / 78.08** | 36.10 / 53.16 | Multi +17.96 mAP |
| Orion | Numeric | 13.85 / 36.91 | **22.76 / 49.03** | Single +8.92 mAP |
| Orion | Drawn | 12.49 / 34.24 | **18.80 / 44.07** | Single +6.31 mAP |

The reversal is larger than the dataset noise thresholds in each direction:
there is no universal per-dataset formulation winner.

For one shared recipe, however, multi-class was selected because:

- its equal-weight two-dataset screening macro at two numeric examples was
  34.31 mAP versus 27.20 for single-class;
- Dreidel's multi-class gain was much larger than Orion's single-class gain;
- multi-class requires one call per image, while single-class requires six
  Dreidel calls or eight Orion calls; and
- the earlier three-dataset screen also favored the one-call combined numeric
  prompt on macro accuracy.

**Decision: one multi-class request for the shared recipe.**

Important limitation: the temperature-zero locked final reran only the four
multi-class finalists. The single-class comparison is based on complete
full-test screening runs, not a temperature-zero finalist rerun. The evidence
is strong enough to reject single-class as the shared macro/efficiency choice,
but not to claim that multi-class is best for every dataset. Orion remains the
clear counterexample.

## Step 7: Test positive plus negative references

The initial complete Orion evaluation compared positive-only references with
positive plus negative references, one request per class:

| Encoding | Positive only | Positive + negative |
|---|---:|---:|
| Numeric | **23.46 / 49.50** | 22.21 / 47.33 |
| Drawn | **21.56 / 47.18** | 16.77 / 42.29 |

Negative references did not improve primary mAP and were not consistently
helpful across later datasets. They also add prompt images and complexity.
They were therefore not advanced to the locked final. **Decision: positive
references only.**

## Step 8: Test box prompting without ground-truth class names

Anonymous prompting maps each hidden class to `Concept A/B/...`; the evaluator
maps those identifiers back to categories only after inference. Two versions
were tested:

- **Explicit:** instruct the model to find objects of the same kind as the
  marked reference.
- **Minimal:** provide references and the output schema without saying
  find/detect/same-kind.

On Dreidel, explicit wording was consistently necessary for competitive
performance. Examples at matched counts include:

| Formulation | Encoding/count | Explicit | Minimal |
|---|---|---:|---:|
| Single-class | Numeric, 2 | **24.03 / 34.55** | 19.11 / 27.47 |
| Single-class | Drawn, 5 | **36.82 / 58.08** | 24.70 / 36.92 |
| Multi-class | Numeric, 2 | **40.34 / 62.71** | 13.77 / 18.00 |
| Multi-class | Drawn, 2 | **33.94 / 57.11** | 20.02 / 31.25 |

So the model can transfer anonymous visual concepts across images, but minimal
box-only prompting is much weaker and produces more malformed/truncated
outputs. If class names are unavailable, use explicit same-kind wording.

The locked final then compared explicit anonymous numeric boxes with the same
numeric boxes plus real class names:

| Semantics | Dreidel | Orion | Macro | Failures | Seconds/image |
|---|---:|---:|---:|---:|---:|
| Real class names | **53.11 / 76.06** | **16.89 / 41.10** | **35.00 / 58.58** | 0 | 11.89 |
| Anonymous explicit | 36.47 / 60.68 | 2.56 / 8.54 | 19.51 / 34.61 | 21 | 55.05 |

Anonymous prompting lost 15.48 macro mAP, generated very long answers, and had
20 failures on Orion alone. **Decision: provide real class names whenever they
are available.**

## Step 9: Test whether model-generated names help

The model generated names from train-only reference examples without seeing
the ground-truth vocabulary. This directly tested whether allowing the model
to name each visual concept before detection could improve anonymous boxes.

On the matched 20-image Dreidel screen:

| Semantics/formulation | Score |
|---|---:|
| Real class names + two drawn boxes, multi-class | **46.86 / 71.53** |
| Real class names only, multi-class | 40.40 / 57.14 |
| Self-generated names only, multi-class | 27.37 / 40.44 |
| Self-generated names + boxes, multi-class | 26.11 / 43.40 |
| Self-generated names + boxes, single-class | 25.63 / 40.74 |

The generated labels were plausible, but fine-grained concepts such as the
different Hebrew symbols were not named precisely enough to recover the
ground-truth semantic advantage. Adding boxes to generated names did not fix
that problem. No self-name arm came within the noise-aware advancement margin.
**Decision: do not self-generate names when real class names are available.**

## Step 10: Run locked finalists on both complete test sets

Four finalists were rerun with the locked settings. All use one request per
target image.

| Finalist | Dreidel | Orion | Macro | Tokens/image | Seconds/image | Failures |
|---|---:|---:|---:|---:|---:|---:|
| **Class names + numeric boxes x2** | **53.11 / 76.06** | 16.89 / **41.10** | **35.00 / 58.58** | 18,797 | 11.89 | 0 |
| Class names + drawn boxes x2 | 51.53 / 74.30 | 10.75 / 33.07 | 31.14 / 53.69 | 18,506 | 12.71 | 0 |
| Class names only | 43.22 / 57.50 | **18.77 / 33.76** | 30.99 / 45.63 | **1,816** | **10.17** | 0 |
| Anonymous explicit + numeric boxes x2 | 36.47 / 60.68 | 2.56 / 8.54 | 19.51 / 34.61 | 22,780 | 55.05 | 21 |

### Why numeric boxes won

Relative to drawn boxes, numeric boxes were tied on Dreidel and materially
better on Orion. Relative to class names only:

- Dreidel improved by **+9.89 mAP / +18.56 mAP50**, well above its noise
  thresholds.
- Orion changed by **-1.89 mAP / +7.34 mAP50**; both differences are within
  Orion's wide noise thresholds.
- The equal-weight macro improved by **+4.00 mAP / +12.95 mAP50**.

Paired-image bootstrap of class names only relative to numeric boxes found:

- a clear Dreidel disadvantage: -11.16 mAP, 95% CI
  `[-18.59, -3.65]`; and
- no resolved Orion mAP difference: +1.36, 95% CI
  `[-3.40, +6.11]`.

Thus boxes provide a real Dreidel improvement without a demonstrated Orion
primary-mAP collapse. Their Orion mAP50 point estimate is also substantially
higher.

## Step 11: Apply the accuracy-versus-efficiency rule

The selected numeric-box prompt and the class-names-only prompt both make one
API call per target image. The accuracy gain is not a near-tie on Dreidel, so
the predeclared selector labels numeric boxes both accuracy-first and
throughput-first.

However, token cost reveals a separate practical tradeoff:

| Recipe | Macro score | Tokens/image | Seconds/image |
|---|---:|---:|---:|
| Numeric boxes x2 | **35.00 / 58.58** | 18,797 | 11.89 |
| Class names only | 30.99 / 45.63 | **1,816** | **10.17** |

Numeric boxes use **10.35x** the tokens and are about **17%** slower per image.
Therefore:

- use numeric boxes x2 when accuracy is the priority;
- use class names only when token cost is the priority; and
- do not describe the numeric recipe as cheapest merely because it is a
  one-call recipe.

## Final answers to the requested decision axes

| Question | Decision | Strength of evidence |
|---|---|---|
| Multi-class vs single-class | Multi-class for one shared macro/efficiency recipe; single-class can win Orion alone | Strong full-set screening; single-class was not rerun as a locked finalist |
| Reasoning level | None | Locked two-dataset stopping-rule test; low failed, medium correctly skipped |
| Number of examples | Two positive examples per class | Full count screens plus noise/efficiency rule |
| Numeric vs drawn boxes | Numeric | Locked two-dataset final; tied Dreidel, clear Orion advantage |
| Text/class-name-only vs box few-shot | Numeric boxes for accuracy; names only for cost | Locked two-dataset final |
| Positive vs positive + negative | Positive only | Complete Orion screen; negatives gave no consistent gain |
| Box-only/minimal anonymous prompt | Do not use; explicit same-kind wording is strongly preferred if names are hidden | Complete Dreidel single- and multi-class screens |
| Anonymous vs real class names | Real names | Locked two-dataset final; large accuracy, latency, and reliability advantage |
| Self-generated names | Do not use when real names exist | Matched stratified causal screen |
| Temperature/seed | `temperature=0`, seed `1234`, but still treat outputs as stochastic | Ten complete repeats per dataset |

## What the result does and does not establish

Established:

- Numeric positive references plus real names are the best locked finalist on
  equal-weight Dreidel/Orion macro accuracy.
- Reasoning is not justified for this task and tested prompt family.
- Two examples are a better shared efficiency point than larger reference
  counts.
- Anonymous/minimal and self-naming approaches are not competitive when real
  class names are available.
- Cost and accuracy recommendations should be reported separately.

Not established:

- That this recipe is optimal for every RF20/RF100 dataset. Orion's formulation
  preference differs from Dreidel's, and Lacrosse earlier favored class names
  on primary mAP.
- That multi-class always beats single-class. Orion's complete screen showed
  the opposite.
- That two examples are always better than one. The decision is for a shared
  recipe under measured noise and efficiency constraints.
- That temperature zero eliminates sampling variance. It demonstrably does
  not for this provider/model combination.
- That the measured class-names-only noise floor is identical for every prompt
  family.

If stronger external-validity confidence is needed, the smallest useful next
experiment is not another broad factorial. It is a locked temperature-zero
evaluation on one new, preselected dataset of:

1. class names only, multi-class;
2. class names + numeric boxes x2, multi-class; and
3. class names + numeric boxes x2, single-class merged.

That would test the chosen recipe and the unresolved formulation interaction
without reopening already-settled reasoning, self-name, anonymous, negative,
or high-box-count branches.

## Canonical artifacts

- Final human-readable report:
  `qwen38-fsod-runs/final-recipe-study/final-analysis/final_report.md`
- Final metrics and bootstrap intervals:
  `qwen38-fsod-runs/final-recipe-study/final-analysis/final_report.json`
- Noise floors:
  `qwen38-fsod-runs/final-recipe-study/noise_floor.json`
- Reasoning decision:
  `qwen38-fsod-runs/final-recipe-study/reasoning_low_decision.json`
- Self-name screen:
  `qwen38-fsod-runs/final-recipe-study/dreidel-self-name-screen/`
- Dreidel and Orion named box-count screens:
  `qwen38-fsod-runs/dreidel-box-count-ablation-v1/` and
  `qwen38-fsod-runs/orion-box-count-ablation-v1/`
- Anonymous single- and multi-class screens:
  `qwen38-fsod-runs/dreidel-exemplar-only-box-combined-v1/` and
  `qwen38-fsod-runs/dreidel-anonymous-multi-screen-v1/`
- Chronological research record: `QWEN38_RF100VL_RESEARCH_LOG.md`
