# Qwen3.8-Max RF20-VL-FSOD research handoff

> **Historical snapshot, superseded.** This file captures the state on
> 2026-08-24 while experiments were still running. Do not use its active-run
> status or recommendations as current. Start with
> [`QWEN38_RF100VL_RESEARCH_LOG.md`](../QWEN38_RF100VL_RESEARCH_LOG.md), which
> contains the completed results, exact Rex-Omni F1 correction, final ranking,
> stop decision, and current artifact index.

## Assignment for the next agent

Read this document and the linked artifacts, then give an independent opinion
on the most promising next experiment. Start with a critique and a ranked plan,
not code. In particular, assess the proposed zero-cost cross-mode consensus
screen in [Recommended next work](#recommended-next-work).

Do not stop, modify, or duplicate the two running experiments described below.
Do not launch paid API inference without explicit approval. Free analysis of
existing predictions is welcome, but any thresholds or policies intended for a
test result must be locked on train/support or validation data before test is
scored.

## Research objective

We want a simple and defensible inference-time scaling or adaptation method for
Qwen3.8-Max object detection on RF20-VL-FSOD that does one of two things:

1. improves COCO mAP50-95 over the best current deployable recipe; or
2. preserves its accuracy while reducing inference cost.

The method should use RF20-VL-FSOD as intended, remain label-free on test
images, avoid a large hand-engineered feature set, and ideally expose a clear
compute-versus-accuracy knob.

## Locked evaluation protocol

- Dataset: all 20 official RF20-VL-FSOD datasets, 3,970 test images.
- Visual references: official train/support split only.
- Detection prompt: one multi-class request per image unless a method explicitly
  studies isolated single-class calls.
- Box format: `{"bbox_2d":[x1,y1,x2,y2],"label":"class"}` with normalized
  0-1000 XYXY coordinates. Numeric references use the same object and coordinate
  format as predictions.
- Reasoning: disabled with both provider controls.
- Sampling: temperature 0 and seed 1234. The hosted API is nevertheless not
  deterministic.
- Scoring: pycocotools, primary metric mAP50-95, secondary mAP50,
  `maxDets=[1,10,500]`.
- Every request, raw response, parsed prediction, usage record, and failure is
  checkpointed.
- Test ground truth may score a frozen policy, but may never select a route,
  threshold, prompt, reference, or stopping point for that same test result.

The core evaluator is [`evaluate_qwen38_recipe.py`](../evaluate_qwen38_recipe.py).

## Current fixed-prompt evidence

Scores are RF20 dataset-macro `mAP50-95 / mAP50`.

| Method | Score | Estimated detector cost | Interpretation |
|---|---:|---:|---|
| Class names only | 24.37 / 43.54 | $22.28 | Cheapest clean baseline |
| Annotator instructions | 24.46 / 44.58 | $26.21 | Within noise overall |
| 1 positive numeric reference/class | 25.35 / 46.73 | $34.20 | Best simple accuracy-cost tradeoff |
| 10 positive numeric references/class | 25.74 / 47.92 | $116.79 | Highest fixed-prompt score; lead over 1-shot is within noise |

The dollar values are reproducible list-price estimates from recorded token
usage, not invoice totals. See
[`QWEN38_CONTEXT_ADAPTATION_REPORT.md`](../QWEN38_CONTEXT_ADAPTATION_REPORT.md).

## Main mechanism discovered

Visual references act like dataset adaptation rather than universal extra
context. They help most when a class name under-specifies appearance, state,
role, or annotation semantics, and can hurt when the name already identifies a
familiar visual object.

| Pre-rated dataset group | Datasets | 1-reference gain | 10-reference gain |
|---|---:|---:|---:|
| No class names under-specified | 9 | -2.52 / -3.46 | -2.28 / -3.05 |
| Some class names under-specified | 8 | +2.72 / +6.77 | +2.17 / +7.10 |
| Every class name under-specified | 3 | +6.82 / +13.57 | **+10.20 / +19.43** |

At class level, one reference gained +6.68 / +12.68 on under-specified
classes but lost 2.22 / 1.84 on sufficient-name classes. Ten references gained
+6.57 / +13.88 on under-specified classes but lost 3.89 / 4.97 on
sufficient-name classes.

Examples of large visual-reference gains include railway defect types, Dreidel
states and symbols, document components, animal age groups, smoke, and opaque
product-family names. Regressions include digits, cars, dogs, fish, penguins,
and wheat heads. All Elements is an important counterexample: its labels can be
semantically specialized, but visual prompting still regressed strongly.

The relationship is predictive, not deterministic. It motivates dataset-level
routing but is not a reliable per-class rule by itself.

## Repeatability and noise

Temperature 0 and a fixed seed do not make the hosted API deterministic.
Six-run matched estimates on larger datasets were:

| Dataset | Names-only score noise | One-reference gain noise |
|---|---:|---:|
| Actions, 409 images | 0.49 / 1.19 | 0.68 / 2.58 |
| Paper Parts, 500 images | 0.95 / 3.87 | 3.01 / 3.86 |
| Defect Detection, 188 images | 2.17 / 2.89 | 3.17 / 4.42 |

These are conservative repeatability thresholds, not an RF20-wide confidence
interval. The complete study is
[`QWEN38_LARGE_DATASET_NOISE_RESULT.md`](../QWEN38_LARGE_DATASET_NOISE_RESULT.md).

The hosted variance is also the motivation for same-prompt candidate consensus:
independent generations contain diversity even at temperature 0.

## Current deployable routing result

The strongest completed simple pipeline is a train/support-calibrated
dataset-level gate:

1. reserve one support object per class as the visual reference;
2. compare clean class-names and 1-shot detectors on the remaining labeled
   support objects using known-object recall, because support annotations may be
   sparse;
3. make one dataset-wide decision;
4. run one clean detector branch on every test image.

| Method | Score | Selected detector cost |
|---|---:|---:|
| Support-calibrated 0/1 | 26.67 / 47.79 | $30.38 |
| Support-calibrated gate replayed as 0/10 | **27.26 / 49.23** | $92.87 |
| Dataset-level 0/10 test oracle | 27.95 / 50.91 | upper bound |

The 0/1 result is descriptive on all 20 because six datasets were used during
policy development, but it was effectively tied with fixed 1-shot on the 14
prospectively held-out datasets while selecting references for only 5/14. The
0/10 extension is accuracy-first and reuses the same gate. Details:
[`QWEN38_SUPPORT_CALIBRATED_ROUTER_RESULT.md`](../QWEN38_SUPPORT_CALIBRATED_ROUTER_RESULT.md).

## Per-class routing: headroom and deployability problem

Existing clean names and 10-shot test predictions show real per-class
compositional headroom:

| Method | Score |
|---|---:|
| Dataset 0/10 support-calibrated | 26.89 / 48.21 |
| Locked dataset 0/1 gate replayed as 0/10 | 27.26 / 49.23 |
| Direct per-class 0/10 support calibration | 26.56 / 47.76 |
| Five-fold per-class AP cross-fit, mean | **28.21 / 51.00** |
| Per-class test oracle | 28.82 / 51.84 |

The 28.21 score is not an untouched-test result: complete annotations from
other test folds calibrated each held-out fold. It estimates what a separate,
fully annotated calibration set could enable. Sparse support calibration is too
noisy per class and chose the correct branch only 62.4% of the time.

A proper validation experiment then selected 0, 1, or 10 per class using all
739 validation images. Splicing the corresponding existing clean test outputs
scored 26.92 / 49.46 and selected 37 classes at zero, 38 at one, and 35 at ten.
This is label-clean but still an output-splicing analysis, not one actual mixed
prompt.

Details:
[`QWEN38_PER_CLASS_ZERO_TEN_RESULT.md`](../QWEN38_PER_CLASS_ZERO_TEN_RESULT.md)
and [`analysis/per-class-zero-one-ten-validation-v1/summary.json`](../analysis/per-class-zero-one-ten-validation-v1/summary.json).

## Actual mixed per-class prompt: currently running

The deployability test places each validation-selected 0/1/10 allocation into
one real multi-class prompt per test image. Classes may therefore receive
different reference counts inside the same context. This tests whether the
output-splicing gain survives cross-class prompt interaction.

Snapshot at 2026-08-24 08:32 UTC: 12/20 datasets complete. The run was resumed
from checkpoints after the host closed a non-lingering user session. User
service lingering is now enabled. The remaining datasets are Recode Waste,
Soda Bottles, Dreidel, Trail Camera, Water Meter, WB Prova, Wildfire Smoke, and
X-Ray ID.

Intermediate macro over the 12 completed datasets:

| Method | mAP50-95 / mAP50 |
|---|---:|
| Class names | 22.22 / 41.77 |
| Fixed 1-shot | **23.31 / 45.77** |
| Fixed 10-shot | 22.77 / 45.34 |
| Mixed 0/1/10 prompt | 22.54 / 44.56 |

Mixed prompting is +0.32 / +2.79 over names on this subset but trails both
uniform visual modes. It averages -2.77 / -3.40 against the independently best
uniform mode per completed dataset. Large primary regressions include All
Elements (-11.87), Orion (-4.27), and FLIR (-4.00). Paper Parts is essentially
tied with its best uniform mode (-0.04 / +0.20). Treat these as provisional
until all 20 finish.

Run root:
[`qwen38-fsod-runs/rf20-validation-routed-zero-one-ten-mixed-v1/`](../qwen38-fsod-runs/rf20-validation-routed-zero-one-ten-mixed-v1/).

## What did not work

### Model self-routing per image

A separate model call saw the target image, the saved names-only prediction,
and the dataset-level prior, then selected names or references. On a held-out
2,209-image evaluation, it scored -0.21 / -0.23 below the dataset prior and
cost an additional $8.02. A one-sided names-to-references variant was an RF20
accuracy tie with extra cost. Low and medium reasoning made decisions worse and
cost more. See
[`QWEN38_PER_IMAGE_ROUTER_RESULT.md`](../QWEN38_PER_IMAGE_ROUTER_RESULT.md).

### Progressive multi-turn shot acquisition

The trajectory used 0, 1, 2, 5, and 10 references and stopped after two
consecutive perfectly stable transitions. It matched its own multi-turn
10-shot endpoint at 56% lower trajectory cost, so box stability is a viable
label-free stopping signal. It did not improve accuracy: fixed 1-shot was both
more accurate and much cheaper on the pilot, and the multi-turn 10-shot endpoint
was substantially worse than the established standalone 10-shot prompt because
the conversation anchored later outputs to earlier predictions. See
[`QWEN38_PROGRESSIVE_TTS_RESULT.md`](../QWEN38_PROGRESSIVE_TTS_RESULT.md).

### Sequential support acquisition

Support-only sequential shot acquisition improved over names on Dreidel and
Lacrosse but was sensitive to support order and had 4.16 / 10.06 macro regret
against the test-grid oracle. It was not advanced to RF20. See
[`QWEN38_SSA_THREE_DATASET_RESULT.md`](../QWEN38_SSA_THREE_DATASET_RESULT.md).

### Instructions as universal context

Correct annotator instructions beat shuffled instructions, proving that the
model uses their content. They still did not improve RF20 reliably. On a
six-dataset matched control, correct instructions scored -2.97 / -4.13 below
class names. Instructions plus a visual reference also failed to stack
reliably. Visual examples transferred dataset-specific appearance more
effectively.

### Hand-engineered per-image confidence features

Missing-class, count-mismatch, confidence, geometry, overlap, and similar cheap
signals did not reliably identify which prompt branch was better. The simple
feature screen produced AUC around 0.39. Expanding the feature hunt was rejected
because it creates researcher degrees of freedom without a transferable
intrinsic signal.

### Reasoning

Reasoning was more than twice as slow in the early prompt screens, consumed many
reasoning tokens, hurt Dreidel by roughly 6-8 mAP, and never produced a gain on
Orion that exceeded its measured noise. Keep reasoning disabled for detection
and routing unless a new, tightly bounded validation overturns this.

## Same-prompt self-ensemble: implemented and queued

The current new direction requests three candidates in one non-thinking API
call using the locked dataset 0/10 prompt. Candidates are parsed independently
and fused only within the same image and class:

- matching threshold: IoU >= 0.5;
- at most one box from each candidate can enter a cluster;
- coordinates are averaged;
- score is dominated by candidate vote fraction (3/3, 2/3, 1/3), with mean
  matched IoU only breaking ties;
- singletons are retained at low confidence;
- raw candidates and fused predictions are both scored and saved.

The implementation is in
[`qwen38_box_ensemble.py`](../qwen38_box_ensemble.py), with focused tests in
[`test_qwen38_box_ensemble.py`](../test_qwen38_box_ensemble.py). Seventy-one
focused evaluator/fusion tests passed and all 20 manifests passed no-API
preflight.

A free screen fused already-saved repeated generations on Dreidel, Orion,
Lacrosse, Actions, Defect Detection, and Paper Parts:

| Candidates | Fusion | Mean delta vs corresponding standalone mean | Positive groups |
|---:|---|---:|---:|
| 2 | soft vote | +1.92 / +3.54 | 17/21 |
| 3 | soft vote | **+2.80 / +5.13** | **10/12** |
| 3 | majority only | +1.49 / +2.99 | 8/12 |
| 5 | soft vote | +3.79 / +6.34 | 9/9 |

This is encouraging but exploratory: the historical repeats were separate API
calls, whereas the live experiment asks for `n=3` choices in one request. The
gain may come from box-coordinate averaging, vote-based confidence ranking, or
both; that mechanism has not yet been separated.

The persistent queue will run a 21-request live smoke, full Dreidel/Orion/
Lacrosse gate, then all RF20 only if the gate passes. It uses the same 3,970 test
requests as the dataset router but returns three candidates per request.

Run roots:

- [`analysis/qwen38-self-ensemble-offline-v1/summary.json`](../analysis/qwen38-self-ensemble-offline-v1/summary.json)
- [`qwen38-fsod-runs/rf20-self-ensemble-n3-router-v1/`](../qwen38-fsod-runs/rf20-self-ensemble-n3-router-v1/)
- [`analysis/qwen38-self-ensemble-rf20-v1/`](../analysis/qwen38-self-ensemble-rf20-v1/)

## Recommended next work

### 1. Free cross-mode consensus screen (highest priority, not yet run)

Fuse the already-saved clean class-names, 1-shot, and 10-shot predictions at
the box level. This is different from putting different references in one
prompt: every detector remains isolated and only its output is combined.

Why it is promising:

- names and references make complementary dataset- and class-specific errors;
- consensus supplies an intrinsic confidence score where every raw VLM box
  currently has score 1.0;
- the current same-prompt offline ensemble shows that vote-ranked box fusion can
  materially improve AP;
- complete validation and test predictions for all three modes already exist,
  so the first study costs $0 in API calls;
- it directly targets higher accuracy rather than asking the model to estimate
  its own confidence.

Use the 739-image validation split to choose exactly one global rule, then score
test once. Predeclare a small matrix before looking at validation results:

1. branches: 0+1, 0+10, 1+10, and 0+1+10;
2. class-aware IoU 0.5 matching;
3. soft vote with singletons retained versus strict majority;
4. one mechanism ablation: vote scores with representative coordinates versus
   vote scores plus coordinate averaging.

Do not select a different fusion policy per dataset or class. Report validation
selection, untouched test result, inference-call count, and list-price cost.
Deployment would require two or three clean detector calls per image, so it is
an accuracy-first method unless a later gate reduces calls.

Existing roots:

- 0 and 1: [`qwen38-fsod-runs/rf20-three-way-matched-v1/`](../qwen38-fsod-runs/rf20-three-way-matched-v1/)
- 10: [`qwen38-fsod-runs/rf20-all-available-explicit-sparse-v1/`](../qwen38-fsod-runs/rf20-all-available-explicit-sparse-v1/)
- validation 0 and 10: [`qwen38-fsod-runs/rf20-validation-zero-ten-v1/`](../qwen38-fsod-runs/rf20-validation-zero-ten-v1/)
- validation 1: [`qwen38-fsod-runs/rf20-validation-one-v1/`](../qwen38-fsod-runs/rf20-validation-one-v1/)

### 2. If live `n=3` succeeds, simulate an `n=2 -> n=3` compute gate

This is the simplest genuine per-image test-time-scaling extension:

1. request two same-prompt candidates;
2. compute one intrinsic scalar, class-aware box agreement;
3. accept fused `n=2` when agreement is high;
4. request a third candidate only when agreement is low, then fuse all three.

First simulate this for $0 from saved `n=3` candidates. Lock one global
agreement threshold on validation or a prospectively held-out development
split. Report accuracy versus the fraction escalated and token cost. Then run a
small live test because an extra standalone third request repeats the image
input and may not have exactly the same distribution as choice 3 from a single
`n=3` call.

This method is preferable to another learned confidence gate: it uses one
model-intrinsic, label-free signal and gives compute a direct per-image knob.

### 3. Isolated selective class rescue (accuracy-first, higher complexity)

The cross-fitted per-class result proves that some classes want a different
branch from their dataset. The actual mixed prompt is losing that benefit,
probably because references for one class perturb detections for other classes.

A cleaner alternative is:

1. run the locked clean dataset-level branch once;
2. use validation mAP50-95 to preselect only strongly evidenced rescue classes;
3. issue isolated single-class alternate-prompt calls only for those classes;
4. replace that class's predictions in output space.

This preserves context isolation and could approach the 28.21 cross-fitted
headroom, but it increases calls per image and needs new single-class inference
for most datasets. It should be attempted only after a cost calculation and a
small pilot on Actions, All Elements, and Paper Parts.

### 4. Global reference-selection study for the 1-shot Pareto mode

The established first reference favors a large object relative to its image.
That may bias one-shot results against small or atypical appearances. If the
goal is to optimize the practical $34 one-shot mode, compare a few globally
predeclared train-only choices, such as largest, medoid crop, and a fixed seeded
sample. Select one policy across datasets rather than per dataset. This is
lower priority because all-available 10-shot avoids subsampling and because
earlier small A/B screens did not establish a universal selector.

### 5. Test-time image augmentation or tiling (separate future direction)

Original-image plus horizontal-flip or deterministic tile inference could help
small objects and supplies natural predictions to fuse. It is model-agnostic
and has a clean compute knob, but it changes image preprocessing and coordinate
inversion, so it should be treated as a separate study after prompt/candidate
consensus is resolved.

## Directions not worth repeating without new evidence

- More reasoning levels for detection or routing.
- Another free-form request asking the model whether it is confident.
- Additional hand-engineered confidence/geometry feature searches on the same
  RF20 pool.
- More multi-turn refinement schedules that carry previous predictions in the
  conversation.
- More shot-count sweeps without solving reference selection or prompt
  interference.
- Per-class reference counts mixed inside one prompt, unless the completed
  20-dataset run unexpectedly reverses the current trend.
- Treating test oracles, cross-fitted test calibration, or test-selected fusion
  thresholds as deployable benchmark scores.

## Suggested decision sequence

1. Wait for the mixed-prompt and live `n=3` results already in progress.
2. Independently run the $0 validation-locked cross-mode consensus analysis.
3. If live `n=3` wins, compare it with cross-mode consensus on accuracy, calls,
   tokens, latency, and robustness.
4. Use saved `n=3` choices to simulate the `n=2 -> n=3` agreement gate.
5. Advance selective class rescue only if neither consensus method closes enough
   of the gap to the 28.21 cross-fitted upper-bound estimate.

The next useful deliverable should be a short recommendation answering:

- Which single experiment has the best expected accuracy gain per dollar?
- Which result would constitute a genuine per-image inference-time-scaling
  method rather than amortized dataset adaptation?
- What validation split and one global policy prevent further test-set tuning?
- What clear stop condition prevents another open-ended feature search?
