# Qwen3.8-Max FSOD support-calibrated routing result

## Result

The semantic class-name router was replaced with a train-only empirical gate.
One RF20-VL-FSOD support object per class is reserved as the exact reference
used by the established one-shot prompt. The class-names-only and one-shot
detectors are compared on the remaining support objects. No test image or test
annotation is used to make the routing decision.

The gate was locked after a six-dataset diagnostic pilot, then evaluated
prospectively on the other 14 RF20 datasets:

| Held-out 14 method | mAP50-95 | mAP50 | Delta vs names | Selected detector cost |
|---|---:|---:|---:|---:|
| Class names only | 24.88 | 43.71 | baseline | $11.67 |
| Fixed one-shot | **26.64** | **47.55** | +1.76 / +3.84 | $16.68 |
| **Support-calibrated gate** | **26.62** | 46.65 | **+1.73 / +2.94** | **$13.13** |
| Test oracle, upper bound | 27.00 | 47.74 | +2.11 / +4.03 | $15.94 |

On the genuinely held-out datasets, the gate was effectively tied with fixed
one-shot on the primary metric, trailing by only 0.03 mAP50-95 while selecting
the one-shot branch for 5/14 datasets instead of all 14. It retained 82% of the
oracle's available mAP50-95 improvement.

The exact higher-mAP50-95 branch matched on 8/14 datasets. All six disagreements
were marginal: the missed one-shot advantages ranged from 0.11 to 2.11
mAP50-95, below the smallest 3.31-point identical-run range measured in the API
variance study. Their mAP50 differences were also within the corresponding
5.44-point range. Therefore, the gate made no clearly wrong above-noise decision
on the held-out 14.

Across all 20 datasets, including the six diagnostic datasets, the descriptive
result is:

| All-20 method | mAP50-95 | mAP50 | Delta vs names | Selected detector cost |
|---|---:|---:|---:|---:|
| Class names only | 24.37 | 43.54 | baseline | $22.28 |
| Fixed one-shot | 25.35 | 46.73 | +0.98 / +3.19 | $34.20 |
| **Support-calibrated gate** | **26.67** | **47.79** | **+2.30 / +4.25** | **$30.38** |
| Test oracle, upper bound | 26.94 | 48.56 | +2.57 / +5.01 | $33.19 |

The gate selected class names for 12 datasets and one-shot for eight. It
captured 90% of the oracle's available mAP50-95 improvement. The all-20 score is
descriptive rather than a fully prospective routing estimate because the first
six datasets were intentionally chosen to contain known successes and failures.

## Accuracy-first extension: route between zero and ten references

The same locked gate can keep its class-names-only decisions but use all ten
available support objects per class wherever it selects the visual-reference
branch. Replaying the already completed ten-reference predictions gives:

| Scope | Calibrated 0/1 | Calibrated 0/10 | Delta from using 10 | 0/10 detector cost |
|---|---:|---:|---:|---:|
| Held-out 14 | 26.62 / 46.65 | **27.39 / 48.26** | **+0.77 / +1.61** | $18.79 |
| All 20 | 26.67 / 47.79 | **27.26 / 49.23** | **+0.60 / +1.44** | $92.87 |

Values are mAP50-95 / mAP50. Ten references beat one on five of the eight
visual-routed datasets and lost on three, so more context does not improve every
dataset. However, ten references remained better than class names on all eight
datasets selected by the gate. This supports an accuracy-first 0/10 mode at
RF20 macro level, not an expectation that every selected dataset improves over
0/1.

The all-20 selected detector cost rises from $30.38 for 0/1 to $92.87 for 0/10.
Including the one-time $10.62 calibration makes the 0/10 total $103.49, compared
with $116.79 for fixed ten-shot on every dataset. The gain over 0/1 is therefore
modest relative to the additional cost. Use 0/1 for efficiency and 0/10 only
when measured accuracy is the priority.

This extension is an exact, no-new-inference replay, but it is post-hoc: the
gate was locked prospectively for choosing class names versus one reference,
not for choosing class names versus ten. It should be treated as a strong
follow-up result rather than a new prospective validation.

## Per-dataset decisions

Support delta is one-shot minus class names using train-only known-object
R50-95. Test delta is the corresponding difference in complete-test
mAP50-95. A positive test delta means one-shot scored higher.

| Phase | Dataset | Support delta | Gate | Test delta |
|---|---|---:|---|---:|
| diagnostic | All Elements | -0.03 | names | -7.59 |
| diagnostic | Defect Detection | +3.61 | one-shot | +4.16 |
| diagnostic | GWHD | -1.11 | names | -2.69 |
| diagnostic | Orion | +2.37 | one-shot | +3.97 |
| diagnostic | Paper Parts | +21.07 | one-shot | +13.60 |
| diagnostic | Water Meter | -36.89 | names | -16.55 |
| held out | Actions | -0.69 | names | +2.11 |
| held out | Aerial Airport | +4.00 | names | -0.14 |
| held out | Aquarium | +0.48 | names | -3.14 |
| held out | DentalAI | +3.33 | one-shot | +0.45 |
| held out | FLIR | +1.11 | names | +0.12 |
| held out | Lacrosse | -4.44 | names | +0.55 |
| held out | New Defects in Wood | -0.89 | names | +1.63 |
| held out | Recode Waste | +4.70 | one-shot | +1.22 |
| held out | Soda Bottles | 0.00 | names | +0.84 |
| held out | Dreidel | +5.00 | one-shot | +9.66 |
| held out | Trail Camera | +1.67 | names | -1.67 |
| held out | WB Prova | +10.74 | one-shot | +12.32 |
| held out | Wildfire Smoke | -2.22 | names | +0.11 |
| held out | X-Ray ID | +3.52 | one-shot | +0.61 |

## Leakage and sparse-label handling

- References and calibration targets come only from the official
  RF20-VL-FSOD train/support split.
- Every image containing any selected reference object is excluded from
  calibration, preventing evaluation on a prompt image.
- The test split remains untouched until after the decisions are locked.
- Because support annotations can be non-exhaustive, calibration uses
  known-object recall rather than invalid COCO AP. Labeled objects are matched
  to same-class predictions at IoU 0.50:0.95; unmatched predictions are ignored
  rather than treating potentially unlabeled objects as false positives.
- The locked gate selects one-shot only when class-macro R50-95 improves by at
  least two points and R50 does not decrease. Otherwise it defaults to names.
- Recode Waste had no independent calibration object for one of six classes
  after reference-image exclusion. Its decision used the other five classes;
  no reference object or test data was reused to fill the gap.

## Cost and execution audit

Calibration made 1,476 paired requests over 738 support images and 913 labeled
objects. All 1,476 succeeded, with no timeout, provider rejection, or malformed
response. Reasoning was disabled, temperature was 0, the seed was 1234, and
the two branch requests were adjacent. Estimated calibration cost was $10.62.

The detector costs in the tables describe the selected clean test branches.
For a single RF20 benchmark, calibration plus selected inference cost $41.01,
so routing is not cheaper than the $34.20 fixed one-shot run. The selected
branch itself is cheaper and can be reused, making routing economically useful
only when the decision is amortized over future images or repeated evaluations.

No test detector requests were repeated for this study. After each route was
locked from support data, it selected the exact saved clean class-names or
one-shot predictions from the completed matched RF20 evaluation.

Complete local artifacts are in:

- `qwen38-fsod-runs/support-calibrated-router-pilot-v2/`
- `qwen38-fsod-runs/support-calibrated-router-heldout14-v1/`
