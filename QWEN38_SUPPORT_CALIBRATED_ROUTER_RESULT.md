# Qwen3.8-Max FSOD support-calibrated routing pilot

## Result

The semantic class-name router was replaced with a train-only empirical gate.
One RF20-VL-FSOD support object per class is reserved as the exact reference
used by the established one-shot prompt. The class-names-only and one-shot
detectors are then compared on the remaining support objects. No test image or
test annotation is used to make the decision.

The pilot selected the higher-mAP50-95 test branch on all six diagnostic
datasets:

| Dataset | Held-out support objects | Support delta, one-shot minus names (R50-95 / R50) | Selected branch | Test delta, one-shot minus names (mAP50-95 / mAP50) | Correct? |
|---|---:|---:|---|---:|---:|
| All Elements | 82 | -0.03 / +3.85 | names | -7.59 / -0.65 | yes |
| Defect Detection | 36 | +3.61 / +5.56 | one-shot | +4.16 / +6.35 | yes |
| GWHD | 9 | -1.11 / 0.00 | names | -2.69 / -3.64 | yes |
| Orion | 55 | +2.37 / +4.81 | one-shot | +3.97 / +18.25 | yes |
| Paper Parts | 169 | +21.07 / +23.98 | one-shot | +13.60 / +19.25 | yes |
| Water Meter | 90 | -36.89 / -52.22 | names | -16.55 / -29.58 | yes |

Across these six datasets, the selected clean test branches scored 26.79
mAP50-95 / 50.47 mAP50. This was +3.62 / +7.31 over always using class names
and +4.47 / +5.65 over always using one reference. The fixed one-reference
branch by itself was worse than class names by 0.85 mAP50-95 on this deliberately
mixed diagnostic set.

## Leakage and sparse-label handling

- References and calibration targets come only from the official
  RF20-VL-FSOD train/support split.
- Every image containing any selected reference object is excluded from
  calibration, preventing the model from being evaluated on a prompt image.
- The target test split is untouched.
- Because FSOD support annotations can be non-exhaustive, calibration uses
  known-object recall rather than COCO AP. Labeled support objects are matched
  to same-class predictions at IoU 0.50:0.95; unmatched predictions are ignored
  instead of incorrectly treating potentially unlabeled objects as false
  positives.
- The predeclared gate selects one-shot only when class-macro R50-95 improves
  by at least two points and R50 does not decrease. Otherwise it defaults to
  the cheaper class-names branch.

## Execution audit

The run made 716 paired requests over 358 support images and 441 labeled
objects. All 716 succeeded, with no provider rejection, timeout, or malformed
response. Reasoning was disabled, temperature was 0, the seed was 1234, and
requests for the two branches were adjacent. The pilot used 10,093,850 input
tokens, of which 8,892,416 were cached, and 240,564 output tokens. Using the
same pricing convention as the RF20 report, estimated API cost was $6.07.

This is a diagnostic, outcome-balanced pilot rather than an unbiased estimate
of routing accuracy: the six datasets intentionally include three important
semantic-router failures and three clear one-shot successes. The 6/6 result
shows that train-support calibration can solve those known failure modes. A
prospectively locked RF20 run is still required to estimate general routing
performance.

Complete local artifacts are in
`qwen38-fsod-runs/support-calibrated-router-pilot-v2/`.
