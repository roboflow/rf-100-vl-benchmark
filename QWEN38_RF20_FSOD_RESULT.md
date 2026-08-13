# Qwen3.8-Max RF20-VL-FSOD result

We evaluated three multi-class, reasoning-off recipes on all 20 RF20-VL-FSOD
test sets: 3,970 images and 11,910 completed predictions.

## RF20 result

| Recipe | Macro mAP50-95 | Macro mAP50 | Delta vs. names | Cost |
|---|---:|---:|---:|---:|
| Class names only | **24.37** | 43.54 | — | **$22.28** |
| One positive numeric-box example/class | 25.35 | **46.73** | +0.98 / +3.19 | $34.20 |
| Two positive numeric-box examples/class | 25.14 | 46.55 | +0.77 / +3.01 | $44.43 |

The few-shot gains are small relative to the residual API noise measured in
our earlier repeat study. One example costs 53% more than names only; two
examples cost almost 2x as much and do not improve on one.

## Cross-run repeatability

We directly compared the new run with prior full-test results on Dreidel,
Orion, and Lacrosse. Values below are new-minus-prior score shifts, shown as
`mAP50-95 / mAP50`.

| Dataset | Measured noise floor | Names-only shift | One-example shift |
|---|---:|---:|---:|
| Dreidel | 4.12 / 5.76 | -0.09 / -0.51 | -1.41 / -3.89 |
| Orion | 4.06 / 7.94 | -0.63 / -2.37 | +0.43 / +0.43 |
| Lacrosse | 3.00 / 5.26 | +0.91 / +0.65 | +0.14 / +0.34 |

Every shift is within the corresponding measured noise floor. The new run is
therefore consistent with the earlier experiments; the smaller RF20-wide gain
reflects broader cross-dataset behavior rather than a reproducibility problem.

## Recommendation

**Use one multi-class class-names-only request per image as the default.** It is
the simplest and cheapest recipe, and the few-shot improvement is too small to
justify making visual examples mandatory. Keep one positive numeric-box
example per class only as an optional accuracy mode when a modest mAP50 gain is
worth the added cost. Do not use two examples by default.
