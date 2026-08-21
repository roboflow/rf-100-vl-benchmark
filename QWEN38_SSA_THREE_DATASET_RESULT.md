# Qwen3.8-Max SSA three-dataset validation

## Outcome

The study completed successfully as an implementation and diagnostic study,
but the current stopping policy is **not ready for the full RF20 run**. Clean
sequential visual context improved the matched three-dataset macro, while the
support-only policy was too sensitive to support order and missed substantially
better prefixes.

Scores are COCO **mAP50-95 / mAP50**, with `maxDets=500`.

| Dataset | Names only | Locked SSA prefix | Locked SSA | Delta vs names | Test-grid oracle | Oracle regret | Prefixes selected across 3 orders |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dreidel | 40.33 / 54.41 | 1 | 43.54 / 60.88 | +3.20 / +6.47 | 53.03 / 75.21 at 8 | 9.50 / 14.33 | 1, 0, 0 |
| Orion | 16.58 / 29.73 | 0 | 16.58 / 29.73 | 0.00 / 0.00 | 17.24 / 41.78 at all | 0.67 / 12.05 | 0, 0, 0 |
| Lacrosse | 31.69 / 50.80 | 6 | 38.07 / 56.16 | +6.39 / +5.36 | 40.40 / 59.96 at 4 | 2.32 / 3.80 | 6, 0, 0 |
| **Macro** | **29.53 / 44.98** | | **32.73 / 48.92** | **+3.20 / +3.94** | **36.89 / 58.98** | **4.16 / 10.06** | |

Against the prior support-calibrated 0/10 router on these datasets, locked SSA
was **+0.24 mAP50-95 but -5.30 mAP50**. The primary difference is within the
previously measured API noise and is not evidence of a real win.

## What passed

- All 1,552 API records reached a successful terminal state.
- Temperature was 0, the inference seed was fixed, and reasoning stayed off.
- No test image or label entered adaptation, and no model prediction entered
  the clean gold trunk.
- References and predictions used normalized 0-1000 XYXY `bbox_2d` JSON.
- There were no invalid boxes or model failures. One hallucinated Orion label
  was ignored by the existing robust parser as designed.
- Actual research spend was **$16.51**, below the authorized $20 budget.

## Why the policy did not pass

The policy chose useful context in only the canonical Dreidel and Lacrosse
orders. The other two orders selected names-only for both datasets. Its macro
regret against the analysis-grid oracle was 4.16 / 10.06 points, which is too
large for the planned success criterion.

The audit also found that the provisional smoother let incomplete early
windows compete with full three-observation windows. A single +100-point event
therefore selected Dreidel prefix 1. Requiring a complete smoothing window
would select prefix 3 on that order, but that correction was identified after
the locked evaluation and must be treated as a new policy, not retroactively
substituted into this result.

## Decision

Do not launch the full RF20 adaptive benchmark with this stopping rule. The
underlying mechanism remains promising because the locked choices improved the
matched macro, but the next small validation must remove the incomplete-window
bias and produce materially better agreement across support orders. A single
RF20 screening pass is appropriate only after that correction passes the same
three-dataset check.
