# Qwen3.8-Max box-prompt uplift versus API noise

## Takeaway

One positive numeric-box reference per class produced a positive mean uplift on all three selected larger RF20-VL-FSOD datasets. After combining the five new paired repeats with the directly matched run from the original RF20 evaluation, the gain is clearly above measured noise on Paper Parts and Actions. Defect Detection remains positive, but its magnitude is within the expanded cross-run noise threshold.

## Primary comparison: uplift versus class-names-only inter-run noise

This table pools the original RF20 run and the five fresh matched repeats. The class-names-only noise threshold measures how far the regular multi-class baseline moved between otherwise identical runs.

| Dataset | Images | Metric | Class names only | One box reference | Uplift | Class-names-only inter-run noise | Uplift / noise | Judgment |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Paper Parts | 500 | mAP50-95 | 14.12 | 27.54 | **+13.43** | 0.95 | 14.1x | Clearly above baseline noise |
| Paper Parts | 500 | mAP50 | 34.06 | 54.13 | **+20.07** | 3.87 | 5.2x | Clearly above baseline noise |
| Actions | 409 | mAP50-95 | 8.03 | 10.24 | **+2.22** | 0.49 | 4.5x | Clearly above baseline noise |
| Actions | 409 | mAP50 | 18.25 | 23.23 | **+4.98** | 1.19 | 4.2x | Clearly above baseline noise |
| Defect Detection | 188 | mAP50-95 | 29.30 | 31.34 | **+2.03** | 2.17 | 0.94x | Within baseline noise |
| Defect Detection | 188 | mAP50 | 45.77 | 49.37 | **+3.60** | 2.89 | 1.25x | Modestly above baseline noise |

Therefore, box prompting has a repeatable above-baseline-noise uplift on Paper Parts and Actions. Defect Detection is mixed and should be treated as inconclusive: its mAP50-95 gain is below baseline noise, while its mAP50 gain is only 1.25x the baseline noise.

## Reconciliation with the original RF20 score set

The original RF20 result contains one full-test run of each recipe. The follow-up contains five fresh paired, full-test repeats. All six observations use the same datasets, annotations, reference examples, prompts, model, temperature 0, seed 1234, reasoning-off setting, parsing, and scoring. Request summaries were verified as identical across every target image; only the condition names differ.

| Dataset | Images | Metric | Original RF20 uplift | Fresh five-repeat mean | Combined six-run mean | Paired-uplift noise | Combined 95% CI | Judgment |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Paper Parts | 500 | mAP50-95 | +13.60 | +13.39 | **+13.43** | 3.01 | [12.37, 14.48] | Clearly above noise |
| Paper Parts | 500 | mAP50 | +19.25 | +20.23 | **+20.07** | 3.86 | [18.61, 21.53] | Clearly above noise |
| Actions | 409 | mAP50-95 | +2.11 | +2.24 | **+2.22** | 0.68 | [1.97, 2.47] | Clearly above noise |
| Actions | 409 | mAP50 | +3.55 | +5.26 | **+4.98** | 2.58 | [4.00, 5.96] | Clearly above noise |
| Defect Detection | 188 | mAP50-95 | +4.16 | +1.61 | **+2.03** | 3.17 | [0.83, 3.24] | Positive, within noise threshold |
| Defect Detection | 188 | mAP50 | +6.35 | +3.05 | **+3.60** | 4.42 | [1.93, 5.27] | Positive, within noise threshold |

Each noise threshold is the larger of the observed cross-run range and the normal-theory 95% repeatability limit. Class-names-only noise measures baseline score instability. Paired-uplift noise measures instability in the estimated difference between box prompting and the baseline. The paired confidence interval estimates the mean uplift. This is why Defect Detection can have a positive mean confidence interval while still failing the stricter run-to-run repeatability comparison.

These datasets were selected because the original RF20 run showed positive box-prompting uplift, so they test whether those particular gains repeat; they are not an unbiased estimate of the typical RF20 effect. The original all-20-dataset macro result remains +0.98 mAP50-95 / +3.19 mAP50 for one-box prompting.

## Run integrity

The five-repeat follow-up completed all 10,970 requests. It recorded zero execution errors. Six isolated model/content failures were handled by the audited failure policy and did not interrupt the experiment.

Raw summary: [`qwen38-fsod-runs/large-dataset-noise-v1/noise_summary.json`](qwen38-fsod-runs/large-dataset-noise-v1/noise_summary.json)

Original RF20 summary: [`qwen38-fsod-runs/rf20-three-way-matched-v1/rf20_summary.json`](qwen38-fsod-runs/rf20-three-way-matched-v1/rf20_summary.json)
