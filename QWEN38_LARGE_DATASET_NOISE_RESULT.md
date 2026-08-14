# Qwen3.8-Max box-prompt uplift versus API noise

## Takeaway

One positive numeric-box reference per class produced a positive mean uplift on all three selected larger RF20-VL-FSOD datasets. After combining the five new paired repeats with the directly matched run from the original RF20 evaluation, the gain is clearly above measured noise on Paper Parts and Actions. Defect Detection remains positive, but its magnitude is within the expanded cross-run noise threshold.

## Results

The original RF20 result contains one full-test run of each recipe. The follow-up contains five fresh paired, full-test repeats. All six observations use the same datasets, annotations, reference examples, prompts, model, temperature 0, seed 1234, reasoning-off setting, parsing, and scoring. Request summaries were verified as identical across every target image; only the condition names differ.

| Dataset | Images | Metric | Original RF20 uplift | Fresh five-repeat mean | Combined six-run mean | Combined noise threshold | Combined 95% CI | Judgment |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Paper Parts | 500 | mAP50-95 | +13.60 | +13.39 | **+13.43** | 3.01 | [12.37, 14.48] | Clearly above noise |
| Paper Parts | 500 | mAP50 | +19.25 | +20.23 | **+20.07** | 3.86 | [18.61, 21.53] | Clearly above noise |
| Actions | 409 | mAP50-95 | +2.11 | +2.24 | **+2.22** | 0.68 | [1.97, 2.47] | Clearly above noise |
| Actions | 409 | mAP50 | +3.55 | +5.26 | **+4.98** | 2.58 | [4.00, 5.96] | Clearly above noise |
| Defect Detection | 188 | mAP50-95 | +4.16 | +1.61 | **+2.03** | 3.17 | [0.83, 3.24] | Positive, within noise threshold |
| Defect Detection | 188 | mAP50 | +6.35 | +3.05 | **+3.60** | 4.42 | [1.93, 5.27] | Positive, within noise threshold |

The noise threshold is the larger of the observed uplift range and the normal-theory 95% repeatability limit across all six matched runs. The confidence interval estimates the mean paired uplift, while the noise threshold measures how far individual run-level uplifts can move. This is why Defect Detection can have a positive confidence interval while still being within the conservative run-to-run noise threshold.

These datasets were selected because the original RF20 run showed positive box-prompting uplift, so they test whether those particular gains repeat; they are not an unbiased estimate of the typical RF20 effect. The original all-20-dataset macro result remains +0.98 mAP50-95 / +3.19 mAP50 for one-box prompting.

## Run integrity

The five-repeat follow-up completed all 10,970 requests. It recorded zero execution errors. Six isolated model/content failures were handled by the audited failure policy and did not interrupt the experiment.

Raw summary: [`qwen38-fsod-runs/large-dataset-noise-v1/noise_summary.json`](qwen38-fsod-runs/large-dataset-noise-v1/noise_summary.json)

Original RF20 summary: [`qwen38-fsod-runs/rf20-three-way-matched-v1/rf20_summary.json`](qwen38-fsod-runs/rf20-three-way-matched-v1/rf20_summary.json)
