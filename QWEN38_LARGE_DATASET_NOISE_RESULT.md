# Qwen3.8-Max box-prompt uplift versus API noise

## Takeaway

One positive numeric-box reference per class improved detection above the measured API noise on all three larger RF20-VL-FSOD datasets. The gain was very large on Paper Parts, clear on Actions, and positive but only narrowly above noise on Defect Detection. Box prompting therefore has a real benefit, but its size remains strongly dataset-dependent.

## Results

Each result is the mean of five paired, full-test-set repeats. Every pair compared class-names-only multi-class prompting with one positive numeric-box reference per class. Runs used temperature 0, seed 1234, and reasoning disabled; the two conditions were interleaved by target image.

| Dataset | Images | Metric | Class names only | One box reference | Paired uplift | Paired noise threshold | Uplift 95% CI | Judgment |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Paper Parts | 500 | mAP50-95 | 14.08 | 27.47 | **+13.39** | 3.10 | [12.00, 14.78] | Clearly above noise |
| Paper Parts | 500 | mAP50 | 33.88 | 54.12 | **+20.23** | 4.13 | [18.38, 22.09] | Clearly above noise |
| Actions | 409 | mAP50-95 | 8.03 | 10.28 | **+2.24** | 0.72 | [1.92, 2.56] | Clearly above noise |
| Actions | 409 | mAP50 | 18.15 | 23.42 | **+5.26** | 1.90 | [4.41, 6.12] | Clearly above noise |
| Defect Detection | 188 | mAP50-95 | 29.60 | 31.21 | **+1.61** | 1.46 | [0.96, 2.26] | Narrowly above noise |
| Defect Detection | 188 | mAP50 | 46.17 | 49.23 | **+3.05** | 2.66 | [1.86, 4.24] | Narrowly above noise |

The paired noise threshold is the 95% repeatability limit of the five observed per-repeat uplifts. Every mean uplift exceeds that threshold, and every paired 95% confidence interval excludes zero.

## Run integrity

The experiment completed all 10,970 requests. It recorded zero execution errors. Six isolated model/content failures were handled by the audited failure policy and did not interrupt or invalidate the experiment.

Raw summary: [`qwen38-fsod-runs/large-dataset-noise-v1/noise_summary.json`](qwen38-fsod-runs/large-dataset-noise-v1/noise_summary.json)

