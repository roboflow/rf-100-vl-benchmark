# Qwen3.8-Max positive box-count ablation

This experiment measures whether adding more cross-image positive box examples
changes Qwen3.8-Max object-detection accuracy. It uses the complete 54-image
test split of RF20-VL-FSOD `the-dreidel-project`, whose six visually confusable
classes each have exactly ten train-only reference boxes on ten distinct
images.

## Controlled matrix

- positive boxes per class: 1, 2, 3, 5, 10
- box representation: numeric normalized XYXY or green boxes drawn on images
- request formulation: one multi-class request per target or one request per
  class followed by prediction merging
- controls: multi-class and single-class class-names-only prompts with zero
  boxes
- inference: `qwen3.8-max`, reasoning `none`, temperature/provider defaults
  used by the established evaluator, fixed seed 1234, 8,192 completion tokens
- scoring: full test split, pycocotools, `maxDets=[1, 10, 500]`

The nonzero box counts are nested. Rank one follows the established
largest-relative-object rule. Later ranks are selected without test results by
greedy appearance diversity over the annotated object crops. Numeric and drawn
conditions use the exact same source images and boxes. One box is drawn or
listed per reference image. In the multi-class condition, ten boxes therefore
means ten boxes for each of six classes: 60 reference images plus one target.

The complete matrix has 22 conditions and 4,158 API requests. Every response,
raw output, usage record, prediction, metric, reference, and manifest is saved
atomically under:

`qwen38-fsod-runs/dreidel-box-count-ablation-v1/`

Run or resume with:

```bash
bash run_qwen38_box_count_ablation.sh
```

Completion is indicated by `_SUCCESS.json`. Consolidated results are written to
`comparison_summary.csv`, `comparison_summary.json`, and
`aggregate_metrics.json`.
