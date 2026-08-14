import json

import analyze_qwen38_large_dataset_box_count as analysis


def make_run(tmp_path, offsets):
    rows = []
    conditions = []
    for repeat in range(1, 4):
        for box_count in analysis.BOX_COUNTS:
            mode = analysis.mode(box_count, repeat)
            score = 10.0 + repeat + offsets[box_count]
            rows.append(
                {
                    "mode": mode,
                    "complete": True,
                    "task_count": 100,
                    "mAP50_95": score,
                    "mAP50": score + 20,
                    "model_failures": 0,
                    "errors": 0,
                }
            )
            conditions.append(
                {
                    "mode": mode,
                    "formulation": "multi",
                    "semantics": "class_names",
                    "representation": "numeric_prediction",
                    "box_count": box_count,
                    "reasoning_effort": "none",
                    "seed": 1234,
                }
            )
    (tmp_path / "comparison_summary.json").write_text(json.dumps({"rows": rows}))
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "common_settings": {"temperature": 0.0},
                "conditions": conditions,
            }
        )
    )


def test_analyze_dataset_computes_paired_count_deltas(tmp_path):
    make_run(tmp_path, {1: 0.0, 2: 2.0, 5: 5.0})

    result = analysis.analyze_dataset("example", tmp_path, 3)

    assert result["test_images"] == 100
    assert result["counts"]["5"]["mAP50_95"]["mean"] == 17.0
    delta = result["comparisons"]["b05_minus_b01"]["mAP50_95"]
    assert delta["mean"] == 5.0
    assert delta["mean_ci95"] == [5.0, 5.0]


def test_macro_summary_averages_datasets_per_repeat(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    make_run(first, {1: 0.0, 2: 2.0, 5: 5.0})
    make_run(second, {1: 0.0, 2: 4.0, 5: 7.0})
    datasets = {
        "first": analysis.analyze_dataset("first", first, 3),
        "second": analysis.analyze_dataset("second", second, 3),
    }

    result = analysis.macro_summary(datasets, 3)

    assert result["comparisons"]["b02_minus_b01"]["mAP50_95"]["mean"] == 3.0
    assert result["comparisons"]["b05_minus_b01"]["mAP50"]["mean"] == 6.0
