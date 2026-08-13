import json

import analyze_qwen38_large_dataset_noise as analysis


def test_analyze_dataset_computes_paired_uplift(tmp_path):
    rows = []
    conditions = []
    for index in range(1, 6):
        for prefix, representation, box_count, score in (
            ("names", "none", 0, 10.0 + index),
            ("box", "numeric_prediction", 1, 14.0 + index),
        ):
            mode = f"{prefix}_repeat_{index:02d}"
            conditions.append(
                {
                    "mode": mode,
                    "formulation": "multi",
                    "semantics": "class_names",
                    "representation": representation,
                    "box_count": box_count,
                    "reasoning_effort": "none",
                    "seed": 1234,
                }
            )
            rows.append(
                {
                    "mode": mode,
                    "complete": True,
                    "task_count": 100,
                    "mAP50_95": score,
                    "mAP50": score + 10,
                    "model_failures": 0,
                    "errors": 0,
                }
            )
    (tmp_path / "_SUCCESS.json").write_text("{}")
    (tmp_path / "comparison_summary.json").write_text(json.dumps({"rows": rows}))
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "common_settings": {"temperature": 0.0},
                "conditions": conditions,
            }
        )
    )

    result = analysis.analyze_dataset("example", tmp_path)

    assert result["test_images"] == 100
    assert result["paired_uplift"]["mAP50_95"]["mean"] == 4.0
    assert result["paired_uplift"]["mAP50"]["mean_ci95"] == [4.0, 4.0]
