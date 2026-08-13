import json
from pathlib import Path

import pytest

import aggregate_qwen38_rf20 as aggregate


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_complete_dataset(dataset_root: Path, run_root: Path, name: str, modes):
    dataset = dataset_root / name
    run = run_root / name
    write(dataset / "test/_annotations.coco.json", {"images": [{"id": 1}], "categories": [{"id": 1}]})
    write(run / "_SUCCESS.json", {"image_count": 1, "condition_count": len(modes)})
    write(run / "progress.json", {"total": {"pending": 0, "error": 0}})
    write(run / "run_manifest.json", {"dataset_directory": str(dataset.resolve())})
    rows = []
    for index, mode in enumerate(modes):
        rows.append(
            {
                "mode": mode,
                "complete": True,
                "task_count": 1,
                "mAP50_95": 10 + index,
                "mAP50": 20 + index,
                "model_failures": 0,
                "errors": 0,
                "prompt_tokens": 100,
                "completion_tokens": 20,
            }
        )
        write(
            run / f"records/{mode}/{mode}__image_1__class_all.json",
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 40},
                }
            },
        )
    write(run / "comparison_summary.json", {"rows": rows})


def test_aggregate_requires_and_validates_all_twenty(tmp_path):
    dataset_root = tmp_path / "datasets"
    run_root = tmp_path / "runs"
    conditions = tmp_path / "conditions.json"
    modes = ["names", "one", "two"]
    write(
        conditions,
        {
            "conditions": [
                {
                    "mode": mode,
                    "formulation": "multi",
                    "semantics": "class_names",
                    "representation": "none" if index == 0 else "numeric",
                    "box_count": index,
                }
                for index, mode in enumerate(modes)
            ]
        },
    )
    for index in range(20):
        make_complete_dataset(dataset_root, run_root, f"dataset-{index:02d}", modes)
    result = aggregate.aggregate(dataset_root, run_root, conditions)
    assert result["dataset_count"] == 20
    assert result["test_image_count"] == 20
    assert result["request_count"] == 60
    assert len(result["per_dataset"]) == 60
    assert [row["macro_mAP50_95"] for row in result["modes"]] == [10, 11, 12]
    assert result["total_estimated_usd"] == pytest.approx(60 * (60 * 2 + 40 * 0.25 + 20 * 6) / 1_000_000)


def test_aggregate_rejects_incomplete_rf20(tmp_path):
    (tmp_path / "datasets/only-one/test").mkdir(parents=True)
    write(tmp_path / "datasets/only-one/test/_annotations.coco.json", {"images": [], "categories": []})
    write(tmp_path / "conditions.json", {"conditions": [{"mode": "names", "formulation": "multi", "semantics": "class_names", "representation": "none", "box_count": 0}]})
    with pytest.raises(ValueError, match="exactly 20"):
        aggregate.aggregate(tmp_path / "datasets", tmp_path / "runs", tmp_path / "conditions.json")
