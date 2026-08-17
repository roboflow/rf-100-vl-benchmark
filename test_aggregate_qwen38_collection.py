import json
from pathlib import Path

import pytest

import aggregate_qwen38_collection as aggregate


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_complete(dataset_root: Path, run_root: Path, name: str, modes):
    dataset = dataset_root / name
    run = run_root / name
    write(
        dataset / "test/_annotations.coco.json",
        {"images": [{"id": 1}], "categories": [{"id": 1}]},
    )
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


def test_aggregate_named_collection(tmp_path):
    dataset_root = tmp_path / "datasets"
    run_root = tmp_path / "runs"
    conditions = tmp_path / "conditions.json"
    names = ["one", "two"]
    modes = ["baseline", "instructions"]
    write(
        conditions,
        {
            "conditions": [
                {
                    "mode": mode,
                    "formulation": "multi",
                    "semantics": "class_names",
                    "representation": "none",
                    "box_count": 0,
                }
                for mode in modes
            ]
        },
    )
    for name in names:
        make_complete(dataset_root, run_root, name, modes)
    result = aggregate.aggregate(dataset_root, run_root, conditions, names)
    assert result["dataset_count"] == 2
    assert result["test_image_count"] == 2
    assert result["request_count"] == 4
    assert [row["macro_mAP50_95"] for row in result["modes"]] == [10, 11]
    per_call = (60 * 2 + 40 * 0.25 + 20 * 6) / 1_000_000
    assert result["total_estimated_usd"] == pytest.approx(4 * per_call)


def test_aggregate_rejects_duplicate_dataset_names(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        aggregate.aggregate(tmp_path, tmp_path, tmp_path / "config.json", ["one", "one"])
