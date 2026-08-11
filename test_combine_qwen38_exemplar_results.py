import json

import combine_qwen38_exemplar_results as combine


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_source(path, counts):
    conditions = []
    rows = []
    modes = {}
    for instruction in ("explicit", "minimal"):
        for representation in ("numeric", "drawn"):
            for count in counts:
                mode = f"{instruction}_{representation}_b{count:02d}"
                condition = {
                    "mode": mode,
                    "instruction": instruction,
                    "representation": representation,
                    "box_count": count,
                }
                conditions.append(condition)
                rows.append(
                    {
                        "mode": mode,
                        "instruction": instruction,
                        "representation": representation,
                        "boxes_per_class": count,
                        "complete": True,
                    }
                )
                modes[mode] = {"complete": True, "condition": condition}
    configuration = {
        "dataset_directory": "/dataset",
        "train_annotation_sha256": "train",
        "test_annotation_sha256": "test",
        "settings": {"model": "qwen3.8-max"},
        "requests_per_minute": 570,
        "tokens_per_minute": 900_000,
    }
    write_json(
        path / "run_manifest.json",
        {
            "prompt_version": "prompt-v1",
            "class_names_exposed_to_model": False,
            "minimal_mode_semantic_instruction": False,
            "configuration": configuration,
            "conditions": conditions,
        },
    )
    write_json(
        path / "_SUCCESS.json",
        {"request_count": 324 * len(conditions)},
    )
    write_json(
        path / "aggregate_metrics.json",
        {"image_count": 54, "class_count": 6, "modes": modes},
    )
    write_json(path / "comparison_summary.json", {"rows": rows})


def test_combine_requires_and_emits_complete_five_count_factorial(tmp_path):
    base = tmp_path / "base"
    extension = tmp_path / "extension"
    output = tmp_path / "combined"
    make_source(base, (1, 2, 5))
    make_source(extension, (7, 10))

    combine.combine(base, extension, output)

    success = json.loads((output / "_SUCCESS.json").read_text())
    comparison = json.loads((output / "comparison_summary.json").read_text())
    assert success["condition_count"] == 20
    assert success["request_count"] == 6480
    assert success["box_counts"] == [1, 2, 5, 7, 10]
    assert len(comparison["rows"]) == 20
