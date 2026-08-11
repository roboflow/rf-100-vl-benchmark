import json
from pathlib import Path

import prepare_qwen38_recipe_study as study

DATASET = Path("RF100VL/rf20-vl-fsod/the-dreidel-project")


def candidate(**overrides):
    value = {
        "mAP50_95": 25.0,
        "mAP50": 40.0,
        "calls_per_image": 1,
        "prompt_tokens_per_image": 100.0,
        "completion_tokens_per_image": 10.0,
        "effective_serial_seconds_per_image": 5.0,
    }
    value.update(overrides)
    return value


def test_efficiency_breaks_practical_accuracy_tie_first():
    expensive = candidate(
        mAP50_95=30.0,
        calls_per_image=6,
        prompt_tokens_per_image=1000,
    )
    cheap = candidate(mAP50_95=29.2, calls_per_image=1)
    outside_margin = candidate(mAP50_95=28.9, calls_per_image=1)
    assert study.efficient_best([expensive, cheap, outside_margin]) is cheap


def test_stratified_subset_is_deterministic_and_covers_all_classes():
    annotation_path = DATASET / "test/_annotations.coco.json"
    first = study.stratified_image_ids(annotation_path, 20)
    second = study.stratified_image_ids(annotation_path, 20)
    assert first == second
    assert len(first) == 20
    coco = json.loads(annotation_path.read_text())
    represented = {
        int(annotation["category_id"])
        for annotation in coco["annotations"]
        if int(annotation["image_id"]) in first
    }
    assert represented == {int(value["id"]) for value in coco["categories"]}


def test_condition_from_row_preserves_recipe_and_changes_runtime_controls():
    row = candidate(
        formulation="single",
        semantics="anonymous_explicit",
        representation="drawn",
        box_count=5,
    )
    condition = study.condition_from_row(row, "new_mode", reasoning="low", seed=7)
    assert condition.mode == "new_mode"
    assert condition.formulation == "single"
    assert condition.semantics == "anonymous_explicit"
    assert condition.representation == "drawn"
    assert condition.box_count == 5
    assert condition.reasoning_effort == "low"
    assert condition.seed == 7
