import json
from pathlib import Path

import pytest

import evaluate_qwen38_box_count_ablation as ablation
import evaluate_qwen38_orion as base

DATASET = Path("RF100VL/rf20-vl-fsod/the-dreidel-project")
ORION = Path("RF100VL/rf20-vl-fsod/orionproducts")
DEFECT = Path("RF100VL/rf20-vl-fsod-fresh-20260813/defect-detection")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    train = base.load_coco(DATASET / "train/_annotations.coco.json")
    test = base.load_coco(DATASET / "test/_annotations.coco.json")
    categories = base.categories_by_id(test)
    references = ablation.select_reference_sequences(train, DATASET / "train")
    assets = ablation.prepare_reference_assets(
        DATASET / "train",
        tmp_path_factory.mktemp("box-count-references"),
        references,
    )
    return train, test, categories, references, assets


def test_dataset_has_ten_distinct_train_references_per_diverse_class(dataset):
    train, test, categories, references, _ = dataset
    base.validate_split_isolation(train, test)
    assert len(categories) == 6
    assert list(categories.values()) == [
        "Dreidel",
        "Gimel",
        "Hay",
        "Nun",
        "Shin",
        "Spinning Dreidel",
    ]
    assert len(test["images"]) == 54
    assert all(len(sequence) == 10 for sequence in references.values())
    assert all(
        len({reference.image_id for reference in sequence}) == 10
        for sequence in references.values()
    )
    assert all(
        [reference.rank for reference in sequence] == list(range(1, 11))
        for sequence in references.values()
    )


def test_reference_order_is_nested_deterministic_and_starts_with_prior_rule(dataset):
    train, _, _, first, _ = dataset
    second = ablation.select_reference_sequences(train, DATASET / "train")
    assert first == second
    previous = base.select_reference_examples(train)
    for category_id, sequence in first.items():
        assert sequence[0].image_id == previous[category_id].image_id
        assert sequence[0].bbox_xyxy_1000 in previous[category_id].boxes_xyxy_1000
        assert sequence[:1] == sequence[:2][:1]
        assert sequence[:2] == sequence[:3][:2]
        assert sequence[:3] == sequence[:5][:3]
        assert sequence[:5] == sequence[:10][:5]


def test_instance_based_reference_selection_supports_orion_ten_shot():
    train = base.load_coco(ORION / "train/_annotations.coco.json")
    references = ablation.select_reference_sequences(
        train,
        ORION / "train",
        distinct_images_only=False,
    )
    assert len(references) == 8
    assert all(len(sequence) == 10 for sequence in references.values())
    assert all(
        len({reference.annotation_id for reference in sequence}) == 10
        for sequence in references.values()
    )
    assert any(
        len({reference.image_id for reference in sequence}) < 10
        for sequence in references.values()
    )


def test_instance_based_selection_preserves_distinct_image_prefix():
    train = base.load_coco(DEFECT / "train/_annotations.coco.json")
    distinct = ablation.select_reference_sequences(
        train, DEFECT / "train", required_count=5
    )
    instances = ablation.select_reference_sequences(
        train,
        DEFECT / "train",
        required_count=10,
        distinct_images_only=False,
    )
    for category_id in distinct:
        assert instances[category_id][:5] == distinct[category_id]
        assert len({item.annotation_id for item in instances[category_id]}) == 10
    assert len({item.image_id for item in instances[1]}) == 6


def test_median_relative_area_strategy_changes_only_first_reference_rule():
    train = base.load_coco(DEFECT / "train/_annotations.coco.json")
    largest = ablation.select_reference_sequences(
        train,
        DEFECT / "train",
        required_count=1,
        distinct_images_only=False,
    )
    median = ablation.select_reference_sequences(
        train,
        DEFECT / "train",
        required_count=1,
        distinct_images_only=False,
        first_strategy="median-relative-area",
    )
    images = {int(image["id"]): image for image in train["images"]}
    annotations = {}
    for annotation in train["annotations"]:
        annotations.setdefault(int(annotation["category_id"]), []).append(annotation)
    for category_id, values in annotations.items():
        ordered = sorted(
            values,
            key=lambda annotation: (
                float(annotation["bbox"][2])
                * float(annotation["bbox"][3])
                / (
                    int(images[int(annotation["image_id"])]["width"])
                    * int(images[int(annotation["image_id"])]["height"])
                ),
                str(images[int(annotation["image_id"])]["file_name"]),
                int(annotation["id"]),
            ),
        )
        expected = int(ordered[(len(ordered) - 1) // 2]["id"])
        assert median[category_id][0].annotation_id == expected
        assert median[category_id][0] != largest[category_id][0]


def test_condition_and_task_matrix_is_complete_and_unique(dataset):
    _, test, categories, _, _ = dataset
    assert len(ablation.CONDITIONS) == 22
    assert len({condition.mode for condition in ablation.CONDITIONS}) == 22
    visual = [condition for condition in ablation.CONDITIONS if condition.box_count]
    assert len(visual) == 20
    assert {
        (condition.formulation, condition.representation, condition.box_count)
        for condition in visual
    } == {
        (formulation, representation, count)
        for formulation in ablation.FORMULATIONS
        for representation in ablation.REPRESENTATIONS
        for count in ablation.BOX_COUNTS
    }
    tasks = ablation.build_tasks(test, categories)
    assert len(tasks) == 4158
    assert len({task.key for task in tasks}) == len(tasks)
    for condition in ablation.CONDITIONS:
        count = sum(task.mode == condition.mode for task in tasks)
        assert count == (324 if condition.single_class else 54)


def test_numeric_and_drawn_prompts_use_identical_nested_boxes(dataset):
    _, test, categories, references, assets = dataset
    image = test["images"][0]
    category_id, category_name = next(iter(categories.items()))
    for count in ablation.BOX_COUNTS:
        for formulation in ablation.FORMULATIONS:
            expected_reference_images = count * (
                1 if formulation == "single" else len(categories)
            )
            for representation in ablation.REPRESENTATIONS:
                mode = f"{formulation}_{representation}_b{count:02d}"
                task = base.Task(
                    mode=mode,
                    image_id=int(image["id"]),
                    file_name=str(image["file_name"]),
                    width=int(image["width"]),
                    height=int(image["height"]),
                    category_id=category_id if formulation == "single" else None,
                    category_name=category_name if formulation == "single" else None,
                )
                messages = ablation.build_messages(
                    task, DATASET / "test", categories, references, assets
                )
                content = messages[0]["content"]
                assert content[-2] == {"type": "text", "text": "TARGET IMAGE:"}
                assert content[-1]["type"] == "image_url"
                assert sum(part["type"] == "image_url" for part in content) == (
                    expected_reference_images + 1
                )
                text = "\n".join(
                    part["text"] for part in content if part["type"] == "text"
                )
                assert "README.dataset" not in text
                assert f"{count} positive train-only reference" in text
                if representation == "numeric":
                    for reference in references[category_id][:count]:
                        assert json.dumps(list(reference.bbox_xyxy_1000)) in text
                else:
                    assert "green box marks one positive example" in text


def test_drawn_assets_preserve_source_size_and_change_pixels(dataset):
    from PIL import Image

    _, _, _, _, assets = dataset
    for paths in assets.values():
        with Image.open(paths["source"]) as source, Image.open(paths["drawn"]) as drawn:
            assert source.size == drawn.size
            assert source.convert("RGB").tobytes() != drawn.convert("RGB").tobytes()


def test_token_pacing_scales_with_actual_reference_image_count():
    estimates = ablation.build_token_estimates(class_count=6)
    assert estimates["multi_names_b00"] == 5_500
    assert estimates["single_names_b00"] == 5_500
    assert estimates["single_numeric_b10"] == 35_500
    assert estimates["multi_numeric_b10"] == 185_500
    assert estimates["multi_drawn_b10"] == 185_500
    assert estimates["multi_numeric_b10"] > estimates["multi_numeric_b01"]


def test_prepare_only_writes_full_resumable_contract(tmp_path):
    output = tmp_path / "ablation"
    assert (
        ablation.main(
            [
                "--dataset-dir",
                str(DATASET),
                "--output-dir",
                str(output),
                "--prepare-only",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "run_manifest.json").read_text())
    progress = json.loads((output / "progress.json").read_text())
    comparison = json.loads((output / "comparison_summary.json").read_text())
    assert manifest["prompt_version"] == ablation.PROMPT_VERSION
    assert manifest["configuration"]["settings"]["reasoning_effort"] == "none"
    assert manifest["reference_selection"]["one_box_per_distinct_train_image"]
    assert progress["total"] == {
        "total": 4158,
        "success": 0,
        "model_failure": 0,
        "error": 0,
        "pending": 4158,
    }
    assert len(comparison["rows"]) == 22
    assert not (output / "_SUCCESS.json").exists()
    assert len(list((output / "metrics").glob("*.json"))) == 22
    assert len(list((output / "predictions").glob("*.json"))) == 22
    assert (output / "comparison_summary.csv").is_file()
