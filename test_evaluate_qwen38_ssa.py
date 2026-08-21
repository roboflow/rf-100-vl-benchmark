import json
from pathlib import Path

import pytest

import evaluate_qwen38_orion as base
import evaluate_qwen38_recipe as recipe
import evaluate_qwen38_ssa as ssa
import evaluate_qwen38_support_calibrated_router as support

DATASET = Path("RF100VL/rf20-vl-fsod-fresh-20260813/the-dreidel-project")


def load_dataset():
    train = base.load_coco(DATASET / "train/_annotations.coco.json")
    categories = base.categories_by_id(train)
    return train, categories


def test_support_order_is_seeded_and_contains_each_labeled_image_once():
    train, _ = load_dataset()
    first = ssa.support_order(train, 1234)
    assert first == ssa.support_order(train, 1234)
    assert first != ssa.support_order(train, 4321)
    assert len(first) == len(set(first))
    assert set(first) == {int(value["image_id"]) for value in train["annotations"]}


def test_gold_annotations_use_prediction_schema_and_normalized_coordinates():
    train, categories = load_dataset()
    images = {int(value["id"]): value for value in train["images"]}
    annotations = ssa.annotations_by_image(train)
    image_id = ssa.support_order(train, 1234)[0]
    payload = json.loads(
        ssa.annotation_json(annotations[image_id], images[image_id], categories)
    )
    assert payload
    for detection in payload:
        assert list(detection) == ["bbox_2d", "label"]
        assert detection["label"] in categories.values()
        assert len(detection["bbox_2d"]) == 4
        assert all(0 <= coordinate <= 1000 for coordinate in detection["bbox_2d"])


def test_clean_trunk_contains_only_official_gold_assistant_messages():
    train, categories = load_dataset()
    order = ssa.support_order(train, 1234)[:3]
    trunk = ssa.build_trunk(order, train, DATASET / "train", categories)
    assert [message["role"] for message in trunk] == [
        "user", "assistant", "user", "assistant", "user", "assistant"
    ]
    assert all(
        isinstance(trunk[index]["content"], str)
        for index in range(1, len(trunk), 2)
    )
    assert "TARGET IMAGE" not in " ".join(
        part["text"]
        for message in trunk
        if isinstance(message["content"], list)
        for part in message["content"]
        if part["type"] == "text"
    )


def test_prequential_target_is_not_in_prefix_and_turn_one_matches_zero_framing():
    train, categories = load_dataset()
    images = {int(value["id"]): value for value in train["images"]}
    order = ssa.support_order(train, 1234)
    first = ssa.build_branch(
        [], images[order[0]], train, DATASET / "train", DATASET / "train", categories
    )
    zero = ssa.build_branch(
        [], images[order[0]], train, DATASET / "train", DATASET / "train", categories
    )
    assert base.request_summary(first) == base.request_summary(zero)
    second = ssa.build_branch(
        order[:1], images[order[1]], train, DATASET / "train", DATASET / "train", categories
    )
    assert len(second) == 3
    assert order[1] not in order[:1]


def test_zero_prefix_prompt_is_the_existing_names_only_prompt():
    _, categories = load_dataset()
    test = base.load_coco(DATASET / "test/_annotations.coco.json")
    image = test["images"][0]
    task = ssa.task_for_image("names_multi", image)
    condition = recipe.Condition(
        "names_multi", "multi", "class_names", "none", 0
    )
    established = recipe.build_messages(
        task, condition, DATASET / "test", categories, {}, {}, {}
    )[0]
    assert base.request_summary([ssa.target_message(image, DATASET / "test", categories)]) == (
        base.request_summary([established])
    )


def test_multiturn_fingerprint_covers_every_gold_and_image_turn():
    train, categories = load_dataset()
    images = {int(value["id"]): value for value in train["images"]}
    order = ssa.support_order(train, 1234)
    one = ssa.build_branch(
        order[:1], images[order[2]], train, DATASET / "train", DATASET / "train", categories
    )
    two = ssa.build_branch(
        order[:2], images[order[2]], train, DATASET / "train", DATASET / "train", categories
    )
    task = ssa.task_for_image("test", images[order[2]])
    settings = {"temperature": 0, "seed": 1234}
    assert ssa.expected_fingerprint(task, one, settings) != ssa.expected_fingerprint(
        task, two, settings
    )
    summary = base.request_summary(two)
    assert len(summary["turns"]) == 5


def test_support_metric_uses_recall5095_and_marks_sparse_precision_diagnostic():
    train, _ = load_dataset()
    image_id = ssa.support_order(train, 1234)[0]
    annotation = ssa.one_image_calibration(train, image_id)["annotations"][0]
    prediction = {
        "image_id": image_id,
        "category_id": int(annotation["category_id"]),
        "bbox": list(annotation["bbox"]),
        "score": 1.0,
    }
    metrics = ssa.support_metrics(train, image_id, [prediction])
    assert "class_macro_recall50_95" in metrics
    assert metrics["precision_is_valid_for_routing"] is False
    assert metrics["unmatched_predictions_ignored"] is True


@pytest.mark.parametrize(
    "dataset_name",
    ["the-dreidel-project", "orionproducts", "lacrosse-object-detection"],
)
def test_normalized_box_round_trip_preserves_geometry(dataset_name):
    dataset = Path("RF100VL/rf20-vl-fsod-fresh-20260813") / dataset_name
    train = base.load_coco(dataset / "train/_annotations.coco.json")
    categories = base.categories_by_id(train)
    images = {int(value["id"]): value for value in train["images"]}
    for annotation in train["annotations"][:10]:
        image = images[int(annotation["image_id"])]
        category_id = int(annotation["category_id"])
        normalized = base.annotation_xywh_to_normalized_xyxy(
            annotation["bbox"], int(image["width"]), int(image["height"])
        )
        converted, diagnostics = base.convert_detections_to_coco(
            [{"bbox_2d": normalized, "label": categories[category_id]}],
            int(image["id"]),
            int(image["width"]),
            int(image["height"]),
            categories,
        )
        assert len(converted) == 1
        assert diagnostics["invalid_boxes"] == 0
        # Integer quantization in the normalized 0-1000 schema costs a small
        # amount of overlap for the tiniest support boxes.
        assert support.intersection_over_union(
            annotation["bbox"], converted[0]["bbox"]
        ) > 0.97


def test_stopping_preview_selects_best_smoothed_support_delta_without_test_data():
    curve = [
        {"prefix_images": 0, "delta": {"class_macro_recall50_95": 0}},
        {"prefix_images": 1, "delta": {"class_macro_recall50_95": 3}},
        {"prefix_images": 2, "delta": {"class_macro_recall50_95": 5}},
        {"prefix_images": 3, "delta": {"class_macro_recall50_95": 1}},
    ]
    result = ssa.simulate_best_prefix(curve, window=2, epsilon=2)
    assert result["selected_prefix"] == 2
    assert result["reason"] == "best_smoothed_support_delta"


def test_support_order_seed_does_not_change_inference_seed(tmp_path, capsys):
    output = tmp_path / "prepared"
    assert ssa.main(
        [
            "--dataset-dir",
            str(DATASET),
            "--output-dir",
            str(output),
            "--seed",
            "4321",
            "--inference-seed",
            "1234",
            "--max-support-turns",
            "2",
            "--test-prefixes",
            "0",
            "--prepare-only",
        ]
    ) == 0
    capsys.readouterr()
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["seed"] == 4321
    assert manifest["inference_seed"] == 1234
    assert manifest["settings"]["seed"] == 1234
