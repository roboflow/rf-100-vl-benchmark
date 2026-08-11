import json
from pathlib import Path

import evaluate_qwen38_orion as base
import evaluate_qwen38_orion_subset as subset

DATASET = Path("RF100VL/rf20-vl-fsod/orionproducts")


def load_fixture():
    train = base.load_coco(DATASET / "train/_annotations.coco.json")
    test = base.load_coco(DATASET / "test/_annotations.coco.json")
    categories = base.categories_by_id(test)
    examples = base.select_reference_examples(train)
    negatives = base.validate_negative_pairs(categories)
    return train, test, categories, examples, negatives


def test_fixed_subset_is_five_images_and_contains_every_class():
    _, test, categories, _, _ = load_fixture()
    image_ids = subset.SUBSET_IMAGE_IDS_BY_NAME["five"]
    ground_truth = subset.subset_ground_truth(test, image_ids)
    assert [image["id"] for image in ground_truth["images"]] == list(
        image_ids
    )
    assert len(ground_truth["images"]) == 5
    assert {annotation["category_id"] for annotation in ground_truth["annotations"]} == set(
        categories
    )
    assert 5 * (1 + 5 * 8) == 205


def test_multi_reference_messages_have_eight_references_and_target_last(tmp_path):
    train, test, categories, examples, negatives = load_fixture()
    assets = base.prepare_reference_assets(
        DATASET / "train", tmp_path / "references", examples, negatives
    )
    tasks = subset.build_tasks(test, subset.SUBSET_IMAGE_IDS_BY_NAME["five"])
    for task in tasks:
        messages = subset.build_multi_reference_messages(
            task, DATASET / "test", categories, examples, assets
        )
        content = messages[0]["content"]
        images = [part for part in content if part["type"] == "image_url"]
        assert len(images) == 9
        assert content[-2] == {"type": "text", "text": "TARGET IMAGE:"}
        assert content[-1] == images[-1]
        text = "\n".join(
            part["text"] for part in content if part["type"] == "text"
        )
        assert all(name in text for name in categories.values())
        assert all("README.dataset" not in part for part in text.splitlines())


def test_prepare_only_rescores_existing_modes_and_makes_no_api_calls(tmp_path):
    output = tmp_path / "subset"
    assert subset.main(["--output-dir", str(output), "--prepare-only"]) == 0
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["existing_mode_task_equivalent"] == 205
    assert manifest["new_api_request_count"] == 20
    for effort in subset.REASONING_EFFORTS:
        aggregate = json.loads(
            (output / effort / "aggregate_metrics.json").read_text()
        )
        assert set(base.MODES).issubset(aggregate["modes"])
        assert all(
            aggregate["modes"][mode]["metrics"] is not None for mode in base.MODES
        )
        assert all(
            aggregate["modes"][mode]["metrics"] is None for mode in subset.NEW_MODES
        )
        assert len(list((output / effort / "predictions").glob("*.json"))) == 8
        assert len(list((output / effort / "metrics").glob("*.json"))) == 8
    comparison = json.loads((output / "comparison_summary.json").read_text())
    assert len(comparison["rows"]) == 16
    assert (output / "comparison_summary.csv").is_file()


def test_twenty_image_subset_is_nested_representative_and_reuses_five(tmp_path):
    _, test, categories, _, _ = load_fixture()
    five = subset.SUBSET_IMAGE_IDS_BY_NAME["five"]
    twenty = subset.SUBSET_IMAGE_IDS_BY_NAME["twenty"]
    assert len(twenty) == 20
    assert twenty[: len(five)] == five
    ground_truth = subset.subset_ground_truth(test, twenty)
    assert {annotation["category_id"] for annotation in ground_truth["annotations"]} == set(
        categories
    )
    output = tmp_path / "twenty"
    assert subset.main([
        "--subset", "twenty",
        "--output-dir", str(output),
        "--prepare-only",
    ]) == 0
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["existing_mode_task_equivalent"] == 820
    assert manifest["new_api_request_count"] == 80
    assert manifest["reused_nested_request_count"] == 20
    for effort in subset.REASONING_EFFORTS:
        for mode in subset.NEW_MODES:
            assert len(list((output / effort / "records" / mode).glob("*.json"))) == 5


def test_full_run_uses_every_test_image_and_can_select_one_new_mode():
    _, test, _, _, _ = load_fixture()
    image_ids = subset.image_ids_for_run(test, "full")
    assert len(image_ids) == 59
    assert image_ids == tuple(sorted(image["id"] for image in test["images"]))
    tasks = subset.build_tasks(
        test,
        image_ids,
        ["multi_class_positive_numeric"],
    )
    assert len(tasks) == 59
    assert {task.mode for task in tasks} == {"multi_class_positive_numeric"}


def test_non_orion_full_run_does_not_reuse_orion_nested_records(tmp_path):
    dataset = Path("RF100VL/rf20-vl-fsod/lacrosse-object-detection")
    output = tmp_path / "lacrosse"
    assert subset.main([
        "--subset", "full",
        "--dataset-dir", str(dataset),
        "--negative-pairs-file",
        "qwen38-fsod-configs/lacrosse-object-detection-negative-pairs.json",
        "--new-modes", "multi_class_positive_numeric",
        "--reasoning-efforts", "none", "low",
        "--output-dir", str(output),
        "--prepare-only",
    ]) == 0
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert len(manifest["image_ids"]) == 50
    assert manifest["new_api_request_count"] == 100
    assert manifest["reused_nested_request_count"] == 0
    assert not list(output.glob("*/records/**/*.json"))
