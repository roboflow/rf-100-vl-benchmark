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
    ground_truth = subset.subset_ground_truth(test)
    assert [image["id"] for image in ground_truth["images"]] == list(
        subset.SUBSET_IMAGE_IDS
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
    tasks = subset.build_tasks(test)
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
