import json
from pathlib import Path

import pytest

import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_orion as base
import evaluate_qwen38_recipe as recipe

DATASET = Path("RF100VL/rf20-vl-fsod/the-dreidel-project")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    train = base.load_coco(DATASET / "train/_annotations.coco.json")
    test = base.load_coco(DATASET / "test/_annotations.coco.json")
    categories = base.categories_by_id(test)
    references = box_ablation.select_reference_sequences(
        train, DATASET / "train", required_count=2
    )
    assets = box_ablation.prepare_reference_assets(
        DATASET / "train",
        tmp_path_factory.mktemp("recipe-references"),
        references,
    )
    self_names = {
        category_id: f"generated visual name {index}"
        for index, category_id in enumerate(categories)
    }
    return test, categories, references, assets, self_names


def make_task(mode, formulation, test, categories):
    image = test["images"][0]
    category_id, category_name = next(iter(categories.items()))
    return base.Task(
        mode=mode,
        image_id=int(image["id"]),
        file_name=str(image["file_name"]),
        width=int(image["width"]),
        height=int(image["height"]),
        category_id=category_id if formulation == "single" else None,
        category_name=category_name if formulation == "single" else None,
    )


@pytest.mark.parametrize("formulation", ["single", "multi"])
@pytest.mark.parametrize("representation", ["numeric", "drawn"])
@pytest.mark.parametrize("semantics", ["class_names", "anonymous_explicit", "anonymous_minimal", "self_name"])
def test_prompt_factorial_is_well_formed(
    formulation, representation, semantics, dataset
):
    test, categories, references, assets, self_names = dataset
    condition = recipe.Condition(
        mode=f"{formulation}_{semantics}_{representation}",
        formulation=formulation,
        semantics=semantics,
        representation=representation,
        box_count=2,
    )
    task = make_task(condition.mode, formulation, test, categories)
    messages = recipe.build_messages(
        task,
        condition,
        DATASET / "test",
        categories,
        self_names,
        references,
        assets,
    )
    content = messages[0]["content"]
    reference_classes = 1 if formulation == "single" else len(categories)
    assert sum(part["type"] == "image_url" for part in content) == (
        2 * reference_classes + 1
    )
    if semantics == "anonymous_minimal":
        assert content[-3]["text"] == "TARGET IMAGE:"
    else:
        assert content[-2]["text"] == "TARGET IMAGE:"
    assert content[-1]["type"] in {"image_url", "text"}
    text = "\n".join(part["text"] for part in content if part["type"] == "text")
    if semantics.startswith("anonymous"):
        assert all(name.casefold() not in text.casefold() for name in categories.values())
        assert "Concept A" in text
    if semantics == "anonymous_minimal":
        assert "same kind" not in text.casefold()
        assert "detect every" not in text.casefold()
    if semantics == "self_name":
        assert "generated visual name" in text


def test_single_and_multi_task_counts(dataset):
    test, categories, *_ = dataset
    conditions = (
        recipe.Condition("single", "single", "class_names", "none", 0),
        recipe.Condition("multi", "multi", "class_names", "none", 0),
    )
    tasks = recipe.build_tasks(test, categories, conditions)
    assert len(tasks) == len(test["images"]) * (len(categories) + 1)
    assert len({task.key for task in tasks}) == len(tasks)
    first_image_id = int(test["images"][0]["id"])
    first_image_tasks = [task for task in tasks if task.image_id == first_image_id]
    assert [task.mode for task in first_image_tasks] == [
        *(["single"] * len(categories)),
        "multi",
    ]
    assert all(task.image_id == first_image_id for task in tasks[: len(first_image_tasks)])


def test_labels_are_unique_and_semantic_names_are_hidden(dataset):
    _, categories, _, _, self_names = dataset
    anonymous = recipe.Condition(
        "anonymous", "multi", "anonymous_explicit", "drawn", 1
    )
    generated = recipe.Condition("generated", "multi", "self_name", "drawn", 1)
    anonymous_labels = recipe.display_labels(anonymous, categories, self_names)
    generated_labels = recipe.display_labels(generated, categories, self_names)
    assert list(anonymous_labels.values()) == [
        "Concept A",
        "Concept B",
        "Concept C",
        "Concept D",
        "Concept E",
        "Concept F",
    ]
    assert len(set(generated_labels.values())) == len(categories)
    assert all(
        semantic.casefold() not in " ".join(anonymous_labels.values()).casefold()
        for semantic in categories.values()
    )


def test_prepare_only_subset_has_locked_deterministic_contract(tmp_path, dataset):
    test, categories, *_ = dataset
    image_ids = [int(image["id"]) for image in test["images"][:2]]
    condition_path = tmp_path / "conditions.json"
    condition_path.write_text(
        json.dumps(
            {
                "conditions": [
                    {
                        "mode": "anonymous_multi",
                        "formulation": "multi",
                        "semantics": "anonymous_explicit",
                        "representation": "numeric",
                        "box_count": 1,
                    },
                    {
                        "mode": "names_single",
                        "formulation": "single",
                        "semantics": "class_names",
                        "representation": "none",
                        "box_count": 0,
                    },
                ]
            }
        )
    )
    output = tmp_path / "prepared"
    arguments = [
        "--dataset-dir",
        str(DATASET),
        "--conditions",
        str(condition_path),
        "--output-dir",
        str(output),
        "--image-ids",
        *map(str, image_ids),
        "--prepare-only",
    ]
    assert recipe.main(arguments) == 0
    manifest = json.loads((output / "run_manifest.json").read_text())
    progress = json.loads((output / "progress.json").read_text())
    ground_truth = json.loads((output / "ground_truth_subset.json").read_text())
    assert manifest["common_settings"]["temperature"] == 0.0
    assert manifest["concurrency"] == 256
    assert manifest["thinking_controls"] == {
        "policy": "reasoning_effort-plus-enable_thinking-v1",
        "none_maps_to_enable_thinking": False,
    }
    assert manifest["max_detections"] == 500
    assert manifest["selected_test_image_ids"] == sorted(image_ids)
    assert len(ground_truth["images"]) == 2
    assert {int(value["image_id"]) for value in ground_truth["annotations"]} <= set(image_ids)
    assert progress["total"]["total"] == 2 * (1 + len(categories))
    assert progress["total"]["pending"] == progress["total"]["total"]
    assert not (output / "_SUCCESS.json").exists()


def test_condition_settings_explicitly_disable_both_thinking_paths():
    condition = recipe.Condition(
        mode="names_multi",
        formulation="multi",
        semantics="class_names",
        representation="none",
        box_count=0,
        reasoning_effort="none",
        seed=1234,
    )
    settings = recipe.condition_settings(condition, {"temperature": 0.0})
    assert settings["reasoning_effort"] == "none"
    assert settings["enable_thinking"] is False


@pytest.mark.parametrize(
    "value",
    [
        {"mode": "bad", "formulation": "bad", "semantics": "class_names", "representation": "none", "box_count": 0},
        {"mode": "bad", "formulation": "multi", "semantics": "anonymous_explicit", "representation": "none", "box_count": 0},
        {"mode": "bad", "formulation": "multi", "semantics": "self_name_only", "representation": "drawn", "box_count": 1},
    ],
)
def test_invalid_conditions_are_rejected(tmp_path, value):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([value]))
    with pytest.raises(ValueError):
        recipe.load_conditions(path)
