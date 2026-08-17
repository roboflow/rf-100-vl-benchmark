import json
from pathlib import Path

import pytest

import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_orion as base
import evaluate_qwen38_recipe as recipe

DATASET = Path("RF100VL/rf20-vl-fsod/the-dreidel-project")
DEFECT = Path("RF100VL/rf20-vl-fsod-fresh-20260813/defect-detection")


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


def test_ten_shot_groups_instances_that_share_a_reference_image(tmp_path):
    train = base.load_coco(DEFECT / "train/_annotations.coco.json")
    test = base.load_coco(DEFECT / "test/_annotations.coco.json")
    categories = base.categories_by_id(test)
    references = box_ablation.select_reference_sequences(
        train,
        DEFECT / "train",
        required_count=10,
        distinct_images_only=False,
    )
    assets = box_ablation.prepare_reference_assets(
        DEFECT / "train", tmp_path / "references", references
    )
    condition = recipe.Condition(
        "box10",
        "multi",
        "class_names",
        "numeric_prediction",
        10,
        group_reference_instances_by_image=True,
    )
    task = make_task(condition.mode, "multi", test, categories)
    content = recipe.build_messages(
        task,
        condition,
        DEFECT / "test",
        categories,
        {},
        references,
        assets,
    )[0]["content"]
    reference_payloads = [
        json.loads(part["text"])
        for part in content
        if part["type"] == "text" and part["text"].startswith('[{"bbox_2d"')
    ]
    assert sum(len(payload) for payload in reference_payloads) == 40
    assert len(reference_payloads) == 34
    assert any(len(payload) > 1 for payload in reference_payloads)
    assert sum(part["type"] == "image_url" for part in content) == 35


def test_all_available_condition_includes_every_official_train_annotation(tmp_path):
    dataset = Path("RF100VL/rf20-vl-fsod-fresh-20260813/all-elements")
    train = base.load_coco(dataset / "train/_annotations.coco.json")
    test = base.load_coco(dataset / "test/_annotations.coco.json")
    categories = base.categories_by_id(test)
    references = box_ablation.select_reference_sequences(
        train,
        dataset / "train",
        required_count=10,
        distinct_images_only=False,
        allow_fewer=True,
    )
    assets = box_ablation.prepare_reference_assets(
        dataset / "train", tmp_path / "references", references
    )
    condition = recipe.Condition(
        "all_available",
        "multi",
        "class_names",
        "numeric_prediction",
        10,
        group_reference_instances_by_image=True,
        explicit_sparse_references=True,
        all_available_references=True,
    )
    task = make_task(condition.mode, "multi", test, categories)
    content = recipe.build_messages(
        task,
        condition,
        dataset / "test",
        categories,
        {},
        references,
        assets,
    )[0]["content"]
    assert "Use all positive reference boxes supplied for each label." in content[0]["text"]
    assert "10 positive reference" not in content[0]["text"]
    payloads = [
        json.loads(part["text"])
        for part in content
        if part["type"] == "text" and part["text"].startswith('[{"bbox_2d"')
    ]
    assert sum(map(len, payloads)) == len(train["annotations"])


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


def test_retry_terminal_provider_failures_is_explicit_opt_in():
    assert recipe.parse_args(
        ["--dataset-dir", ".", "--conditions", "conditions.json", "--output-dir", "out"]
    ).retry_terminal_provider_failures is False
    assert recipe.parse_args(
        [
            "--dataset-dir", ".",
            "--conditions", "conditions.json",
            "--output-dir", "out",
            "--retry-terminal-provider-failures",
        ]
    ).retry_terminal_provider_failures is True


def test_prediction_shaped_numeric_reference_exactly_matches_output_schema(dataset):
    test, categories, references, assets, self_names = dataset
    condition = recipe.Condition(
        mode="numeric_prediction",
        formulation="multi",
        semantics="class_names",
        representation="numeric_prediction",
        box_count=1,
    )
    task = make_task(condition.mode, "multi", test, categories)
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
    reference_annotations = []
    for part in content:
        if part["type"] != "text" or not part["text"].startswith("[{"):
            continue
        value = json.loads(part["text"])
        assert isinstance(value, list) and len(value) == 1
        assert list(value[0]) == ["bbox_2d", "label"]
        assert isinstance(value[0]["bbox_2d"], list) and len(value[0]["bbox_2d"]) == 4
        assert value[0]["label"] in categories.values()
        reference_annotations.append(value[0])
    assert len(reference_annotations) == len(categories)


def test_detection_reference_and_output_examples_share_one_serializer():
    serialized = recipe.detection_list_json(
        [([100, 200, 300, 400], "widget")]
    )
    assert serialized == '[{"bbox_2d":[100,200,300,400],"label":"widget"}]'
    assert (
        '[{"bbox_2d":[x1,y1,x2,y2],"label":"exact requested label"}]'
        in recipe._output_contract(["widget"])
    )


def test_explicit_sparse_condition_changes_only_reference_semantics(dataset):
    test, categories, references, assets, _ = dataset
    implicit = recipe.Condition(
        "implicit", "multi", "class_names", "numeric_prediction", 2
    )
    explicit = recipe.Condition(
        "explicit",
        "multi",
        "class_names",
        "numeric_prediction",
        2,
        explicit_sparse_references=True,
    )
    implicit_task = make_task(implicit.mode, "multi", test, categories)
    explicit_task = make_task(explicit.mode, "multi", test, categories)
    implicit_content = recipe.build_messages(
        implicit_task,
        implicit,
        DATASET / "test",
        categories,
        {},
        references,
        assets,
    )[0]["content"]
    explicit_content = recipe.build_messages(
        explicit_task,
        explicit,
        DATASET / "test",
        categories,
        {},
        references,
        assets,
    )[0]["content"]
    assert implicit_content[1:] == explicit_content[1:]
    sparse_clause = (
        "The marked boxes are sparse positive exemplars. Treat all unmarked "
        "objects and regions in reference images as unlabeled, not as negative "
        "examples or exhaustive annotations."
    )
    assert sparse_clause not in implicit_content[0]["text"]
    assert sparse_clause in explicit_content[0]["text"]
    assert "explicit_sparse_references" not in recipe.condition_payload(implicit)
    assert recipe.condition_payload(explicit)["explicit_sparse_references"] is True


def test_correct_instructions_append_the_exact_dataset_readme(dataset):
    test, categories, references, assets, _ = dataset
    readme = (DATASET / "README.dataset.txt").read_text(encoding="utf-8").strip()
    condition = recipe.Condition(
        "instructions", "multi", "class_names", "none", 0,
        instruction_mode="correct",
    )
    task = make_task(condition.mode, "multi", test, categories)
    content = recipe.build_messages(
        task,
        condition,
        DATASET / "test",
        categories,
        {},
        references,
        assets,
        readme,
    )[0]["content"]
    assert f"DATASET ANNOTATOR GUIDE:\n{readme}" in content[0]["text"]
    assert content[0]["text"].index("END DATASET ANNOTATOR GUIDE") < content[0]["text"].index(
        "FINAL DETECTION REQUEST"
    )
    assert content[0]["text"].endswith(
        "Do not explain, restate the guide, or describe the detected objects in prose."
    )
    assert content[-2] == {"type": "text", "text": "TARGET IMAGE:"}
    assert content[-1]["type"] == "image_url"


def test_permuted_instructions_preserve_content_but_break_section_mapping(dataset):
    readme = (DATASET / "README.dataset.txt").read_text(encoding="utf-8").strip()
    permuted = recipe.permute_class_instruction_sections(readme)
    assert permuted != readme
    assert sorted(permuted.split()) == sorted(readme.split())
    assert permuted[: permuted.index("# Object Classes") + len("# Object Classes")] == (
        readme[: readme.index("# Object Classes") + len("# Object Classes")]
    )
    original_first = readme.split("## Dreidel", 1)[1].split("## Gimel", 1)[0]
    permuted_first = permuted.split("## Dreidel", 1)[1].split("## Gimel", 1)[0]
    original_second = readme.split("## Gimel", 1)[1].split("## Hay", 1)[0]
    assert permuted_first == original_second
    assert permuted_first != original_first


def test_strict_permutation_also_breaks_introduction_definition_mapping(dataset):
    readme = (DATASET / "README.dataset.txt").read_text(encoding="utf-8").strip()
    detailed_only = recipe.permute_class_instruction_sections(readme)
    strict = recipe.permute_all_class_guidance(readme)
    assert strict != detailed_only
    assert strict != readme
    assert sorted(strict.split()) == sorted(readme.split())
    original_definition = readme.split("- **Dreidel**:", 1)[1].splitlines()[0]
    strict_definition = strict.split("- **Dreidel**:", 1)[1].splitlines()[0]
    assert strict_definition != original_definition


def test_instruction_modes_are_validated_and_manifested(tmp_path):
    path = tmp_path / "conditions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "mode": "correct",
                    "formulation": "multi",
                    "semantics": "class_names",
                    "representation": "none",
                    "box_count": 0,
                    "instruction_mode": "correct",
                }
            ]
        )
    )
    condition = recipe.load_conditions(path)[0]
    assert recipe.condition_payload(condition)["instruction_mode"] == "correct"
    invalid = json.loads(path.read_text())
    invalid[0]["instruction_mode"] = "unknown"
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError):
        recipe.load_conditions(path)


@pytest.mark.parametrize(
    "value",
    [
        {"mode": "bad", "formulation": "bad", "semantics": "class_names", "representation": "none", "box_count": 0},
        {"mode": "bad", "formulation": "multi", "semantics": "anonymous_explicit", "representation": "none", "box_count": 0},
        {"mode": "bad", "formulation": "multi", "semantics": "self_name_only", "representation": "drawn", "box_count": 1},
        {"mode": "bad", "formulation": "multi", "semantics": "class_names", "representation": "none", "box_count": 0, "explicit_sparse_references": True},
    ],
)
def test_invalid_conditions_are_rejected(tmp_path, value):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([value]))
    with pytest.raises(ValueError):
        recipe.load_conditions(path)
