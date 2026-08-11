import json
from pathlib import Path

import pytest

import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_exemplar_only_ablation as exemplar
import evaluate_qwen38_orion as base

DATASET = Path("RF100VL/rf20-vl-fsod/the-dreidel-project")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    train = base.load_coco(DATASET / "train/_annotations.coco.json")
    test = base.load_coco(DATASET / "test/_annotations.coco.json")
    categories = base.categories_by_id(test)
    references = box_ablation.select_reference_sequences(
        train,
        DATASET / "train",
        required_count=max(exemplar.BOX_COUNTS),
    )
    assets = box_ablation.prepare_reference_assets(
        DATASET / "train",
        tmp_path_factory.mktemp("exemplar-only-references"),
        references,
    )
    return test, categories, references, assets


def test_condition_and_task_matrix_is_complete(dataset):
    test, categories, _, _ = dataset
    assert len(exemplar.CONDITIONS) == 12
    assert {
        (condition.instruction, condition.representation, condition.box_count)
        for condition in exemplar.CONDITIONS
    } == {
        (instruction, representation, count)
        for instruction in exemplar.INSTRUCTIONS
        for representation in exemplar.REPRESENTATIONS
        for count in exemplar.BOX_COUNTS
    }
    tasks = exemplar.build_tasks(test, categories)
    assert len(tasks) == 3888
    assert len({task.key for task in tasks}) == len(tasks)
    assert all(
        sum(task.mode == condition.mode for task in tasks) == 324
        for condition in exemplar.CONDITIONS
    )


@pytest.mark.parametrize("mode", exemplar.MODES)
def test_prompts_never_expose_semantic_class_names(mode, dataset):
    test, categories, references, assets = dataset
    image = test["images"][0]
    category_id, category_name = next(iter(categories.items()))
    task = base.Task(
        mode=mode,
        image_id=int(image["id"]),
        file_name=str(image["file_name"]),
        width=int(image["width"]),
        height=int(image["height"]),
        category_id=category_id,
        category_name=category_name,
    )
    messages = exemplar.build_messages(
        task,
        DATASET / "test",
        references,
        assets,
    )
    condition = exemplar.CONDITIONS_BY_MODE[mode]
    content = messages[0]["content"]
    text = "\n".join(part["text"] for part in content if part["type"] == "text")
    assert all(name.casefold() not in text.casefold() for name in categories.values())
    assert sum(part["type"] == "image_url" for part in content) == (
        condition.box_count + 1
    )
    assert content[-1 if condition.instruction == "explicit" else -2][
        "type"
    ] == "image_url"

    if condition.instruction == "explicit":
        assert exemplar.EXPLICIT_PROMPT in text
        assert "same kind" in text.casefold()
        assert text.endswith("TARGET IMAGE:")
    else:
        assert exemplar.EXPLICIT_PROMPT not in text
        assert "same kind" not in text.casefold()
        assert "find" not in text.casefold()
        assert "detect" not in text.casefold()
        assert text.endswith("XYXY integers normalized 0..1000.")

    if condition.representation == "numeric":
        for reference in references[category_id][: condition.box_count]:
            assert json.dumps(
                {"bbox_2d": list(reference.bbox_xyxy_1000)},
                separators=(",", ":"),
            ) in text
    else:
        assert not any(
            part.get("text", "").startswith('{"bbox_2d":') for part in content
        )


def test_minimal_protocol_has_no_semantic_detection_instruction():
    value = exemplar.minimal_output_protocol().casefold()
    for forbidden in ("find", "detect", "same kind", "class", "reference"):
        assert forbidden not in value


def test_seven_and_ten_box_extension_is_a_complete_balanced_factorial():
    original = exemplar.BOX_COUNTS
    try:
        exemplar.configure_box_counts((7, 10))
        assert len(exemplar.CONDITIONS) == 8
        assert {
            (condition.instruction, condition.representation, condition.box_count)
            for condition in exemplar.CONDITIONS
        } == {
            (instruction, representation, count)
            for instruction in exemplar.INSTRUCTIONS
            for representation in exemplar.REPRESENTATIONS
            for count in (7, 10)
        }
    finally:
        exemplar.configure_box_counts(original)


def test_prepare_only_writes_resumable_contract(tmp_path):
    output = tmp_path / "exemplar-only"
    assert (
        exemplar.main(
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
    assert not manifest["class_names_exposed_to_model"]
    assert not manifest["minimal_mode_semantic_instruction"]
    assert manifest["explicit_prompt"] == exemplar.EXPLICIT_PROMPT
    assert progress["total"] == {
        "total": 3888,
        "success": 0,
        "model_failure": 0,
        "error": 0,
        "pending": 3888,
    }
    assert len(comparison["rows"]) == 12
    assert not (output / "_SUCCESS.json").exists()
