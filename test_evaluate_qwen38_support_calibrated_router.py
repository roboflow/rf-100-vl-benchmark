import pytest

import evaluate_qwen38_support_calibrated_router as gate


def _reference(image_id):
    class Reference:
        pass

    value = Reference()
    value.image_id = image_id
    return (value,)


def test_calibration_excludes_every_reference_source_image():
    train = {
        "images": [
            {"id": 1, "file_name": "ref.jpg"},
            {"id": 2, "file_name": "cal.jpg"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1},
            {"id": 2, "image_id": 1, "category_id": 2},
            {"id": 3, "image_id": 2, "category_id": 1},
        ],
        "categories": [{"id": 1, "name": "one"}, {"id": 2, "name": "two"}],
    }
    split, audit = gate.build_calibration_split(
        train, {1: _reference(1), 2: _reference(1)}
    )
    assert [value["id"] for value in split["images"]] == [2]
    assert [value["id"] for value in split["annotations"]] == [3]
    assert audit["reference_calibration_image_overlap"] == []
    assert audit["classes_without_calibration_objects"] == [2]


def test_known_object_recall_ignores_unmatched_predictions():
    calibration = {
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"image_id": 1, "category_id": 1, "bbox": [20, 20, 10, 10]},
        ]
    }
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
        {"image_id": 1, "category_id": 1, "bbox": [100, 100, 10, 10]},
    ]
    result = gate.known_object_recall(calibration, predictions)
    assert result["micro_recall50"] == 0.5
    assert result["class_macro_recall50"] == 0.5
    assert result["unmatched_predictions_ignored"] is True


def test_greedy_matching_does_not_reuse_one_prediction():
    annotations = [
        {"bbox": [0, 0, 10, 10]},
        {"bbox": [0, 0, 10, 10]},
    ]
    predictions = [{"bbox": [0, 0, 10, 10]}]
    assert gate.greedy_matches(annotations, predictions, 0.5) == 1


def test_route_requires_material_primary_gain_without_recall50_loss():
    names = {"class_macro_recall50_95": 0.40, "class_macro_recall50": 0.60}
    references = {"class_macro_recall50_95": 0.43, "class_macro_recall50": 0.61}
    selected, deltas = gate.choose_route(names, references, 2.0)
    assert selected == gate.REFERENCE_MODE
    assert deltas["class_macro_recall50_95"] == pytest.approx(3.0)

    references["class_macro_recall50"] = 0.59
    selected, _ = gate.choose_route(names, references, 2.0)
    assert selected == gate.NAMES_MODE


def test_gate_uses_the_established_one_shot_prompt_condition():
    reference = gate.CONDITION_BY_MODE[gate.REFERENCE_MODE]
    assert reference.representation == "numeric_prediction"
    assert reference.box_count == 1
    assert reference.formulation == "multi"
    assert reference.reasoning_effort == "none"
    assert reference.explicit_sparse_references is False
