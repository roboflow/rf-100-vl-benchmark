import evaluate_qwen38_reference_count_calibration as calibration
import evaluate_qwen38_box_count_ablation as box


def reference(category_id, rank, image_id):
    return box.ReferenceBox(
        category_id=category_id,
        category_name=str(category_id),
        rank=rank,
        annotation_id=rank,
        image_id=image_id,
        file_name=f"{image_id}.jpg",
        width=100,
        height=100,
        bbox_xyxy_1000=(0, 0, 100, 100),
    )


def test_target_image_is_removed_from_every_class_reference_pool():
    references = {
        1: (reference(1, 1, 10), reference(1, 2, 11)),
        2: (reference(2, 1, 12), reference(2, 2, 10)),
    }
    result = calibration.references_without_target(references, 10)
    assert [[value.image_id for value in values] for values in result.values()] == [[11], [12]]


def test_target_exclusion_refuses_a_class_without_independent_support():
    references = {1: (reference(1, 1, 10),)}
    try:
        calibration.references_without_target(references, 10)
    except ValueError as error:
        assert "no independent reference" in str(error)
    else:
        raise AssertionError("Expected target/reference exclusion failure.")


def test_all_calibration_conditions_disable_reasoning_and_use_sparse_references():
    for condition in calibration.CONDITIONS:
        assert condition.reasoning_effort == "none"
        assert condition.formulation == "multi"
        if condition.box_count:
            assert condition.representation == "numeric_prediction"
            assert condition.explicit_sparse_references is True
