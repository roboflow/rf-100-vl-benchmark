from pathlib import Path

import preflight_qwen38_adaptive_rf20 as preflight


def test_adaptive_contract_passes_fresh_official_rf20_fsod():
    result = preflight.validate(
        Path("RF100VL/rf20-vl-fsod-fresh-20260813").resolve(),
        max_examples_per_class=10,
    )
    assert result["benchmark"] == "RF20-VL-FSOD"
    assert result["dataset_count"] == 20
    assert result["test_images"] == 3970
    assert result["test_objects"] == 57285
    assert result["classes"] == 110
    assert result["available_train_references"] == 1099
    assert result["initial_examples_per_class"] == 0
    assert result["max_examples_per_class"] == 10
    assert result["all_references_train_only"]
    assert result["all_reference_payloads_match_prediction_schema"]
    assert result["initial_turn_contains_only_class_names_and_target"]
    assert result["prediction_feedback"] is False
    assert result["test_ground_truth_visible"] is False
    assert result["final_detection_only_scored"]
    assert result["reasoning_disabled"]
    assert result["temperature"] == 0
    assert result["max_detections"] == 500
