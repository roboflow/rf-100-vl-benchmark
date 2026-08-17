from pathlib import Path

import preflight_qwen38_rf20_all_available as preflight


def test_locked_all_available_contract_passes_fresh_official_rf20_fsod():
    result = preflight.validate(
        Path("RF100VL/rf20-vl-fsod-fresh-20260813").resolve(),
        Path("qwen38-fsod-configs/rf20-all-available-explicit-sparse.json").resolve(),
    )
    assert result["benchmark"] == "RF20-VL-FSOD"
    assert result["dataset_count"] == 20
    assert result["test_images"] == 3970
    assert result["classes"] == 110
    assert result["available_train_references"] == 1099
    assert result["test_objects"] == 57285
    assert result["requests"] == 3970
    assert result["reference_object_transmissions"] == 245070
    assert result["all_official_train_annotations_included_once_per_request"]
    assert result["all_references_train_only"]
    assert result["all_reference_payloads_match_prediction_schema"]
    assert result["explicit_sparse_reference_semantics"]
    assert result["target_image_is_last"]
    assert result["reasoning_disabled"]
