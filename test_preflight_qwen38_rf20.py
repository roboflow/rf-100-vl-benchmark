from pathlib import Path

import preflight_qwen38_rf20 as preflight


def test_locked_rf20_launch_contract_passes_complete_local_dataset():
    result = preflight.validate(
        Path("RF100VL/rf20-vl-fsod").resolve(),
        Path("qwen38-fsod-configs/rf20-three-way.json").resolve(),
    )
    assert result["dataset_count"] == 20
    assert result["test_images"] == 3970
    assert result["classes"] == 110
    assert result["test_objects"] == 57285
    assert result["requests"] == 11910
    assert result["all_references_train_only"]
    assert result["all_reference_payloads_match_prediction_schema"]
    assert result["target_image_is_last"]
