from pathlib import Path

import preflight_qwen38_strict_instruction_control as preflight


def test_strict_control_preflight():
    result = preflight.validate(
        Path("RF100VL/rf20-vl-fsod-fresh-20260813").resolve(),
        Path("qwen38-fsod-configs/instruction-study-strict-permuted-control.json").resolve(),
    )
    assert result["dataset_count"] == 6
    assert result["test_images"] == 1737
    assert result["request_count"] == 1737
    assert result["same_readme_vocabulary"] is True
    assert result["introduction_definitions_permuted"] is True
    assert result["detailed_class_sections_permuted"] is True
