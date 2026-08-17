import json
from pathlib import Path

import pytest

import preflight_qwen38_instruction_study as preflight

ROOT = Path("RF100VL/rf20-vl-fsod-fresh-20260813")
FULL = Path("qwen38-fsod-configs/rf20-instructions-only.json")
SUBSET = Path("qwen38-fsod-configs/instruction-study-six-dataset-five-arm.json")
RATINGS = Path("qwen38-fsod-configs/rf20-label-sufficiency-ratings.json")


def test_locked_instruction_study_preflight():
    result = preflight.validate(
        ROOT.resolve(), FULL.resolve(), SUBSET.resolve(), RATINGS.resolve()
    )
    assert result["dataset_count"] == 20
    assert result["test_images"] == 3970
    assert result["classes"] == 110
    assert result["subset_images"] == 1737
    assert result["full_requests"] == 3970
    assert result["subset_requests"] == 8685
    assert result["conditions_interleaved_within_image"] is True
    assert result["target_image_last"] is True


def test_preflight_rejects_changed_ratings(tmp_path):
    ratings = json.loads(RATINGS.read_text())
    ratings["datasets"]["actions"]["classes"][0] = "changed"
    path = tmp_path / "ratings.json"
    path.write_text(json.dumps(ratings))
    with pytest.raises(ValueError, match="inventory mismatch"):
        preflight.validate(ROOT.resolve(), FULL.resolve(), SUBSET.resolve(), path)
