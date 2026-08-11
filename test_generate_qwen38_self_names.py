import json
from pathlib import Path

import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_orion as base
import generate_qwen38_self_names as self_names

DATASET = Path("RF100VL/rf20-vl-fsod/the-dreidel-project")


def test_parse_name_accepts_json_fence_and_short_fallback():
    assert self_names.parse_name('{"name":"striped spinning toy"}') == "striped spinning toy"
    assert self_names.parse_name('```json\n{"name":"blue package"}\n```') == "blue package"
    assert self_names.parse_name("red foil candy packet") == "red foil candy packet"


def test_self_name_prompt_never_contains_ground_truth_names(tmp_path):
    train = base.load_coco(DATASET / "train/_annotations.coco.json")
    test = base.load_coco(DATASET / "test/_annotations.coco.json")
    categories = base.categories_by_id(test)
    references = box_ablation.select_reference_sequences(
        train, DATASET / "train", required_count=1
    )
    assets = box_ablation.prepare_reference_assets(
        DATASET / "train", tmp_path / "references", references
    )
    for representation in ("numeric", "drawn"):
        category_id = next(iter(categories))
        messages = self_names.build_messages(
            category_id, representation, 1, references, assets
        )
        content = messages[0]["content"]
        text = "\n".join(part["text"] for part in content if part["type"] == "text")
        assert all(name.casefold() not in text.casefold() for name in categories.values())
        assert sum(part["type"] == "image_url" for part in content) == 1
        if representation == "numeric":
            assert "bbox_2d" in text
        else:
            assert "bbox_2d" not in text
