import pytest

import evaluate_qwen38_strict_binary_router as router

LABELS = ["author", "equation number", "figure caption", "paragraph"]


def test_prompt_is_conservative_dataset_level_and_score_blind():
    prompt = router.build_router_prompt(LABELS)
    assert "dataset-level router" in prompt
    assert "EVERY label" in prompt
    assert "A false class_names_only decision is more costly" in prompt
    assert "Do not predict boxes" in prompt
    assert all(label in prompt for label in LABELS)


def test_parse_accepts_strict_routes_and_exact_labels():
    decision = router.parse_router_decision(
        '{"route":"visual_references","confidence":0.9,'
        '"labels_requiring_visual_context":["equation number"],'
        '"reason":"annotation semantics"}',
        LABELS,
    )
    assert decision["route"] == "visual_references"
    assert decision["labels_requiring_visual_context"] == ["equation number"]
    zero = router.parse_router_decision(
        '{"route":"class_names_only","confidence":1,'
        '"labels_requiring_visual_context":[],"reason":"standard objects"}',
        LABELS,
    )
    assert zero["route"] == "class_names_only"


def test_parse_rejects_inconsistent_or_unknown_decisions():
    with pytest.raises(ValueError, match="cannot request visual context"):
        router.parse_router_decision(
            '{"route":"class_names_only","confidence":0.5,'
            '"labels_requiring_visual_context":["author"]}',
            LABELS,
        )
    with pytest.raises(ValueError, match="unknown label"):
        router.parse_router_decision(
            '{"route":"visual_references","confidence":0.5,'
            '"labels_requiring_visual_context":["bogus"]}',
            LABELS,
        )
