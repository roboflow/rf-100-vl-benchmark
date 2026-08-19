import pytest

import evaluate_qwen38_strict_binary_router_rf20 as rf20_router


def test_majority_route_is_stable_when_all_repeats_agree():
    assert rf20_router.majority_route(["class_names_only"] * 3) == (
        "class_names_only",
        True,
    )
    assert rf20_router.majority_route(["visual_references"] * 3) == (
        "visual_references",
        True,
    )


def test_majority_route_uses_majority_and_conservative_tie():
    assert rf20_router.majority_route(
        ["class_names_only", "visual_references", "visual_references"]
    ) == ("visual_references", False)
    assert rf20_router.majority_route(
        ["class_names_only", "visual_references"]
    ) == ("visual_references", False)
    with pytest.raises(ValueError, match="missing or invalid"):
        rf20_router.majority_route([])


def test_router_cost_uses_existing_rf20_price_convention():
    usage = {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 1_000_000,
        "prompt_tokens_details": {"cached_tokens": 400_000},
    }
    assert rf20_router.estimated_cost(usage) == pytest.approx(7.3)
