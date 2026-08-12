import math

import pytest

import analyze_qwen38_noise_floor as noise


def test_metric_summary_uses_observed_or_normal_repeatability_floor():
    result = noise.metric_summary([10.0, 10.5, 9.5, 10.2, 9.8])
    assert result["mean"] == pytest.approx(10.0)
    assert result["observed_range"] == 1.0
    assert result["tie_threshold"] == max(
        1.0, 1.96 * math.sqrt(2) * result["sample_sd"]
    )


def test_canonical_sha_is_order_insensitive_for_objects():
    assert noise.canonical_sha({"a": 1, "b": 2}) == noise.canonical_sha(
        {"b": 2, "a": 1}
    )


def test_at_least_three_repeats_are_required():
    with pytest.raises(ValueError):
        noise.metric_summary([1.0, 2.0])


def test_prediction_identity_ignores_detection_order():
    first = {"category_id": 1, "bbox": [0, 0, 10, 10], "score": 1.0}
    second = {"category_id": 2, "bbox": [20, 20, 5, 5], "score": 1.0}
    assert noise.canonical_predictions([first, second]) == noise.canonical_predictions(
        [second, first]
    )


def test_pairwise_detection_agreement_matches_only_same_class_at_iou_threshold():
    left = [
        {"category_id": 1, "bbox": [0, 0, 10, 10]},
        {"category_id": 2, "bbox": [20, 20, 10, 10]},
    ]
    right = [
        {"category_id": 1, "bbox": [1, 1, 10, 10]},
        {"category_id": 3, "bbox": [20, 20, 10, 10]},
    ]
    result = noise.pairwise_detection_agreement(left, right)
    assert result["matched_count"] == 1
    assert result["f1"] == 0.5
    assert result["matched_ious"] == pytest.approx([81 / 119])


def test_empty_detection_sets_have_perfect_pairwise_agreement():
    result = noise.pairwise_detection_agreement([], [])
    assert result["f1"] == 1.0
    assert result["matched_ious"] == []
