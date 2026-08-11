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
