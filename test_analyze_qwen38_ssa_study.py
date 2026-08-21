import analyze_qwen38_ssa_study as analysis


def curve(values):
    return [
        {
            "prefix_images": index,
            "delta": {analysis.PRIMARY_DELTA: value},
        }
        for index, value in enumerate(values)
    ]


def test_stopping_policy_uses_best_observed_prefix_before_patience_stop():
    policy = analysis.Policy(window=2, patience=2, minimum_prefix=4, epsilon=1)
    result = analysis.simulate_policy(
        curve([0, 2, 8, 9, 4, 3, 2, 1]), policy, full_prefix=8
    )
    assert result["stop_prefix"] < 8
    assert result["selected_prefix"] == 3
    assert result["reason"] == "best_observed_prefix_before_stop"


def test_stopping_policy_chooses_names_when_signal_never_clears_noise():
    policy = analysis.Policy(window=2, patience=1, minimum_prefix=4, epsilon=3)
    result = analysis.simulate_policy(
        curve([0, 1, 2, 1, 0, -1]), policy, full_prefix=6
    )
    assert result["selected_prefix"] == 0
    assert result["reason"] == "no_support_signal_above_noise"


def test_no_stop_with_material_signal_uses_exhaustion_boundary():
    policy = analysis.Policy(window=2, patience=2, minimum_prefix=20, epsilon=1)
    result = analysis.simulate_policy(
        curve([0, 3, 5, 4]), policy, full_prefix=4
    )
    assert result["stop_prefix"] == 4
    assert result["selected_prefix"] == 4
    assert result["reason"] == "support_exhausted_with_material_signal"


def test_percentile_interpolates():
    assert analysis.percentile([0, 10], 0.25) == 2.5
