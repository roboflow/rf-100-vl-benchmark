import pytest

from qwen38_calibrated_counts import COUNTS, MODE_BY_COUNT, choose_count


def metrics(primary, recall50):
    return {
        "class_macro_recall50_95": primary / 100,
        "class_macro_recall50": recall50 / 100,
    }


def test_contract_has_all_candidate_counts_and_unique_modes():
    assert COUNTS == (0, 1, 2, 5, 10)
    assert len(set(MODE_BY_COUNT.values())) == len(COUNTS)


def test_choose_count_accepts_material_later_gains_and_skips_small_gains():
    values = {
        0: metrics(20, 40),
        1: metrics(23, 45),
        2: metrics(24, 46),
        5: metrics(26, 50),
        10: metrics(27, 52),
    }
    selected, trace = choose_count(values, 2)
    assert selected == 5
    assert [row["accepted"] for row in trace] == [True, False, True, False]


def test_choose_count_requires_nondecreasing_recall50():
    values = {count: metrics(20, 40) for count in COUNTS}
    values[10] = metrics(30, 39)
    selected, _ = choose_count(values, 2)
    assert selected == 0


def test_choose_count_rejects_missing_candidates():
    with pytest.raises(ValueError):
        choose_count({0: metrics(20, 40)}, 2)
