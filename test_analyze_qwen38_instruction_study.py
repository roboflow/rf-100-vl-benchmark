import json
from pathlib import Path

import pytest

import analyze_qwen38_instruction_study as analysis


def test_per_category_score_matches_perfect_and_empty_predictions(tmp_path):
    annotation_path = tmp_path / "ground_truth.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "width": 100, "height": 100}],
                "categories": [
                    {"id": 1, "name": "one"},
                    {"id": 2, "name": "two"},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
                    {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 10, 10], "area": 100, "iscrowd": 0},
                ],
            }
        )
    )
    perfect = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 1.0}]
    scores = analysis.score_category(annotation_path, perfect, 1)
    assert scores["mAP50_95"] == pytest.approx(100.0)
    assert scores["mAP50"] == pytest.approx(100.0)
    empty = analysis.score_category(annotation_path, [], 1)
    assert empty == {"mAP50_95": 0.0, "mAP50": 0.0}


def test_expand_ratings_and_cluster_bootstrap_are_deterministic():
    ratings = {
        "datasets": {
            "one": {
                "classes": ["a", "b"],
                "name_alone_insufficient": ["a"],
                "requires_state_role_or_context": [],
                "requires_special_boundary_rule": ["a"],
                "unusual_visual_domain": [],
            }
        }
    }
    expanded = analysis.expand_ratings(ratings)
    assert expanded[("one", "a")]["challenge_count"] == 2
    assert expanded[("one", "b")]["challenge_count"] == 0

    rows = [
        {"dataset": "one", "flag": 1, "gain": 4.0},
        {"dataset": "one", "flag": 0, "gain": 1.0},
        {"dataset": "two", "flag": 1, "gain": 6.0},
        {"dataset": "two", "flag": 0, "gain": 3.0},
    ]
    first = analysis._cluster_bootstrap(
        rows, analysis._binary_difference("flag", "gain"), repeats=100, seed=7
    )
    second = analysis._cluster_bootstrap(
        rows, analysis._binary_difference("flag", "gain"), repeats=100, seed=7
    )
    assert first == second
    assert first["estimate"] == pytest.approx(3.0)
