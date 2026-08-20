"""Shared contracts for support-calibrated Qwen3.8 reference-count routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import evaluate_qwen38_recipe as recipe

COUNTS = (0, 1, 2, 5, 10)
CONFIG_BY_COUNT = {
    count: Path(__file__).parent
    / "qwen38-fsod-configs"
    / f"calibrated-count-b{count:02d}.json"
    for count in COUNTS
}
CONDITION_BY_COUNT = {
    count: recipe.load_conditions(path)[0] for count, path in CONFIG_BY_COUNT.items()
}
MODE_BY_COUNT = {
    count: condition.mode for count, condition in CONDITION_BY_COUNT.items()
}
COUNT_BY_MODE = {mode: count for count, mode in MODE_BY_COUNT.items()}


def choose_count(
    metrics_by_count: dict[int, dict[str, Any]], minimum_gain_points: float = 2.0
) -> tuple[int, list[dict[str, Any]]]:
    """Select the smallest count that provides each material monotonic upgrade."""

    if set(metrics_by_count) != set(COUNTS):
        raise ValueError(f"Expected metrics for counts {COUNTS}.")
    selected = 0
    trace: list[dict[str, Any]] = []
    for candidate in COUNTS[1:]:
        current = metrics_by_count[selected]
        proposed = metrics_by_count[candidate]
        primary_delta = 100 * (
            proposed["class_macro_recall50_95"]
            - current["class_macro_recall50_95"]
        )
        recall50_delta = 100 * (
            proposed["class_macro_recall50"] - current["class_macro_recall50"]
        )
        accepted = (
            primary_delta >= minimum_gain_points and recall50_delta >= 0.0
        )
        trace.append(
            {
                "current_count": selected,
                "candidate_count": candidate,
                "delta_recall50_95": primary_delta,
                "delta_recall50": recall50_delta,
                "accepted": accepted,
            }
        )
        if accepted:
            selected = candidate
    return selected, trace


def read_route_rows(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Route summary must contain a nonempty rows list.")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("dataset"), str):
            raise ValueError("Invalid route row.")
        count = int(row.get("selected_count", -1))
        if count not in COUNTS:
            raise ValueError(f"Unsupported selected count: {count}")
        if row["dataset"] in result:
            raise ValueError(f"Duplicate route dataset: {row['dataset']}")
        result[row["dataset"]] = row
    return result
