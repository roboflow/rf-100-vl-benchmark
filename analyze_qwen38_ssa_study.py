#!/usr/bin/env python3
"""Audit and simulate stopping policies from a three-dataset SSA collection.

Policy selection in this file reads support curves only. Test-grid scores are
loaded afterwards solely to measure whether the locked provisional policy and
the support signal predict useful test prefixes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any

DATASETS = (
    "the-dreidel-project",
    "orionproducts",
    "lacrosse-object-detection",
)
SEEDS = (1234, 4321, 2026)
PRIMARY_DELTA = "class_macro_recall50_95"


@dataclass(frozen=True)
class Policy:
    window: int
    patience: int
    minimum_prefix: int
    epsilon: float


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of no values.")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def rolling_means(values: Sequence[float], window: int) -> list[float]:
    if window < 1:
        raise ValueError("window must be positive")
    return [fmean(values[max(0, index - window + 1) : index + 1]) for index in range(len(values))]


def simulate_policy(
    curve: Sequence[dict[str, Any]], policy: Policy, full_prefix: int
) -> dict[str, Any]:
    points = [row for row in curve if int(row["prefix_images"]) > 0]
    if not points:
        return {
            "selected_prefix": 0,
            "stop_prefix": 0,
            "reason": "no_informative_points",
        }
    values = [float(row["delta"][PRIMARY_DELTA]) for row in points]
    smoothed = rolling_means(values, policy.window)
    weak_windows = 0
    stop_index: int | None = None
    for index in range(len(points)):
        prefix = int(points[index]["prefix_images"])
        if prefix < policy.minimum_prefix or index + 1 < 2 * policy.window:
            continue
        recent = fmean(values[index - policy.window + 1 : index + 1])
        previous = fmean(values[index - 2 * policy.window + 1 : index - policy.window + 1])
        weak_windows = weak_windows + 1 if recent - previous < policy.epsilon else 0
        if weak_windows >= policy.patience:
            stop_index = index
            break

    observed_end = stop_index if stop_index is not None else len(points) - 1
    best_index = max(range(observed_end + 1), key=lambda index: smoothed[index])
    best_signal = smoothed[best_index]
    if best_signal < policy.epsilon:
        selected_prefix = 0
        reason = "no_support_signal_above_noise"
    elif stop_index is None:
        selected_prefix = full_prefix
        reason = "support_exhausted_with_material_signal"
    else:
        selected_prefix = int(points[best_index]["prefix_images"])
        reason = "best_observed_prefix_before_stop"
    return {
        "selected_prefix": selected_prefix,
        "stop_prefix": (
            int(points[stop_index]["prefix_images"])
            if stop_index is not None
            else full_prefix
        ),
        "reason": reason,
        "best_observed_prefix": int(points[best_index]["prefix_images"]),
        "best_smoothed_delta_recall50_95": best_signal,
        "policy": asdict(policy),
    }


def order_noise_scales(curves: dict[str, dict[int, list[dict[str, Any]]]]) -> list[float]:
    """Estimate support-order variability at matched relative progress points."""

    scales = []
    for dataset_curves in curves.values():
        for fraction in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            values = []
            for curve in dataset_curves.values():
                informative = [row for row in curve if int(row["prefix_images"]) > 0]
                index = min(len(informative) - 1, round(fraction * (len(informative) - 1)))
                values.append(float(informative[index]["delta"][PRIMARY_DELTA]))
            if len(values) >= 2:
                scales.append(stdev(values))
    return scales


def validate_curve(
    curve: Sequence[dict[str, Any]], manifest: dict[str, Any]
) -> list[str]:
    violations = []
    order = [int(value) for value in manifest["support_image_order"]]
    if len(curve) != len(order):
        violations.append("curve_length_mismatch")
    if len(order) != len(set(order)):
        violations.append("duplicate_support_images")
    for index, row in enumerate(curve):
        if int(row["prefix_images"]) != index:
            violations.append("prefix_index_mismatch")
        if not row.get("target_absent_from_prefix"):
            violations.append("prequential_target_leakage")
        if row.get("branch_status") not in {"success", "model_failure"}:
            violations.append("nonterminal_branch")
        if row.get("zero_status") not in {"success", "model_failure"}:
            violations.append("nonterminal_zero_probe")
    if curve and curve[0]["delta"][PRIMARY_DELTA] != 0:
        violations.append("turn_one_delta_not_zero")
    if manifest.get("test_images_used_during_adaptation") is not False:
        violations.append("test_images_used_during_adaptation")
    if manifest.get("test_annotations_used_during_adaptation") is not False:
        violations.append("test_annotations_used_during_adaptation")
    if manifest.get("clean_trunk") is not True:
        violations.append("unclean_trunk")
    if manifest.get("temperature") != 0.0:
        violations.append("nonzero_temperature")
    if manifest.get("reasoning_disabled") is not True:
        violations.append("reasoning_enabled")
    return sorted(set(violations))


def read_test_grid(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as file:
        return [
            {
                **row,
                "prefix_images": int(row["prefix_images"]),
                "mAP50_95": float(row["mAP50_95"]),
                "mAP50": float(row["mAP50"]),
            }
            for row in csv.DictReader(file)
        ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    output = args.output or run_root / "stopping_validation.json"

    curves: dict[str, dict[int, list[dict[str, Any]]]] = {}
    manifests: dict[str, dict[int, dict[str, Any]]] = {}
    violations: dict[str, list[str]] = {}
    actual_cost = 0.0
    for dataset in DATASETS:
        curves[dataset] = {}
        manifests[dataset] = {}
        for seed in SEEDS:
            directory = run_root / dataset / f"seed-{seed}"
            manifest = read_json(directory / "run_manifest.json")
            curve = read_jsonl(directory / "adaptation_curve.jsonl")
            summary = read_json(directory / "summary.json")
            manifests[dataset][seed] = manifest
            curves[dataset][seed] = curve
            current_violations = validate_curve(curve, manifest)
            if current_violations:
                violations[f"{dataset}/{seed}"] = current_violations
            actual_cost += float(summary["invocation_usage"]["estimated_usd"])

    scales = order_noise_scales(curves)
    epsilon_quantiles = {
        "q25": percentile(scales, 0.25),
        "q50": median(scales),
        "q75": percentile(scales, 0.75),
    }
    # The provisional policy is fixed from support curves alone. We inspect its
    # test result only after this selection is materialized in the output.
    policy = Policy(window=3, patience=2, minimum_prefix=8, epsilon=epsilon_quantiles["q50"])

    locked_decisions: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        decisions = {}
        for seed in SEEDS:
            full_prefix = len(manifests[dataset][seed]["support_image_order"])
            decisions[str(seed)] = simulate_policy(curves[dataset][seed], policy, full_prefix)
        locked_decisions[dataset] = decisions
    locked_support_policy = {
        "support_only_policy_selection": True,
        "test_labels_used_to_choose_policy": False,
        "policy": asdict(policy),
        "order_noise_scale_points": len(scales),
        "epsilon_quantiles": epsilon_quantiles,
        "decisions": locked_decisions,
    }
    # This file is deliberately materialized before any test-grid file is read.
    # It is the auditable boundary between support-only selection and evaluation.
    locked_path = run_root / "locked_support_policy.json"
    locked_path.write_text(
        json.dumps(locked_support_policy, indent=2) + "\n", encoding="utf-8"
    )

    datasets = {}
    for dataset in DATASETS:
        decisions = locked_decisions[dataset]
        grid = read_test_grid(run_root / dataset / "seed-1234" / "test_grid.csv")
        oracle_primary = max(grid, key=lambda row: row["mAP50_95"])
        oracle_secondary = max(grid, key=lambda row: row["mAP50"])
        selected = decisions["1234"]["selected_prefix"]
        selected_grid = next((row for row in grid if row["prefix_images"] == selected), None)
        datasets[dataset] = {
            "decisions": decisions,
            "selected_prefixes": [decisions[str(seed)]["selected_prefix"] for seed in SEEDS],
            "canonical_selected_grid_result": selected_grid,
            "test_grid": grid,
            "test_grid_oracle_mAP50_95": oracle_primary,
            "test_grid_oracle_mAP50": oracle_secondary,
        }

    result = {
        "study": "three-dataset SSA stopping validation",
        "support_only_policy_selection": True,
        "test_labels_used_to_choose_policy": False,
        "policy": asdict(policy),
        "locked_support_policy": str(locked_path),
        "order_noise_scale_points": len(scales),
        "epsilon_quantiles": epsilon_quantiles,
        "guardrail_violations": violations,
        "datasets": datasets,
        "actual_api_cost_usd": actual_cost,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
