#!/usr/bin/env python3
"""Analyze paired 1/2/5-reference repeats on larger RF20 datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import evaluate_qwen38_orion as base
from analyze_qwen38_noise_floor import metric_summary


BOX_COUNTS = (1, 2, 5)
METRICS = ("mAP50_95", "mAP50")
T_CRITICAL_95 = {3: 4.302652729696142, 5: 2.7764451051977987}


def mode(box_count: int, repeat: int) -> str:
    return f"box_b{box_count:02d}_repeat_{repeat:02d}"


def confidence_interval(values: list[float]) -> list[float]:
    critical = T_CRITICAL_95.get(len(values))
    if critical is None:
        raise ValueError("Only three- and five-repeat analyses are supported.")
    mean = fmean(values)
    margin = critical * stdev(values) / math.sqrt(len(values))
    return [mean - margin, mean + margin]


def analyze_dataset(
    name: str, run_directory: Path, repeat_count: int
) -> dict[str, Any]:
    comparison = json.loads((run_directory / "comparison_summary.json").read_text())
    rows = {str(row["mode"]): row for row in comparison["rows"]}
    expected = {
        mode(box_count, repeat)
        for repeat in range(1, repeat_count + 1)
        for box_count in BOX_COUNTS
    }
    missing = expected - set(rows)
    if missing:
        raise ValueError(f"Missing modes in {run_directory}: {sorted(missing)}")
    if not all(rows[value].get("complete") for value in expected):
        raise ValueError(f"Incomplete selected condition in {run_directory}")

    manifest = json.loads((run_directory / "run_manifest.json").read_text())
    settings = manifest["common_settings"]
    if settings.get("temperature") != 0.0:
        raise ValueError("Box-count experiment must use temperature zero.")
    conditions = {value["mode"]: value for value in manifest["conditions"]}
    for repeat in range(1, repeat_count + 1):
        for box_count in BOX_COUNTS:
            condition = conditions[mode(box_count, repeat)]
            actual = (
                condition["formulation"],
                condition["semantics"],
                condition["representation"],
                int(condition["box_count"]),
                condition["reasoning_effort"],
                int(condition["seed"]),
            )
            wanted = (
                "multi",
                "class_names",
                "numeric_prediction",
                box_count,
                "none",
                1234,
            )
            if actual != wanted:
                raise ValueError(f"Invalid condition: {mode(box_count, repeat)}")

    image_counts = {int(rows[value]["task_count"]) for value in expected}
    if len(image_counts) != 1:
        raise ValueError(f"Repeat image counts differ in {run_directory}")

    score_values: dict[int, dict[str, list[float]]] = {
        box_count: {
            metric: [
                float(rows[mode(box_count, repeat)][metric])
                for repeat in range(1, repeat_count + 1)
            ]
            for metric in METRICS
        }
        for box_count in BOX_COUNTS
    }
    counts = {
        str(box_count): {
            metric: metric_summary(score_values[box_count][metric])
            for metric in METRICS
        }
        for box_count in BOX_COUNTS
    }
    comparisons: dict[str, Any] = {}
    for left, right in ((2, 1), (5, 1), (5, 2)):
        key = f"b{left:02d}_minus_b{right:02d}"
        comparisons[key] = {}
        for metric in METRICS:
            deltas = [
                left_score - right_score
                for left_score, right_score in zip(
                    score_values[left][metric],
                    score_values[right][metric],
                    strict=True,
                )
            ]
            comparisons[key][metric] = {
                **metric_summary(deltas),
                "mean_ci95": confidence_interval(deltas),
            }

    return {
        "dataset": name,
        "run_directory": str(run_directory.resolve()),
        "test_images": image_counts.pop(),
        "repeat_count": repeat_count,
        "counts": counts,
        "comparisons": comparisons,
        "model_failures": sum(int(rows[value]["model_failures"]) for value in expected),
        "errors": sum(int(rows[value]["errors"]) for value in expected),
    }


def macro_summary(datasets: dict[str, dict[str, Any]], repeat_count: int) -> dict[str, Any]:
    result: dict[str, Any] = {"counts": {}, "comparisons": {}}
    for box_count in BOX_COUNTS:
        result["counts"][str(box_count)] = {}
        for metric in METRICS:
            values = [
                fmean(
                    dataset["counts"][str(box_count)][metric]["values"][repeat]
                    for dataset in datasets.values()
                )
                for repeat in range(repeat_count)
            ]
            result["counts"][str(box_count)][metric] = metric_summary(values)
    for left, right in ((2, 1), (5, 1), (5, 2)):
        key = f"b{left:02d}_minus_b{right:02d}"
        result["comparisons"][key] = {}
        for metric in METRICS:
            values = [
                result["counts"][str(left)][metric]["values"][repeat]
                - result["counts"][str(right)][metric]["values"][repeat]
                for repeat in range(repeat_count)
            ]
            result["comparisons"][key][metric] = {
                **metric_summary(values),
                "mean_ci95": confidence_interval(values),
            }
    return result


def write_csv(path: Path, datasets: dict[str, dict[str, Any]], macro: dict[str, Any]) -> None:
    rows = []
    for name, dataset in [*datasets.items(), ("macro", macro)]:
        for box_count in BOX_COUNTS:
            for metric in METRICS:
                summary = dataset["counts"][str(box_count)][metric]
                rows.append(
                    {
                        "dataset": name,
                        "metric": metric,
                        "boxes_per_class": box_count,
                        "mean": summary["mean"],
                        "noise_threshold": summary["tie_threshold"],
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=RUN_DIRECTORY")
    parser.add_argument("--repeat-count", type=int, choices=sorted(T_CRITICAL_95), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    datasets = {}
    for specification in args.run:
        if "=" not in specification:
            raise ValueError("--run must use NAME=RUN_DIRECTORY.")
        name, raw_path = specification.split("=", 1)
        if not name or name in datasets:
            raise ValueError(f"Invalid or duplicate dataset name: {name!r}")
        datasets[name] = analyze_dataset(name, Path(raw_path), args.repeat_count)
    macro = macro_summary(datasets, args.repeat_count)
    result = {
        "created_at": base.utc_now(),
        "method": (
            f"{args.repeat_count} paired full-test repeats of one, two, and five "
            "positive numeric-box examples per class."
        ),
        "datasets": datasets,
        "macro": macro,
    }
    base.atomic_write_json(args.output, result)
    write_csv(args.csv, datasets, macro)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
