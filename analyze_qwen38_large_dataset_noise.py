#!/usr/bin/env python3
"""Summarize paired class-name and one-box repeats on larger RF20 datasets."""

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


REPEATS = 5
T_CRITICAL_95 = 2.7764451051977987


def confidence_interval(values: list[float]) -> list[float]:
    if len(values) != REPEATS:
        raise ValueError(f"Expected {REPEATS} paired values; found {len(values)}.")
    margin = T_CRITICAL_95 * stdev(values) / math.sqrt(len(values))
    mean = fmean(values)
    return [mean - margin, mean + margin]


def expected_modes(prefix: str) -> list[str]:
    return [f"{prefix}_repeat_{index:02d}" for index in range(1, REPEATS + 1)]


def analyze_dataset(name: str, run_directory: Path) -> dict[str, Any]:
    if not (run_directory / "_SUCCESS.json").is_file():
        raise ValueError(f"Incomplete run: {run_directory}")
    comparison = json.loads((run_directory / "comparison_summary.json").read_text())
    rows = {str(row["mode"]): row for row in comparison["rows"]}
    names_modes = expected_modes("names")
    box_modes = expected_modes("box")
    if set(rows) != set(names_modes + box_modes):
        raise ValueError(f"Unexpected modes in {run_directory}: {sorted(rows)}")
    if not all(row.get("complete") for row in rows.values()):
        raise ValueError(f"Incomplete condition in {run_directory}")

    manifest = json.loads((run_directory / "run_manifest.json").read_text())
    settings = manifest["common_settings"]
    if settings.get("temperature") != 0.0:
        raise ValueError("Noise experiment must use temperature zero.")
    conditions = {value["mode"]: value for value in manifest["conditions"]}
    for mode in names_modes:
        condition = conditions[mode]
        if (condition["representation"], int(condition["box_count"])) != ("none", 0):
            raise ValueError(f"Invalid names-only condition: {mode}")
    for mode in box_modes:
        condition = conditions[mode]
        if (condition["representation"], int(condition["box_count"])) != (
            "numeric_prediction",
            1,
        ):
            raise ValueError(f"Invalid one-box condition: {mode}")

    def values(modes: list[str], metric: str) -> list[float]:
        return [float(rows[mode][metric]) for mode in modes]

    names_ap = values(names_modes, "mAP50_95")
    names_ap50 = values(names_modes, "mAP50")
    box_ap = values(box_modes, "mAP50_95")
    box_ap50 = values(box_modes, "mAP50")
    delta_ap = [box - names for box, names in zip(box_ap, names_ap, strict=True)]
    delta_ap50 = [box - names for box, names in zip(box_ap50, names_ap50, strict=True)]
    image_counts = {int(row["task_count"]) for row in rows.values()}
    if len(image_counts) != 1:
        raise ValueError(f"Repeat image counts differ in {run_directory}.")

    return {
        "dataset": name,
        "run_directory": str(run_directory.resolve()),
        "test_images": image_counts.pop(),
        "repeat_count": REPEATS,
        "temperature": settings["temperature"],
        "seed": 1234,
        "reasoning_effort": "none",
        "names_only": {
            "mAP50_95": metric_summary(names_ap),
            "mAP50": metric_summary(names_ap50),
        },
        "one_box": {
            "mAP50_95": metric_summary(box_ap),
            "mAP50": metric_summary(box_ap50),
        },
        "paired_uplift": {
            "mAP50_95": {
                **metric_summary(delta_ap),
                "mean_ci95": confidence_interval(delta_ap),
            },
            "mAP50": {
                **metric_summary(delta_ap50),
                "mean_ci95": confidence_interval(delta_ap50),
            },
        },
        "model_failures": sum(int(row["model_failures"]) for row in rows.values()),
        "errors": sum(int(row["errors"]) for row in rows.values()),
    }


def write_csv(path: Path, datasets: dict[str, dict[str, Any]]) -> None:
    rows = []
    for value in datasets.values():
        uplift_ap = value["paired_uplift"]["mAP50_95"]
        uplift_ap50 = value["paired_uplift"]["mAP50"]
        rows.append(
            {
                "dataset": value["dataset"],
                "test_images": value["test_images"],
                "names_mean_mAP50_95": value["names_only"]["mAP50_95"]["mean"],
                "names_noise_floor_mAP50_95": value["names_only"]["mAP50_95"]["tie_threshold"],
                "box_mean_mAP50_95": value["one_box"]["mAP50_95"]["mean"],
                "box_noise_floor_mAP50_95": value["one_box"]["mAP50_95"]["tie_threshold"],
                "mean_uplift_mAP50_95": uplift_ap["mean"],
                "uplift_ci95_low_mAP50_95": uplift_ap["mean_ci95"][0],
                "uplift_ci95_high_mAP50_95": uplift_ap["mean_ci95"][1],
                "mean_uplift_mAP50": uplift_ap50["mean"],
                "uplift_ci95_low_mAP50": uplift_ap50["mean_ci95"][0],
                "uplift_ci95_high_mAP50": uplift_ap50["mean_ci95"][1],
                "model_failures": value["model_failures"],
                "errors": value["errors"],
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
        datasets[name] = analyze_dataset(name, Path(raw_path))
    result = {
        "created_at": base.utc_now(),
        "method": (
            "Five paired full-test repeats of class-names-only and one positive "
            "numeric-box example per class. Conditions are interleaved by target "
            "image with temperature=0, seed=1234, and reasoning disabled."
        ),
        "datasets": datasets,
    }
    base.atomic_write_json(args.output, result)
    write_csv(args.csv, datasets)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
