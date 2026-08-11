#!/usr/bin/env python3
"""Combine disjoint exemplar-only Qwen runs into one verified comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import evaluate_qwen38_orion as base


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def combine(base_directory: Path, extension_directory: Path, output: Path) -> None:
    sources = (base_directory.resolve(), extension_directory.resolve())
    manifests = [load_json(source / "run_manifest.json") for source in sources]
    successes = [load_json(source / "_SUCCESS.json") for source in sources]
    aggregates = [load_json(source / "aggregate_metrics.json") for source in sources]
    comparisons = [load_json(source / "comparison_summary.json") for source in sources]

    stable_configuration_keys = (
        "dataset_directory",
        "train_annotation_sha256",
        "test_annotation_sha256",
        "settings",
        "requests_per_minute",
        "tokens_per_minute",
    )
    first = manifests[0]
    if any(
        manifest.get("prompt_version") != first.get("prompt_version")
        or manifest.get("class_names_exposed_to_model") is not False
        or manifest.get("minimal_mode_semantic_instruction") is not False
        or {
            key: manifest["configuration"].get(key)
            for key in stable_configuration_keys
        }
        != {
            key: first["configuration"].get(key)
            for key in stable_configuration_keys
        }
        for manifest in manifests
    ):
        raise ValueError("Source runs are not compatible exemplar-only evaluations.")

    conditions = [
        condition for manifest in manifests for condition in manifest["conditions"]
    ]
    modes = [str(condition["mode"]) for condition in conditions]
    if len(set(modes)) != len(modes):
        raise ValueError("Source runs contain overlapping conditions.")
    expected_counts = {1, 2, 5, 7, 10}
    if {int(condition["box_count"]) for condition in conditions} != expected_counts:
        raise ValueError("Combined runs do not cover box counts 1, 2, 5, 7, and 10.")
    expected_factorial = {
        (instruction, representation, count)
        for instruction in ("explicit", "minimal")
        for representation in ("numeric", "drawn")
        for count in expected_counts
    }
    actual_factorial = {
        (
            str(condition["instruction"]),
            str(condition["representation"]),
            int(condition["box_count"]),
        )
        for condition in conditions
    }
    if actual_factorial != expected_factorial:
        raise ValueError("Combined runs do not form the expected 20-condition factorial.")

    aggregate_modes = {
        mode: summary
        for aggregate in aggregates
        for mode, summary in aggregate["modes"].items()
    }
    if set(aggregate_modes) != set(modes) or not all(
        summary.get("complete") for summary in aggregate_modes.values()
    ):
        raise ValueError("Source aggregates are incomplete or inconsistent.")
    rows = [row for comparison in comparisons for row in comparison["rows"]]
    if {str(row["mode"]) for row in rows} != set(modes) or not all(
        row.get("complete") is True for row in rows
    ):
        raise ValueError("Source comparisons are incomplete or inconsistent.")
    rows.sort(
        key=lambda row: (
            str(row["instruction"]),
            str(row["representation"]),
            int(row["boxes_per_class"]),
        )
    )

    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "created_at": base.utc_now(),
        "prompt_version": first["prompt_version"],
        "sources": [str(source) for source in sources],
        "source_success_markers": successes,
        "condition_count": len(modes),
        "request_count": sum(int(marker["request_count"]) for marker in successes),
        "box_counts": sorted(expected_counts),
    }
    base.atomic_write_json(output / "combined_manifest.json", provenance)
    base.atomic_write_json(
        output / "aggregate_metrics.json",
        {
            "updated_at": base.utc_now(),
            "prompt_version": first["prompt_version"],
            "image_count": aggregates[0]["image_count"],
            "class_count": aggregates[0]["class_count"],
            "modes": aggregate_modes,
        },
    )
    base.atomic_write_json(
        output / "comparison_summary.json",
        {"updated_at": base.utc_now(), "rows": rows},
    )
    csv_path = output / "comparison_summary.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    base.atomic_write_json(output / "_SUCCESS.json", provenance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    combine(args.base_dir, args.extension_dir, args.output_dir)


if __name__ == "__main__":
    main()
