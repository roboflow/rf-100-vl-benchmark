#!/usr/bin/env python3
"""Analyze RF20 instruction, visual-reference, and matched-control results."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Sequence

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import evaluate_qwen38_orion as base

RF20_MODES = {
    "names": ("three_way", "names_multi"),
    "instructions": ("instructions", "instructions_multi"),
    "one_reference": ("three_way", "numeric_prediction_b01_multi"),
    "ten_references": ("ten_reference", "numeric_prediction_all_available_multi_explicit_sparse"),
}
MATCHED_MODES = (
    "names_multi",
    "instructions_multi",
    "numeric_prediction_b01_multi",
    "instructions_numeric_prediction_b01_multi",
    "permuted_instructions_multi",
)
RATING_FIELDS = (
    "name_alone_insufficient",
    "requires_state_role_or_context",
    "requires_special_boundary_rule",
    "unusual_visual_domain",
)
METRICS = ("mAP50_95", "mAP50")
BOOTSTRAP_SEED = 1234
BOOTSTRAP_REPEATS = 10_000


def _read(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _summary_row(run_root: Path, dataset: str, mode: str) -> dict[str, Any]:
    rows = _read(run_root / dataset / "comparison_summary.json")["rows"]
    matches = [row for row in rows if row["mode"] == mode]
    if len(matches) != 1 or not matches[0].get("complete"):
        raise ValueError(f"Missing complete summary for {run_root}/{dataset}/{mode}.")
    return matches[0]


def _predictions(run_root: Path, dataset: str, mode: str) -> list[dict[str, Any]]:
    value = _read(run_root / dataset / "predictions" / f"{mode}.json")
    if not isinstance(value, list):
        raise TypeError(f"Predictions are not a list for {dataset}/{mode}.")
    return value


def _mean_valid(values: Any) -> float:
    valid = values[values > -1]
    return float(np.mean(valid)) if valid.size else -1.0


def score_category(
    annotation_path: Path,
    predictions: list[dict[str, Any]],
    category_id: int,
) -> dict[str, float]:
    """Match the shared maxDets=500 COCO calculation for one category."""

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        ground_truth = COCO(str(annotation_path))
        if predictions:
            detections = ground_truth.loadRes(predictions)
        else:
            detections = COCO()
            detections.dataset = {
                "images": ground_truth.dataset.get("images", []),
                "categories": ground_truth.dataset.get("categories", []),
                "annotations": [],
            }
            detections.createIndex()
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.maxDets = [1, 10, 500]
        evaluator.params.catIds = [category_id]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, 2]
    iou_50 = np.where(np.isclose(evaluator.params.iouThrs, 0.5))[0]
    return {
        "mAP50_95": 100 * _mean_valid(precision),
        "mAP50": 100 * _mean_valid(precision[iou_50]),
    }


def expand_ratings(ratings: dict[str, Any]) -> dict[tuple[str, str], dict[str, int]]:
    result = {}
    for dataset, values in ratings["datasets"].items():
        classes = values["classes"]
        for class_name in classes:
            row = {
                field: int(class_name in values[field])
                for field in RATING_FIELDS
            }
            row["challenge_count"] = sum(row.values())
            result[(dataset, class_name)] = row
    return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    statistic: Callable[[Sequence[dict[str, Any]]], float],
    *,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    datasets = sorted(grouped)
    observed = statistic(rows)
    rng = random.Random(seed)
    estimates = []
    for _ in range(repeats):
        sample = []
        for draw_index in range(len(datasets)):
            dataset = rng.choice(datasets)
            sample.extend({**row, "_draw": draw_index} for row in grouped[dataset])
        value = statistic(sample)
        if math.isfinite(value):
            estimates.append(value)
    if len(estimates) < repeats * 0.95:
        raise ValueError("Too many invalid clustered bootstrap draws.")
    return {
        "estimate": observed,
        "ci95_low": _percentile(estimates, 2.5),
        "ci95_high": _percentile(estimates, 97.5),
        "bootstrap_repeats": len(estimates),
    }


def _binary_difference(field: str, outcome: str) -> Callable[[Sequence[dict[str, Any]]], float]:
    def statistic(rows: Sequence[dict[str, Any]]) -> float:
        positive = [float(row[outcome]) for row in rows if int(row[field]) == 1]
        negative = [float(row[outcome]) for row in rows if int(row[field]) == 0]
        if not positive or not negative:
            return math.nan
        return fmean(positive) - fmean(negative)

    return statistic


def _linear_slope(field: str, outcome: str) -> Callable[[Sequence[dict[str, Any]]], float]:
    def statistic(rows: Sequence[dict[str, Any]]) -> float:
        x = np.asarray([float(row[field]) for row in rows])
        y = np.asarray([float(row[outcome]) for row in rows])
        denominator = float(np.sum((x - np.mean(x)) ** 2))
        if denominator == 0:
            return math.nan
        return float(np.sum((x - np.mean(x)) * (y - np.mean(y))) / denominator)

    return statistic


def _mode_sources(
    three_way_root: Path,
    instruction_root: Path,
    ten_reference_root: Path,
) -> dict[str, tuple[Path, str]]:
    roots = {
        "three_way": three_way_root,
        "instructions": instruction_root,
        "ten_reference": ten_reference_root,
    }
    return {
        name: (roots[root_name], mode)
        for name, (root_name, mode) in RF20_MODES.items()
    }


def analyze(
    dataset_root: Path,
    three_way_root: Path,
    instruction_root: Path,
    ten_reference_root: Path,
    matched_root: Path,
    ratings_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    for success in (
        three_way_root / "_RF20_SUCCESS.json",
        instruction_root / "_RF20_SUCCESS.json",
        ten_reference_root / "_RF20_SUCCESS.json",
        matched_root / "_COLLECTION_SUCCESS.json",
    ):
        _read(success)
    ratings_raw = _read(ratings_path)
    ratings = expand_ratings(ratings_raw)
    datasets = sorted(ratings_raw["datasets"])
    matched_summary = _read(matched_root / "collection_summary.json")
    matched_datasets = list(matched_summary["datasets"])
    mode_sources = _mode_sources(three_way_root, instruction_root, ten_reference_root)

    dataset_rows = []
    class_rows = []
    for dataset in datasets:
        annotation_path = dataset_root / dataset / "test/_annotations.coco.json"
        ground_truth = _read(annotation_path)
        categories = {
            int(category["id"]): str(category["name"])
            for category in ground_truth["categories"]
        }
        dataset_row: dict[str, Any] = {
            "dataset": dataset,
            "test_images": len(ground_truth["images"]),
            "classes": len(categories),
            "test_objects": len(ground_truth["annotations"]),
        }
        mode_predictions = {}
        for alias, (root, mode) in mode_sources.items():
            summary = _summary_row(root, dataset, mode)
            dataset_row[f"{alias}_mAP50_95"] = float(summary["mAP50_95"])
            dataset_row[f"{alias}_mAP50"] = float(summary["mAP50"])
            mode_predictions[alias] = _predictions(root, dataset, mode)
        for alias in ("instructions", "one_reference", "ten_references"):
            for metric in METRICS:
                dataset_row[f"{alias}_gain_{metric}"] = (
                    dataset_row[f"{alias}_{metric}"] - dataset_row[f"names_{metric}"]
                )
        dataset_rows.append(dataset_row)

        for category_id, class_name in categories.items():
            rating = ratings[(dataset, class_name)]
            class_row: dict[str, Any] = {
                "dataset": dataset,
                "category_id": category_id,
                "class_name": class_name,
                **rating,
            }
            for alias, predictions in mode_predictions.items():
                filtered = [
                    prediction
                    for prediction in predictions
                    if int(prediction["category_id"]) == category_id
                ]
                scores = score_category(annotation_path, filtered, category_id)
                for metric, value in scores.items():
                    class_row[f"{alias}_{metric}"] = value
            for alias in ("instructions", "one_reference", "ten_references"):
                for metric in METRICS:
                    class_row[f"{alias}_gain_{metric}"] = (
                        class_row[f"{alias}_{metric}"] - class_row[f"names_{metric}"]
                    )
            class_rows.append(class_row)

    matched_dataset_rows = []
    matched_class_rows = []
    for dataset in matched_datasets:
        annotation_path = dataset_root / dataset / "test/_annotations.coco.json"
        ground_truth = _read(annotation_path)
        categories = {
            int(category["id"]): str(category["name"])
            for category in ground_truth["categories"]
        }
        row: dict[str, Any] = {
            "dataset": dataset,
            "test_images": len(ground_truth["images"]),
            "classes": len(categories),
        }
        predictions_by_mode = {}
        for mode in MATCHED_MODES:
            summary = _summary_row(matched_root, dataset, mode)
            for metric in METRICS:
                row[f"{mode}_{metric}"] = float(summary[metric])
            predictions_by_mode[mode] = _predictions(matched_root, dataset, mode)
        for metric in METRICS:
            baseline = row[f"names_multi_{metric}"]
            correct = row[f"instructions_multi_{metric}"]
            visual = row[f"numeric_prediction_b01_multi_{metric}"]
            combined = row[f"instructions_numeric_prediction_b01_multi_{metric}"]
            permuted = row[f"permuted_instructions_multi_{metric}"]
            row[f"instruction_gain_{metric}"] = correct - baseline
            row[f"visual_gain_{metric}"] = visual - baseline
            row[f"combined_gain_{metric}"] = combined - baseline
            row[f"correct_vs_permuted_{metric}"] = correct - permuted
            row[f"instruction_visual_interaction_{metric}"] = combined - correct - visual + baseline
        matched_dataset_rows.append(row)

        for category_id, class_name in categories.items():
            class_row: dict[str, Any] = {
                "dataset": dataset,
                "category_id": category_id,
                "class_name": class_name,
                **ratings[(dataset, class_name)],
            }
            for mode, predictions in predictions_by_mode.items():
                filtered = [
                    prediction
                    for prediction in predictions
                    if int(prediction["category_id"]) == category_id
                ]
                scores = score_category(annotation_path, filtered, category_id)
                for metric, value in scores.items():
                    class_row[f"{mode}_{metric}"] = value
            for metric in METRICS:
                baseline = class_row[f"names_multi_{metric}"]
                correct = class_row[f"instructions_multi_{metric}"]
                visual = class_row[f"numeric_prediction_b01_multi_{metric}"]
                combined = class_row[f"instructions_numeric_prediction_b01_multi_{metric}"]
                permuted = class_row[f"permuted_instructions_multi_{metric}"]
                class_row[f"instruction_gain_{metric}"] = correct - baseline
                class_row[f"visual_gain_{metric}"] = visual - baseline
                class_row[f"combined_gain_{metric}"] = combined - baseline
                class_row[f"correct_vs_permuted_{metric}"] = correct - permuted
                class_row[f"instruction_visual_interaction_{metric}"] = (
                    combined - correct - visual + baseline
                )
            matched_class_rows.append(class_row)

    rating_effect_rows = []
    for outcome_prefix in ("instructions_gain", "one_reference_gain", "ten_references_gain"):
        for metric in METRICS:
            outcome = f"{outcome_prefix}_{metric}"
            for field in RATING_FIELDS:
                result = _cluster_bootstrap(class_rows, _binary_difference(field, outcome))
                rating_effect_rows.append(
                    {
                        "outcome": outcome,
                        "rating": field,
                        "interpretation": "mean gain for rated classes minus mean gain for unrated classes",
                        **result,
                    }
                )
            result = _cluster_bootstrap(class_rows, _linear_slope("challenge_count", outcome))
            rating_effect_rows.append(
                {
                    "outcome": outcome,
                    "rating": "challenge_count",
                    "interpretation": "linear gain per additional challenge flag",
                    **result,
                }
            )

    rf20_macro = []
    for alias in mode_sources:
        rf20_macro.append(
            {
                "mode": alias,
                "dataset_count": len(dataset_rows),
                "macro_mAP50_95": fmean(row[f"{alias}_mAP50_95"] for row in dataset_rows),
                "macro_mAP50": fmean(row[f"{alias}_mAP50"] for row in dataset_rows),
                "delta_vs_names_mAP50_95": fmean(
                    row[f"{alias}_mAP50_95"] - row["names_mAP50_95"] for row in dataset_rows
                ),
                "delta_vs_names_mAP50": fmean(
                    row[f"{alias}_mAP50"] - row["names_mAP50"] for row in dataset_rows
                ),
            }
        )
    matched_macro = []
    for mode in MATCHED_MODES:
        matched_macro.append(
            {
                "mode": mode,
                "dataset_count": len(matched_dataset_rows),
                "macro_mAP50_95": fmean(row[f"{mode}_mAP50_95"] for row in matched_dataset_rows),
                "macro_mAP50": fmean(row[f"{mode}_mAP50"] for row in matched_dataset_rows),
            }
        )
    matched_effects = {
        name: {
            metric: fmean(row[f"{name}_{metric}"] for row in matched_dataset_rows)
            for metric in METRICS
        }
        for name in (
            "instruction_gain",
            "visual_gain",
            "combined_gain",
            "correct_vs_permuted",
            "instruction_visual_interaction",
        )
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "rf20_per_dataset.csv", dataset_rows)
    _write_csv(output_directory / "rf20_per_class.csv", class_rows)
    _write_csv(output_directory / "matched_subset_per_dataset.csv", matched_dataset_rows)
    _write_csv(output_directory / "matched_subset_per_class.csv", matched_class_rows)
    _write_csv(output_directory / "rating_effects_cluster_bootstrap.csv", rating_effect_rows)
    _write_csv(output_directory / "rf20_macro.csv", rf20_macro)
    _write_csv(output_directory / "matched_subset_macro.csv", matched_macro)
    result = {
        "created_at": base.utc_now(),
        "ratings_version": ratings_raw["version"],
        "ratings_locked_at": ratings_raw["locked_at"],
        "bootstrap": {
            "unit": "dataset",
            "repeats": BOOTSTRAP_REPEATS,
            "seed": BOOTSTRAP_SEED,
        },
        "rf20_macro": rf20_macro,
        "matched_subset_macro": matched_macro,
        "matched_subset_effects": matched_effects,
        "rating_effects": rating_effect_rows,
        "provider_failures": {
            "instructions_rf20": sum(
                int(_summary_row(instruction_root, dataset, "instructions_multi").get("model_failures") or 0)
                for dataset in datasets
            ),
            "matched_subset": sum(
                int(_summary_row(matched_root, dataset, mode).get("model_failures") or 0)
                for dataset in matched_datasets
                for mode in MATCHED_MODES
            ),
        },
    }
    base.atomic_write_json(output_directory / "analysis.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--three-way-root", type=Path, required=True)
    parser.add_argument("--instruction-root", type=Path, required=True)
    parser.add_argument("--ten-reference-root", type=Path, required=True)
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.dataset_root.resolve(),
        args.three_way_root.resolve(),
        args.instruction_root.resolve(),
        args.ten_reference_root.resolve(),
        args.matched_root.resolve(),
        args.ratings.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
