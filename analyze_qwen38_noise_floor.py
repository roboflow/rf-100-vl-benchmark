#!/usr/bin/env python3
"""Measure residual score variance across identical deterministic Qwen runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import evaluate_qwen38_orion as base


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make semantically identical detection lists insensitive to output order."""
    return sorted(
        predictions,
        key=lambda value: (
            int(value["category_id"]),
            *(float(coordinate) for coordinate in value["bbox"]),
            float(value.get("score", 1.0)),
        ),
    )


def xywh_iou(left: list[float], right: list[float]) -> float:
    left_x1, left_y1, left_width, left_height = map(float, left)
    right_x1, right_y1, right_width, right_height = map(float, right)
    left_x2, left_y2 = left_x1 + left_width, left_y1 + left_height
    right_x2, right_y2 = right_x1 + right_width, right_y1 + right_height
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    union = left_width * left_height + right_width * right_height - intersection
    return intersection / union if union > 0 else 0.0


def pairwise_detection_agreement(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compare two unordered detection sets using same-class greedy IoU matching."""
    candidates = []
    for left_index, left_detection in enumerate(left):
        for right_index, right_detection in enumerate(right):
            if int(left_detection["category_id"]) != int(right_detection["category_id"]):
                continue
            iou = xywh_iou(left_detection["bbox"], right_detection["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, left_index, right_index))
    matched_left: set[int] = set()
    matched_right: set[int] = set()
    matched_ious = []
    for iou, left_index, right_index in sorted(candidates, reverse=True):
        if left_index in matched_left or right_index in matched_right:
            continue
        matched_left.add(left_index)
        matched_right.add(right_index)
        matched_ious.append(iou)
    match_count = len(matched_ious)
    denominator = len(left) + len(right)
    f1 = 1.0 if denominator == 0 else 2.0 * match_count / denominator
    return {
        "left_count": len(left),
        "right_count": len(right),
        "matched_count": match_count,
        "f1": f1,
        "matched_ious": matched_ious,
    }


def category_count_signature(predictions: list[dict[str, Any]]) -> tuple[tuple[int, int], ...]:
    counts: dict[int, int] = {}
    for prediction in predictions:
        category_id = int(prediction["category_id"])
        counts[category_id] = counts.get(category_id, 0) + 1
    return tuple(sorted(counts.items()))


def metric_summary(values: list[float]) -> dict[str, Any]:
    if len(values) < 3:
        raise ValueError("At least three identical repeats are required.")
    sample_sd = stdev(values)
    observed_range = max(values) - min(values)
    pairwise = [abs(left - right) for left, right in combinations(values, 2)]
    repeatability_limit = 1.96 * math.sqrt(2) * sample_sd
    # With a finite repeat set, never claim a floor below a difference actually
    # observed. The normal-theory repeatability limit protects beyond that
    # finite set when variance is nonzero.
    tie_threshold = max(observed_range, repeatability_limit)
    return {
        "values": values,
        "mean": fmean(values),
        "sample_sd": sample_sd,
        "observed_range": observed_range,
        "max_observed_pairwise_difference": max(pairwise),
        "repeatability_limit_95": repeatability_limit,
        "tie_threshold": tie_threshold,
    }


def records_by_image(run_directory: Path, mode: str) -> dict[int, dict[str, Any]]:
    result = {}
    for path in (run_directory / "records" / mode).glob("*.json"):
        record = json.loads(path.read_text())
        image_id = int(record["task"]["image_id"])
        result[image_id] = record
    return result


def analyze_run(name: str, run_directory: Path) -> dict[str, Any]:
    comparison = json.loads((run_directory / "comparison_summary.json").read_text())
    rows = sorted(comparison["rows"], key=lambda value: value["mode"])
    if len(rows) != 10 or not all(row.get("complete") for row in rows):
        raise ValueError(f"Noise-floor run is incomplete: {run_directory}")
    structural = {
        (
            row["formulation"],
            row["semantics"],
            row["representation"],
            int(row["boxes_per_class"]),
            row["reasoning_effort"],
            int(row["seed"]),
        )
        for row in rows
    }
    if len(structural) != 1:
        raise ValueError("Noise-floor repeats are not structurally identical.")
    manifest = json.loads((run_directory / "run_manifest.json").read_text())
    if manifest["common_settings"].get("temperature") != 0.0:
        raise ValueError("Noise-floor run must explicitly use temperature zero.")

    modes = [str(row["mode"]) for row in rows]
    records = {mode: records_by_image(run_directory, mode) for mode in modes}
    image_sets = {tuple(sorted(values)) for values in records.values()}
    if len(image_sets) != 1:
        raise ValueError("Noise-floor repeats cover different image sets.")
    image_ids = list(next(iter(image_sets)))
    raw_all_equal = 0
    predictions_all_equal = 0
    detection_counts_all_equal = 0
    category_counts_all_equal = 0
    pairwise_f1 = []
    matched_ious = []
    for image_id in image_ids:
        raw_hashes = {
            canonical_sha(records[mode][image_id].get("raw_response")) for mode in modes
        }
        predictions = {
            mode: records[mode][image_id].get("predictions", []) for mode in modes
        }
        prediction_hashes = {
            canonical_sha(canonical_predictions(predictions[mode])) for mode in modes
        }
        raw_all_equal += len(raw_hashes) == 1
        predictions_all_equal += len(prediction_hashes) == 1
        detection_counts_all_equal += len({len(value) for value in predictions.values()}) == 1
        category_counts_all_equal += len(
            {category_count_signature(value) for value in predictions.values()}
        ) == 1
        for left_mode, right_mode in combinations(modes, 2):
            agreement = pairwise_detection_agreement(
                predictions[left_mode], predictions[right_mode]
            )
            pairwise_f1.append(float(agreement["f1"]))
            matched_ious.extend(float(value) for value in agreement["matched_ious"])

    metrics = {
        "AP": metric_summary([float(row["mAP50_95"]) for row in rows]),
        "AP50": metric_summary([float(row["mAP50"]) for row in rows]),
    }
    return {
        "name": name,
        "run_directory": str(run_directory.resolve()),
        "repeat_count": len(rows),
        "image_count": len(image_ids),
        "concurrency": manifest.get("concurrency"),
        "thinking_controls": manifest.get("thinking_controls"),
        "temperature": 0.0,
        "seed": rows[0]["seed"],
        "reasoning_effort": rows[0]["reasoning_effort"],
        "metrics": metrics,
        "response_repeatability": {
            "images_with_identical_raw_response_across_all_repeats": raw_all_equal,
            "images_with_identical_predictions_across_all_repeats": predictions_all_equal,
            "images_with_identical_detection_count_across_all_repeats": detection_counts_all_equal,
            "images_with_identical_per_class_counts_across_all_repeats": category_counts_all_equal,
            "total_images": len(image_ids),
            "raw_identity_rate": raw_all_equal / len(image_ids),
            "prediction_identity_rate": predictions_all_equal / len(image_ids),
            "detection_count_identity_rate": detection_counts_all_equal / len(image_ids),
            "per_class_count_identity_rate": category_counts_all_equal / len(image_ids),
            "pairwise_image_repeat_comparisons": len(pairwise_f1),
            "mean_pairwise_detection_f1_at_iou_50": fmean(pairwise_f1),
            "mean_iou_of_same_class_matches_at_iou_50": (
                fmean(matched_ious) if matched_ious else None
            ),
            "same_class_matches_at_iou_50": len(matched_ious),
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Dataset name and run directory as NAME=PATH; repeat for each dataset.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = {}
    for specification in args.run:
        if "=" not in specification:
            raise ValueError("--run must use NAME=PATH.")
        name, raw_path = specification.split("=", 1)
        if not name or name in datasets:
            raise ValueError(f"Invalid or duplicate dataset name: {name!r}")
        datasets[name] = analyze_run(name, Path(raw_path))
    result = {
        "created_at": base.utc_now(),
        "method": (
            "Ten identical full-test repeats at temperature=0, seed=1234, and "
            "reasoning=none. The per-dataset tie threshold is max(observed score "
            "range, 1.96*sqrt(2)*sample SD)."
        ),
        "scope": (
            "Residual API/inference stochasticity on the fixed test set. This is "
            "separate from finite-dataset sampling/generalization uncertainty."
        ),
        "datasets": datasets,
    }
    base.atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
