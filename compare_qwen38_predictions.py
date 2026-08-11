#!/usr/bin/env python3
"""Compare two COCO prediction sets with a paired image bootstrap."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path
from statistics import fmean
from typing import Any

import evaluate_qwen38_orion as base


def remap_sample(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    sampled_image_ids: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    images = {int(image["id"]): image for image in ground_truth["images"]}
    annotations: dict[int, list[dict[str, Any]]] = {}
    detections: dict[int, list[dict[str, Any]]] = {}
    for annotation in ground_truth["annotations"]:
        annotations.setdefault(int(annotation["image_id"]), []).append(annotation)
    for prediction in predictions:
        detections.setdefault(int(prediction["image_id"]), []).append(prediction)
    remapped_images: list[dict[str, Any]] = []
    remapped_annotations: list[dict[str, Any]] = []
    remapped_predictions: list[dict[str, Any]] = []
    annotation_id = 1
    for new_image_id, original_image_id in enumerate(sampled_image_ids, start=1):
        remapped_images.append({**images[original_image_id], "id": new_image_id})
        for annotation in annotations.get(original_image_id, []):
            remapped_annotations.append(
                {**annotation, "id": annotation_id, "image_id": new_image_id}
            )
            annotation_id += 1
        for prediction in detections.get(original_image_id, []):
            remapped_predictions.append({**prediction, "image_id": new_image_id})
    return (
        {
            **ground_truth,
            "images": remapped_images,
            "annotations": remapped_annotations,
        },
        remapped_predictions,
    )


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap(
    annotation_path: Path,
    baseline_predictions: list[dict[str, Any]],
    candidate_predictions: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    ground_truth = base.load_coco(annotation_path)
    image_ids = [int(image["id"]) for image in ground_truth["images"]]
    rng = random.Random(seed)
    differences: dict[str, list[float]] = {"AP": [], "AP50": []}
    with tempfile.TemporaryDirectory(prefix="qwen-bootstrap-") as temporary:
        temporary_path = Path(temporary) / "ground_truth.json"
        for _ in range(iterations):
            sampled = rng.choices(image_ids, k=len(image_ids))
            remapped_ground_truth, remapped_baseline = remap_sample(
                ground_truth, baseline_predictions, sampled
            )
            _, remapped_candidate = remap_sample(
                ground_truth, candidate_predictions, sampled
            )
            base.atomic_write_json(temporary_path, remapped_ground_truth)
            baseline_metrics = base.score_coco(temporary_path, remapped_baseline)
            candidate_metrics = base.score_coco(temporary_path, remapped_candidate)
            for metric in differences:
                differences[metric].append(
                    (candidate_metrics[metric] - baseline_metrics[metric]) * 100
                )
    return {
        metric: {
            "mean_difference_points": fmean(values),
            "ci95_difference_points": [
                percentile(values, 0.025),
                percentile(values, 0.975),
            ],
            "probability_candidate_better": sum(value > 0 for value in values)
            / len(values),
        }
        for metric, values in differences.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--practical-equivalence-map", type=float, default=1.0)
    args = parser.parse_args()
    if args.iterations < 20:
        raise ValueError("At least 20 bootstrap iterations are required.")
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    if not isinstance(baseline, list) or not isinstance(candidate, list):
        raise ValueError("Prediction files must contain COCO detection lists.")
    baseline_point = base.score_coco(args.annotations, baseline)
    candidate_point = base.score_coco(args.annotations, candidate)
    bootstrap = paired_bootstrap(
        args.annotations,
        baseline,
        candidate,
        iterations=args.iterations,
        seed=args.seed,
    )
    primary_difference = (candidate_point["AP"] - baseline_point["AP"]) * 100
    primary_ci = bootstrap["AP"]["ci95_difference_points"]
    statistically_resolved = primary_ci[0] > 0 or primary_ci[1] < 0
    practically_equivalent = abs(primary_difference) <= args.practical_equivalence_map
    # Statistical non-resolution is not evidence of practical equivalence.
    # Only the predeclared effect-size margin can pass the calibration gate.
    calibration_pass = practically_equivalent
    result = {
        "baseline": {
            "predictions": str(args.baseline.resolve()),
            "mAP50_95": baseline_point["AP"] * 100,
            "mAP50": baseline_point["AP50"] * 100,
        },
        "candidate": {
            "predictions": str(args.candidate.resolve()),
            "mAP50_95": candidate_point["AP"] * 100,
            "mAP50": candidate_point["AP50"] * 100,
        },
        "point_difference_candidate_minus_baseline": {
            "mAP50_95": primary_difference,
            "mAP50": (candidate_point["AP50"] - baseline_point["AP50"]) * 100,
        },
        "paired_image_bootstrap": {
            "iterations": args.iterations,
            "seed": args.seed,
            **bootstrap,
        },
        "decision": {
            "practical_equivalence_margin_mAP50_95": args.practical_equivalence_map,
            "primary_ci_excludes_zero": statistically_resolved,
            "calibration_pass": calibration_pass,
            "meaning": (
                "Skip temperature reruns of other screened recipes under the explicit "
                "cost-saving assumption."
                if calibration_pass
                else "Temperature calibration was not stable enough to extrapolate."
            ),
        },
        "created_at": base.utc_now(),
    }
    base.atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
