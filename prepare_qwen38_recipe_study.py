#!/usr/bin/env python3
"""Build adaptive Qwen3.8 recipe-study configs from completed screen results."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import evaluate_qwen38_orion as base
from evaluate_qwen38_recipe import Condition


def read_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text())
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Missing rows in {path}")
    return [row for row in rows if row.get("complete") and row.get("mAP50_95") is not None]


def normalize_named(rows: list[dict[str, Any]], image_count: int) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        mode = str(row["mode"])
        formulation = str(row["formulation"])
        representation = str(row["representation"])
        if representation == "class_names":
            representation = "none"
        result.append(
            {
                **row,
                "source": "named_box_screen",
                "semantics": "class_names",
                "formulation": formulation,
                "representation": representation,
                "box_count": int(row["boxes_per_class"]),
                "image_count": image_count,
                "prompt_tokens_per_image": float(row.get("prompt_tokens") or 0) / image_count,
                "completion_tokens_per_image": float(row.get("completion_tokens") or 0) / image_count,
                "effective_serial_seconds_per_image": float(row.get("mean_inference_seconds") or 0)
                * int(row["calls_per_image"]),
                "legacy_mode": mode,
            }
        )
    return result


def normalize_anonymous_single(
    rows: list[dict[str, Any]], image_count: int
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        instruction = str(row["instruction"])
        result.append(
            {
                **row,
                "source": "anonymous_single_screen",
                "semantics": f"anonymous_{instruction}",
                "formulation": "single",
                "representation": str(row["representation"]),
                "box_count": int(row["boxes_per_class"]),
                "image_count": image_count,
                "prompt_tokens_per_image": float(row.get("prompt_tokens") or 0) / image_count,
                "completion_tokens_per_image": float(row.get("completion_tokens") or 0) / image_count,
                "effective_serial_seconds_per_image": float(row.get("mean_inference_seconds") or 0)
                * int(row["calls_per_image"]),
                "legacy_mode": str(row["mode"]),
            }
        )
    return result


def normalize_recipe(rows: list[dict[str, Any]], image_count: int, source: str) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        result.append(
            {
                **row,
                "source": source,
                "box_count": int(row["boxes_per_class"]),
                "image_count": image_count,
                "prompt_tokens_per_image": float(row.get("prompt_tokens") or 0) / image_count,
                "completion_tokens_per_image": float(row.get("completion_tokens") or 0) / image_count,
                "effective_serial_seconds_per_image": float(row.get("mean_inference_seconds") or 0)
                * int(row["calls_per_image"]),
                "legacy_mode": str(row["mode"]),
            }
        )
    return result


def efficient_best(rows: list[dict[str, Any]], margin: float = 1.0) -> dict[str, Any]:
    if not rows:
        raise ValueError("No completed candidates were supplied.")
    best_accuracy = max(float(row["mAP50_95"]) for row in rows)
    close = [row for row in rows if float(row["mAP50_95"]) >= best_accuracy - margin]
    return min(
        close,
        key=lambda row: (
            int(row["calls_per_image"]),
            float(row["prompt_tokens_per_image"]) + float(row["completion_tokens_per_image"]),
            float(row["effective_serial_seconds_per_image"]),
            -float(row["mAP50_95"]),
        ),
    )


def condition_from_row(row: dict[str, Any], mode: str, *, reasoning: str = "none", seed: int = 1234) -> Condition:
    return Condition(
        mode=mode,
        formulation=str(row["formulation"]),
        semantics=str(row["semantics"]),
        representation=str(row["representation"]),
        box_count=int(row["box_count"]),
        reasoning_effort=reasoning,
        seed=seed,
    )


def write_conditions(path: Path, conditions: list[Condition]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base.atomic_write_json(path, {"conditions": [asdict(value) for value in conditions]})


def stratified_image_ids(annotation_path: Path, count: int) -> list[int]:
    coco = base.load_coco(annotation_path)
    category_ids = [int(value["id"]) for value in coco["categories"]]
    by_image: dict[int, dict[int, int]] = {
        int(image["id"]): {category_id: 0 for category_id in category_ids}
        for image in coco["images"]
    }
    for annotation in coco["annotations"]:
        by_image[int(annotation["image_id"])][int(annotation["category_id"])] += 1
    selected: list[int] = []
    totals = {category_id: 0 for category_id in category_ids}
    candidates = set(by_image)
    while candidates and len(selected) < min(count, len(candidates) + len(selected)):
        def priority(image_id: int) -> tuple[Any, ...]:
            counts = by_image[image_id]
            new_categories = sum(counts[category_id] > 0 and totals[category_id] == 0 for category_id in category_ids)
            balance = sum(counts[category_id] / (1 + totals[category_id]) for category_id in category_ids)
            return (new_categories, balance, sum(counts.values()), -image_id)

        chosen = max(candidates, key=priority)
        selected.append(chosen)
        candidates.remove(chosen)
        for category_id in category_ids:
            totals[category_id] += by_image[chosen][category_id]
    if any(totals[category_id] == 0 for category_id in category_ids):
        raise ValueError(f"Selected {count} images do not cover every category.")
    return sorted(selected)


def prepare_screen(args: argparse.Namespace) -> None:
    named = normalize_named(read_rows(args.named_summary), args.image_count)
    anonymous_single = normalize_anonymous_single(
        read_rows(args.anonymous_single_summary), args.image_count
    )
    anonymous_multi = normalize_recipe(
        read_rows(args.anonymous_multi_summary), args.image_count, "anonymous_multi_screen"
    )
    rows = named + anonymous_single + anonymous_multi
    ranked = sorted(rows, key=lambda row: float(row["mAP50_95"]), reverse=True)
    best_overall = efficient_best(ranked)
    best_box = efficient_best([row for row in ranked if int(row["box_count"]) > 0])
    best_numeric = efficient_best([row for row in ranked if row["representation"] == "numeric"])
    best_drawn = efficient_best([row for row in ranked if row["representation"] == "drawn"])
    best_anonymous = efficient_best(
        [row for row in ranked if str(row["semantics"]).startswith("anonymous")]
    )
    fast = efficient_best([row for row in ranked if int(row["calls_per_image"]) == 1 and int(row["box_count"]) == 0])

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in ranked for key in row})
    with (output / "screen_ranking.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked)
    selection = {
        "created_at": base.utc_now(),
        "practical_equivalence_margin_mAP50_95": 1.0,
        "selection_rule": "Within one mAP of the maximum, prefer fewer calls, then fewer tokens and lower latency.",
        "best_overall": best_overall,
        "best_box": best_box,
        "best_numeric": best_numeric,
        "best_drawn": best_drawn,
        "best_anonymous": best_anonymous,
        "fast": fast,
    }
    base.atomic_write_json(output / "screen_selection.json", selection)

    representation = str(best_box["representation"])
    box_count = int(best_box["box_count"])
    conditions = [
        Condition("screen_fast_names_multi", "multi", "class_names", "none", 0),
        Condition("screen_names_single_boxes", "single", "class_names", representation, box_count),
        Condition("screen_names_multi_boxes", "multi", "class_names", representation, box_count),
        Condition("screen_anonymous_explicit_single", "single", "anonymous_explicit", representation, box_count),
        Condition("screen_anonymous_explicit_multi", "multi", "anonymous_explicit", representation, box_count),
        Condition("screen_anonymous_minimal_single", "single", "anonymous_minimal", representation, box_count),
        Condition("screen_anonymous_minimal_multi", "multi", "anonymous_minimal", representation, box_count),
        Condition("screen_self_name_single_boxes", "single", "self_name", representation, box_count),
        Condition("screen_self_name_multi_boxes", "multi", "self_name", representation, box_count),
        Condition("screen_self_name_only_single", "single", "self_name_only", "none", 0),
        Condition("screen_self_name_only_multi", "multi", "self_name_only", "none", 0),
    ]
    write_conditions(output / "self_name_screen_conditions.json", conditions)
    dreidel_ids = stratified_image_ids(args.dreidel_annotations, args.subset_size)
    orion_ids = stratified_image_ids(args.orion_annotations, args.subset_size)
    base.atomic_write_json(
        output / "subset_image_ids.json",
        {"dreidel": dreidel_ids, "orion": orion_ids, "selection_uses_predictions": False},
    )


def prepare_finalists(args: argparse.Namespace) -> None:
    study = args.output_dir
    screen_selection = json.loads((study / "screen_selection.json").read_text())
    self_rows = normalize_recipe(
        read_rows(args.self_name_screen_summary), args.subset_size, "self_name_subset_screen"
    )
    self_best = efficient_best(self_rows)
    best_self = efficient_best([row for row in self_rows if str(row["semantics"]).startswith("self_name")])

    legacy_rows = [
        screen_selection["fast"],
        screen_selection["best_numeric"],
        screen_selection["best_drawn"],
    ]
    anonymous_candidates = [screen_selection["best_anonymous"]]
    candidate_rows = legacy_rows + anonymous_candidates
    # Carry the best self-name arm if it is within one mAP of the best matched
    # subset control; otherwise it cannot be a practical winner.
    if float(best_self["mAP50_95"]) >= max(float(row["mAP50_95"]) for row in self_rows) - 1.0:
        candidate_rows.append(best_self)
    structural: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in candidate_rows:
        key = (
            row["formulation"],
            row["semantics"],
            row["representation"],
            int(row["box_count"]),
        )
        structural[key] = row
    finalists = [
        condition_from_row(row, f"final_{index:02d}")
        for index, row in enumerate(structural.values(), start=1)
    ]
    write_conditions(study / "finalist_conditions.json", finalists)

    top_row = self_best
    top_none = condition_from_row(top_row, "reasoning_top_none", reasoning="none")
    top_low = condition_from_row(top_row, "reasoning_top_low", reasoning="low")
    fast_row = screen_selection["fast"]
    fast_none = condition_from_row(fast_row, "reasoning_fast_none", reasoning="none")
    fast_low = condition_from_row(fast_row, "reasoning_fast_low", reasoning="low")
    write_conditions(
        study / "reasoning_gate_conditions.json",
        [top_none, top_low, fast_none, fast_low],
    )
    determinism: list[Condition] = []
    for label, row in (("top", top_row), ("fast", fast_row)):
        determinism.extend(
            [
                condition_from_row(row, f"det_{label}_seed1234_a", seed=1234),
                condition_from_row(row, f"det_{label}_seed1234_b", seed=1234),
                condition_from_row(row, f"det_{label}_seed4321", seed=4321),
            ]
        )
    write_conditions(study / "determinism_conditions.json", determinism)
    base.atomic_write_json(
        study / "finalist_selection.json",
        {
            "created_at": base.utc_now(),
            "self_name_subset_best": self_best,
            "best_self_name_arm": best_self,
            "finalists": [asdict(value) for value in finalists],
            "reasoning_gate": [asdict(value) for value in [top_none, top_low, fast_none, fast_low]],
            "determinism": [asdict(value) for value in determinism],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    screen = subparsers.add_parser("prepare-screen")
    screen.add_argument("--named-summary", type=Path, required=True)
    screen.add_argument("--anonymous-single-summary", type=Path, required=True)
    screen.add_argument("--anonymous-multi-summary", type=Path, required=True)
    screen.add_argument("--dreidel-annotations", type=Path, required=True)
    screen.add_argument("--orion-annotations", type=Path, required=True)
    screen.add_argument("--image-count", type=int, default=54)
    screen.add_argument("--subset-size", type=int, default=20)
    screen.add_argument("--output-dir", type=Path, required=True)
    finalist = subparsers.add_parser("prepare-finalists")
    finalist.add_argument("--self-name-screen-summary", type=Path, required=True)
    finalist.add_argument("--subset-size", type=int, default=20)
    finalist.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare-screen":
        prepare_screen(args)
    else:
        prepare_finalists(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
