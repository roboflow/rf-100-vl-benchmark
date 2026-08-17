#!/usr/bin/env python3
"""Validate the controlled largest-versus-median one-shot selector experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_orion as base
import evaluate_qwen38_recipe as recipe

DATASETS = ("paper-parts", "actions", "defect-detection")
MODE = "numeric_prediction_b01_multi_median_area"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _assets(
    train_directory: Path,
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]],
) -> dict[tuple[int, int], dict[str, Path]]:
    return {
        (category_id, reference.rank): {"source": train_directory / reference.file_name}
        for category_id, sequence in references.items()
        for reference in sequence
    }


def _relative_areas(train: dict[str, Any]) -> dict[int, list[tuple[float, int]]]:
    images = {int(image["id"]): image for image in train["images"]}
    result: dict[int, list[tuple[float, int]]] = {
        int(category["id"]): [] for category in train["categories"]
    }
    for annotation in train["annotations"]:
        image = images[int(annotation["image_id"])]
        area = (
            float(annotation["bbox"][2])
            * float(annotation["bbox"][3])
            / (int(image["width"]) * int(image["height"]))
        )
        result[int(annotation["category_id"])].append((area, int(annotation["id"])))
    return result


def validate(dataset_root: Path, run_root: Path, largest_run_root: Path) -> dict[str, Any]:
    condition = recipe.Condition(
        mode=MODE,
        formulation="multi",
        semantics="class_names",
        representation="numeric_prediction",
        box_count=1,
        reasoning_effort="none",
        seed=1234,
    )
    largest_condition = recipe.Condition(
        mode="numeric_prediction_b01_multi",
        formulation="multi",
        semantics="class_names",
        representation="numeric_prediction",
        box_count=1,
        reasoning_effort="none",
        seed=1234,
    )
    rows: list[dict[str, Any]] = []
    total_images = 0
    for dataset_name in DATASETS:
        dataset = dataset_root / dataset_name
        train_directory = dataset / "train"
        test_directory = dataset / "test"
        train_path = train_directory / "_annotations.coco.json"
        test_path = test_directory / "_annotations.coco.json"
        train = base.load_coco(train_path)
        test = base.load_coco(test_path)
        base.validate_split_isolation(train, test)
        categories = base.categories_by_id(test)
        if categories != base.categories_by_id(train):
            raise ValueError(f"Train/test categories differ for {dataset_name}.")

        manifest = _read(run_root / dataset_name / "run_manifest.json")
        old_manifest = _read(largest_run_root / dataset_name / "run_manifest.json")
        if manifest["train_annotation_sha256"] != old_manifest["train_annotation_sha256"]:
            raise ValueError(f"Train annotations differ for {dataset_name}.")
        if manifest["test_annotation_sha256"] != old_manifest["test_annotation_sha256"]:
            raise ValueError(f"Test annotations differ for {dataset_name}.")
        if manifest["reference_selection"].get("first_reference_strategy") != "median-relative-area":
            raise ValueError(f"Median selector is not frozen for {dataset_name}.")
        if manifest["common_settings"] != old_manifest["common_settings"]:
            raise ValueError(f"Inference settings differ from RF20 for {dataset_name}.")

        largest = box_ablation.select_reference_sequences(
            train,
            train_directory,
            required_count=1,
            distinct_images_only=False,
        )
        middle = box_ablation.select_reference_sequences(
            train,
            train_directory,
            required_count=1,
            distinct_images_only=False,
            first_strategy="median-relative-area",
        )
        old_classes = old_manifest["reference_selection"]["classes"]
        for category_id in categories:
            if largest[category_id][0].annotation_id != int(
                old_classes[str(category_id)][0]["annotation_id"]
            ):
                raise ValueError(f"Completed largest reference changed for {dataset_name}.")

        first_image = min(test["images"], key=lambda value: int(value["id"]))
        common_task = {
            "image_id": int(first_image["id"]),
            "file_name": str(first_image["file_name"]),
            "width": int(first_image["width"]),
            "height": int(first_image["height"]),
        }
        largest_message = recipe.build_messages(
            base.Task(mode=largest_condition.mode, **common_task),
            largest_condition,
            test_directory,
            categories,
            {},
            largest,
            _assets(train_directory, largest),
        )[0]["content"]
        middle_message = recipe.build_messages(
            base.Task(mode=condition.mode, **common_task),
            condition,
            test_directory,
            categories,
            {},
            middle,
            _assets(train_directory, middle),
        )[0]["content"]
        if largest_message[0] != middle_message[0]:
            raise ValueError(f"Prompt wording changed for {dataset_name}.")
        largest_markers = [
            part.get("text")
            for part in largest_message
            if part.get("type") == "text" and not str(part.get("text", "")).startswith("[{")
        ]
        middle_markers = [
            part.get("text")
            for part in middle_message
            if part.get("type") == "text" and not str(part.get("text", "")).startswith("[{")
        ]
        if largest_markers != middle_markers:
            raise ValueError(f"Prompt structure changed for {dataset_name}.")
        if largest_message[-1]["image_url"] != middle_message[-1]["image_url"]:
            raise ValueError(f"Target image changed for {dataset_name}.")

        areas = _relative_areas(train)
        largest_values = []
        middle_values = []
        changed = 0
        for category_id in categories:
            by_annotation = {annotation_id: area for area, annotation_id in areas[category_id]}
            largest_id = largest[category_id][0].annotation_id
            middle_id = middle[category_id][0].annotation_id
            largest_values.append(by_annotation[largest_id])
            middle_values.append(by_annotation[middle_id])
            changed += largest_id != middle_id
        if changed != len(categories):
            raise ValueError(f"Not every exemplar changed for {dataset_name}.")

        image_count = len(test["images"])
        total_images += image_count
        rows.append(
            {
                "dataset": dataset_name,
                "test_images": image_count,
                "classes": len(categories),
                "changed_exemplars": changed,
                "largest_reference_median_relative_area": median(largest_values),
                "median_reference_median_relative_area": median(middle_values),
            }
        )

    if total_images != 1097:
        raise ValueError(f"Expected 1,097 complete-test requests; found {total_images}.")
    return {
        "created_at": base.utc_now(),
        "experiment": "one-shot-largest-vs-median-relative-area-selector",
        "dataset_count": len(rows),
        "request_count": total_images,
        "largest_arm": "six completed observations per dataset",
        "new_arm": "one median-relative-area observation per dataset",
        "only_intended_change": "train exemplar selection",
        "prompt_wording_identical": True,
        "inference_settings_identical": True,
        "complete_test_splits": True,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--largest-run-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.dataset_root.resolve(),
        args.run_root.resolve(),
        args.largest_run_root.resolve(),
    )
    base.atomic_write_json(args.report.resolve(), result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
