#!/usr/bin/env python3
"""Validate the exact Qwen3.8-Max RF20 three-way launch contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_orion as base
import evaluate_qwen38_recipe as recipe

EXPECTED_DATASETS = {
    "actions",
    "aerial-airport",
    "all-elements",
    "aquarium-combined",
    "defect-detection",
    "dentalai",
    "flir-camera-objects",
    "gwhd2021",
    "lacrosse-object-detection",
    "new-defects-in-wood",
    "orionproducts",
    "paper-parts",
    "recode-waste",
    "soda-bottles",
    "the-dreidel-project",
    "trail-camera",
    "water-meter",
    "wb-prova",
    "wildfire-smoke",
    "x-ray-id",
}
EXPECTED_CONDITIONS = (
    recipe.Condition("names_multi", "multi", "class_names", "none", 0),
    recipe.Condition(
        "numeric_prediction_b01_multi",
        "multi",
        "class_names",
        "numeric_prediction",
        1,
    ),
    recipe.Condition(
        "numeric_prediction_b02_multi",
        "multi",
        "class_names",
        "numeric_prediction",
        2,
    ),
)


def _assets(
    train_directory: Path,
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]],
) -> dict[tuple[int, int], dict[str, Path]]:
    return {
        (category_id, reference.rank): {"source": train_directory / reference.file_name}
        for category_id, sequence in references.items()
        for reference in sequence
    }


def _reference_annotations(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for part in content:
        if part.get("type") != "text" or not str(part.get("text", "")).startswith("[{"):
            continue
        value = json.loads(part["text"])
        if not isinstance(value, list) or len(value) != 1:
            raise ValueError("A reference annotation is not a one-object JSON list.")
        detection = value[0]
        if not isinstance(detection, dict) or list(detection) != ["bbox_2d", "label"]:
            raise ValueError(
                "A reference does not use the canonical bbox_2d/label schema."
            )
        result.append(detection)
    return result


def validate(dataset_root: Path, conditions_path: Path) -> dict[str, Any]:
    conditions = recipe.load_conditions(conditions_path)
    if conditions != EXPECTED_CONDITIONS:
        raise ValueError(
            "The RF20 launch conditions differ from the locked three-way contract."
        )
    datasets = {
        path.name: path
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "test/_annotations.coco.json").is_file()
    }
    if set(datasets) != EXPECTED_DATASETS:
        raise ValueError(
            "RF20 dataset inventory mismatch: "
            f"missing={sorted(EXPECTED_DATASETS - set(datasets))}, "
            f"extra={sorted(set(datasets) - EXPECTED_DATASETS)}"
        )

    rows = []
    total_images = total_classes = total_objects = total_requests = 0
    for name, dataset in sorted(datasets.items()):
        train_directory = dataset / "train"
        test_directory = dataset / "test"
        train = base.load_coco(train_directory / "_annotations.coco.json")
        test = base.load_coco(test_directory / "_annotations.coco.json")
        base.validate_split_isolation(train, test)
        categories = base.categories_by_id(test)
        if categories != base.categories_by_id(train):
            raise ValueError(f"Train/test categories differ for {name}.")
        for split, images in (
            (train_directory, train["images"]),
            (test_directory, test["images"]),
        ):
            for image in images:
                if not (split / str(image["file_name"])).is_file():
                    raise FileNotFoundError(split / str(image["file_name"]))

        references = box_ablation.select_reference_sequences(
            train, train_directory, required_count=2, distinct_images_only=True
        )
        if any(
            len({value.image_id for value in sequence}) != 2
            for sequence in references.values()
        ):
            raise ValueError(
                f"References are not two distinct train images per class for {name}."
            )
        assets = _assets(train_directory, references)
        first_image = min(test["images"], key=lambda value: int(value["id"]))
        for condition in conditions:
            task = base.Task(
                mode=condition.mode,
                image_id=int(first_image["id"]),
                file_name=str(first_image["file_name"]),
                width=int(first_image["width"]),
                height=int(first_image["height"]),
            )
            messages = recipe.build_messages(
                task,
                condition,
                test_directory,
                categories,
                {},
                references,
                assets,
            )
            content = messages[0]["content"]
            if content[-2] != {"type": "text", "text": "TARGET IMAGE:"}:
                raise ValueError(
                    f"Target marker is not penultimate for {name}/{condition.mode}."
                )
            if content[-1].get("type") != "image_url":
                raise ValueError(
                    f"Target image is not last for {name}/{condition.mode}."
                )
            labels = list(categories.values())
            if recipe._output_contract(labels) not in content[0]["text"]:
                raise ValueError(
                    f"Output contract mismatch for {name}/{condition.mode}."
                )
            annotations = _reference_annotations(content)
            expected = [
                recipe.detection_object(
                    reference.bbox_xyxy_1000, categories[category_id]
                )
                for category_id in categories
                for reference in references[category_id][: condition.box_count]
            ]
            if annotations != expected:
                raise ValueError(
                    f"Reference payload mismatch for {name}/{condition.mode}."
                )
            image_parts = sum(part.get("type") == "image_url" for part in content)
            if image_parts != 1 + condition.box_count * len(categories):
                raise ValueError(
                    f"Reference image count mismatch for {name}/{condition.mode}."
                )

        image_count = len(test["images"])
        class_count = len(categories)
        object_count = len(test["annotations"])
        request_count = image_count * len(conditions)
        total_images += image_count
        total_classes += class_count
        total_objects += object_count
        total_requests += request_count
        rows.append(
            {
                "dataset": name,
                "test_images": image_count,
                "classes": class_count,
                "test_objects": object_count,
                "requests": request_count,
                "matched_reference_schema": True,
            }
        )

    expected_totals = (3970, 110, 57285, 11910)
    totals = (total_images, total_classes, total_objects, total_requests)
    if totals != expected_totals:
        raise ValueError(
            f"RF20 totals changed: expected={expected_totals}, actual={totals}"
        )
    return {
        "created_at": base.utc_now(),
        "dataset_root": str(dataset_root),
        "conditions_path": str(conditions_path),
        "dataset_count": len(rows),
        "test_images": total_images,
        "classes": total_classes,
        "test_objects": total_objects,
        "requests": total_requests,
        "all_references_train_only": True,
        "all_reference_payloads_match_prediction_schema": True,
        "target_image_is_last": True,
        "datasets": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.dataset_root.resolve(), args.conditions.resolve())
    base.atomic_write_json(args.report.resolve(), result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
