#!/usr/bin/env python3
"""Validate the locked Qwen3.8-Max RF20-VL-FSOD all-available contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
EXPECTED_CONDITION = recipe.Condition(
    mode="numeric_prediction_all_available_multi_explicit_sparse",
    formulation="multi",
    semantics="class_names",
    representation="numeric_prediction",
    box_count=10,
    reasoning_effort="none",
    seed=1234,
    group_reference_instances_by_image=True,
    explicit_sparse_references=True,
    all_available_references=True,
)
SPARSE_CLAUSE = (
    "The marked boxes are sparse positive exemplars. Treat all unmarked objects "
    "and regions in reference images as unlabeled, not as negative examples or "
    "exhaustive annotations."
)


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


def _reference_annotations(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for part in content:
        text = str(part.get("text", ""))
        if part.get("type") != "text" or not text.startswith("[{"):
            continue
        values = json.loads(text)
        if not isinstance(values, list) or not values:
            raise ValueError("A reference annotation is not a nonempty JSON list.")
        for detection in values:
            if not isinstance(detection, dict) or list(detection) != ["bbox_2d", "label"]:
                raise ValueError("A reference does not use bbox_2d/label prediction schema.")
            bbox = detection["bbox_2d"]
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(not isinstance(value, int) for value in bbox)
                or not (0 <= bbox[0] < bbox[2] <= 1000)
                or not (0 <= bbox[1] < bbox[3] <= 1000)
            ):
                raise ValueError(f"Invalid normalized XYXY reference box: {bbox}")
            result.append(detection)
    return result


def validate(dataset_root: Path, conditions_path: Path) -> dict[str, Any]:
    conditions = recipe.load_conditions(conditions_path)
    if conditions != (EXPECTED_CONDITION,):
        raise ValueError("Conditions differ from the locked all-available FSOD contract.")
    datasets = {
        path.name: path
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "test/_annotations.coco.json").is_file()
    }
    if set(datasets) != EXPECTED_DATASETS:
        raise ValueError(
            "RF20-VL-FSOD inventory mismatch: "
            f"missing={sorted(EXPECTED_DATASETS - set(datasets))}, "
            f"extra={sorted(set(datasets) - EXPECTED_DATASETS)}"
        )

    rows: list[dict[str, Any]] = []
    total_images = total_classes = total_train_objects = total_test_objects = 0
    total_reference_transmissions = 0
    max_reference_images = 0
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
                path = split / str(image["file_name"])
                if not path.is_file():
                    raise FileNotFoundError(path)

        train_ids: dict[int, set[int]] = {category_id: set() for category_id in categories}
        for annotation in train["annotations"]:
            train_ids[int(annotation["category_id"])].add(int(annotation["id"]))
        references = box_ablation.select_reference_sequences(
            train,
            train_directory,
            required_count=10,
            distinct_images_only=False,
            allow_fewer=True,
        )
        for category_id, sequence in references.items():
            selected = {reference.annotation_id for reference in sequence}
            if selected != train_ids[category_id]:
                raise ValueError(f"{name}/{category_id} omitted or repeated a train object.")

        condition = conditions[0]
        first_image = min(test["images"], key=lambda value: int(value["id"]))
        task = base.Task(
            mode=condition.mode,
            image_id=int(first_image["id"]),
            file_name=str(first_image["file_name"]),
            width=int(first_image["width"]),
            height=int(first_image["height"]),
        )
        content = recipe.build_messages(
            task,
            condition,
            test_directory,
            categories,
            {},
            references,
            _assets(train_directory, references),
        )[0]["content"]
        prompt = content[0]["text"]
        if "Use all positive reference boxes supplied for each label." not in prompt:
            raise ValueError(f"All-available wording is absent for {name}.")
        if SPARSE_CLAUSE not in prompt:
            raise ValueError(f"Sparse-reference wording is absent for {name}.")
        if recipe._output_contract(list(categories.values())) not in prompt:
            raise ValueError(f"Output contract mismatch for {name}.")
        if content[-2] != {"type": "text", "text": "TARGET IMAGE:"}:
            raise ValueError(f"Target marker is not penultimate for {name}.")
        if content[-1].get("type") != "image_url":
            raise ValueError(f"Target image is not last for {name}.")

        actual = _reference_annotations(content)
        expected = [
            recipe.detection_object(reference.bbox_xyxy_1000, categories[category_id])
            for category_id in categories
            for reference in references[category_id]
        ]
        signature = lambda value: (value["label"], tuple(value["bbox_2d"]))
        if Counter(map(signature, actual)) != Counter(map(signature, expected)):
            raise ValueError(f"Reference payload mismatch for {name}.")
        reference_images = sum(
            len({reference.image_id for reference in sequence})
            for sequence in references.values()
        )
        if sum(part.get("type") == "image_url" for part in content) != reference_images + 1:
            raise ValueError(f"Grouped reference-image count mismatch for {name}.")

        image_count = len(test["images"])
        train_object_count = len(train["annotations"])
        total_images += image_count
        total_classes += len(categories)
        total_train_objects += train_object_count
        total_test_objects += len(test["annotations"])
        total_reference_transmissions += image_count * train_object_count
        max_reference_images = max(max_reference_images, reference_images)
        rows.append(
            {
                "dataset": name,
                "test_images": image_count,
                "classes": len(categories),
                "available_train_references": train_object_count,
                "reference_images_per_request": reference_images,
                "requests": image_count,
            }
        )

    expected_totals = (3970, 110, 1099, 57285, 245070)
    actual_totals = (
        total_images,
        total_classes,
        total_train_objects,
        total_test_objects,
        total_reference_transmissions,
    )
    if actual_totals != expected_totals:
        raise ValueError(f"RF20 totals changed: expected={expected_totals}, actual={actual_totals}")
    return {
        "created_at": base.utc_now(),
        "benchmark": "RF20-VL-FSOD",
        "dataset_root": str(dataset_root),
        "conditions_path": str(conditions_path),
        "dataset_count": len(rows),
        "test_images": total_images,
        "classes": total_classes,
        "available_train_references": total_train_objects,
        "test_objects": total_test_objects,
        "requests": total_images,
        "reference_object_transmissions": total_reference_transmissions,
        "max_reference_images_per_request": max_reference_images,
        "all_official_train_annotations_included_once_per_request": True,
        "all_references_train_only": True,
        "all_reference_payloads_match_prediction_schema": True,
        "explicit_sparse_reference_semantics": True,
        "target_image_is_last": True,
        "reasoning_disabled": True,
        "seed": 1234,
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
