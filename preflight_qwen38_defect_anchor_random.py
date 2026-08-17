#!/usr/bin/env python3
"""Validate the Defect Detection anchor-plus-random five-shot experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_orion as base
import evaluate_qwen38_recipe as recipe

MODE = "numeric_prediction_b05_multi_anchor_random_explicit_sparse"
SPARSE_TEXT = "The marked boxes are sparse positive exemplars."


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def validate(dataset: Path, run_root: Path, one_shot_run_root: Path) -> dict[str, Any]:
    train_directory = dataset / "train"
    test_directory = dataset / "test"
    train = base.load_coco(train_directory / "_annotations.coco.json")
    test = base.load_coco(test_directory / "_annotations.coco.json")
    base.validate_split_isolation(train, test)
    categories = base.categories_by_id(test)
    if categories != base.categories_by_id(train):
        raise ValueError("Train/test categories differ.")
    if len(test["images"]) != 188 or len(categories) != 4:
        raise ValueError("Defect Detection full-test inventory changed.")

    manifest = _read(run_root / "run_manifest.json")
    if manifest["reference_selection"]["method"] != (
        "largest-relative-area-then-seeded-random-object-order-v1"
    ):
        raise ValueError("The seeded-random reference method is not frozen.")
    if manifest["reference_selection"].get("random_seed") != 1234:
        raise ValueError("The reference seed is not 1234.")
    if manifest["selected_test_image_ids"] is not None:
        raise ValueError("The run is not configured for the complete test split.")
    settings = manifest["common_settings"]
    expected_settings = {
        "model": "qwen3.8-max",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "max_completion_tokens": 8192,
        "temperature": 0.0,
        "vl_high_resolution_images": False,
        "timeout_seconds": 180.0,
    }
    if settings != expected_settings:
        raise ValueError("Inference settings differ from the locked contract.")

    references = box_ablation.select_reference_sequences(
        train,
        train_directory,
        required_count=5,
        distinct_images_only=False,
        first_strategy="largest-then-seeded-random",
        random_seed=1234,
    )
    old_manifest = _read(one_shot_run_root / "run_manifest.json")
    old_classes = old_manifest["reference_selection"]["classes"]
    selected_ids: dict[str, list[int]] = {}
    for category_id, sequence in references.items():
        ids = [reference.annotation_id for reference in sequence]
        if len(ids) != 5 or len(set(ids)) != 5:
            raise ValueError(f"Class {category_id} does not have five unique objects.")
        if ids[0] != int(old_classes[str(category_id)][0]["annotation_id"]):
            raise ValueError(f"Class {category_id} changed the established one-shot anchor.")
        selected_ids[str(category_id)] = ids

    condition = recipe.Condition(
        mode=MODE,
        formulation="multi",
        semantics="class_names",
        representation="numeric_prediction",
        box_count=5,
        reasoning_effort="none",
        seed=1234,
        group_reference_instances_by_image=True,
        explicit_sparse_references=True,
    )
    assets = {
        (category_id, reference.rank): {"source": train_directory / reference.file_name}
        for category_id, sequence in references.items()
        for reference in sequence
    }
    image = min(test["images"], key=lambda value: int(value["id"]))
    task = base.Task(
        mode=MODE,
        image_id=int(image["id"]),
        file_name=str(image["file_name"]),
        width=int(image["width"]),
        height=int(image["height"]),
    )
    content = recipe.build_messages(
        task,
        condition,
        test_directory,
        categories,
        {},
        references,
        assets,
    )[0]["content"]
    if SPARSE_TEXT not in content[0]["text"]:
        raise ValueError("Explicit sparse-reference wording is absent.")
    if content[-2] != {"type": "text", "text": "TARGET IMAGE:"}:
        raise ValueError("Target marker is not penultimate.")
    if content[-1].get("type") != "image_url":
        raise ValueError("Target image is not last.")
    detections = []
    for part in content:
        text = str(part.get("text", ""))
        if part.get("type") == "text" and text.startswith("[{"):
            detections.extend(json.loads(text))
    if len(detections) != 20:
        raise ValueError(f"Expected 20 reference objects; found {len(detections)}.")
    if any(list(value) != ["bbox_2d", "label"] for value in detections):
        raise ValueError("Reference payload does not match prediction schema.")
    expected_images = 1 + sum(
        len({reference.image_id for reference in sequence})
        for sequence in references.values()
    )
    actual_images = sum(part.get("type") == "image_url" for part in content)
    if actual_images != expected_images:
        raise ValueError("Grouped reference image count is incorrect.")

    return {
        "created_at": base.utc_now(),
        "dataset": "defect-detection",
        "test_images": 188,
        "classes": 4,
        "requests": 188,
        "reference_objects_per_class": 5,
        "first_reference_matches_completed_one_shot": True,
        "remaining_references_seeded_uniformly_without_replacement": True,
        "reference_random_seed": 1234,
        "all_references_train_only": True,
        "complete_test_split": True,
        "explicit_sparse_reference_semantics": True,
        "prediction_shaped_reference_json": True,
        "target_image_last": True,
        "selected_annotation_ids": selected_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--one-shot-run-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.dataset.resolve(),
        args.run_root.resolve(),
        args.one_shot_run_root.resolve(),
    )
    base.atomic_write_json(args.report.resolve(), result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
