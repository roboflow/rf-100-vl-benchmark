#!/usr/bin/env python3
"""Validate the locked Qwen3.8-Max RF20 instruction study before API calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_orion as base
import evaluate_qwen38_recipe as recipe
from preflight_qwen38_rf20 import EXPECTED_DATASETS

SUBSET_DATASETS = (
    "actions",
    "all-elements",
    "defect-detection",
    "flir-camera-objects",
    "paper-parts",
    "water-meter",
)
RATING_FIELDS = (
    "name_alone_insufficient",
    "requires_state_role_or_context",
    "requires_special_boundary_rule",
    "unusual_visual_domain",
)
FULL_CONDITIONS = (
    recipe.Condition(
        "instructions_multi",
        "multi",
        "class_names",
        "none",
        0,
        instruction_mode="correct",
    ),
)
SUBSET_CONDITIONS = (
    recipe.Condition("names_multi", "multi", "class_names", "none", 0),
    recipe.Condition(
        "instructions_multi",
        "multi",
        "class_names",
        "none",
        0,
        instruction_mode="correct",
    ),
    recipe.Condition(
        "numeric_prediction_b01_multi",
        "multi",
        "class_names",
        "numeric_prediction",
        1,
        explicit_sparse_references=True,
    ),
    recipe.Condition(
        "instructions_numeric_prediction_b01_multi",
        "multi",
        "class_names",
        "numeric_prediction",
        1,
        explicit_sparse_references=True,
        instruction_mode="correct",
    ),
    recipe.Condition(
        "permuted_instructions_multi",
        "multi",
        "class_names",
        "none",
        0,
        instruction_mode="permuted",
    ),
)


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _assets(
    train_directory: Path,
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]],
) -> dict[tuple[int, int], dict[str, Path]]:
    return {
        (category_id, reference.rank): {"source": train_directory / reference.file_name}
        for category_id, sequence in references.items()
        for reference in sequence
    }


def validate(
    dataset_root: Path,
    full_conditions_path: Path,
    subset_conditions_path: Path,
    ratings_path: Path,
) -> dict[str, Any]:
    full_conditions = recipe.load_conditions(full_conditions_path)
    subset_conditions = recipe.load_conditions(subset_conditions_path)
    if full_conditions != FULL_CONDITIONS:
        raise ValueError("Full RF20 instruction condition differs from the locked contract.")
    if subset_conditions != SUBSET_CONDITIONS:
        raise ValueError("Six-dataset conditions differ from the locked five-arm contract.")

    datasets = {
        path.name: path
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "test/_annotations.coco.json").is_file()
    }
    if set(datasets) != EXPECTED_DATASETS:
        raise ValueError(
            "RF20 inventory mismatch: "
            f"missing={sorted(EXPECTED_DATASETS - set(datasets))}, "
            f"extra={sorted(set(datasets) - EXPECTED_DATASETS)}"
        )

    ratings = _read(ratings_path)
    if ratings.get("version") != "rf20-label-sufficiency-v1":
        raise ValueError("Unexpected label-rating version.")
    if ratings.get("score_blind") is not True:
        raise ValueError("Label ratings were not declared score blind.")
    rating_datasets = ratings.get("datasets") or {}
    if set(rating_datasets) != EXPECTED_DATASETS:
        raise ValueError("Label-rating dataset inventory does not match RF20.")

    rows: list[dict[str, Any]] = []
    total_images = total_classes = total_objects = 0
    for name, dataset in sorted(datasets.items()):
        train_directory = dataset / "train"
        test_directory = dataset / "test"
        train_path = train_directory / "_annotations.coco.json"
        test_path = test_directory / "_annotations.coco.json"
        readme_path = dataset / "README.dataset.txt"
        train = base.load_coco(train_path)
        test = base.load_coco(test_path)
        base.validate_split_isolation(train, test)
        categories = base.categories_by_id(test)
        if categories != base.categories_by_id(train):
            raise ValueError(f"Train/test category mismatch for {name}.")
        class_names = list(categories.values())
        rating = rating_datasets[name]
        if rating.get("classes") != class_names:
            raise ValueError(f"Locked class inventory mismatch for {name}.")
        for field in RATING_FIELDS:
            values = rating.get(field)
            if not isinstance(values, list) or len(values) != len(set(values)):
                raise ValueError(f"Invalid {field} ratings for {name}.")
            if not set(values) <= set(class_names):
                raise ValueError(f"Unknown class in {field} ratings for {name}.")
        if not readme_path.is_file() or not readme_path.read_text(encoding="utf-8").strip():
            raise FileNotFoundError(readme_path)
        for split, coco in ((train_directory, train), (test_directory, test)):
            for image in coco["images"]:
                if not (split / str(image["file_name"])).is_file():
                    raise FileNotFoundError(split / str(image["file_name"]))

        if name in SUBSET_DATASETS:
            readme = readme_path.read_text(encoding="utf-8").strip()
            permuted = recipe.permute_class_instruction_sections(readme)
            if sorted(permuted.split()) != sorted(readme.split()):
                raise AssertionError(f"Permuted control changed tokens for {name}.")
            references = box_ablation.select_reference_sequences(
                train,
                train_directory,
                required_count=1,
                distinct_images_only=True,
            )
            first_image = min(test["images"], key=lambda value: int(value["id"]))
            task_template = {
                "image_id": int(first_image["id"]),
                "file_name": str(first_image["file_name"]),
                "width": int(first_image["width"]),
                "height": int(first_image["height"]),
            }
            assets = _assets(train_directory, references)
            prompts = {}
            for condition in subset_conditions:
                task = base.Task(mode=condition.mode, **task_template)
                messages = recipe.build_messages(
                    task,
                    condition,
                    test_directory,
                    categories,
                    {},
                    references,
                    assets,
                    readme,
                )
                content = messages[0]["content"]
                if content[-2] != {"type": "text", "text": "TARGET IMAGE:"}:
                    raise ValueError(f"Target marker position mismatch for {name}/{condition.mode}.")
                if content[-1].get("type") != "image_url":
                    raise ValueError(f"Target image position mismatch for {name}/{condition.mode}.")
                prompts[condition.mode] = content[0]["text"]
            if readme not in prompts["instructions_multi"]:
                raise ValueError(f"Correct README missing from instruction prompt for {name}.")
            if readme in prompts["permuted_instructions_multi"]:
                raise ValueError(f"Permuted control retained exact README for {name}.")
            if permuted not in prompts["permuted_instructions_multi"]:
                raise ValueError(f"Permuted README missing from control prompt for {name}.")
            if "sparse positive exemplars" not in prompts["numeric_prediction_b01_multi"]:
                raise ValueError(f"Sparse-reference clarification missing for {name}.")

        image_count = len(test["images"])
        class_count = len(categories)
        object_count = len(test["annotations"])
        total_images += image_count
        total_classes += class_count
        total_objects += object_count
        rows.append(
            {
                "dataset": name,
                "test_images": image_count,
                "classes": class_count,
                "test_objects": object_count,
                "subset": name in SUBSET_DATASETS,
            }
        )

    if (total_images, total_classes, total_objects) != (3970, 110, 57285):
        raise ValueError("RF20 totals differ from the locked study inputs.")
    subset_images = sum(row["test_images"] for row in rows if row["subset"])
    if subset_images != 1737:
        raise ValueError(f"Six-dataset image count changed: {subset_images}.")
    return {
        "created_at": base.utc_now(),
        "dataset_root": str(dataset_root),
        "full_condition_sha256": base.sha256_file(full_conditions_path),
        "subset_condition_sha256": base.sha256_file(subset_conditions_path),
        "ratings_sha256": base.sha256_file(ratings_path),
        "ratings_locked_at": ratings["locked_at"],
        "dataset_count": len(rows),
        "test_images": total_images,
        "classes": total_classes,
        "test_objects": total_objects,
        "full_requests": total_images,
        "subset_datasets": list(SUBSET_DATASETS),
        "subset_images": subset_images,
        "subset_requests": subset_images * len(SUBSET_CONDITIONS),
        "conditions_interleaved_within_image": True,
        "temperature": 0,
        "seed": 1234,
        "reasoning": "off",
        "target_image_last": True,
        "reference_split": "train-only",
        "permuted_control": "within-dataset class-section body rotation",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--full-conditions", type=Path, required=True)
    parser.add_argument("--subset-conditions", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.dataset_root.resolve(),
        args.full_conditions.resolve(),
        args.subset_conditions.resolve(),
        args.ratings.resolve(),
    )
    base.atomic_write_json(args.report.resolve(), result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
