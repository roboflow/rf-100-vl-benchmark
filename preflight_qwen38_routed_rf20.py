#!/usr/bin/env python3
"""Validate an RF20 route and every selected Qwen detection prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_orion as base
import evaluate_qwen38_recipe as recipe
from qwen38_calibrated_counts import CONDITION_BY_COUNT, MODE_BY_COUNT, read_route_rows


def validate(dataset_root: Path, route_path: Path) -> dict:
    route = json.loads(route_path.read_text(encoding="utf-8"))
    route_rows = read_route_rows(route)
    datasets = {
        path.name: path
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "test/_annotations.coco.json").is_file()
    }
    if len(datasets) != 20 or set(datasets) != set(route_rows):
        raise ValueError(
            "RF20 route inventory mismatch: "
            f"datasets={len(datasets)}, routes={len(route_rows)}, "
            f"missing={sorted(set(datasets) - set(route_rows))}, "
            f"extra={sorted(set(route_rows) - set(datasets))}"
        )
    rows = []
    total_images = total_objects = total_requests = 0
    for name, dataset in sorted(datasets.items()):
        row = route_rows[name]
        count = int(row["selected_count"])
        condition = CONDITION_BY_COUNT[count]
        if row.get("selected_mode") != MODE_BY_COUNT[count]:
            raise ValueError(f"Mode/count mismatch for {name}.")
        train_directory = dataset / "train"
        test_directory = dataset / "test"
        train = base.load_coco(train_directory / "_annotations.coco.json")
        test = base.load_coco(test_directory / "_annotations.coco.json")
        base.validate_split_isolation(train, test)
        categories = base.categories_by_id(train)
        if categories != base.categories_by_id(test):
            raise ValueError(f"Train/test category mismatch for {name}.")
        references = {}
        assets = {}
        if count:
            references = box_ablation.select_reference_sequences(
                train,
                train_directory,
                required_count=count,
                distinct_images_only=False,
                first_strategy="largest-relative-area",
                allow_fewer=condition.all_available_references,
            )
            assets = {
                (category_id, reference.rank): {
                    "source": train_directory / reference.file_name
                }
                for category_id, sequence in references.items()
                for reference in sequence
            }
        image = min(test["images"], key=lambda value: int(value["id"]))
        task = base.Task(
            mode=condition.mode,
            image_id=int(image["id"]),
            file_name=str(image["file_name"]),
            width=int(image["width"]),
            height=int(image["height"]),
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
            raise ValueError(f"Target marker ordering mismatch for {name}.")
        if content[-1].get("type") != "image_url":
            raise ValueError(f"Target image is not last for {name}.")
        prompt = str(content[0].get("text", ""))
        if recipe._output_contract(list(categories.values())) not in prompt:
            raise ValueError(f"Detection output contract mismatch for {name}.")
        if count and "sparse positive exemplars" not in prompt:
            raise ValueError(f"Sparse-reference clarification missing for {name}.")
        if count == 10 and "Use all positive reference boxes" not in prompt:
            raise ValueError(f"All-reference wording missing for {name}.")
        reference_ids = {
            reference.image_id
            for sequence in references.values()
            for reference in sequence[:count]
        }
        train_image_ids = {int(value["id"]) for value in train["images"]}
        test_image_ids = {int(value["id"]) for value in test["images"]}
        if not reference_ids <= train_image_ids:
            raise ValueError(f"A reference is not train-only for {name}.")
        image_count = len(test["images"])
        total_images += image_count
        total_objects += len(test["annotations"])
        total_requests += image_count
        rows.append(
            {
                "dataset": name,
                "selected_count": count,
                "selected_mode": condition.mode,
                "test_images": image_count,
                "test_objects": len(test["annotations"]),
                "classes": len(categories),
                "reference_images": len(reference_ids),
                "train_test_image_id_overlap_is_irrelevant": bool(
                    train_image_ids & test_image_ids
                ),
            }
        )
    if (total_images, total_objects, total_requests) != (3970, 57285, 3970):
        raise ValueError("RF20 totals changed.")
    return {
        "created_at": base.utc_now(),
        "benchmark": "RF20-VL-FSOD",
        "route_path": str(route_path),
        "dataset_count": len(rows),
        "test_images": total_images,
        "test_objects": total_objects,
        "requests": total_requests,
        "test_annotations_used_for_routing": False,
        "all_references_train_only": True,
        "matched_reference_and_prediction_schema": True,
        "explicit_sparse_reference_semantics": True,
        "reasoning_disabled": True,
        "temperature": 0,
        "seed": 1234,
        "max_detections": 500,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    value = validate(args.dataset_root.resolve(), args.route.resolve())
    base.atomic_write_json(args.report.resolve(), value)
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
