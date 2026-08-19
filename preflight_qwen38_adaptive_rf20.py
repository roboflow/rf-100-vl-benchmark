#!/usr/bin/env python3
"""Validate the complete RF20 adaptive no-feedback benchmark contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate_qwen38_adaptive_no_feedback as adaptive
import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_orion as base


def validate(dataset_root: Path, max_examples_per_class: int = 10) -> dict[str, Any]:
    datasets = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "test/_annotations.coco.json").is_file()
    )
    if len(datasets) != 20:
        raise ValueError(f"RF20-VL-FSOD requires exactly 20 datasets; found {len(datasets)}.")
    dataset_rows = []
    test_images = test_objects = classes = available_references = 0
    for dataset in datasets:
        train_directory = dataset / "train"
        test_directory = dataset / "test"
        train = base.load_coco(train_directory / "_annotations.coco.json")
        test = base.load_coco(test_directory / "_annotations.coco.json")
        base.validate_split_isolation(train, test)
        categories = base.categories_by_id(test)
        if categories != base.categories_by_id(train):
            raise ValueError(f"Train/test category mismatch in {dataset.name}.")
        references = box_ablation.select_reference_sequences(
            train,
            train_directory,
            required_count=max_examples_per_class,
            distinct_images_only=False,
            allow_fewer=True,
        )
        if set(references) != set(categories):
            raise ValueError(f"Reference/category mismatch in {dataset.name}.")
        assets = {
            (category_id, reference.rank): {
                "source": train_directory / reference.file_name
            }
            for category_id, sequence in references.items()
            for reference in sequence
        }
        task = adaptive.build_tasks({**test, "images": test["images"][:1]})[0]
        initial = adaptive.build_initial_messages(task, test_directory, categories)
        initial_content = initial[0]["content"]
        if sum(part["type"] == "image_url" for part in initial_content) != 1:
            raise ValueError(f"Initial turn is not a clean zero-shot target in {dataset.name}.")
        initial_text = "\n".join(
            part["text"] for part in initial_content if part["type"] == "text"
        )
        if "zero labeled visual examples" not in initial_text or "Do not output detections yet" not in initial_text:
            raise ValueError(f"Initial adaptive policy wording is missing in {dataset.name}.")
        first_ids = [category_id for category_id, sequence in references.items() if sequence]
        added = [(category_id, references[category_id][0]) for category_id in first_ids]
        counts = {category_id: int(category_id in first_ids) for category_id in categories}
        reference_message = adaptive.build_reference_message(
            added, counts, categories, assets
        )
        payloads = []
        for part in reference_message["content"]:
            if part["type"] != "text" or not part["text"].startswith('[{"bbox_2d"'):
                continue
            payloads.extend(json.loads(part["text"]))
        if len(payloads) != len(added):
            raise ValueError(f"Reference grouping lost objects in {dataset.name}.")
        if any(set(payload) != {"bbox_2d", "label"} for payload in payloads):
            raise ValueError(f"Reference payload schema mismatch in {dataset.name}.")
        if any(
            len(payload["bbox_2d"]) != 4
            or any(not isinstance(value, int) or not 0 <= value <= 1000 for value in payload["bbox_2d"])
            or payload["label"] not in categories.values()
            for payload in payloads
        ):
            raise ValueError(f"Invalid adaptive reference geometry in {dataset.name}.")
        final_text = adaptive.build_final_message(categories)["content"][0]["text"]
        if "bbox_2d" not in final_text or "normalized independently from 0 to 1000" not in final_text:
            raise ValueError(f"Final detection contract mismatch in {dataset.name}.")
        train_image_ids = {int(image["id"]) for image in train["images"]}
        if any(reference.image_id not in train_image_ids for sequence in references.values() for reference in sequence):
            raise ValueError(f"A reference is not train-only in {dataset.name}.")
        row = {
            "dataset": dataset.name,
            "test_images": len(test["images"]),
            "test_objects": len(test["annotations"]),
            "classes": len(categories),
            "available_reference_objects": sum(len(sequence) for sequence in references.values()),
            "minimum_references_for_any_class": min(map(len, references.values())),
            "maximum_references_for_any_class": max(map(len, references.values())),
        }
        dataset_rows.append(row)
        test_images += row["test_images"]
        test_objects += row["test_objects"]
        classes += row["classes"]
        available_references += row["available_reference_objects"]
    return {
        "benchmark": "RF20-VL-FSOD",
        "mode": adaptive.MODE,
        "prompt_version": adaptive.PROMPT_VERSION,
        "dataset_count": len(datasets),
        "test_images": test_images,
        "test_objects": test_objects,
        "classes": classes,
        "available_train_references": available_references,
        "initial_examples_per_class": 0,
        "max_examples_per_class": max_examples_per_class,
        "reference_increment_per_requested_class": 1,
        "all_references_train_only": True,
        "all_reference_payloads_match_prediction_schema": True,
        "initial_turn_contains_only_class_names_and_target": True,
        "structured_json_decisions": True,
        "prediction_feedback": False,
        "test_ground_truth_visible": False,
        "final_detection_only_scored": True,
        "reasoning_disabled": True,
        "temperature": 0,
        "max_detections": 500,
        "datasets": dataset_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--max-examples-per-class", type=int, default=10)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.dataset_root.resolve(), args.max_examples_per_class)
    base.atomic_write_json(args.report.resolve(), result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
