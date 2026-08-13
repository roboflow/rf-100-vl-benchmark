#!/usr/bin/env python3
"""Validate a fresh RF20-VL-FSOD download and compare it byte-for-byte."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

import evaluate_qwen38_orion as base

SPLITS = ("train", "valid", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_split(split: Path) -> dict[str, Any]:
    annotation_path = split / "_annotations.coco.json"
    coco = base.load_coco(annotation_path)
    images = {int(value["id"]): value for value in coco["images"]}
    categories = {int(value["id"]): value for value in coco["categories"]}
    if len(images) != len(coco["images"]):
        raise ValueError(f"Duplicate image IDs in {annotation_path}")
    if len(categories) != len(coco["categories"]):
        raise ValueError(f"Duplicate category IDs in {annotation_path}")
    annotation_ids = [int(value["id"]) for value in coco["annotations"]]
    if len(set(annotation_ids)) != len(annotation_ids):
        raise ValueError(f"Duplicate annotation IDs in {annotation_path}")
    expected_files = {str(value["file_name"]) for value in images.values()}
    actual_files = {
        path.name
        for path in split.iterdir()
        if path.is_file() and path.name != annotation_path.name
    }
    if actual_files != expected_files:
        raise ValueError(
            f"Image inventory mismatch in {split}: "
            f"missing={sorted(expected_files - actual_files)[:5]}, "
            f"extra={sorted(actual_files - expected_files)[:5]}"
        )
    total_bytes = 0
    for image in images.values():
        path = split / str(image["file_name"])
        with Image.open(path) as opened:
            opened.verify()
        with Image.open(path) as opened:
            if opened.size != (int(image["width"]), int(image["height"])):
                raise ValueError(
                    f"Image dimensions disagree with COCO metadata: {path}"
                )
        total_bytes += path.stat().st_size
    degenerate_boxes = 0
    for annotation in coco["annotations"]:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in images or category_id not in categories:
            raise ValueError(f"Dangling annotation reference in {annotation_path}")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Invalid COCO bbox in {annotation_path}")
        x, y, width, height = (float(value) for value in bbox)
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError(f"Non-finite COCO bbox in {annotation_path}")
        if width <= 0 or height <= 0:
            # Preserve and report official annotations exactly as downloaded.
            # Removing them would change the benchmark ground truth.
            degenerate_boxes += 1
        image = images[image_id]
        if (
            x < -1
            or y < -1
            or x + width > float(image["width"]) + 1
            or y + height > float(image["height"]) + 1
        ):
            raise ValueError(f"Out-of-bounds COCO bbox in {annotation_path}")
    return {
        "images": len(images),
        "annotations": len(coco["annotations"]),
        "categories": len(categories),
        "degenerate_boxes": degenerate_boxes,
        "image_bytes": total_bytes,
        "annotation_sha256": _sha256(annotation_path),
    }


def validate(reference_root: Path, fresh_root: Path) -> dict[str, Any]:
    reference_datasets = {
        path.name: path for path in reference_root.iterdir() if path.is_dir()
    }
    fresh_datasets = {path.name: path for path in fresh_root.iterdir() if path.is_dir()}
    if set(reference_datasets) != set(fresh_datasets):
        raise ValueError("Fresh and reference dataset inventories differ.")
    rows = []
    image_files_compared = image_bytes_compared = 0
    for name in sorted(reference_datasets):
        reference = reference_datasets[name]
        fresh = fresh_datasets[name]
        split_rows = {}
        split_names: dict[str, set[str]] = {}
        for split_name in SPLITS:
            reference_split = reference / split_name
            fresh_split = fresh / split_name
            fresh_stats = _validate_split(fresh_split)
            reference_annotation = reference_split / "_annotations.coco.json"
            fresh_annotation = fresh_split / "_annotations.coco.json"
            if json.loads(reference_annotation.read_text()) != json.loads(
                fresh_annotation.read_text()
            ):
                raise ValueError(f"COCO annotations changed for {name}/{split_name}.")
            image_names = {
                path.name
                for path in fresh_split.iterdir()
                if path.is_file() and path.name != fresh_annotation.name
            }
            split_names[split_name] = image_names
            for image_name in sorted(image_names):
                reference_image = reference_split / image_name
                fresh_image = fresh_split / image_name
                if not reference_image.is_file() or _sha256(reference_image) != _sha256(
                    fresh_image
                ):
                    raise ValueError(
                        f"Image bytes changed: {name}/{split_name}/{image_name}"
                    )
                image_files_compared += 1
                image_bytes_compared += fresh_image.stat().st_size
            split_rows[split_name] = fresh_stats
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                overlap = split_names[left] & split_names[right]
                if overlap:
                    raise ValueError(
                        f"Split leakage in {name}: {left}/{right}: {sorted(overlap)[:5]}"
                    )
        rows.append({"dataset": name, "splits": split_rows})
    degenerate_boxes = sum(
        split["degenerate_boxes"] for row in rows for split in row["splits"].values()
    )
    return {
        "created_at": base.utc_now(),
        "reference_root": str(reference_root),
        "fresh_root": str(fresh_root),
        "dataset_count": len(rows),
        "image_files_compared": image_files_compared,
        "image_bytes_compared": image_bytes_compared,
        "annotations_exactly_equal": True,
        "images_byte_identical": True,
        "all_images_decode_and_match_coco_dimensions": True,
        "nonfinite_or_out_of_bounds_boxes": 0,
        "official_degenerate_boxes_preserved": degenerate_boxes,
        "all_splits_isolated": True,
        "datasets": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--fresh-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.reference_root.resolve(), args.fresh_root.resolve())
    base.atomic_write_json(args.report.resolve(), result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
