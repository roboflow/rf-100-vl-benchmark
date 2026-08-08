#!/usr/bin/env python3
"""Validate RF100VL, the Cosmos endpoint, and real GCS before GPU evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence
import uuid

from evaluate_cosmos import (
    GCSArtifactStore,
    MODEL_ID,
    PROMPT_VERSION,
    atomic_write_json,
    discover_datasets,
    normalize_label,
    parse_gcs_uri,
    resolve_annotation_file,
    resolve_image_path,
    sha256_file,
)


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer.")
    result = int(value)
    if result <= 0 or result != value:
        raise ValueError(f"{field} must be a positive integer.")
    return result


def validate_dataset(dataset_directory: Path) -> dict[str, Any]:
    """Validate one test split, including every image and annotation reference."""

    test_directory = dataset_directory / "test"
    annotation_path = resolve_annotation_file(test_directory)
    with annotation_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    categories = coco.get("categories")
    images = coco.get("images")
    annotations = coco.get("annotations")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"{annotation_path}: categories must be a non-empty list.")
    if not isinstance(images, list) or not images:
        raise ValueError(f"{annotation_path}: images must be a non-empty list.")
    if not isinstance(annotations, list):
        raise ValueError(f"{annotation_path}: annotations must be a list.")

    category_ids: set[int] = set()
    normalized_names: set[str] = set()
    for category in categories:
        category_id = int(category["id"])
        name = str(category["name"])
        normalized = normalize_label(name)
        if category_id in category_ids:
            raise ValueError(f"{annotation_path}: duplicate category id {category_id}.")
        if not normalized or normalized in normalized_names:
            raise ValueError(
                f"{annotation_path}: category names are empty or ambiguous after normalization."
            )
        category_ids.add(category_id)
        normalized_names.add(normalized)

    image_dimensions: dict[int | str, tuple[int, int]] = {}
    for image in images:
        image_id = image["id"]
        if image_id in image_dimensions:
            raise ValueError(f"{annotation_path}: duplicate image id {image_id!r}.")
        width = _positive_integer(image["width"], f"image {image_id!r} width")
        height = _positive_integer(image["height"], f"image {image_id!r} height")
        image_path = resolve_image_path(test_directory, str(image["file_name"]))
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "Pillow is required. Install requirements-cosmos.txt."
            ) from error
        with Image.open(image_path) as opened:
            if opened.size != (width, height):
                raise ValueError(
                    f"{image_path}: pixels are {opened.size}, COCO metadata is "
                    f"{(width, height)}."
                )
            opened.verify()
        image_dimensions[image_id] = (width, height)

    annotation_ids: set[int | str] = set()
    degenerate_annotation_count = 0
    for index, annotation in enumerate(annotations):
        annotation_id = annotation.get("id", f"index-{index}")
        if annotation_id in annotation_ids:
            raise ValueError(
                f"{annotation_path}: duplicate annotation id {annotation_id!r}."
            )
        annotation_ids.add(annotation_id)
        image_id = annotation.get("image_id")
        if image_id not in image_dimensions:
            raise ValueError(
                f"{annotation_path}: annotation {annotation_id!r} references "
                f"unknown image {image_id!r}."
            )
        category_id = int(annotation.get("category_id"))
        if category_id not in category_ids:
            raise ValueError(
                f"{annotation_path}: annotation {annotation_id!r} references "
                f"unknown category {category_id}."
            )
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(
                f"{annotation_path}: annotation {annotation_id!r} has invalid bbox."
            )
        try:
            x, y, width, height = (float(value) for value in bbox)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{annotation_path}: annotation {annotation_id!r} has non-numeric bbox."
            ) from error
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError(
                f"{annotation_path}: annotation {annotation_id!r} has non-finite bbox."
            )
        image_width, image_height = image_dimensions[image_id]
        tolerance = 1e-6
        if (
            x < -tolerance
            or y < -tolerance
            or width < 0
            or height < 0
            or x + width > image_width + tolerance
            or y + height > image_height + tolerance
        ):
            raise ValueError(
                f"{annotation_path}: annotation {annotation_id!r} bbox {bbox} "
                f"is outside image {(image_width, image_height)}."
            )
        # RF100VL's canonical COCO export currently contains a small number of
        # zero-area ground-truth boxes. Keep the source annotation unchanged so
        # Cosmos is scored against exactly the same COCO data as other models,
        # but make the condition explicit in the preflight report. Negative or
        # out-of-image boxes remain fatal above.
        if width == 0 or height == 0:
            degenerate_annotation_count += 1

    return {
        "dataset": dataset_directory.name,
        "annotation_sha256": sha256_file(annotation_path),
        "category_count": len(categories),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "degenerate_annotation_count": degenerate_annotation_count,
    }


def validate_dataset_root(root: Path, expected_datasets: int) -> dict[str, Any]:
    dataset_directories = discover_datasets(root, None)
    if len(dataset_directories) != expected_datasets:
        raise ValueError(
            f"Expected {expected_datasets} RF100VL test datasets, found "
            f"{len(dataset_directories)} below {root}."
        )
    datasets = [validate_dataset(path) for path in dataset_directories]
    return {
        "expected_dataset_count": expected_datasets,
        "dataset_count": len(datasets),
        "image_count": sum(item["image_count"] for item in datasets),
        "annotation_count": sum(item["annotation_count"] for item in datasets),
        "degenerate_annotation_count": sum(
            item["degenerate_annotation_count"] for item in datasets
        ),
        "datasets": datasets,
    }


def validate_endpoint(
    base_url: str,
    api_key: str,
    expected_model_id: str,
    timeout: float,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The openai package is required. Install requirements-cosmos.txt."
        ) from error
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
    )
    model_ids = sorted(model.id for model in client.models.list().data)
    if expected_model_id not in model_ids:
        raise ValueError(
            f"Endpoint {base_url} does not advertise {expected_model_id!r}; "
            f"reported models: {model_ids}."
        )
    return {
        "base_url": base_url,
        "expected_model_id": expected_model_id,
        "advertised_model_ids": model_ids,
    }


def validate_gcs(parent_uri: str) -> dict[str, Any]:
    """Exercise GCS CRUD/list/restore below an isolated UUID child prefix."""

    parse_gcs_uri(parent_uri)
    child_name = f"preflight-storage-{uuid.uuid4().hex}"
    store = GCSArtifactStore(f"{parent_uri.rstrip('/')}/{child_name}")
    touched = [
        "run_access_probe.json",
        "dataset/records/one.json",
        "_SUCCESS.json",
    ]
    try:
        store.verify_access()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "one.json"
            source.write_text('{"version":1}\n', encoding="utf-8")
            store.upload_file(source, "dataset/records/one.json")
            source.write_text('{"version":2}\n', encoding="utf-8")
            store.upload_file(source, "dataset/records/one.json")
            restored = root / "restored"
            restored_count = store.restore_prefix("dataset/records", restored)
            if restored_count != 1:
                raise ValueError(
                    f"Expected one restored GCS object, got {restored_count}."
                )
            if (restored / "one.json").read_text(encoding="utf-8") != '{"version":2}\n':
                raise ValueError("The updated GCS object did not round-trip exactly.")
            success = root / "_SUCCESS.json"
            success.write_text('{"status":"test-only"}\n', encoding="utf-8")
            store.upload_file(success, "_SUCCESS.json")
            store.delete_if_exists("_SUCCESS.json")
    finally:
        for relative_path in touched:
            store.delete_if_exists(relative_path)
    return {
        "parent_uri": parent_uri,
        "isolated_child": child_name,
        "operations": ["create", "update", "list", "read", "restore", "delete"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight RF100VL data, Cosmos vLLM, and GCS before inference."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-datasets", type=int, default=100)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("COSMOS_API_KEY", "EMPTY"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--gcs-test-uri",
        required=True,
        help="A gs://bucket/prefix parent used only for isolated preflight objects.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("cosmos_preflight_report.json"),
    )
    args = parser.parse_args(argv)
    if args.expected_datasets < 1:
        parser.error("--expected-datasets must be positive.")
    if args.timeout <= 0:
        parser.error("--timeout must be positive.")
    try:
        parse_gcs_uri(args.gcs_test_uri)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_dir.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    gcs_result = validate_gcs(args.gcs_test_uri)
    gcs_result["report_uri"] = f"{args.gcs_test_uri.rstrip('/')}/preflight_report.json"
    report = {
        "schema_version": 1,
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "prompt_version": PROMPT_VERSION,
        "dataset": validate_dataset_root(dataset_root, args.expected_datasets),
        "endpoint": validate_endpoint(
            args.base_url,
            args.api_key,
            args.model_id,
            args.timeout,
        ),
        "gcs": gcs_result,
    }
    report_path = args.report.resolve()
    atomic_write_json(report_path, report)
    GCSArtifactStore(args.gcs_test_uri).upload_file(
        report_path, "preflight_report.json"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
