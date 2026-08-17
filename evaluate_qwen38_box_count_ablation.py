#!/usr/bin/env python3
"""Ablate positive box-reference count for Qwen3.8-Max on RF20-VL-FSOD.

The experiment is a nested 0/1/2/3/5/10-shot comparison. Zero-shot controls
use class names only. Positive-reference conditions cross numeric versus drawn
boxes with single-class versus multi-class requests. Every nonzero count is the
number of train-only reference boxes per class. Predictions are evaluated on
the complete test split with the repository's COCO maxDets=500 scorer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import logging
import math
import os
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import evaluate_qwen38_orion as base

MODEL_ID = "qwen3.8-max"
PROMPT_VERSION = "qwen3.8-max-positive-box-count-v1"
BOX_COUNTS = (1, 2, 3, 5, 10)
FORMULATIONS = ("multi", "single")
REPRESENTATIONS = ("numeric", "drawn")
DEFAULT_DATASET = Path("RF100VL/rf20-vl-fsod/the-dreidel-project")
DEFAULT_OUTPUT = Path("qwen38-fsod-runs/dreidel-box-count-ablation-v1")
TERMINAL_STATUSES = base.TERMINAL_STATUSES

LOGGER = logging.getLogger("qwen38_box_count")


@dataclass(frozen=True)
class Condition:
    mode: str
    formulation: str
    representation: str | None
    box_count: int

    @property
    def single_class(self) -> bool:
        return self.formulation == "single"


@dataclass(frozen=True)
class ReferenceBox:
    category_id: int
    category_name: str
    rank: int
    annotation_id: int
    image_id: int
    file_name: str
    width: int
    height: int
    bbox_xyxy_1000: tuple[int, int, int, int]


def build_conditions() -> tuple[Condition, ...]:
    result = [
        Condition("multi_names_b00", "multi", None, 0),
        Condition("single_names_b00", "single", None, 0),
    ]
    for formulation in FORMULATIONS:
        for representation in REPRESENTATIONS:
            for count in BOX_COUNTS:
                result.append(
                    Condition(
                        f"{formulation}_{representation}_b{count:02d}",
                        formulation,
                        representation,
                        count,
                    )
                )
    return tuple(result)


CONDITIONS = build_conditions()
CONDITIONS_BY_MODE = {condition.mode: condition for condition in CONDITIONS}
MODES = tuple(CONDITIONS_BY_MODE)


def _crop_feature(image_path: Path, bbox: Sequence[float]) -> tuple[float, ...]:
    """Return a small deterministic appearance descriptor for diversity ordering."""

    from PIL import Image

    x, y, width, height = (float(value) for value in bbox)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        left = max(0, math.floor(x))
        top = max(0, math.floor(y))
        right = min(image.width, math.ceil(x + width))
        bottom = min(image.height, math.ceil(y + height))
        if right <= left or bottom <= top:
            raise ValueError(f"Degenerate reference crop in {image_path}: {bbox}")
        crop = image.crop((left, top, right, bottom)).resize((16, 16))
        return tuple(channel / 255.0 for channel in crop.tobytes())


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum((a - b) ** 2 for a, b in zip(left, right))


def select_reference_sequences(
    train: dict[str, Any],
    train_directory: Path,
    *,
    required_count: int = max(BOX_COUNTS),
    distinct_images_only: bool = True,
    first_strategy: str = "largest-relative-area",
    random_seed: int = 1234,
    allow_fewer: bool = False,
) -> dict[int, tuple[ReferenceBox, ...]]:
    """Select nested, diverse, train-only reference boxes.

    Rank one defaults to the existing experiment's largest-relative-area rule.
    A median-relative-area first reference is available for controlled selector
    sensitivity experiments. ``largest-then-seeded-random`` preserves the
    established rank-one reference and orders all remaining object annotations
    by a stable seeded hash, producing a uniform, immutable nested sample.
    Remaining ranks use deterministic farthest-point sampling over object-crop
    pixels. By default, each box comes from a distinct image; instance-based
    FSOD datasets can explicitly allow shared source images. The ordering never
    uses test images or model predictions.
    """

    images = {int(image["id"]): image for image in train["images"]}
    categories = base.categories_by_id(train)
    annotations_by_category: dict[int, list[dict[str, Any]]] = {
        category_id: [] for category_id in categories
    }
    for annotation in train["annotations"]:
        category_id = int(annotation["category_id"])
        if category_id in annotations_by_category:
            annotations_by_category[category_id].append(annotation)

    sequences: dict[int, tuple[ReferenceBox, ...]] = {}
    for category_id, category_name in categories.items():
        # Prefer one representative box per image when the dataset supports
        # it. RF20's K-shot definition is instance based, so datasets with
        # fewer than K distinct source images can explicitly opt into using
        # all annotated boxes, including multiple boxes from one
        # source image.
        best_by_image: dict[int, dict[str, Any]] = {}
        for annotation in annotations_by_category[category_id]:
            image_id = int(annotation["image_id"])
            area = float(annotation["bbox"][2]) * float(annotation["bbox"][3])
            current = best_by_image.get(image_id)
            if current is None:
                best_by_image[image_id] = annotation
                continue
            current_area = float(current["bbox"][2]) * float(current["bbox"][3])
            if (area, -int(annotation["id"])) > (
                current_area,
                -int(current["id"]),
            ):
                best_by_image[image_id] = annotation
        if distinct_images_only:
            selected_annotations = list(best_by_image.values())
        else:
            selected_annotations = list(annotations_by_category[category_id])
        target_count = (
            min(required_count, len(selected_annotations))
            if allow_fewer
            else required_count
        )
        if len(selected_annotations) < target_count:
            raise ValueError(
                f"{category_name!r} has {len(selected_annotations)} eligible "
                f"positive train boxes; {target_count} are required."
            )

        candidates: list[dict[str, Any]] = []
        for annotation in selected_annotations:
            image_id = int(annotation["image_id"])
            image = images[image_id]
            image_path = train_directory / str(image["file_name"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            relative_area = (
                float(annotation["bbox"][2])
                * float(annotation["bbox"][3])
                / (int(image["width"]) * int(image["height"]))
            )
            candidates.append(
                {
                    "annotation": annotation,
                    "image": image,
                    "relative_area": relative_area,
                    "feature": _crop_feature(image_path, annotation["bbox"]),
                }
            )

        if first_strategy in {
            "largest-relative-area",
            "largest-then-seeded-random",
        }:
            candidates.sort(
                key=lambda item: (
                    -item["relative_area"],
                    str(item["image"]["file_name"]),
                    int(item["annotation"]["id"]),
                )
            )
            selected = [candidates.pop(0)]
            if first_strategy == "largest-then-seeded-random":
                dataset_name = train_directory.parent.name
                candidates.sort(
                    key=lambda item: hashlib.sha256(
                        (
                            f"{random_seed}\0{dataset_name}\0{category_id}\0"
                            f"{int(item['annotation']['id'])}"
                        ).encode("utf-8")
                    ).digest()
                )
                selected.extend(candidates[: target_count - 1])
                del candidates[: target_count - 1]
        elif first_strategy == "median-relative-area":
            candidates.sort(
                key=lambda item: (
                    item["relative_area"],
                    str(item["image"]["file_name"]),
                    int(item["annotation"]["id"]),
                )
            )
            selected = [candidates.pop((len(candidates) - 1) // 2)]
        else:
            raise ValueError(f"Unknown first-reference strategy: {first_strategy}")
        preferred_annotation_ids = {
            int(annotation["id"]) for annotation in best_by_image.values()
        }
        while len(selected) < target_count:
            # Preserve the existing nested 1/2/5-shot sequence: exhaust one
            # representative instance from every distinct image before adding
            # further annotated instances from already-used images.
            preferred = [
                item
                for item in candidates
                if int(item["annotation"]["id"]) in preferred_annotation_ids
            ]
            pool = preferred or candidates
            pool.sort(
                key=lambda item: (
                    -min(
                        _squared_distance(item["feature"], chosen["feature"])
                        for chosen in selected
                    ),
                    str(item["image"]["file_name"]),
                    int(item["annotation"]["id"]),
                )
            )
            choice = pool[0]
            candidates.remove(choice)
            selected.append(choice)

        references = []
        for rank, item in enumerate(selected, start=1):
            annotation = item["annotation"]
            image = item["image"]
            references.append(
                ReferenceBox(
                    category_id=category_id,
                    category_name=category_name,
                    rank=rank,
                    annotation_id=int(annotation["id"]),
                    image_id=int(image["id"]),
                    file_name=str(image["file_name"]),
                    width=int(image["width"]),
                    height=int(image["height"]),
                    bbox_xyxy_1000=base.annotation_xywh_to_normalized_xyxy(
                        annotation["bbox"],
                        int(image["width"]),
                        int(image["height"]),
                    ),
                )
            )
        sequences[category_id] = tuple(references)
    return sequences


def prepare_reference_assets(
    train_directory: Path,
    output_directory: Path,
    references: dict[int, tuple[ReferenceBox, ...]],
) -> dict[tuple[int, int], dict[str, Path]]:
    assets: dict[tuple[int, int], dict[str, Path]] = {}
    for category_id, sequence in references.items():
        for reference in sequence:
            source = train_directory / reference.file_name
            drawn = (
                output_directory
                / f"class_{category_id}"
                / f"rank_{reference.rank:02d}.jpg"
            )
            base.render_reference(
                source,
                [reference.bbox_xyxy_1000],
                drawn,
                positive=True,
            )
            assets[(category_id, reference.rank)] = {
                "source": source,
                "drawn": drawn,
            }
    return assets


def build_tasks(
    test: dict[str, Any], categories: dict[int, str]
) -> list[base.Task]:
    tasks: list[base.Task] = []
    for condition in CONDITIONS:
        for image in sorted(test["images"], key=lambda value: int(value["id"])):
            common = {
                "mode": condition.mode,
                "image_id": int(image["id"]),
                "file_name": str(image["file_name"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
            }
            if condition.single_class:
                for category_id, category_name in categories.items():
                    tasks.append(
                        base.Task(
                            **common,
                            category_id=category_id,
                            category_name=category_name,
                        )
                    )
            else:
                tasks.append(base.Task(**common))
    if len({task.key for task in tasks}) != len(tasks):
        raise ValueError("Generated task keys are not unique.")
    return tasks


def build_messages(
    task: base.Task,
    test_directory: Path,
    categories: dict[int, str],
    references: dict[int, tuple[ReferenceBox, ...]],
    assets: dict[tuple[int, int], dict[str, Path]],
) -> list[dict[str, Any]]:
    condition = CONDITIONS_BY_MODE[task.mode]
    target = test_directory / task.file_name
    if not target.is_file():
        raise FileNotFoundError(target)

    if condition.single_class:
        if task.category_id is None or task.category_name is None:
            raise ValueError(f"Single-class task is missing its category: {task}")
        requested = {task.category_id: task.category_name}
        prompt = f'Detect every instance of "{task.category_name}" in the TARGET IMAGE. '
    else:
        requested = categories
        prompt = "Detect every instance of the listed classes in the TARGET IMAGE. "

    if condition.box_count:
        prompt += (
            f"Use the {condition.box_count} positive train-only reference "
            f"box{'es' if condition.box_count != 1 else ''} supplied per class. "
        )
    prompt += base.output_contract(list(requested.values()))
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

    for category_id, category_name in requested.items():
        for reference in references[category_id][: condition.box_count]:
            label = (
                f'POSITIVE REFERENCE FOR "{category_name}" '
                f"({reference.rank}/{condition.box_count})"
            )
            if condition.representation == "numeric":
                text = (
                    f"{label}: normalized XYXY box "
                    f"{json.dumps(list(reference.bbox_xyxy_1000))} marks one "
                    f"positive example of {category_name}."
                )
                image_path = assets[(category_id, reference.rank)]["source"]
            elif condition.representation == "drawn":
                text = f"{label}: the green box marks one positive example of {category_name}."
                image_path = assets[(category_id, reference.rank)]["drawn"]
            else:
                raise ValueError(f"Unknown representation: {condition.representation}")
            content.extend(
                [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": base.data_url(image_path)},
                    },
                ]
            )
    content.extend(
        [
            {"type": "text", "text": "TARGET IMAGE:"},
            {"type": "image_url", "image_url": {"url": base.data_url(target)}},
        ]
    )
    return [{"role": "user", "content": content}]


def expected_images_per_request(condition: Condition, class_count: int) -> int:
    reference_classes = 1 if condition.single_class else class_count
    return 1 + condition.box_count * reference_classes


def build_token_estimates(class_count: int) -> dict[str, int]:
    """Build conservative per-mode totals used by the shared TPM limiter."""

    return {
        condition.mode: 3_000 * expected_images_per_request(condition, class_count)
        + 2_500
        for condition in CONDITIONS
    }


class TaskRateLimiter:
    """Apply one task's visual-input estimate to the shared dual limiter."""

    def __init__(self, shared: base.SmoothDualRateLimiter, estimated_tokens: int) -> None:
        self.shared = shared
        self.estimated_tokens = estimated_tokens

    def acquire(self, _base_estimate: int) -> None:
        self.shared.acquire(self.estimated_tokens)


def record_path(output_directory: Path, task: base.Task) -> Path:
    return output_directory / "records" / task.mode / f"{task.key}.json"


def summarize_records(
    tasks: Sequence[base.Task], output_directory: Path
) -> dict[str, Any]:
    result: dict[str, dict[str, int]] = {}
    for task in tasks:
        counts = result.setdefault(
            task.mode,
            {"total": 0, "success": 0, "model_failure": 0, "error": 0, "pending": 0},
        )
        counts["total"] += 1
        record = base.load_record(record_path(output_directory, task))
        status = record.get("status") if record else "pending"
        counts[status if status in counts else "error"] += 1
    total = {
        key: sum(value[key] for value in result.values())
        for key in ("total", "success", "model_failure", "error", "pending")
    }
    return {"updated_at": base.utc_now(), "total": total, "modes": result}


def _usage_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    usage = [record.get("usage") or {} for record in records]
    inference_times = [
        float(record["inference_seconds"])
        for record in records
        if record.get("inference_seconds") is not None
    ]
    return {
        "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in usage),
        "completion_tokens": sum(
            int(item.get("completion_tokens") or 0) for item in usage
        ),
        "reasoning_tokens": sum(
            int(
                (item.get("completion_tokens_details") or {}).get("reasoning_tokens")
                or 0
            )
            for item in usage
        ),
        "mean_inference_seconds": fmean(inference_times) if inference_times else None,
    }


def finalize(
    all_tasks: Sequence[base.Task],
    annotation_path: Path,
    output_directory: Path,
    *,
    image_count: int,
    class_count: int,
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        tasks = [task for task in all_tasks if task.mode == condition.mode]
        predictions: list[dict[str, Any]] = []
        statuses: dict[str, int] = {}
        records: list[dict[str, Any]] = []
        for task in tasks:
            record = base.load_record(record_path(output_directory, task))
            status = record.get("status", "missing") if record else "missing"
            statuses[status] = statuses.get(status, 0) + 1
            if record and status in TERMINAL_STATUSES:
                records.append(record)
                predictions.extend(record.get("predictions", []))
        complete = sum(statuses.get(status, 0) for status in TERMINAL_STATUSES) == len(
            tasks
        )
        predictions_path = output_directory / "predictions" / f"{condition.mode}.json"
        base.atomic_write_json(predictions_path, predictions)
        metrics = base.score_coco(annotation_path, predictions) if complete else None
        summary = {
            "condition": asdict(condition),
            "complete": complete,
            "task_count": len(tasks),
            "calls_per_image": class_count if condition.single_class else 1,
            "reference_images_per_request": expected_images_per_request(
                condition, class_count
            )
            - 1,
            "statuses": statuses,
            "prediction_count": len(predictions),
            "usage": _usage_summary(records),
            "metrics": metrics,
        }
        modes[condition.mode] = summary
        base.atomic_write_json(
            output_directory / "metrics" / f"{condition.mode}.json", summary
        )
        rows.append(
            {
                "mode": condition.mode,
                "formulation": condition.formulation,
                "representation": condition.representation or "class_names",
                "boxes_per_class": condition.box_count,
                "calls_per_image": summary["calls_per_image"],
                "reference_images_per_request": summary[
                    "reference_images_per_request"
                ],
                "task_count": len(tasks),
                "complete": complete,
                "mAP50_95": metrics["AP"] * 100 if metrics else None,
                "mAP50": metrics["AP50"] * 100 if metrics else None,
                "model_failures": statuses.get("model_failure", 0),
                "errors": statuses.get("error", 0) + statuses.get("missing", 0),
                "mean_inference_seconds": summary["usage"][
                    "mean_inference_seconds"
                ],
                "prompt_tokens": summary["usage"]["prompt_tokens"],
                "completion_tokens": summary["usage"]["completion_tokens"],
            }
        )
    aggregate = {
        "updated_at": base.utc_now(),
        "prompt_version": PROMPT_VERSION,
        "image_count": image_count,
        "class_count": class_count,
        "modes": modes,
    }
    base.atomic_write_json(output_directory / "aggregate_metrics.json", aggregate)
    comparison = {"updated_at": base.utc_now(), "rows": rows}
    base.atomic_write_json(output_directory / "comparison_summary.json", comparison)
    csv_path = output_directory / "comparison_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    if all(summary["complete"] for summary in modes.values()):
        base.atomic_write_json(
            output_directory / "_SUCCESS.json",
            {
                "completed_at": base.utc_now(),
                "prompt_version": PROMPT_VERSION,
                "dataset": str(annotation_path.parents[1]),
                "image_count": image_count,
                "class_count": class_count,
                "condition_count": len(CONDITIONS),
                "request_count": sum(row["task_count"] for row in rows),
            },
        )
    return aggregate


def write_or_validate_manifest(
    path: Path,
    configuration: dict[str, Any],
    references: dict[int, tuple[ReferenceBox, ...]],
    train_directory: Path,
    *,
    distinct_images_only: bool,
) -> None:
    expected = {
        "prompt_version": PROMPT_VERSION,
        "configuration": configuration,
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "reference_selection": {
            "method": "largest-relative-area-then-greedy-crop-diversity-v1",
            "nested_counts": list(BOX_COUNTS),
            "one_box_per_distinct_train_image": distinct_images_only,
            "classes": {
                str(category_id): [
                    {
                        **asdict(reference),
                        "source_sha256": base.sha256_file(
                            train_directory / reference.file_name
                        ),
                    }
                    for reference in sequence
                ]
                for category_id, sequence in references.items()
            },
        },
    }
    expected = json.loads(json.dumps(expected, ensure_ascii=False))
    existing = base.load_record(path)
    if existing:
        comparable = {key: existing.get(key) for key in expected}
        if comparable != expected:
            raise ValueError(
                f"Existing manifest does not match this experiment: {path}"
            )
        return
    base.atomic_write_json(path, {**expected, "created_at": base.utc_now()})


def configure_logging(output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        handlers=[
            logging.FileHandler(output_directory / "experiment.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--base-url",
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--requests-per-minute", type=float, default=570.0)
    parser.add_argument("--tokens-per-minute", type=float, default=900_000.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--temperature",
        type=float,
        help=(
            "Explicit sampling temperature. Omit to preserve provider-default "
            "behavior in established exploratory runs."
        ),
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--image-ids", nargs="+", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-per-mode", type=int)
    parser.add_argument(
        "--allow-shared-reference-images",
        action="store_true",
        help=(
            "Use distinct train annotations when fewer than ten distinct source "
            "images exist for a class."
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.concurrency < 1 or args.max_retries < 0:
        raise ValueError("Concurrency must be positive and retries nonnegative.")
    if args.requests_per_minute <= 0 or args.tokens_per_minute <= 0:
        raise ValueError("RPM and TPM limits must be positive.")
    if args.temperature is not None and not 0 <= args.temperature < 2:
        raise ValueError("Temperature must be in [0, 2).")
    if args.limit is not None and args.limit_per_mode is not None:
        raise ValueError("--limit and --limit-per-mode are mutually exclusive.")
    if not os.getenv("DASHSCOPE_API_KEY") and not args.prepare_only:
        raise RuntimeError("DASHSCOPE_API_KEY is required for inference.")

    dataset_directory = args.dataset_dir.resolve()
    output_directory = args.output_dir.resolve()
    configure_logging(output_directory)
    lock_file = (output_directory / ".run.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another ablation process owns {output_directory}.") from error

    train_directory = dataset_directory / "train"
    test_directory = dataset_directory / "test"
    train_path = train_directory / "_annotations.coco.json"
    test_path = test_directory / "_annotations.coco.json"
    train = base.load_coco(train_path)
    test = base.load_coco(test_path)
    base.validate_split_isolation(train, test)
    categories = base.categories_by_id(test)
    if categories != base.categories_by_id(train):
        raise ValueError("Train/test category definitions differ.")
    distinct_images_only = not args.allow_shared_reference_images
    references = select_reference_sequences(
        train,
        train_directory,
        distinct_images_only=distinct_images_only,
    )
    assets = prepare_reference_assets(
        train_directory, output_directory / "references", references
    )
    token_estimates = build_token_estimates(len(categories))
    all_tasks = build_tasks(test, categories)
    selected_modes = set(args.modes)
    tasks = [task for task in all_tasks if task.mode in selected_modes]
    if args.image_ids is not None:
        requested = set(args.image_ids)
        available = {int(image["id"]) for image in test["images"]}
        if requested - available:
            raise ValueError(f"Unknown test image IDs: {sorted(requested - available)}")
        tasks = [task for task in tasks if task.image_id in requested]

    settings = {
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "seed": args.seed,
        "max_completion_tokens": args.max_completion_tokens,
        "reasoning_effort": "none",
        "vl_high_resolution_images": False,
        "timeout_seconds": args.timeout_seconds,
    }
    if args.temperature is not None:
        settings["temperature"] = args.temperature
    configuration = {
        "dataset_directory": str(dataset_directory),
        "train_annotation_sha256": base.sha256_file(train_path),
        "test_annotation_sha256": base.sha256_file(test_path),
        "settings": settings,
        "token_estimates_by_mode": token_estimates,
        "requests_per_minute": args.requests_per_minute,
        "tokens_per_minute": args.tokens_per_minute,
    }
    write_or_validate_manifest(
        output_directory / "run_manifest.json",
        configuration,
        references,
        train_directory,
        distinct_images_only=distinct_images_only,
    )
    base.atomic_write_json(
        output_directory / "progress.json",
        summarize_records(all_tasks, output_directory),
    )
    LOGGER.info(
        "Prepared %d requests across %d conditions, %d images, and %d classes.",
        len(all_tasks),
        len(CONDITIONS),
        len(test["images"]),
        len(categories),
    )
    if args.prepare_only:
        finalize(
            all_tasks,
            test_path,
            output_directory,
            image_count=len(test["images"]),
            class_count=len(categories),
        )
        return 0

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=settings["base_url"],
        timeout=settings["timeout_seconds"],
        max_retries=0,
    )
    pending: list[base.Task] = []
    for task in tasks:
        existing = base.load_record(record_path(output_directory, task))
        if existing and existing.get("status") in TERMINAL_STATUSES:
            messages = build_messages(
                task, test_directory, categories, references, assets
            )
            expected = base.request_fingerprint(
                task, base.request_summary(messages), settings
            )
            if (
                existing.get("task_key") != task.key
                or existing.get("request_fingerprint") != expected
            ):
                raise ValueError(
                    "A terminal checkpoint has a mismatched request fingerprint: "
                    f"{task.key}"
                )
            continue
        pending.append(task)
    if args.limit is not None:
        pending = pending[: args.limit]
    elif args.limit_per_mode is not None:
        pending = [
            task
            for mode in MODES
            for task in [item for item in pending if item.mode == mode][
                : args.limit_per_mode
            ]
        ]

    rate_limiter = base.SmoothDualRateLimiter(
        args.requests_per_minute, args.tokens_per_minute
    )
    LOGGER.info(
        "Starting %d pending requests with concurrency=%d, RPM=%.1f, TPM=%.0f.",
        len(pending),
        args.concurrency,
        args.requests_per_minute,
        args.tokens_per_minute,
    )

    def execute(task: base.Task) -> dict[str, Any]:
        messages = build_messages(task, test_directory, categories, references, assets)
        return base.execute_task(
            task,
            client,
            test_directory,
            categories,
            {},
            {},
            {},
            settings,
            args.max_retries,
            TaskRateLimiter(rate_limiter, token_estimates[task.mode]),
            messages_override=messages,
        )

    write_lock = threading.Lock()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(execute, task): task for task in pending}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001
                record = {
                    "status": "error",
                    "error": f"WorkerFailure: {type(error).__name__}: {error}",
                    "task": asdict(task),
                    "task_key": task.key,
                    "predictions": [],
                    "completed_at": base.utc_now(),
                }
            with write_lock:
                base.atomic_write_json(record_path(output_directory, task), record)
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    progress = summarize_records(all_tasks, output_directory)
                    base.atomic_write_json(output_directory / "progress.json", progress)
                    LOGGER.info(
                        "Checkpoint %d/%d; overall terminal=%d/%d, errors=%d.",
                        completed,
                        len(pending),
                        progress["total"]["success"]
                        + progress["total"]["model_failure"],
                        progress["total"]["total"],
                        progress["total"]["error"],
                    )

    progress = summarize_records(all_tasks, output_directory)
    base.atomic_write_json(output_directory / "progress.json", progress)
    finalize(
        all_tasks,
        test_path,
        output_directory,
        image_count=len(test["images"]),
        class_count=len(categories),
    )
    selected_progress = summarize_records(tasks, output_directory)
    unresolved = (
        selected_progress["total"]["error"]
        + selected_progress["total"]["pending"]
    )
    LOGGER.info(
        "Invocation finished: selected terminal=%d/%d, unresolved=%d.",
        selected_progress["total"]["success"]
        + selected_progress["total"]["model_failure"],
        selected_progress["total"]["total"],
        unresolved,
    )
    return 0 if unresolved == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
