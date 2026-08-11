#!/usr/bin/env python3
"""Compare Qwen3.8-Max prompt modes on a fixed five-image Orion subset.

The existing six modes are rescored from their completed full-test records,
without making new API calls. Two missing modes are then evaluated:

* multi_class_positive_numeric: eight class examples, numeric boxes, one call
* multi_class_positive_drawn: eight class examples, drawn boxes, one call

Both modes use the same train-only reference selected for each class by the
full experiment. Each request contains eight reference images followed by one
target image, and is run with reasoning effort ``none`` and ``low``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import logging
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import evaluate_qwen38_orion as base

SUBSET_VERSION = "orion-five-image-all-class-v1"
SUBSET_IMAGE_IDS = (36, 2, 27, 30, 0)
NEW_MODES = (
    "multi_class_positive_numeric",
    "multi_class_positive_drawn",
)
REASONING_EFFORTS = ("none", "low")
DEFAULT_FULL_RUNS = {
    "none": Path("qwen38-orion-runs/orion-prompt-modes-v1-no-thinking"),
    "low": Path("qwen38-orion-runs/orion-prompt-modes-v1"),
}

LOGGER = logging.getLogger("qwen38_orion_subset")


def subset_ground_truth(test: dict[str, Any]) -> dict[str, Any]:
    selected = set(SUBSET_IMAGE_IDS)
    images_by_id = {int(image["id"]): image for image in test["images"]}
    missing = selected - set(images_by_id)
    if missing:
        raise ValueError(f"Subset contains unknown test image IDs: {sorted(missing)}")
    result = {
        key: value
        for key, value in test.items()
        if key not in {"images", "annotations"}
    }
    result["images"] = [images_by_id[image_id] for image_id in SUBSET_IMAGE_IDS]
    result["annotations"] = [
        annotation
        for annotation in test["annotations"]
        if int(annotation["image_id"]) in selected
    ]
    present_categories = {
        int(annotation["category_id"]) for annotation in result["annotations"]
    }
    expected_categories = {int(category["id"]) for category in test["categories"]}
    if present_categories != expected_categories:
        raise ValueError("The fixed subset must contain every Orion class.")
    return result


def build_tasks(test: dict[str, Any]) -> list[base.Task]:
    images = {int(image["id"]): image for image in test["images"]}
    return [
        base.Task(
            mode=mode,
            image_id=image_id,
            file_name=str(images[image_id]["file_name"]),
            width=int(images[image_id]["width"]),
            height=int(images[image_id]["height"]),
        )
        for mode in NEW_MODES
        for image_id in SUBSET_IMAGE_IDS
    ]


def build_multi_reference_messages(
    task: base.Task,
    test_directory: Path,
    categories: dict[int, str],
    examples: dict[int, base.ReferenceExample],
    assets: dict[int, dict[str, Path]],
) -> list[dict[str, Any]]:
    if task.mode not in NEW_MODES:
        raise ValueError(f"Unknown subset mode: {task.mode}")
    target = test_directory / task.file_name
    if not target.is_file():
        raise FileNotFoundError(target)
    class_names = [categories[category_id] for category_id in sorted(categories)]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Detect every instance of the listed classes in the TARGET IMAGE. "
                "Use the positive reference example supplied for every class. "
                + base.output_contract(class_names)
            ),
        }
    ]
    for category_id in sorted(categories):
        name = categories[category_id]
        example = examples[category_id]
        if task.mode == "multi_class_positive_numeric":
            boxes = [list(box) for box in example.boxes_xyxy_1000]
            text = (
                f"POSITIVE REFERENCE FOR {name}: normalized XYXY boxes "
                f"mark examples of {name}: {json.dumps(boxes)}"
            )
            image_path = assets[category_id]["positive_source"]
        else:
            text = f"POSITIVE REFERENCE FOR {name}: green boxes mark examples of {name}."
            image_path = assets[category_id]["positive_drawn"]
        content.extend(
            [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": base.data_url(image_path)}},
            ]
        )
    content.extend(
        [
            {"type": "text", "text": "TARGET IMAGE:"},
            {"type": "image_url", "image_url": {"url": base.data_url(target)}},
        ]
    )
    return [{"role": "user", "content": content}]


def filtered_predictions(full_run: Path, mode: str) -> list[dict[str, Any]]:
    path = full_run / "predictions" / f"{mode}.json"
    predictions = json.loads(path.read_text(encoding="utf-8"))
    selected = set(SUBSET_IMAGE_IDS)
    return [
        prediction
        for prediction in predictions
        if int(prediction["image_id"]) in selected
    ]


def record_path(output_directory: Path, effort: str, task: base.Task) -> Path:
    return output_directory / effort / "records" / task.mode / f"{task.key}.json"


def score_all_modes(
    output_directory: Path,
    ground_truth_path: Path,
    effort: str,
    tasks: Sequence[base.Task],
    full_run: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in base.MODES:
        predictions = filtered_predictions(full_run, mode)
        metrics = base.score_coco(ground_truth_path, predictions)
        predictions_path = output_directory / effort / "predictions" / f"{mode}.json"
        base.atomic_write_json(predictions_path, predictions)
        result[mode] = {
            "source": "rescored_completed_full_run",
            "prediction_count": len(predictions),
            "predictions_path": str(predictions_path),
            "metrics": metrics,
        }
    for mode in NEW_MODES:
        predictions: list[dict[str, Any]] = []
        statuses: dict[str, int] = {}
        for task in tasks:
            if task.mode != mode:
                continue
            record = base.load_record(record_path(output_directory, effort, task))
            status = record.get("status", "missing") if record else "missing"
            statuses[status] = statuses.get(status, 0) + 1
            if record and status in base.TERMINAL_STATUSES:
                predictions.extend(record.get("predictions", []))
        complete = sum(statuses.get(status, 0) for status in base.TERMINAL_STATUSES) == len(
            SUBSET_IMAGE_IDS
        )
        predictions_path = output_directory / effort / "predictions" / f"{mode}.json"
        base.atomic_write_json(predictions_path, predictions)
        result[mode] = {
            "source": "new_subset_run",
            "statuses": statuses,
            "prediction_count": len(predictions),
            "predictions_path": str(predictions_path),
            "metrics": base.score_coco(ground_truth_path, predictions) if complete else None,
        }
    for mode, mode_summary in result.items():
        base.atomic_write_json(
            output_directory / effort / "metrics" / f"{mode}.json",
            {"mode": mode, **mode_summary},
        )
    summary = {
        "subset_version": SUBSET_VERSION,
        "image_ids": list(SUBSET_IMAGE_IDS),
        "reasoning_effort": effort,
        "modes": result,
    }
    base.atomic_write_json(output_directory / effort / "aggregate_metrics.json", summary)
    return summary


def write_comparison(
    output_directory: Path, summaries: dict[str, dict[str, Any]]
) -> None:
    rows = []
    for effort in REASONING_EFFORTS:
        if effort not in summaries:
            continue
        for mode, value in summaries[effort]["modes"].items():
            metrics = value.get("metrics")
            rows.append(
                {
                    "mode": mode,
                    "reasoning_effort": effort,
                    "source": value["source"],
                    "prediction_count": value["prediction_count"],
                    "mAP50_95": None if metrics is None else metrics["AP"],
                    "mAP50": None if metrics is None else metrics["AP50"],
                }
            )
    base.atomic_write_json(
        output_directory / "comparison_summary.json",
        {
            "subset_version": SUBSET_VERSION,
            "image_ids": list(SUBSET_IMAGE_IDS),
            "rows": rows,
        },
    )
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    destination = output_directory / "comparison_summary.csv"
    temporary = destination.with_suffix(".csv.tmp")
    temporary.write_text(stream.getvalue(), encoding="utf-8")
    os.replace(temporary, destination)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("RF100VL/rf20-vl-fsod/orionproducts"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qwen38-orion-runs/orion-five-image-single-prompt-v1"),
    )
    parser.add_argument("--model", default=base.MODEL_ID)
    parser.add_argument(
        "--base-url",
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument(
        "--reasoning-efforts",
        nargs="+",
        choices=REASONING_EFFORTS,
        default=list(REASONING_EFFORTS),
    )
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.concurrency <= 0 or args.max_retries < 0:
        raise ValueError("Concurrency must be positive and retries nonnegative.")
    if not args.prepare_only and not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is required for inference.")

    dataset_directory = args.dataset_dir.resolve()
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        handlers=[
            logging.FileHandler(output_directory / "experiment.log"),
            logging.StreamHandler(),
        ],
    )
    train_directory = dataset_directory / "train"
    test_directory = dataset_directory / "test"
    train = base.load_coco(train_directory / "_annotations.coco.json")
    test = base.load_coco(test_directory / "_annotations.coco.json")
    categories = base.categories_by_id(test)
    examples = base.select_reference_examples(train)
    negative_ids = base.validate_negative_pairs(categories)
    assets = base.prepare_reference_assets(
        train_directory, output_directory / "references", examples, negative_ids
    )
    subset = subset_ground_truth(test)
    ground_truth_path = output_directory / "subset_ground_truth.json"
    base.atomic_write_json(ground_truth_path, subset)
    tasks = build_tasks(test)

    manifest = {
        "subset_version": SUBSET_VERSION,
        "image_ids": list(SUBSET_IMAGE_IDS),
        "existing_mode_task_equivalent": len(SUBSET_IMAGE_IDS)
        * (1 + len(base.SINGLE_CLASS_MODES) * len(categories)),
        "new_api_request_count": len(tasks) * len(args.reasoning_efforts),
        "new_modes": list(NEW_MODES),
        "reasoning_efforts": list(args.reasoning_efforts),
        "reference_examples": {
            str(category_id): asdict(example)
            for category_id, example in examples.items()
        },
    }
    base.atomic_write_json(output_directory / "run_manifest.json", manifest)

    summaries = {}
    for effort in args.reasoning_efforts:
        full_run = DEFAULT_FULL_RUNS[effort].resolve()
        summaries[effort] = score_all_modes(
            output_directory, ground_truth_path, effort, tasks, full_run
        )
    write_comparison(output_directory, summaries)
    if args.prepare_only:
        LOGGER.info("Prepared and rescored the fixed subset without API calls.")
        return 0

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=args.base_url.rstrip("/"),
        timeout=args.timeout_seconds,
        max_retries=0,
    )
    limiter = base.SmoothDualRateLimiter(570, 900_000)
    pending: list[tuple[str, base.Task, list[dict[str, Any]], dict[str, Any]]] = []
    for effort in args.reasoning_efforts:
        settings = {
            "model": args.model,
            "base_url": args.base_url.rstrip("/"),
            "seed": args.seed,
            "max_completion_tokens": args.max_completion_tokens,
            "reasoning_effort": effort,
            "vl_high_resolution_images": False,
            "timeout_seconds": args.timeout_seconds,
        }
        for task in tasks:
            messages = build_multi_reference_messages(
                task, test_directory, categories, examples, assets
            )
            path = record_path(output_directory, effort, task)
            expected = base.request_fingerprint(
                task, base.request_summary(messages), settings
            )
            existing = base.load_record(path)
            if existing and existing.get("status") in base.TERMINAL_STATUSES:
                if existing.get("request_fingerprint") != expected:
                    raise ValueError(f"Checkpoint fingerprint mismatch: {path}")
                continue
            pending.append((effort, task, messages, settings))

    LOGGER.info("Starting %d new subset requests.", len(pending))
    write_lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                base.execute_task,
                task,
                client,
                test_directory,
                categories,
                examples,
                negative_ids,
                assets,
                settings,
                args.max_retries,
                limiter,
                messages,
            ): (effort, task)
            for effort, task, messages, settings in pending
        }
        for future in concurrent.futures.as_completed(futures):
            effort, task = futures[future]
            record = future.result()
            with write_lock:
                base.atomic_write_json(record_path(output_directory, effort, task), record)
                LOGGER.info("Saved %s/%s: %s", effort, task.key, record["status"])

    unresolved = 0
    summaries = {}
    for effort in args.reasoning_efforts:
        summary = score_all_modes(
            output_directory,
            ground_truth_path,
            effort,
            tasks,
            DEFAULT_FULL_RUNS[effort].resolve(),
        )
        summaries[effort] = summary
        for mode in NEW_MODES:
            statuses = summary["modes"][mode]["statuses"]
            unresolved += statuses.get("missing", 0) + statuses.get("error", 0)
    write_comparison(output_directory, summaries)
    return 0 if unresolved == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
