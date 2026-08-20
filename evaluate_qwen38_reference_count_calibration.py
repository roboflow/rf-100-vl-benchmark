#!/usr/bin/env python3
"""Choose 0/1/2/5/10 Qwen references using train-only support predictions.

Every labeled RF20 support image is predicted under every candidate count. The
current target support image is removed from that request's reference pool, so
its annotations remain unseen until scoring. Unmatched predictions are ignored
because RF20 FSOD support annotations can be sparse. Test images and test
annotations are never used to choose the count.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_orion as base
import evaluate_qwen38_recipe as recipe
import evaluate_qwen38_support_calibrated_router as support
from aggregate_qwen38_rf20 import PRICES_PER_MILLION
from qwen38_calibrated_counts import COUNTS, CONDITION_BY_COUNT, MODE_BY_COUNT, choose_count

PROMPT_VERSION = "qwen3.8-max-fsod-loio-count-calibration-v1"
CONDITIONS = tuple(CONDITION_BY_COUNT[count] for count in COUNTS)
CONDITION_BY_MODE = {condition.mode: condition for condition in CONDITIONS}


@dataclass
class DatasetContext:
    name: str
    directory: Path
    output_directory: Path
    train_directory: Path
    train: dict[str, Any]
    categories: dict[int, str]
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]]
    assets: dict[tuple[int, int], dict[str, Path]]
    tasks: list[base.Task]
    audit: dict[str, Any]


def references_without_target(
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]], image_id: int
) -> dict[int, tuple[box_ablation.ReferenceBox, ...]]:
    result = {
        category_id: tuple(
            reference for reference in sequence if reference.image_id != image_id
        )
        for category_id, sequence in references.items()
    }
    empty = [category_id for category_id, sequence in result.items() if not sequence]
    if empty:
        raise ValueError(
            f"Target image {image_id} leaves no independent reference for classes {empty}."
        )
    return result


def prepare_context(dataset: Path, output_root: Path) -> DatasetContext:
    train_directory = dataset / "train"
    train_path = train_directory / "_annotations.coco.json"
    test_path = dataset / "test/_annotations.coco.json"
    train = base.load_coco(train_path)
    test = base.load_coco(test_path)
    base.validate_split_isolation(train, test)
    categories = base.categories_by_id(train)
    if categories != base.categories_by_id(test):
        raise ValueError(f"Train/test categories differ for {dataset.name}.")
    references = box_ablation.select_reference_sequences(
        train,
        train_directory,
        required_count=10,
        distinct_images_only=False,
        first_strategy="largest-relative-area",
        allow_fewer=True,
    )
    assets = {
        (category_id, reference.rank): {
            "source": train_directory / reference.file_name
        }
        for category_id, sequence in references.items()
        for reference in sequence
    }
    labeled_image_ids = {int(value["image_id"]) for value in train["annotations"]}
    calibration = {
        **train,
        "images": [
            image for image in train["images"] if int(image["id"]) in labeled_image_ids
        ],
    }
    tasks = recipe.build_tasks(calibration, categories, CONDITIONS)
    remaining_counts: list[int] = []
    for image_id in sorted(labeled_image_ids):
        remaining_counts.extend(
            len(sequence)
            for sequence in references_without_target(references, image_id).values()
        )
    audit = {
        "train_annotation_sha256": base.sha256_file(train_path),
        "test_annotation_sha256": base.sha256_file(test_path),
        "support_images": len(calibration["images"]),
        "support_objects": len(train["annotations"]),
        "classes": len(categories),
        "tasks": len(tasks),
        "target_reference_image_overlap": 0,
        "minimum_available_references_after_target_exclusion": min(remaining_counts),
        "maximum_available_references_after_target_exclusion": max(remaining_counts),
    }
    context = DatasetContext(
        name=dataset.name,
        directory=dataset,
        output_directory=output_root / dataset.name,
        train_directory=train_directory,
        train=train,
        categories=categories,
        references=references,
        assets=assets,
        tasks=tasks,
        audit=audit,
    )
    first_image_id = min(labeled_image_ids)
    for condition in CONDITIONS:
        task = next(
            value
            for value in tasks
            if value.image_id == first_image_id and value.mode == condition.mode
        )
        messages, selected = task_messages(context, task)
        content = messages[0]["content"]
        if content[-2] != {"type": "text", "text": "TARGET IMAGE:"}:
            raise ValueError(f"Target marker order is invalid for {dataset.name}.")
        if content[-1].get("type") != "image_url":
            raise ValueError(f"Target image is not last for {dataset.name}.")
        if condition.box_count and "sparse positive exemplars" not in str(
            content[0].get("text", "")
        ):
            raise ValueError(f"Sparse-reference clause missing for {dataset.name}.")
        if any(
            reference.image_id == first_image_id
            for sequence in selected.values()
            for reference in sequence[: condition.box_count]
        ):
            raise ValueError(f"Support target leaked for {dataset.name}.")
    context.audit["validated_prompt_contracts"] = len(CONDITIONS)
    return context


def task_messages(
    context: DatasetContext, task: base.Task
) -> tuple[list[dict[str, Any]], dict[int, tuple[box_ablation.ReferenceBox, ...]]]:
    condition = CONDITION_BY_MODE[task.mode]
    references = references_without_target(context.references, task.image_id)
    if any(
        reference.image_id == task.image_id
        for sequence in references.values()
        for reference in sequence[: condition.box_count]
    ):
        raise AssertionError("A calibration target image leaked into its references.")
    messages = recipe.build_messages(
        task,
        condition,
        context.train_directory,
        context.categories,
        {},
        references,
        context.assets,
    )
    return messages, references


def record_path(context: DatasetContext, task: base.Task) -> Path:
    return context.output_directory / "records" / task.mode / f"{task.key}.json"


def record_usage(contexts: list[DatasetContext]) -> dict[str, int | float]:
    prompt = cached = completion = reasoning = 0
    for context in contexts:
        for task in context.tasks:
            record = base.load_record(record_path(context, task)) or {}
            if record.get("status") not in base.TERMINAL_STATUSES:
                continue
            usage = record.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            completion_details = usage.get("completion_tokens_details") or {}
            prompt += int(usage.get("prompt_tokens") or 0)
            cached += int(details.get("cached_tokens") or 0)
            completion += int(usage.get("completion_tokens") or 0)
            reasoning += int(completion_details.get("reasoning_tokens") or 0)
    cost = (
        (prompt - cached) * PRICES_PER_MILLION["uncached_prompt"]
        + cached * PRICES_PER_MILLION["implicit_cached_prompt"]
        + completion * PRICES_PER_MILLION["completion"]
    ) / 1_000_000
    return {
        "prompt_tokens": prompt,
        "cached_prompt_tokens": cached,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "estimated_usd": cost,
    }


def summarize_context(
    context: DatasetContext, minimum_gain_points: float
) -> tuple[dict[str, Any], int]:
    metrics_by_count: dict[int, dict[str, Any]] = {}
    statuses_by_count: dict[int, dict[str, int]] = {}
    unresolved = 0
    for count in COUNTS:
        mode = MODE_BY_COUNT[count]
        predictions: list[dict[str, Any]] = []
        statuses: dict[str, int] = {}
        for task in context.tasks:
            if task.mode != mode:
                continue
            record = base.load_record(record_path(context, task))
            status = record.get("status", "missing") if record else "missing"
            statuses[status] = statuses.get(status, 0) + 1
            if status in base.TERMINAL_STATUSES and record:
                predictions.extend(record.get("predictions", []))
            else:
                unresolved += 1
        statuses_by_count[count] = statuses
        metrics_by_count[count] = support.known_object_recall(
            context.train, predictions
        )
        base.atomic_write_json(
            context.output_directory / "metrics" / f"count_{count:02d}.json",
            {
                "count": count,
                "mode": mode,
                "statuses": statuses,
                "metrics": metrics_by_count[count],
            },
        )
    selected_count, trace = choose_count(metrics_by_count, minimum_gain_points)
    row = {
        "dataset": context.name,
        "selected_count": selected_count,
        "selected_mode": MODE_BY_COUNT[selected_count],
        "support_images": len(context.train["images"]),
        "support_objects": len(context.train["annotations"]),
        "calibration_scores": {
            str(count): {
                "recall50_95": 100
                * metrics_by_count[count]["class_macro_recall50_95"],
                "recall50": 100
                * metrics_by_count[count]["class_macro_recall50"],
            }
            for count in COUNTS
        },
        "selection_trace": trace,
        "statuses": {str(count): statuses for count, statuses in statuses_by_count.items()},
    }
    base.atomic_write_json(
        context.output_directory / "summary.json",
        {"audit": context.audit, "decision": row},
    )
    return row, unresolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--minimum-gain-points", type=float, default=2.0)
    parser.add_argument("--model", default="qwen3.8-max")
    parser.add_argument(
        "--base-url",
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--requests-per-minute", type=float, default=6750.0)
    parser.add_argument("--tokens-per-minute", type=float, default=900_000.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1 or args.minimum_gain_points < 0:
        raise ValueError("Concurrency must be positive and gain nonnegative.")
    if not args.prepare_only and not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is required.")

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_file = (output_root / ".run.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another process owns {output_root}.") from error

    available = {
        path.name: path
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "train/_annotations.coco.json").is_file()
    }
    selected_names = args.datasets or sorted(available)
    unknown = set(selected_names) - set(available)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    contexts = [
        prepare_context(available[name], output_root) for name in selected_names
    ]
    common_settings = {
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": 0.0,
        "vl_high_resolution_images": False,
        "timeout_seconds": args.timeout_seconds,
    }
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "dataset_root": str(dataset_root),
        "datasets": selected_names,
        "conditions": [recipe.condition_payload(value) for value in CONDITIONS],
        "common_settings": common_settings,
        "minimum_gain_points": args.minimum_gain_points,
        "routing_metric": "leave-one-support-image-out-known-object-class-macro-recall",
        "selection_policy": "material-primary-gain-and-nondecreasing-r50-v1",
        "reference_source": "RF20-VL-FSOD train objects only",
        "target_support_image_excluded_from_its_reference_pool": True,
        "test_images_sent": False,
        "test_annotations_used_for_routing": False,
        "audits": {context.name: context.audit for context in contexts},
        "concurrency": args.concurrency,
        "requests_per_minute": args.requests_per_minute,
        "tokens_per_minute": args.tokens_per_minute,
        "max_detections": 500,
    }
    recipe.write_or_validate_manifest(output_root / "run_manifest.json", manifest)
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return 0

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=common_settings["base_url"],
        timeout=args.timeout_seconds,
        max_retries=0,
    )
    limiter = base.SmoothDualRateLimiter(
        args.requests_per_minute, args.tokens_per_minute
    )
    pending: list[tuple[DatasetContext, base.Task]] = []
    for context in contexts:
        for task in context.tasks:
            condition = CONDITION_BY_MODE[task.mode]
            settings = recipe.condition_settings(condition, common_settings)
            messages, _ = task_messages(context, task)
            existing = base.load_record(record_path(context, task))
            if existing:
                terminal = base.terminalize_provider_failure(existing)
                if terminal is not existing:
                    base.atomic_write_json(record_path(context, task), terminal)
                    existing = terminal
            if existing and existing.get("status") in base.TERMINAL_STATUSES:
                expected = base.request_fingerprint(
                    task, base.request_summary(messages), settings
                )
                if (
                    existing.get("task_key") != task.key
                    or existing.get("request_fingerprint") != expected
                ):
                    raise ValueError(
                        f"Mismatched checkpoint: {context.name}/{task.key}"
                    )
                continue
            pending.append((context, task))

    def execute(context: DatasetContext, task: base.Task) -> dict[str, Any]:
        condition = CONDITION_BY_MODE[task.mode]
        settings = recipe.condition_settings(condition, common_settings)
        messages, references = task_messages(context, task)
        estimate = recipe.token_estimate(
            condition, len(context.categories), references
        )
        return base.execute_task(
            task,
            client,
            context.train_directory,
            context.categories,
            {},
            {},
            {},
            settings,
            args.max_retries,
            recipe.TaskRateLimiter(limiter, estimate),
            messages_override=messages,
        )

    completed = 0
    write_lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = {
            executor.submit(execute, context, task): (context, task)
            for context, task in pending
        }
        for future in concurrent.futures.as_completed(futures):
            context, task = futures[future]
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
                base.atomic_write_json(record_path(context, task), record)
                completed += 1
                if completed % 25 == 0 or completed == len(pending):
                    print(f"checkpoint {completed}/{len(pending)}", flush=True)

    rows = []
    unresolved = 0
    for context in contexts:
        row, current_unresolved = summarize_context(
            context, args.minimum_gain_points
        )
        rows.append(row)
        unresolved += current_unresolved
    usage = record_usage(contexts)
    summary = {
        "completed_at": base.utc_now(),
        "route": "support-calibrated-reference-count-v1",
        "dataset_count": len(rows),
        "minimum_gain_points": args.minimum_gain_points,
        "test_data_used_for_route": False,
        "usage": usage,
        "counts": {
            str(count): sum(row["selected_count"] == count for row in rows)
            for count in COUNTS
        },
        "rows": sorted(rows, key=lambda row: row["dataset"]),
    }
    base.atomic_write_json(output_root / "summary.json", summary)
    if unresolved == 0:
        base.atomic_write_json(
            output_root / "_SUCCESS.json",
            {
                "completed_at": base.utc_now(),
                "dataset_count": len(rows),
                "request_count": sum(len(context.tasks) for context in contexts),
            },
        )
    print(json.dumps(summary, indent=2))
    return 0 if unresolved == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
