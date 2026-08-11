#!/usr/bin/env python3
"""Evaluate semantic-label-free box exemplars with Qwen3.8-Max.

Every request is single-class, but the model is never shown that class's name.
The explicit conditions ask for objects of the same kind as the marked example.
The minimal conditions supply only the reference image/box payload, target image,
and a label-free output protocol. Parsed boxes are assigned to the task's hidden
category externally before standard COCO maxDets=500 scoring.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import json
import logging
import os
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_orion as base

MODEL_ID = "qwen3.8-max"
PROMPT_VERSION = "qwen3.8-max-exemplar-only-box-v1"
SUPPORTED_BOX_COUNTS = (1, 2, 5, 7, 10)
DEFAULT_BOX_COUNTS = (1, 2, 5)
BOX_COUNTS = DEFAULT_BOX_COUNTS
INSTRUCTIONS = ("explicit", "minimal")
REPRESENTATIONS = ("numeric", "drawn")
EXPLICIT_PROMPT = (
    "Find every object in the target image that is the same kind as the object "
    "marked in the reference image."
)
DEFAULT_DATASET = Path("RF100VL/rf20-vl-fsod/the-dreidel-project")
DEFAULT_OUTPUT = Path("qwen38-fsod-runs/dreidel-exemplar-only-box-v1")
TERMINAL_STATUSES = base.TERMINAL_STATUSES

LOGGER = logging.getLogger("qwen38_exemplar_only")


@dataclass(frozen=True)
class Condition:
    mode: str
    instruction: str
    representation: str
    box_count: int


def build_conditions() -> tuple[Condition, ...]:
    return tuple(
        Condition(
            mode=f"{instruction}_{representation}_b{count:02d}",
            instruction=instruction,
            representation=representation,
            box_count=count,
        )
        for instruction in INSTRUCTIONS
        for representation in REPRESENTATIONS
        for count in BOX_COUNTS
    )


CONDITIONS = build_conditions()
CONDITIONS_BY_MODE = {condition.mode: condition for condition in CONDITIONS}
MODES = tuple(CONDITIONS_BY_MODE)
ALL_MODES = tuple(
    f"{instruction}_{representation}_b{count:02d}"
    for instruction in INSTRUCTIONS
    for representation in REPRESENTATIONS
    for count in SUPPORTED_BOX_COUNTS
)


def configure_box_counts(box_counts: Sequence[int]) -> None:
    """Configure one invocation's factorial while keeping resumable manifests exact."""

    normalized = tuple(dict.fromkeys(int(count) for count in box_counts))
    if not normalized:
        raise ValueError("At least one box count is required.")
    unsupported = set(normalized) - set(SUPPORTED_BOX_COUNTS)
    if unsupported:
        raise ValueError(f"Unsupported box counts: {sorted(unsupported)}")
    global BOX_COUNTS, CONDITIONS, CONDITIONS_BY_MODE, MODES
    BOX_COUNTS = normalized
    CONDITIONS = build_conditions()
    CONDITIONS_BY_MODE = {condition.mode: condition for condition in CONDITIONS}
    MODES = tuple(CONDITIONS_BY_MODE)


def generic_output_contract() -> str:
    return (
        "Return only a JSON list exactly like "
        '[{"bbox_2d":[x1,y1,x2,y2],"label":"object"}]. '
        "Use XYXY integer coordinates normalized independently from 0 to 1000 "
        "relative to the target image, with the origin at top-left. Return [] "
        "if none are present."
    )


def minimal_output_protocol() -> str:
    # Deliberately contains no class name, visual-concept description, or
    # find/detect/same-kind instruction. It exists only to make coordinates
    # machine-scorable and declares the last image as the coordinate frame.
    return (
        'OUTPUT(last image): [{"bbox_2d":[x1,y1,x2,y2]}] | []; '
        "XYXY integers normalized 0..1000."
    )


def build_tasks(
    test: dict[str, Any], categories: dict[int, str]
) -> list[base.Task]:
    tasks: list[base.Task] = []
    for condition in CONDITIONS:
        for image in sorted(test["images"], key=lambda value: int(value["id"])):
            for category_id, category_name in categories.items():
                tasks.append(
                    base.Task(
                        mode=condition.mode,
                        image_id=int(image["id"]),
                        file_name=str(image["file_name"]),
                        width=int(image["width"]),
                        height=int(image["height"]),
                        category_id=category_id,
                        category_name=category_name,
                    )
                )
    if len({task.key for task in tasks}) != len(tasks):
        raise ValueError("Generated task keys are not unique.")
    return tasks


def build_messages(
    task: base.Task,
    test_directory: Path,
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]],
    assets: dict[tuple[int, int], dict[str, Path]],
) -> list[dict[str, Any]]:
    if task.category_id is None:
        raise ValueError("Exemplar-only evaluation requires single-class tasks.")
    condition = CONDITIONS_BY_MODE[task.mode]
    target = test_directory / task.file_name
    if not target.is_file():
        raise FileNotFoundError(target)

    content: list[dict[str, Any]] = []
    if condition.instruction == "explicit":
        content.append(
            {
                "type": "text",
                "text": f"{EXPLICIT_PROMPT} {generic_output_contract()}",
            }
        )

    for reference in references[task.category_id][: condition.box_count]:
        if condition.representation == "numeric":
            payload = {
                "bbox_2d": list(reference.bbox_xyxy_1000),
            }
            content.extend(
                [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base.data_url(
                                assets[(task.category_id, reference.rank)]["source"]
                            )
                        },
                    },
                    {
                        "type": "text",
                        "text": json.dumps(payload, separators=(",", ":")),
                    },
                ]
            )
        elif condition.representation == "drawn":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base.data_url(
                            assets[(task.category_id, reference.rank)]["drawn"]
                        )
                    },
                }
            )
        else:
            raise ValueError(f"Unknown representation: {condition.representation}")

    if condition.instruction == "explicit":
        content.append({"type": "text", "text": "TARGET IMAGE:"})
    content.append(
        {"type": "image_url", "image_url": {"url": base.data_url(target)}}
    )
    if condition.instruction == "minimal":
        content.append({"type": "text", "text": minimal_output_protocol()})
    return [{"role": "user", "content": content}]


def build_token_estimates() -> dict[str, int]:
    return {
        condition.mode: 3_000 * (condition.box_count + 1) + 2_500
        for condition in CONDITIONS
    }


class TaskRateLimiter:
    def __init__(self, shared: base.SmoothDualRateLimiter, estimated_tokens: int):
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
        records: list[dict[str, Any]] = []
        statuses: dict[str, int] = {}
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
        base.atomic_write_json(
            output_directory / "predictions" / f"{condition.mode}.json",
            predictions,
        )
        metrics = base.score_coco(annotation_path, predictions) if complete else None
        summary = {
            "condition": asdict(condition),
            "complete": complete,
            "task_count": len(tasks),
            "calls_per_image": class_count,
            "statuses": statuses,
            "prediction_count": len(predictions),
            "usage": box_ablation._usage_summary(records),
            "metrics": metrics,
        }
        modes[condition.mode] = summary
        base.atomic_write_json(
            output_directory / "metrics" / f"{condition.mode}.json", summary
        )
        rows.append(
            {
                "mode": condition.mode,
                "instruction": condition.instruction,
                "representation": condition.representation,
                "boxes_per_class": condition.box_count,
                "calls_per_image": class_count,
                "task_count": len(tasks),
                "complete": complete,
                "mAP50_95": metrics["AP"] * 100 if metrics else None,
                "mAP50": metrics["AP50"] * 100 if metrics else None,
                "model_failures": statuses.get("model_failure", 0),
                "errors": statuses.get("error", 0) + statuses.get("missing", 0),
                "mean_inference_seconds": summary["usage"]["mean_inference_seconds"],
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
    base.atomic_write_json(
        output_directory / "comparison_summary.json",
        {"updated_at": base.utc_now(), "rows": rows},
    )
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
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]],
    train_directory: Path,
) -> None:
    expected = {
        "prompt_version": PROMPT_VERSION,
        "configuration": configuration,
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "class_names_exposed_to_model": False,
        "minimal_mode_semantic_instruction": False,
        "explicit_prompt": EXPLICIT_PROMPT,
        "reference_selection": {
            "method": "largest-relative-area-then-greedy-crop-diversity-v1",
            "nested_counts": list(BOX_COUNTS),
            "one_box_per_distinct_train_image": True,
            "classes": {
                str(category_id): [
                    {
                        **asdict(reference),
                        "source_sha256": base.sha256_file(
                            train_directory / reference.file_name
                        ),
                    }
                    for reference in sequence[: max(BOX_COUNTS)]
                ]
                for category_id, sequence in references.items()
            },
        },
    }
    expected = json.loads(json.dumps(expected, ensure_ascii=False))
    existing = base.load_record(path)
    if existing:
        if {key: existing.get(key) for key in expected} != expected:
            raise ValueError(f"Existing manifest does not match this experiment: {path}")
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
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--box-counts",
        nargs="+",
        type=int,
        choices=SUPPORTED_BOX_COUNTS,
        default=list(DEFAULT_BOX_COUNTS),
    )
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES)
    parser.add_argument("--image-ids", nargs="+", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-per-mode", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_box_counts(args.box_counts)
    if args.concurrency < 1 or args.max_retries < 0:
        raise ValueError("Concurrency must be positive and retries nonnegative.")
    if args.requests_per_minute <= 0 or args.tokens_per_minute <= 0:
        raise ValueError("RPM and TPM limits must be positive.")
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
        raise RuntimeError(f"Another process owns {output_directory}.") from error

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
    references = box_ablation.select_reference_sequences(
        train,
        train_directory,
        required_count=max(BOX_COUNTS),
    )
    assets = box_ablation.prepare_reference_assets(
        train_directory, output_directory / "references", references
    )
    token_estimates = build_token_estimates()
    all_tasks = build_tasks(test, categories)
    selected_modes = set(args.modes or MODES)
    unknown_selected_modes = selected_modes - set(MODES)
    if unknown_selected_modes:
        raise ValueError(
            "Selected modes do not belong to the configured box counts: "
            f"{sorted(unknown_selected_modes)}"
        )
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
        "force_single_category_labels": True,
    }
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
    )
    base.atomic_write_json(
        output_directory / "progress.json",
        summarize_records(all_tasks, output_directory),
    )
    LOGGER.info(
        "Prepared %d requests across %d conditions, %d images, and %d hidden classes.",
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
            messages = build_messages(task, test_directory, references, assets)
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
        messages = build_messages(task, test_directory, references, assets)
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
            force_single_category_labels=True,
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
    unresolved = selected_progress["total"]["error"] + selected_progress["total"][
        "pending"
    ]
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
