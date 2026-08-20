#!/usr/bin/env python3
"""Route Qwen3.8-Max zero/one-shot detection using RF20 FSOD support objects.

One train-only object per class is kept as the exact visual reference used by
the established RF20 one-shot recipe.  Images containing any chosen reference
are excluded from calibration.  Both clean detector branches are then run on
the remaining support images.

RF20 support annotations are object-level and may be non-exhaustive.  Ordinary
COCO AP would therefore count valid detections of unlabeled objects as false
positives.  This evaluator instead reports *known-object recall*: predictions
are greedily matched to labeled held-out support objects of the same class,
while unmatched predictions are ignored.  Test annotations are never read for
routing (except category-schema validation), and test images are never sent.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_orion as base
import evaluate_qwen38_recipe as recipe

NAMES_MODE = "names_multi"
REFERENCE_MODE = "numeric_prediction_b01_multi"
PROMPT_VERSION = "qwen3.8-max-fsod-support-calibrated-router-v2"
THRESHOLDS = tuple(round(0.5 + 0.05 * index, 2) for index in range(10))
CONDITIONS = (
    recipe.Condition(
        mode=NAMES_MODE,
        formulation="multi",
        semantics="class_names",
        representation="none",
        box_count=0,
        reasoning_effort="none",
        seed=1234,
    ),
    recipe.Condition(
        mode=REFERENCE_MODE,
        formulation="multi",
        semantics="class_names",
        representation="numeric_prediction",
        box_count=1,
        reasoning_effort="none",
        seed=1234,
    ),
)
CONDITION_BY_MODE = {condition.mode: condition for condition in CONDITIONS}


@dataclass
class DatasetContext:
    name: str
    directory: Path
    output_directory: Path
    train: dict[str, Any]
    categories: dict[int, str]
    calibration: dict[str, Any]
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]]
    assets: dict[tuple[int, int], dict[str, Path]]
    tasks: list[base.Task]


def build_calibration_split(
    train: dict[str, Any],
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exclude every source image used by the multi-class reference prompt."""

    reference_image_ids = {
        sequence[0].image_id for sequence in references.values() if sequence
    }
    if len(reference_image_ids) == 0:
        raise ValueError("No reference source images were selected.")
    annotations = [
        annotation
        for annotation in train["annotations"]
        if int(annotation["image_id"]) not in reference_image_ids
    ]
    calibration_image_ids = {int(value["image_id"]) for value in annotations}
    images = [
        image for image in train["images"] if int(image["id"]) in calibration_image_ids
    ]
    counts = {
        category_id: sum(
            int(annotation["category_id"]) == category_id
            for annotation in annotations
        )
        for category_id in references
    }
    split = {**train, "images": images, "annotations": annotations}
    audit = {
        "reference_image_ids": sorted(reference_image_ids),
        "calibration_image_count": len(images),
        "calibration_object_count": len(annotations),
        "calibration_objects_per_class": {
            str(category_id): count for category_id, count in counts.items()
        },
        "classes_without_calibration_objects": [
            category_id for category_id, count in counts.items() if count == 0
        ],
        "reference_calibration_image_overlap": sorted(
            reference_image_ids & calibration_image_ids
        ),
    }
    if audit["reference_calibration_image_overlap"]:
        raise AssertionError("A reference source image leaked into calibration.")
    if not images or not annotations:
        raise ValueError("No support data remains after excluding references.")
    return split, audit


def intersection_over_union(left: Sequence[float], right: Sequence[float]) -> float:
    lx, ly, lw, lh = (float(value) for value in left)
    rx, ry, rw, rh = (float(value) for value in right)
    left_x2, left_y2 = lx + max(0.0, lw), ly + max(0.0, lh)
    right_x2, right_y2 = rx + max(0.0, rw), ry + max(0.0, rh)
    width = max(0.0, min(left_x2, right_x2) - max(lx, rx))
    height = max(0.0, min(left_y2, right_y2) - max(ly, ry))
    intersection = width * height
    union = max(0.0, lw) * max(0.0, lh) + max(0.0, rw) * max(0.0, rh) - intersection
    return intersection / union if union > 0 else 0.0


def greedy_matches(
    annotations: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    threshold: float,
) -> int:
    pairs = sorted(
        (
            intersection_over_union(annotation["bbox"], prediction["bbox"]),
            annotation_index,
            prediction_index,
        )
        for annotation_index, annotation in enumerate(annotations)
        for prediction_index, prediction in enumerate(predictions)
    )
    used_annotations: set[int] = set()
    used_predictions: set[int] = set()
    matched = 0
    for overlap, annotation_index, prediction_index in reversed(pairs):
        if overlap < threshold:
            break
        if annotation_index in used_annotations or prediction_index in used_predictions:
            continue
        used_annotations.add(annotation_index)
        used_predictions.add(prediction_index)
        matched += 1
    return matched


def known_object_recall(
    calibration: dict[str, Any], predictions: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    ground_truth: dict[tuple[int, int], list[dict[str, Any]]] = {}
    detections: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for annotation in calibration["annotations"]:
        key = (int(annotation["image_id"]), int(annotation["category_id"]))
        ground_truth.setdefault(key, []).append(annotation)
    for prediction in predictions:
        key = (int(prediction["image_id"]), int(prediction["category_id"]))
        detections.setdefault(key, []).append(prediction)

    category_ids = sorted({category_id for _, category_id in ground_truth})
    per_threshold: dict[str, Any] = {}
    per_class_matches = {
        category_id: {threshold: 0 for threshold in THRESHOLDS}
        for category_id in category_ids
    }
    per_class_totals = {
        category_id: sum(
            len(values)
            for (image_id, current_category_id), values in ground_truth.items()
            if current_category_id == category_id
        )
        for category_id in category_ids
    }
    for threshold in THRESHOLDS:
        matches = 0
        for key, annotations in ground_truth.items():
            current = greedy_matches(annotations, detections.get(key, []), threshold)
            matches += current
            per_class_matches[key[1]][threshold] += current
        total = len(calibration["annotations"])
        class_recalls = [
            per_class_matches[category_id][threshold] / per_class_totals[category_id]
            for category_id in category_ids
            if per_class_totals[category_id]
        ]
        per_threshold[str(threshold)] = {
            "matched": matches,
            "known_objects": total,
            "micro_recall": matches / total,
            "class_macro_recall": fmean(class_recalls),
        }

    macro_values = [
        per_threshold[str(threshold)]["class_macro_recall"] for threshold in THRESHOLDS
    ]
    micro_values = [
        per_threshold[str(threshold)]["micro_recall"] for threshold in THRESHOLDS
    ]
    per_class = {}
    for category_id in category_ids:
        values = [
            per_class_matches[category_id][threshold] / per_class_totals[category_id]
            for threshold in THRESHOLDS
        ]
        per_class[str(category_id)] = {
            "known_objects": per_class_totals[category_id],
            "recall50_95": fmean(values),
            "recall50": values[0],
        }
    return {
        "known_object_count": len(calibration["annotations"]),
        "evaluated_class_count": len(category_ids),
        "class_macro_recall50_95": fmean(macro_values),
        "class_macro_recall50": macro_values[0],
        "micro_recall50_95": fmean(micro_values),
        "micro_recall50": micro_values[0],
        "per_threshold": per_threshold,
        "per_class": per_class,
        "unmatched_predictions_ignored": True,
    }


def choose_route(
    names: dict[str, Any], references: dict[str, Any], minimum_gain_points: float
) -> tuple[str, dict[str, float]]:
    deltas = {
        "class_macro_recall50_95": 100
        * (
            references["class_macro_recall50_95"]
            - names["class_macro_recall50_95"]
        ),
        "class_macro_recall50": 100
        * (references["class_macro_recall50"] - names["class_macro_recall50"]),
    }
    # References cost more, so require a material primary gain and no loss at
    # IoU=.50.  This policy is fixed before inspecting held-out test scores.
    selected = (
        REFERENCE_MODE
        if deltas["class_macro_recall50_95"] >= minimum_gain_points
        and deltas["class_macro_recall50"] >= 0.0
        else NAMES_MODE
    )
    return selected, deltas


def record_path(context: DatasetContext, task: base.Task) -> Path:
    return context.output_directory / "records" / task.mode / f"{task.key}.json"


def prepare_context(dataset: Path, output_root: Path) -> tuple[DatasetContext, dict[str, Any]]:
    train_directory = dataset / "train"
    train = base.load_coco(train_directory / "_annotations.coco.json")
    test = base.load_coco(dataset / "test/_annotations.coco.json")
    base.validate_split_isolation(train, test)
    categories = base.categories_by_id(train)
    if categories != base.categories_by_id(test):
        raise ValueError(f"Train/test categories differ for {dataset.name}.")
    references = box_ablation.select_reference_sequences(
        train,
        train_directory,
        required_count=1,
        distinct_images_only=True,
        first_strategy="largest-relative-area",
    )
    calibration, audit = build_calibration_split(train, references)
    output_directory = output_root / dataset.name
    assets = box_ablation.prepare_reference_assets(
        train_directory, output_directory / "references", references
    )
    tasks = recipe.build_tasks(calibration, categories, CONDITIONS)
    context = DatasetContext(
        name=dataset.name,
        directory=dataset,
        output_directory=output_directory,
        train=train,
        categories=categories,
        calibration=calibration,
        references=references,
        assets=assets,
        tasks=tasks,
    )
    return context, audit


def load_test_scores(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None:
        return {}
    with path.open(encoding="utf-8", newline="") as file:
        return {
            (row["dataset"], row["mode"]): row
            for row in csv.DictReader(file)
            if row["mode"] in {NAMES_MODE, REFERENCE_MODE}
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--fixed-test-summary", type=Path)
    parser.add_argument("--minimum-gain-points", type=float, default=2.0)
    parser.add_argument("--model", default="qwen3.8-max")
    parser.add_argument(
        "--base-url",
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--requests-per-minute", type=float, default=570.0)
    parser.add_argument("--tokens-per-minute", type=float, default=900_000.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1 or args.minimum_gain_points < 0:
        raise ValueError("Concurrency must be positive and gain margin nonnegative.")
    if not args.prepare_only and not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is required.")

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_dir.resolve()
    available = {
        path.name: path
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "train/_annotations.coco.json").is_file()
    }
    selected_names = args.datasets or sorted(available)
    unknown = set(selected_names) - set(available)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    output_root.mkdir(parents=True, exist_ok=True)
    contexts: dict[str, DatasetContext] = {}
    audits: dict[str, Any] = {}
    for name in selected_names:
        contexts[name], audits[name] = prepare_context(available[name], output_root)

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
        "routing_metric": "federated-known-object-class-macro-recall",
        "reference_source": "RF20-VL-FSOD train objects only",
        "test_images_sent": False,
        "test_annotations_used_for_routing": False,
        "audits": audits,
    }
    recipe.write_or_validate_manifest(output_root / "run_manifest.json", manifest)
    for context in contexts.values():
        base.atomic_write_json(
            context.output_directory / "calibration_ground_truth.json",
            context.calibration,
        )
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
    limiter = base.SmoothDualRateLimiter(args.requests_per_minute, args.tokens_per_minute)
    pending: list[tuple[DatasetContext, base.Task]] = []
    for context in contexts.values():
        for task in context.tasks:
            condition = CONDITION_BY_MODE[task.mode]
            settings = recipe.condition_settings(condition, common_settings)
            messages = recipe.build_messages(
                task,
                condition,
                context.directory / "train",
                context.categories,
                {},
                context.references,
                context.assets,
            )
            record = base.load_record(record_path(context, task))
            if record and record.get("status") in base.TERMINAL_STATUSES:
                expected = base.request_fingerprint(
                    task, base.request_summary(messages), settings
                )
                if record.get("request_fingerprint") != expected:
                    raise ValueError(f"Mismatched checkpoint: {context.name}/{task.key}")
                continue
            pending.append((context, task))

    def execute(context: DatasetContext, task: base.Task) -> dict[str, Any]:
        condition = CONDITION_BY_MODE[task.mode]
        settings = recipe.condition_settings(condition, common_settings)
        messages = recipe.build_messages(
            task,
            condition,
            context.directory / "train",
            context.categories,
            {},
            context.references,
            context.assets,
        )
        estimate = recipe.token_estimate(
            condition, len(context.categories), context.references
        )
        return base.execute_task(
            task,
            client,
            context.directory / "train",
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
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
                    print(f"checkpoint {completed}/{len(pending)}")

    test_scores = load_test_scores(
        args.fixed_test_summary.resolve() if args.fixed_test_summary else None
    )
    rows = []
    for context in contexts.values():
        mode_scores = {}
        statuses = {}
        for mode in (NAMES_MODE, REFERENCE_MODE):
            predictions = []
            mode_statuses: dict[str, int] = {}
            for task in context.tasks:
                if task.mode != mode:
                    continue
                record = base.load_record(record_path(context, task))
                status = record.get("status", "missing") if record else "missing"
                mode_statuses[status] = mode_statuses.get(status, 0) + 1
                if record and status in base.TERMINAL_STATUSES:
                    predictions.extend(record.get("predictions", []))
            statuses[mode] = mode_statuses
            mode_scores[mode] = known_object_recall(context.calibration, predictions)
            base.atomic_write_json(
                context.output_directory / "metrics" / f"{mode}.json",
                {"statuses": mode_statuses, "metrics": mode_scores[mode]},
            )
        selected, deltas = choose_route(
            mode_scores[NAMES_MODE],
            mode_scores[REFERENCE_MODE],
            args.minimum_gain_points,
        )
        row: dict[str, Any] = {
            "dataset": context.name,
            "calibration_images": len(context.calibration["images"]),
            "calibration_objects": len(context.calibration["annotations"]),
            "calibration_classes": mode_scores[NAMES_MODE]["evaluated_class_count"],
            "total_classes": len(context.categories),
            "names_recall50_95": 100
            * mode_scores[NAMES_MODE]["class_macro_recall50_95"],
            "names_recall50": 100 * mode_scores[NAMES_MODE]["class_macro_recall50"],
            "reference_recall50_95": 100
            * mode_scores[REFERENCE_MODE]["class_macro_recall50_95"],
            "reference_recall50": 100
            * mode_scores[REFERENCE_MODE]["class_macro_recall50"],
            "support_delta_recall50_95": deltas["class_macro_recall50_95"],
            "support_delta_recall50": deltas["class_macro_recall50"],
            "selected_mode": selected,
        }
        names_test = test_scores.get((context.name, NAMES_MODE))
        reference_test = test_scores.get((context.name, REFERENCE_MODE))
        if names_test and reference_test:
            row.update(
                {
                    "test_delta_mAP50_95": float(reference_test["mAP50_95"])
                    - float(names_test["mAP50_95"]),
                    "test_delta_mAP50": float(reference_test["mAP50"])
                    - float(names_test["mAP50"]),
                    "test_primary_winner": (
                        REFERENCE_MODE
                        if float(reference_test["mAP50_95"])
                        > float(names_test["mAP50_95"])
                        else NAMES_MODE
                    ),
                    "router_primary_correct": selected
                    == (
                        REFERENCE_MODE
                        if float(reference_test["mAP50_95"])
                        > float(names_test["mAP50_95"])
                        else NAMES_MODE
                    ),
                }
            )
        rows.append(row)
        base.atomic_write_json(
            context.output_directory / "summary.json",
            {
                "dataset": context.name,
                "audit": audits[context.name],
                "modes": mode_scores,
                "decision": row,
                "statuses": statuses,
            },
        )

    summary = {
        "completed_at": base.utc_now(),
        "dataset_count": len(rows),
        "minimum_gain_points": args.minimum_gain_points,
        "rows": rows,
    }
    if all("router_primary_correct" in row for row in rows):
        summary["router_primary_accuracy"] = fmean(
            float(row["router_primary_correct"]) for row in rows
        )
    base.atomic_write_json(output_root / "summary.json", summary)
    csv_path = output_root / "per_dataset.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    unresolved = sum(
        1
        for context in contexts.values()
        for task in context.tasks
        if (base.load_record(record_path(context, task)) or {}).get("status")
        not in base.TERMINAL_STATUSES
    )
    if unresolved == 0:
        base.atomic_write_json(
            output_root / "_SUCCESS.json",
            {"completed_at": base.utc_now(), "dataset_count": len(rows)},
        )
    print(json.dumps(summary, indent=2))
    return 0 if unresolved == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
