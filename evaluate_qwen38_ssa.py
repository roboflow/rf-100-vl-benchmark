#!/usr/bin/env python3
"""Collect and evaluate a clean-trunk sequential-support Qwen3.8 curve.

This is the small-scale and resumable implementation of Sequential Support
Acquisition (SSA). A support image is predicted before its gold annotations
enter the conversation. Model predictions are always made on disposable
branches; only official support images and gold annotations enter the trunk.

The routing signal is train-only class-macro known-object recall averaged over
IoU 0.50:0.95. Final prefix quality is evaluated with standard COCO mAP50-95
and mAP50 on untouched test images. Unmatched support predictions are ignored
because RF20-VL-FSOD support annotations can be sparse.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import json
import os
import random
import threading
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import evaluate_qwen38_orion as base
import evaluate_qwen38_recipe as recipe
import evaluate_qwen38_support_calibrated_router as support
from aggregate_qwen38_rf20 import PRICES_PER_MILLION, PRICING_SOURCE

MODEL_ID = "qwen3.8-max"
PROMPT_VERSION = "qwen3.8-max-ssa-clean-trunk-v1"
MODE = "ssa_clean_trunk"
TERMINAL_STATUSES = base.TERMINAL_STATUSES


def annotation_json(
    annotations: Sequence[dict[str, Any]],
    image: dict[str, Any],
    categories: dict[int, str],
) -> str:
    detections = []
    for annotation in sorted(annotations, key=lambda value: int(value["id"])):
        category_id = int(annotation["category_id"])
        detections.append(
            (
                base.annotation_xywh_to_normalized_xyxy(
                    annotation["bbox"], int(image["width"]), int(image["height"])
                ),
                categories[category_id],
            )
        )
    return recipe.detection_list_json(detections)


def target_message(
    image: dict[str, Any], image_directory: Path, categories: dict[int, str]
) -> dict[str, Any]:
    task = base.Task(
        mode=MODE,
        image_id=int(image["id"]),
        file_name=str(image["file_name"]),
        width=int(image["width"]),
        height=int(image["height"]),
    )
    condition = recipe.Condition(
        mode=MODE,
        formulation="multi",
        semantics="class_names",
        representation="none",
        box_count=0,
        reasoning_effort="none",
        seed=1234,
    )
    return recipe.build_messages(
        task, condition, image_directory, categories, {}, {}, {}
    )[0]


def support_user_message(image: dict[str, Any], train_directory: Path) -> dict[str, Any]:
    image_path = train_directory / str(image["file_name"])
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "LABELED SUPPORT IMAGE. Learn the dataset-specific visual and "
                    "annotation conventions from the following gold assistant "
                    "response. Its boxes are sparse positive annotations: treat "
                    "all unmarked objects and regions as unlabeled, not as negative "
                    "examples or exhaustive annotations. Boxes use the same "
                    "normalized 0-1000 XYXY bbox_2d JSON schema as final detections."
                ),
            },
            {"type": "image_url", "image_url": {"url": base.data_url(image_path)}},
        ],
    }


def gold_assistant_message(
    image: dict[str, Any],
    annotations: Sequence[dict[str, Any]],
    categories: dict[int, str],
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": annotation_json(annotations, image, categories),
    }


def support_order(train: dict[str, Any], seed: int) -> list[int]:
    labeled = sorted({int(value["image_id"]) for value in train["annotations"]})
    random.Random(seed).shuffle(labeled)
    return labeled


def annotations_by_image(train: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in train["annotations"]:
        result[int(annotation["image_id"])].append(annotation)
    return dict(result)


def build_trunk(
    prefix_image_ids: Sequence[int],
    train: dict[str, Any],
    train_directory: Path,
    categories: dict[int, str],
) -> list[dict[str, Any]]:
    images = {int(value["id"]): value for value in train["images"]}
    annotations = annotations_by_image(train)
    trunk: list[dict[str, Any]] = []
    for image_id in prefix_image_ids:
        if image_id not in annotations:
            raise ValueError(f"Support image {image_id} has no gold annotations.")
        trunk.extend(
            [
                support_user_message(images[image_id], train_directory),
                gold_assistant_message(
                    images[image_id], annotations[image_id], categories
                ),
            ]
        )
    assert_clean_trunk(trunk)
    return trunk


def build_branch(
    prefix_image_ids: Sequence[int],
    target: dict[str, Any],
    train: dict[str, Any],
    train_directory: Path,
    target_directory: Path,
    categories: dict[int, str],
) -> list[dict[str, Any]]:
    if int(target["id"]) in set(prefix_image_ids) and target_directory == train_directory:
        raise ValueError("Support target leaked into its own prequential prefix.")
    messages = build_trunk(prefix_image_ids, train, train_directory, categories)
    messages.append(target_message(target, target_directory, categories))
    return messages


def assert_clean_trunk(messages: Sequence[dict[str, Any]]) -> None:
    if len(messages) % 2:
        raise ValueError("A clean trunk must contain complete user/gold pairs.")
    for index in range(0, len(messages), 2):
        user = messages[index]
        gold = messages[index + 1]
        if user.get("role") != "user" or gold.get("role") != "assistant":
            raise ValueError("Clean trunk roles must alternate user/assistant.")
        content = gold.get("content")
        if not isinstance(content, str):
            raise TypeError("Gold assistant output must be canonical JSON text.")
        value = json.loads(content)
        if not isinstance(value, list) or any(
            not isinstance(item, dict)
            or list(item) != ["bbox_2d", "label"]
            for item in value
        ):
            raise ValueError("Noncanonical assistant content entered the clean trunk.")


def prefix_object_counts(
    prefix_image_ids: Sequence[int], train: dict[str, Any]
) -> dict[str, int]:
    selected = set(prefix_image_ids)
    counts = Counter(
        int(annotation["category_id"])
        for annotation in train["annotations"]
        if int(annotation["image_id"]) in selected
    )
    return {str(category_id): counts[category_id] for category_id in sorted(counts)}


def one_image_calibration(train: dict[str, Any], image_id: int) -> dict[str, Any]:
    return {
        **train,
        "images": [
            image for image in train["images"] if int(image["id"]) == image_id
        ],
        "annotations": [
            value
            for value in train["annotations"]
            if int(value["image_id"]) == image_id
        ],
    }


def support_metrics(
    train: dict[str, Any], image_id: int, predictions: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    calibration = one_image_calibration(train, image_id)
    metrics = support.known_object_recall(calibration, predictions)
    prediction_count = sum(
        int(value["image_id"]) == image_id for value in predictions
    )
    matched50 = int(metrics["per_threshold"]["0.5"]["matched"])
    # This is intentionally diagnostic only. Unmatched predictions may be
    # valid objects omitted by the sparse FSOD support annotations.
    apparent_precision50 = matched50 / prediction_count if prediction_count else None
    return {
        **metrics,
        "prediction_count": prediction_count,
        "apparent_sparse_precision50": apparent_precision50,
        "precision_is_valid_for_routing": False,
    }


def paired_delta(
    branch: dict[str, Any], zero: dict[str, Any]
) -> dict[str, float]:
    return {
        "class_macro_recall50_95": 100
        * (
            branch["class_macro_recall50_95"]
            - zero["class_macro_recall50_95"]
        ),
        "class_macro_recall50": 100
        * (branch["class_macro_recall50"] - zero["class_macro_recall50"]),
    }


def simulate_best_prefix(
    curve: Sequence[dict[str, Any]], *, window: int = 2, epsilon: float = 2.0
) -> dict[str, Any]:
    points = [row for row in curve if int(row["prefix_images"]) > 0]
    if not points:
        return {"selected_prefix": 0, "reason": "no_informative_points"}
    values = [float(row["delta"]["class_macro_recall50_95"]) for row in points]
    smoothed = []
    for index, value in enumerate(values):
        start = max(0, index - window + 1)
        smoothed.append(fmean(values[start : index + 1]))
    best_index = max(range(len(points)), key=lambda index: smoothed[index])
    if smoothed[best_index] < epsilon:
        selected = 0
        reason = "no_material_support_gain"
    else:
        selected = int(points[best_index]["prefix_images"])
        reason = "best_smoothed_support_delta"
    return {
        "selected_prefix": selected,
        "reason": reason,
        "window": window,
        "epsilon": epsilon,
        "smoothed_delta_recall50_95": smoothed,
        "candidate_prefixes": [int(row["prefix_images"]) for row in points],
    }


class TokenLimiter:
    def __init__(self, shared: base.SmoothDualRateLimiter, image_count: int):
        self.shared = shared
        self.estimate = 3_000 * image_count + 2_500

    def acquire(self, _unused: int) -> None:
        self.shared.acquire(self.estimate)


def task_for_image(mode: str, image: dict[str, Any]) -> base.Task:
    return base.Task(
        mode=mode,
        image_id=int(image["id"]),
        file_name=str(image["file_name"]),
        width=int(image["width"]),
        height=int(image["height"]),
    )


def execute_messages(
    *,
    task: base.Task,
    messages: list[dict[str, Any]],
    client: Any,
    image_directory: Path,
    categories: dict[int, str],
    settings: dict[str, Any],
    max_retries: int,
    limiter: base.SmoothDualRateLimiter,
) -> dict[str, Any]:
    return base.execute_task(
        task,
        client,
        image_directory,
        categories,
        {},
        {},
        {},
        settings,
        max_retries,
        TokenLimiter(limiter, sum(
            1
            for message in messages
            for part in (
                message.get("content")
                if isinstance(message.get("content"), list)
                else []
            )
            if isinstance(part, dict) and part.get("type") == "image_url"
        )),
        messages_override=messages,
    )


def record_cost(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    prompt = cached = completion = reasoning = 0
    for record in records:
        usage = record.get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        cached += int(prompt_details.get("cached_tokens") or 0)
        reasoning += int(completion_details.get("reasoning_tokens") or 0)
    estimated = (
        (prompt - cached) * PRICES_PER_MILLION["uncached_prompt"]
        + cached * PRICES_PER_MILLION["implicit_cached_prompt"]
        + completion * PRICES_PER_MILLION["completion"]
    ) / 1_000_000
    return {
        "prompt_tokens": prompt,
        "cached_prompt_tokens": cached,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "estimated_usd": estimated,
    }


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_terminal(path: Path, expected_fingerprint: str) -> dict[str, Any] | None:
    record = base.load_record(path)
    if not record or record.get("status") not in TERMINAL_STATUSES:
        return None
    if record.get("request_fingerprint") != expected_fingerprint:
        raise ValueError(f"Checkpoint fingerprint mismatch: {path}")
    return record


def expected_fingerprint(
    task: base.Task, messages: list[dict[str, Any]], settings: dict[str, Any]
) -> str:
    return base.request_fingerprint(task, base.request_summary(messages), settings)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Support-image order seed. This does not change generation sampling.",
    )
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=1234,
        help="Fixed generation seed, independent from the support-image order.",
    )
    parser.add_argument(
        "--zero-cache-dir",
        type=Path,
        help="Optional dataset-level cache shared by all support-order seeds.",
    )
    parser.add_argument("--max-support-turns", type=int)
    parser.add_argument("--test-prefixes", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--test-image-limit", type=int)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--base-url",
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--requests-per-minute", type=float, default=6750.0)
    parser.add_argument("--tokens-per-minute", type=float, default=900_000.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--adaptation-only",
        action="store_true",
        help="Collect the full support curve but skip all test-image requests.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.concurrency < 1 or args.max_retries < 0:
        raise ValueError("Concurrency must be positive and retries nonnegative.")
    if args.max_support_turns is not None and args.max_support_turns < 1:
        raise ValueError("max-support-turns must be positive.")
    if not args.prepare_only and not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is required for inference.")

    dataset = args.dataset_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_file = (output / ".run.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another process owns {output}.") from error

    train_directory = dataset / "train"
    test_directory = dataset / "test"
    train_path = train_directory / "_annotations.coco.json"
    test_path = test_directory / "_annotations.coco.json"
    train = base.load_coco(train_path)
    test = base.load_coco(test_path)
    base.validate_split_isolation(train, test)
    categories = base.categories_by_id(train)
    if categories != base.categories_by_id(test):
        raise ValueError("Train/test categories differ.")
    train_images = {int(value["id"]): value for value in train["images"]}
    order = support_order(train, args.seed)
    if args.max_support_turns is not None:
        order = order[: args.max_support_turns]
    prefixes = sorted(set(args.test_prefixes))
    if prefixes[0] < 0 or prefixes[-1] > len(order):
        raise ValueError("Test prefixes must be within the collected support order.")
    test_images = sorted(test["images"], key=lambda value: int(value["id"]))
    if args.test_image_limit is not None:
        test_images = test_images[: args.test_image_limit]
    selected_test_ids = {int(value["id"]) for value in test_images}
    test_subset = {
        **test,
        "images": test_images,
        "annotations": [
            value
            for value in test["annotations"]
            if int(value["image_id"]) in selected_test_ids
        ],
    }
    ground_truth_path = output / "test_ground_truth.json"
    base.atomic_write_json(ground_truth_path, test_subset)
    settings = {
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": 0.0,
        "seed": args.inference_seed,
        "reasoning_effort": "none",
        "enable_thinking": False,
        "vl_high_resolution_images": False,
        "timeout_seconds": args.timeout_seconds,
    }
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "dataset": str(dataset),
        "train_annotation_sha256": base.sha256_file(train_path),
        "test_annotation_sha256": base.sha256_file(test_path),
        "seed": args.seed,
        "inference_seed": args.inference_seed,
        "support_image_order": order,
        "support_object_counts_at_full_prefix": prefix_object_counts(order, train),
        "test_prefixes": prefixes,
        "selected_test_image_ids": sorted(selected_test_ids),
        "settings": settings,
        "routing_primary_metric": "known-object-class-macro-recall50-95",
        "routing_guard_metric": "known-object-class-macro-recall50",
        "final_primary_metric": "COCO-mAP50-95-maxDets500",
        "final_secondary_metric": "COCO-mAP50-maxDets500",
        "support_unmatched_predictions_ignored": True,
        "test_images_used_during_adaptation": False,
        "test_annotations_used_during_adaptation": False,
        "clean_trunk": True,
        "reasoning_disabled": True,
        "temperature": 0.0,
    }
    recipe.write_or_validate_manifest(output / "run_manifest.json", manifest)
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return 0

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=settings["base_url"],
        timeout=settings["timeout_seconds"],
        max_retries=0,
    )
    limiter = base.SmoothDualRateLimiter(
        args.requests_per_minute, args.tokens_per_minute
    )
    curve = []
    adaptation_records: list[dict[str, Any]] = []
    invocation_records: list[dict[str, Any]] = []
    for turn, image_id in enumerate(order, start=1):
        prefix = order[: turn - 1]
        target = train_images[image_id]
        branch_messages = build_branch(
            prefix,
            target,
            train,
            train_directory,
            train_directory,
            categories,
        )
        branch_task = task_for_image(f"ssa_turn_{turn:03d}", target)
        branch_path = output / "adaptation" / "branch" / f"turn_{turn:03d}.json"
        branch_fingerprint = expected_fingerprint(branch_task, branch_messages, settings)
        branch_record = load_terminal(branch_path, branch_fingerprint)
        if branch_record is None:
            branch_record = execute_messages(
                task=branch_task,
                messages=branch_messages,
                client=client,
                image_directory=train_directory,
                categories=categories,
                settings=settings,
                max_retries=args.max_retries,
                limiter=limiter,
            )
            base.atomic_write_json(branch_path, branch_record)
            invocation_records.append(branch_record)
        adaptation_records.append(branch_record)

        zero_root = (
            args.zero_cache_dir.resolve()
            if args.zero_cache_dir is not None
            else output / "adaptation" / "zero"
        )
        if turn == 1:
            zero_record = branch_record
            zero_reused_from_branch = True
        else:
            zero_messages = build_branch(
                [], target, train, train_directory, train_directory, categories
            )
            zero_task = task_for_image("ssa_zero_probe", target)
            zero_path = zero_root / f"image_{image_id}.json"
            zero_fingerprint = expected_fingerprint(zero_task, zero_messages, settings)
            zero_record = load_terminal(zero_path, zero_fingerprint)
            if zero_record is None:
                zero_record = execute_messages(
                    task=zero_task,
                    messages=zero_messages,
                    client=client,
                    image_directory=train_directory,
                    categories=categories,
                    settings=settings,
                    max_retries=args.max_retries,
                    limiter=limiter,
                )
                base.atomic_write_json(zero_path, zero_record)
                invocation_records.append(zero_record)
            adaptation_records.append(zero_record)
            zero_reused_from_branch = False
        branch_metrics = support_metrics(train, image_id, branch_record.get("predictions", []))
        zero_metrics = support_metrics(train, image_id, zero_record.get("predictions", []))
        curve.append(
            {
                "turn": turn,
                "prefix_images": turn - 1,
                "target_image_id": image_id,
                "target_ground_truth_objects": len(
                    one_image_calibration(train, image_id)["annotations"]
                ),
                "prefix_object_counts": prefix_object_counts(prefix, train),
                "target_absent_from_prefix": image_id not in prefix,
                "branch_status": branch_record.get("status"),
                "zero_status": zero_record.get("status"),
                "zero_reused_from_branch": zero_reused_from_branch,
                "branch": branch_metrics,
                "zero": zero_metrics,
                "delta": paired_delta(branch_metrics, zero_metrics),
            }
        )
        write_jsonl(output / "adaptation_curve.jsonl", curve)
        print(f"support checkpoint {turn}/{len(order)}", flush=True)

    simulation = simulate_best_prefix(curve)
    base.atomic_write_json(output / "stopping_preview.json", simulation)

    adaptation_usage = record_cost(adaptation_records)
    if args.adaptation_only:
        result = {
            "completed_at": base.utc_now(),
            "dataset": dataset.name,
            "support_turns": len(order),
            "test_images": 0,
            "test_prefixes": [],
            "stopping_preview": simulation,
            "adaptation_usage": adaptation_usage,
            "test_grid": [],
            "research_usage": adaptation_usage,
            "invocation_usage": record_cost(invocation_records),
            "pricing": {
                "per_million_tokens_usd": PRICES_PER_MILLION,
                "source": PRICING_SOURCE,
            },
        }
        base.atomic_write_json(output / "summary.json", result)
        base.atomic_write_json(
            output / "_SUCCESS.json",
            {
                "completed_at": base.utc_now(),
                "support_turns": len(order),
                "test_images": 0,
                "test_requests": 0,
            },
        )
        print(json.dumps(result, indent=2))
        return 0

    write_lock = threading.Lock()

    def run_test(
        prefix_count: int, image: dict[str, Any]
    ) -> tuple[Path, dict[str, Any], bool]:
        prefix = order[:prefix_count]
        messages = build_branch(
            prefix,
            image,
            train,
            train_directory,
            test_directory,
            categories,
        )
        task = task_for_image(f"ssa_prefix_{prefix_count:03d}", image)
        path = output / "test" / f"prefix_{prefix_count:03d}" / f"image_{task.image_id}.json"
        fingerprint = expected_fingerprint(task, messages, settings)
        existing = load_terminal(path, fingerprint)
        if existing is not None:
            return path, existing, False
        record = execute_messages(
            task=task,
            messages=messages,
            client=client,
            image_directory=test_directory,
            categories=categories,
            settings=settings,
            max_retries=args.max_retries,
            limiter=limiter,
        )
        return path, record, True

    jobs = [(prefix, image) for prefix in prefixes for image in test_images]
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(run_test, prefix, image): (prefix, image)
            for prefix, image in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            path, record, was_new = future.result()
            with write_lock:
                base.atomic_write_json(path, record)
                if was_new:
                    invocation_records.append(record)
                completed += 1
                if completed % 10 == 0 or completed == len(jobs):
                    print(f"test checkpoint {completed}/{len(jobs)}", flush=True)

    rows = []
    all_test_records = []
    for prefix in prefixes:
        records = [
            base.load_record(
                output / "test" / f"prefix_{prefix:03d}" / f"image_{int(image['id'])}.json"
            )
            for image in test_images
        ]
        if any(record is None or record.get("status") not in TERMINAL_STATUSES for record in records):
            raise RuntimeError(f"Unresolved test records for prefix {prefix}.")
        terminal = [record for record in records if record is not None]
        all_test_records.extend(terminal)
        predictions = [
            prediction
            for record in terminal
            for prediction in record.get("predictions", [])
        ]
        prediction_path = output / "predictions" / f"prefix_{prefix:03d}.json"
        base.atomic_write_json(prediction_path, predictions)
        metrics = base.score_coco(ground_truth_path, predictions)
        usage = record_cost(terminal)
        rows.append(
            {
                "prefix_images": prefix,
                "reference_objects": sum(prefix_object_counts(order[:prefix], train).values()),
                "mAP50_95": 100 * metrics["AP"],
                "mAP50": 100 * metrics["AP50"],
                "model_failures": sum(record.get("status") == "model_failure" for record in terminal),
                **usage,
            }
        )
    write_csv(output / "test_grid.csv", rows)
    result = {
        "completed_at": base.utc_now(),
        "dataset": dataset.name,
        "support_turns": len(order),
        "test_images": len(test_images),
        "test_prefixes": prefixes,
        "stopping_preview": simulation,
        "adaptation_usage": adaptation_usage,
        "test_grid": rows,
        "research_usage": record_cost([*adaptation_records, *all_test_records]),
        "invocation_usage": record_cost(invocation_records),
        "pricing": {"per_million_tokens_usd": PRICES_PER_MILLION, "source": PRICING_SOURCE},
    }
    base.atomic_write_json(output / "summary.json", result)
    base.atomic_write_json(
        output / "_SUCCESS.json",
        {
            "completed_at": base.utc_now(),
            "support_turns": len(order),
            "test_images": len(test_images),
            "test_requests": len(jobs),
        },
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
