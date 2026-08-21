#!/usr/bin/env python3
"""Evaluate an SSA prefix locked from support-only curves on untouched test data."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import evaluate_qwen38_orion as base
import evaluate_qwen38_ssa as ssa


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--requests-per-minute", type=float, default=6750.0)
    parser.add_argument("--tokens-per-minute", type=float, default=900_000.0)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args(argv)
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is required for inference.")

    root = args.run_root.resolve()
    lock = json.loads((root / "locked_support_policy.json").read_text())
    decision = lock["decisions"][args.dataset][str(args.seed)]
    prefix_count = int(decision["selected_prefix"])
    source = root / args.dataset / f"seed-{args.seed}"
    manifest = json.loads((source / "run_manifest.json").read_text())
    dataset = Path(manifest["dataset"])
    if not dataset.is_absolute():
        dataset = (Path.cwd() / dataset).resolve()
    train_directory = dataset / "train"
    test_directory = dataset / "test"
    train = base.load_coco(train_directory / "_annotations.coco.json")
    test = base.load_coco(test_directory / "_annotations.coco.json")
    base.validate_split_isolation(train, test)
    categories = base.categories_by_id(train)
    if categories != base.categories_by_id(test):
        raise ValueError("Train/test categories differ.")
    order = [int(value) for value in manifest["support_image_order"]]
    if prefix_count < 0 or prefix_count > len(order):
        raise ValueError("Locked prefix is outside the recorded support order.")
    settings = manifest["settings"]
    if settings["temperature"] != 0.0 or settings["reasoning_effort"] != "none":
        raise ValueError("Locked SSA evaluation must be deterministic and reasoning-free.")

    output = source / "locked_selected" / f"prefix_{prefix_count:03d}"
    output.mkdir(parents=True, exist_ok=True)
    client_module = __import__("openai")
    client = client_module.OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=settings["base_url"],
        timeout=settings["timeout_seconds"],
        max_retries=0,
    )
    limiter = base.SmoothDualRateLimiter(
        args.requests_per_minute, args.tokens_per_minute
    )
    invocation: list[dict[str, Any]] = []
    write_lock = threading.Lock()

    def run(image: dict[str, Any]) -> tuple[Path, dict[str, Any], bool]:
        messages = ssa.build_branch(
            order[:prefix_count],
            image,
            train,
            train_directory,
            test_directory,
            categories,
        )
        task = ssa.task_for_image(f"ssa_prefix_{prefix_count:03d}", image)
        path = output / "records" / f"image_{int(image['id'])}.json"
        fingerprint = ssa.expected_fingerprint(task, messages, settings)
        existing = ssa.load_terminal(path, fingerprint)
        if existing is not None:
            return path, existing, False
        record = ssa.execute_messages(
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

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(run, image) for image in test["images"]]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            path, record, was_new = future.result()
            with write_lock:
                base.atomic_write_json(path, record)
                records.append(record)
                if was_new:
                    invocation.append(record)
            if completed % 10 == 0 or completed == len(futures):
                print(f"locked-prefix checkpoint {completed}/{len(futures)}", flush=True)

    if any(record.get("status") not in ssa.TERMINAL_STATUSES for record in records):
        raise RuntimeError("Locked-prefix evaluation contains unresolved requests.")
    predictions = [
        prediction for record in records for prediction in record.get("predictions", [])
    ]
    prediction_path = output / "predictions.json"
    base.atomic_write_json(prediction_path, predictions)
    metrics = base.score_coco(test_directory / "_annotations.coco.json", predictions)
    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "selected_prefix": prefix_count,
        "selection_reason": decision["reason"],
        "mAP50_95": 100 * metrics["AP"],
        "mAP50": 100 * metrics["AP50"],
        "model_failures": sum(record.get("status") == "model_failure" for record in records),
        "logical_usage": ssa.record_cost(records),
        "invocation_usage": ssa.record_cost(invocation),
        "test_labels_used_to_choose_prefix": False,
        "final_primary_metric": "COCO-mAP50-95-maxDets500",
    }
    base.atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
