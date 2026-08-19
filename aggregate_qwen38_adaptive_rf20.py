#!/usr/bin/env python3
"""Validate and aggregate a complete adaptive Qwen3.8-Max RF20 run."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import evaluate_qwen38_adaptive_no_feedback as adaptive
import evaluate_qwen38_orion as base

PRICES_PER_MILLION = {
    "uncached_prompt": 2.0,
    "implicit_cached_prompt": 0.25,
    "completion": 6.0,
}
PRICING_SOURCE = "https://www.qwencloud.com/models/qwen3.8-max"


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing required artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def aggregate(dataset_root: Path, run_root: Path) -> dict[str, Any]:
    datasets = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "test/_annotations.coco.json").is_file()
    )
    if len(datasets) != 20:
        raise ValueError(f"RF20 contract requires exactly 20 datasets; found {len(datasets)}.")
    rows = []
    stop_reasons: dict[str, int] = defaultdict(int)
    total_selected_references = 0
    total_class_images = 0
    for dataset in datasets:
        test = _read(dataset / "test/_annotations.coco.json")
        image_count = len(test.get("images", []))
        class_count = len(test.get("categories", []))
        run = run_root / dataset.name
        success = _read(run / "_SUCCESS.json")
        progress = _read(run / "progress.json")
        summary = _read(run / "comparison_summary.json")
        metrics = _read(run / "metrics" / f"{adaptive.MODE}.json")
        manifest = _read(run / "run_manifest.json")
        if Path(manifest["dataset_directory"]).resolve() != dataset.resolve():
            raise ValueError(f"Dataset manifest mismatch for {dataset.name}.")
        if manifest.get("prompt_version") != adaptive.PROMPT_VERSION:
            raise ValueError(f"Prompt-version mismatch for {dataset.name}.")
        if success.get("image_count") != image_count or success.get("condition_count") != 1:
            raise ValueError(f"Success contract mismatch for {dataset.name}.")
        totals = progress.get("total", {})
        if totals.get("pending") or totals.get("in_progress") or totals.get("error"):
            raise ValueError(f"Unresolved adaptive requests remain for {dataset.name}.")
        summary_rows = summary.get("rows", [])
        if len(summary_rows) != 1 or summary_rows[0].get("mode") != adaptive.MODE:
            raise ValueError(f"Adaptive mode summary mismatch for {dataset.name}.")
        row = summary_rows[0]
        if not row.get("complete") or int(row.get("task_count") or 0) != image_count:
            raise ValueError(f"Incomplete adaptive result for {dataset.name}.")
        usage = metrics.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        cached_tokens = int(usage.get("cached_prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        uncached_tokens = prompt_tokens - cached_tokens
        if uncached_tokens < 0:
            raise ValueError(f"Invalid cached-token accounting for {dataset.name}.")
        estimated_cost = (
            uncached_tokens * PRICES_PER_MILLION["uncached_prompt"]
            + cached_tokens * PRICES_PER_MILLION["implicit_cached_prompt"]
            + completion_tokens * PRICES_PER_MILLION["completion"]
        ) / 1_000_000
        selected = int(metrics.get("total_selected_reference_objects") or 0)
        total_selected_references += selected
        total_class_images += image_count * class_count
        for reason, count in (metrics.get("stop_reasons") or {}).items():
            stop_reasons[str(reason)] += int(count)
        rows.append(
            {
                "dataset": dataset.name,
                "test_images": image_count,
                "classes": class_count,
                "mode": adaptive.MODE,
                "mAP50_95": float(row["mAP50_95"]),
                "mAP50": float(row["mAP50"]),
                "model_failures": int(row.get("model_failures") or 0),
                "errors": int(row.get("errors") or 0),
                "zero_shot_images": int(row.get("zero_shot_images") or 0),
                "total_selected_reference_objects": selected,
                "mean_selected_references_per_image": float(
                    row.get("mean_selected_references_per_image") or 0
                ),
                "mean_selected_references_per_class_image": float(
                    row.get("mean_selected_references_per_class_image") or 0
                ),
                "request_count": int(usage.get("request_count") or 0),
                "prompt_tokens": prompt_tokens,
                "cached_prompt_tokens": cached_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
                "estimated_usd": estimated_cost,
            }
        )
    total_images = sum(row["test_images"] for row in rows)
    mode_summary = {
        "mode": adaptive.MODE,
        "dataset_count": len(rows),
        "macro_mAP50_95": fmean(row["mAP50_95"] for row in rows),
        "macro_mAP50": fmean(row["mAP50"] for row in rows),
        "test_image_count": total_images,
        "zero_shot_images": sum(row["zero_shot_images"] for row in rows),
        "total_selected_reference_objects": total_selected_references,
        "mean_selected_references_per_image": total_selected_references / total_images,
        "mean_selected_references_per_class_image": total_selected_references / total_class_images,
        "stop_reasons": dict(stop_reasons),
        "request_count": sum(row["request_count"] for row in rows),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "cached_prompt_tokens": sum(row["cached_prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
        "reasoning_tokens": sum(row["reasoning_tokens"] for row in rows),
        "model_failures": sum(row["model_failures"] for row in rows),
        "errors": sum(row["errors"] for row in rows),
        "estimated_usd": sum(row["estimated_usd"] for row in rows),
    }
    return {
        "created_at": base.utc_now(),
        "dataset_count": len(datasets),
        "test_image_count": total_images,
        "request_count": mode_summary["request_count"],
        "pricing": {
            "per_million_tokens_usd": PRICES_PER_MILLION,
            "source": PRICING_SOURCE,
        },
        "modes": [mode_summary],
        "total_estimated_usd": mode_summary["estimated_usd"],
        "per_dataset": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.dataset_root.resolve(), args.run_root.resolve())
    args.run_root.mkdir(parents=True, exist_ok=True)
    base.atomic_write_json(args.run_root / "rf20_summary.json", result)
    _write_csv(args.run_root / "rf20_per_dataset.csv", result["per_dataset"])
    _write_csv(args.run_root / "rf20_macro_summary.csv", result["modes"])
    base.atomic_write_json(
        args.run_root / "_RF20_SUCCESS.json",
        {
            "completed_at": base.utc_now(),
            "dataset_count": result["dataset_count"],
            "test_image_count": result["test_image_count"],
            "request_count": result["request_count"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
