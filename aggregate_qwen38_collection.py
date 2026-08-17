#!/usr/bin/env python3
"""Validate and macro-average a complete named Qwen3.8 dataset collection."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

import evaluate_qwen38_orion as base
from aggregate_qwen38_rf20 import PRICES_PER_MILLION, PRICING_SOURCE
from evaluate_qwen38_recipe import load_conditions


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


def _usage(run_directory: Path, mode: str) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "cached_prompt_tokens": 0, "completion_tokens": 0}
    for path in sorted((run_directory / "records" / mode).glob("*.json")):
        record = _read(path)
        usage = record.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        totals["cached_prompt_tokens"] += int(details.get("cached_tokens") or 0)
        totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
    return totals


def aggregate(
    dataset_root: Path,
    run_root: Path,
    conditions_path: Path,
    dataset_names: Sequence[str],
) -> dict[str, Any]:
    if not dataset_names or len(set(dataset_names)) != len(dataset_names):
        raise ValueError("Dataset names must be a nonempty unique sequence.")
    datasets = [dataset_root / name for name in dataset_names]
    missing = [path.name for path in datasets if not (path / "test/_annotations.coco.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing requested datasets: {missing}")
    conditions = load_conditions(conditions_path)
    modes = [condition.mode for condition in conditions]
    rows: list[dict[str, Any]] = []
    expected_requests = 0
    total_images = 0
    for dataset in datasets:
        test = _read(dataset / "test/_annotations.coco.json")
        image_count = len(test.get("images", []))
        class_count = len(test.get("categories", []))
        total_images += image_count
        run = run_root / dataset.name
        success = _read(run / "_SUCCESS.json")
        progress = _read(run / "progress.json")
        summary = _read(run / "comparison_summary.json")
        manifest = _read(run / "run_manifest.json")
        if Path(manifest["dataset_directory"]).resolve() != dataset.resolve():
            raise ValueError(f"Dataset manifest mismatch for {dataset.name}.")
        if success.get("image_count") != image_count or success.get("condition_count") != len(modes):
            raise ValueError(f"Success contract mismatch for {dataset.name}.")
        if progress.get("total", {}).get("pending") or progress.get("total", {}).get("error"):
            raise ValueError(f"Unresolved requests remain for {dataset.name}.")
        by_mode = {str(row["mode"]): row for row in summary.get("rows", [])}
        if set(by_mode) != set(modes):
            raise ValueError(f"Mode mismatch for {dataset.name}: {sorted(by_mode)}")
        expected_requests += image_count * len(modes)
        for mode in modes:
            row = by_mode[mode]
            if not row.get("complete") or int(row.get("task_count") or 0) != image_count:
                raise ValueError(f"Incomplete {dataset.name}/{mode}.")
            usage = _usage(run, mode)
            if usage["prompt_tokens"] != int(row.get("prompt_tokens") or 0):
                raise ValueError(f"Prompt-token mismatch for {dataset.name}/{mode}.")
            if usage["completion_tokens"] != int(row.get("completion_tokens") or 0):
                raise ValueError(f"Completion-token mismatch for {dataset.name}/{mode}.")
            uncached = usage["prompt_tokens"] - usage["cached_prompt_tokens"]
            estimated_cost = (
                uncached * PRICES_PER_MILLION["uncached_prompt"]
                + usage["cached_prompt_tokens"] * PRICES_PER_MILLION["implicit_cached_prompt"]
                + usage["completion_tokens"] * PRICES_PER_MILLION["completion"]
            ) / 1_000_000
            rows.append(
                {
                    "dataset": dataset.name,
                    "test_images": image_count,
                    "classes": class_count,
                    "mode": mode,
                    "mAP50_95": float(row["mAP50_95"]),
                    "mAP50": float(row["mAP50"]),
                    "model_failures": int(row.get("model_failures") or 0),
                    "errors": int(row.get("errors") or 0),
                    **usage,
                    "estimated_usd": estimated_cost,
                }
            )
    mode_rows = []
    for mode in modes:
        selected = [row for row in rows if row["mode"] == mode]
        mode_rows.append(
            {
                "mode": mode,
                "dataset_count": len(selected),
                "macro_mAP50_95": fmean(row["mAP50_95"] for row in selected),
                "macro_mAP50": fmean(row["mAP50"] for row in selected),
                "prompt_tokens": sum(row["prompt_tokens"] for row in selected),
                "cached_prompt_tokens": sum(row["cached_prompt_tokens"] for row in selected),
                "completion_tokens": sum(row["completion_tokens"] for row in selected),
                "model_failures": sum(row["model_failures"] for row in selected),
                "errors": sum(row["errors"] for row in selected),
                "estimated_usd": sum(row["estimated_usd"] for row in selected),
            }
        )
    return {
        "created_at": base.utc_now(),
        "datasets": list(dataset_names),
        "dataset_count": len(datasets),
        "test_image_count": total_images,
        "request_count": expected_requests,
        "pricing": {"per_million_tokens_usd": PRICES_PER_MILLION, "source": PRICING_SOURCE},
        "modes": mode_rows,
        "total_estimated_usd": sum(row["estimated_usd"] for row in rows),
        "per_dataset": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()
    result = aggregate(
        args.dataset_root.resolve(),
        args.run_root.resolve(),
        args.conditions.resolve(),
        args.datasets,
    )
    args.run_root.mkdir(parents=True, exist_ok=True)
    base.atomic_write_json(args.run_root / "collection_summary.json", result)
    _write_csv(args.run_root / "collection_per_dataset.csv", result["per_dataset"])
    _write_csv(args.run_root / "collection_macro_summary.csv", result["modes"])
    base.atomic_write_json(
        args.run_root / "_COLLECTION_SUCCESS.json",
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
