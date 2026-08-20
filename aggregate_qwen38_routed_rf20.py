#!/usr/bin/env python3
"""Validate and macro-average a routed RF20 Qwen3.8 benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Any

import evaluate_qwen38_orion as base
from aggregate_qwen38_rf20 import PRICES_PER_MILLION, PRICING_SOURCE
from qwen38_calibrated_counts import MODE_BY_COUNT, read_route_rows


def read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def usage(run: Path, mode: str) -> dict[str, int]:
    result = {"prompt_tokens": 0, "cached_prompt_tokens": 0, "completion_tokens": 0}
    for path in sorted((run / "records" / mode).glob("*.json")):
        record = read(path)
        current = record.get("usage") or {}
        details = current.get("prompt_tokens_details") or {}
        result["prompt_tokens"] += int(current.get("prompt_tokens") or 0)
        result["cached_prompt_tokens"] += int(details.get("cached_tokens") or 0)
        result["completion_tokens"] += int(current.get("completion_tokens") or 0)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def aggregate(dataset_root: Path, run_root: Path, route_path: Path) -> dict[str, Any]:
    route = read(route_path)
    route_rows = read_route_rows(route)
    datasets = sorted(
        path for path in dataset_root.iterdir()
        if path.is_dir() and (path / "test/_annotations.coco.json").is_file()
    )
    if len(datasets) != 20 or {path.name for path in datasets} != set(route_rows):
        raise ValueError("RF20 routed aggregate inventory mismatch.")
    rows = []
    for dataset in datasets:
        count = int(route_rows[dataset.name]["selected_count"])
        mode = MODE_BY_COUNT[count]
        run = run_root / dataset.name
        success = read(run / "_SUCCESS.json")
        progress = read(run / "progress.json")
        summary = read(run / "comparison_summary.json")
        manifest = read(run / "run_manifest.json")
        test = read(dataset / "test/_annotations.coco.json")
        image_count = len(test["images"])
        if Path(manifest["dataset_directory"]).resolve() != dataset.resolve():
            raise ValueError(f"Dataset manifest mismatch for {dataset.name}.")
        if success.get("image_count") != image_count or success.get("condition_count") != 1:
            raise ValueError(f"Success contract mismatch for {dataset.name}.")
        if progress.get("total", {}).get("pending") or progress.get("total", {}).get("error"):
            raise ValueError(f"Unresolved requests for {dataset.name}.")
        values = summary.get("rows", [])
        if len(values) != 1 or values[0].get("mode") != mode:
            raise ValueError(f"Selected mode mismatch for {dataset.name}.")
        value = values[0]
        if not value.get("complete") or int(value.get("task_count") or 0) != image_count:
            raise ValueError(f"Incomplete routed branch for {dataset.name}.")
        tokens = usage(run, mode)
        uncached = tokens["prompt_tokens"] - tokens["cached_prompt_tokens"]
        cost = (
            uncached * PRICES_PER_MILLION["uncached_prompt"]
            + tokens["cached_prompt_tokens"] * PRICES_PER_MILLION["implicit_cached_prompt"]
            + tokens["completion_tokens"] * PRICES_PER_MILLION["completion"]
        ) / 1_000_000
        rows.append(
            {
                "dataset": dataset.name,
                "test_images": image_count,
                "classes": len(test["categories"]),
                "selected_count": count,
                "selected_mode": mode,
                "mAP50_95": float(value["mAP50_95"]),
                "mAP50": float(value["mAP50"]),
                "model_failures": int(value.get("model_failures") or 0),
                "errors": int(value.get("errors") or 0),
                **tokens,
                "estimated_usd": cost,
            }
        )
    result = {
        "created_at": base.utc_now(),
        "route": route.get("route"),
        "dataset_count": 20,
        "test_image_count": sum(row["test_images"] for row in rows),
        "request_count": sum(row["test_images"] for row in rows),
        "macro_mAP50_95": fmean(row["mAP50_95"] for row in rows),
        "macro_mAP50": fmean(row["mAP50"] for row in rows),
        "selected_counts": {
            str(count): sum(row["selected_count"] == count for row in rows)
            for count in MODE_BY_COUNT
        },
        "model_failures": sum(row["model_failures"] for row in rows),
        "errors": sum(row["errors"] for row in rows),
        "total_estimated_usd": sum(row["estimated_usd"] for row in rows),
        "pricing": {"per_million_tokens_usd": PRICES_PER_MILLION, "source": PRICING_SOURCE},
        "per_dataset": rows,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.dataset_root.resolve(), args.run_root.resolve(), args.route.resolve())
    args.run_root.mkdir(parents=True, exist_ok=True)
    base.atomic_write_json(args.run_root / "rf20_summary.json", result)
    write_csv(args.run_root / "rf20_per_dataset.csv", result["per_dataset"])
    base.atomic_write_json(
        args.run_root / "_RF20_SUCCESS.json",
        {
            "completed_at": base.utc_now(),
            "dataset_count": result["dataset_count"],
            "test_image_count": result["test_image_count"],
            "request_count": result["request_count"],
        },
    )
    print(json.dumps({key: value for key, value in result.items() if key != "per_dataset"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
