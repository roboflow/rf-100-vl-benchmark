#!/usr/bin/env python3
"""Evaluate strict dataset-level binary routing over all RF20 datasets.

The score-blind router sees class names only.  Three identical sequential calls
measure API stability; majority vote selects the exact saved class-names-only or
one-reference detector result for each dataset.  No image inference is run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

import evaluate_qwen38_orion as base
import evaluate_qwen38_strict_binary_router as router

NAMES_MODE = "names_multi"
REFERENCE_MODE = "numeric_prediction_b01_multi"
ROUTE_TO_MODE = {
    "class_names_only": NAMES_MODE,
    "visual_references": REFERENCE_MODE,
}


def request_fingerprint(
    dataset: str,
    prompt: str,
    settings: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "dataset": dataset,
            "prompt_version": router.PROMPT_VERSION,
            "prompt": prompt,
            "settings": settings,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def majority_route(routes: list[str]) -> tuple[str, bool]:
    if not routes or any(route not in router.ROUTES for route in routes):
        raise ValueError("Cannot vote over missing or invalid routes.")
    counts = Counter(routes)
    maximum = max(counts.values())
    winners = [route for route, count in counts.items() if count == maximum]
    # A conservative tie requests visual context.
    selected = (
        "visual_references"
        if "visual_references" in winners
        else "class_names_only"
    )
    return selected, len(counts) == 1


def estimated_cost(usage: dict[str, Any]) -> float:
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    return ((prompt - cached) * 2 + cached * 0.25 + completion * 6) / 1_000_000


def call_router(
    client: Any,
    prompt: str,
    settings: dict[str, Any],
    labels: list[str],
    max_retries: int,
) -> dict[str, Any]:
    attempts = []
    for attempt in range(1, max_retries + 2):
        started = time.monotonic()
        try:
            inference = base.stream_inference(
                client,
                [{"role": "user", "content": prompt}],
                settings,
            )
            if inference.get("finish_reason") == "length":
                raise RuntimeError("Router response was truncated.")
            decision = router.parse_router_decision(inference["response"], labels)
            return {
                "status": "success",
                "decision": decision,
                "raw_response": inference["response"],
                "finish_reason": inference.get("finish_reason"),
                "usage": inference.get("usage") or {},
                "elapsed_seconds": inference.get("elapsed_seconds"),
                "attempts": attempts + [{"attempt": attempt, "status": "success"}],
            }
        except Exception as error:
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            if attempt > max_retries or base.terminal_provider_rejection(error):
                raise
            time.sleep(min(30.0, 2 ** (attempt - 1) + random.random()))
    raise AssertionError("Unreachable router retry loop.")


def load_fixed_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    fixed = {
        (row["dataset"], row["mode"]): row
        for row in rows
        if row["mode"] in {NAMES_MODE, REFERENCE_MODE}
    }
    datasets = {dataset for dataset, _ in fixed}
    if len(datasets) != 20 or len(fixed) != 40:
        raise ValueError("Fixed RF20 summary lacks both detector branches for 20 datasets.")
    return fixed


def summarize(
    datasets: list[Path],
    records: dict[str, list[dict[str, Any]]],
    fixed: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    rows = []
    for dataset in datasets:
        dataset_records = records[dataset.name]
        routes = [record["decision"]["route"] for record in dataset_records]
        selected_route, stable = majority_route(routes)
        selected_mode = ROUTE_TO_MODE[selected_route]
        names = fixed[(dataset.name, NAMES_MODE)]
        reference = fixed[(dataset.name, REFERENCE_MODE)]
        selected = fixed[(dataset.name, selected_mode)]
        routing_cost = sum(estimated_cost(record.get("usage") or {}) for record in dataset_records)
        row = {
            "dataset": dataset.name,
            "class_count": len(base.categories_by_id(base.load_coco(dataset / "test/_annotations.coco.json"))),
            "repeat_routes": routes,
            "stable": stable,
            "selected_route": selected_route,
            "selected_mode": selected_mode,
            "selected_mAP50_95": float(selected["mAP50_95"]),
            "selected_mAP50": float(selected["mAP50"]),
            "delta_vs_names_mAP50_95": float(selected["mAP50_95"])
            - float(names["mAP50_95"]),
            "delta_vs_names_mAP50": float(selected["mAP50"])
            - float(names["mAP50"]),
            "names_mAP50_95": float(names["mAP50_95"]),
            "names_mAP50": float(names["mAP50"]),
            "reference_mAP50_95": float(reference["mAP50_95"]),
            "reference_mAP50": float(reference["mAP50"]),
            "selected_detector_cost_usd": float(selected["estimated_usd"]),
            "routing_cost_usd": routing_cost,
            "selected_total_cost_usd": float(selected["estimated_usd"])
            + routing_cost,
        }
        rows.append(row)

    def macro(key: str, values: list[dict[str, Any]] = rows) -> float:
        return fmean(float(row[key]) for row in values)

    held_out = [row for row in rows if row["dataset"] != "paper-parts"]
    summary = {
        "completed_at": base.utc_now(),
        "prompt_version": router.PROMPT_VERSION,
        "dataset_count": len(rows),
        "repeats_per_dataset": len(next(iter(records.values()))),
        "stable_dataset_count": sum(row["stable"] for row in rows),
        "selected_route_counts": dict(Counter(row["selected_route"] for row in rows)),
        "macro_mAP50_95": macro("selected_mAP50_95"),
        "macro_mAP50": macro("selected_mAP50"),
        "names_macro_mAP50_95": macro("names_mAP50_95"),
        "names_macro_mAP50": macro("names_mAP50"),
        "reference_macro_mAP50_95": macro("reference_mAP50_95"),
        "reference_macro_mAP50": macro("reference_mAP50"),
        "delta_vs_names_mAP50_95": macro("delta_vs_names_mAP50_95"),
        "delta_vs_names_mAP50": macro("delta_vs_names_mAP50"),
        "detector_cost_usd": sum(row["selected_detector_cost_usd"] for row in rows),
        "routing_cost_usd": sum(row["routing_cost_usd"] for row in rows),
        "total_cost_usd": sum(row["selected_total_cost_usd"] for row in rows),
        "held_out_19": {
            "dataset_count": len(held_out),
            "macro_mAP50_95": macro("selected_mAP50_95", held_out),
            "macro_mAP50": macro("selected_mAP50", held_out),
            "delta_vs_names_mAP50_95": macro("delta_vs_names_mAP50_95", held_out),
            "delta_vs_names_mAP50": macro("delta_vs_names_mAP50", held_out),
        },
        "datasets": rows,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--fixed-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default=router.MODEL_ID)
    parser.add_argument(
        "--base-url",
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_retries < 0:
        raise ValueError("Repeats must be positive and retries nonnegative.")
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is required.")

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    datasets = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "test/_annotations.coco.json").is_file()
    )
    if len(datasets) != 20:
        raise ValueError(f"Expected 20 RF20 datasets, found {len(datasets)}.")
    fixed = load_fixed_rows(args.fixed_summary.resolve())
    settings = {
        "model": args.model,
        "seed": 1234,
        "max_completion_tokens": 1024,
        "temperature": 0.0,
        "reasoning_effort": "none",
        "enable_thinking": False,
        "vl_high_resolution_images": False,
        "timeout_seconds": args.timeout_seconds,
        "response_format": {"type": "json_object"},
    }
    manifest = {
        "prompt_version": router.PROMPT_VERSION,
        "dataset_root": str(dataset_root),
        "fixed_summary": str(args.fixed_summary.resolve()),
        "repeats": args.repeats,
        "settings": settings,
        "score_blind_router": True,
        "images_sent_to_router": False,
        "ground_truth_sent_to_router": False,
        "detector_inference_run": False,
        "paper_parts_role": "development_dataset",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    existing_manifest = base.load_record(manifest_path)
    if existing_manifest and {key: existing_manifest.get(key) for key in manifest} != manifest:
        raise ValueError("Existing router manifest does not match this experiment.")
    if not existing_manifest:
        base.atomic_write_json(manifest_path, {**manifest, "created_at": base.utc_now()})

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=args.base_url.rstrip("/"),
        timeout=args.timeout_seconds,
        max_retries=0,
    )
    records: dict[str, list[dict[str, Any]]] = {}
    for dataset in datasets:
        test = base.load_coco(dataset / "test/_annotations.coco.json")
        labels = list(base.categories_by_id(test).values())
        prompt = router.build_router_prompt(labels)
        fingerprint = request_fingerprint(dataset.name, prompt, settings)
        records[dataset.name] = []
        for repeat in range(1, args.repeats + 1):
            record_path = output_dir / "records" / dataset.name / f"repeat_{repeat:02d}.json"
            record = base.load_record(record_path)
            if record:
                if record.get("request_fingerprint") != fingerprint:
                    raise ValueError(f"Mismatched saved routing call: {record_path}")
            else:
                result = call_router(
                    client,
                    prompt,
                    settings,
                    labels,
                    args.max_retries,
                )
                record = {
                    **result,
                    "completed_at": base.utc_now(),
                    "dataset": dataset.name,
                    "repeat": repeat,
                    "class_names": labels,
                    "prompt": prompt,
                    "request_fingerprint": fingerprint,
                }
                base.atomic_write_json(record_path, record)
            records[dataset.name].append(record)
            print(
                f"{dataset.name} repeat {repeat}/{args.repeats}: "
                f"{record['decision']['route']}"
            )

    summary = summarize(datasets, records, fixed)
    base.atomic_write_json(output_dir / "summary.json", summary)
    csv_path = output_dir / "per_dataset.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "dataset",
            "repeat_routes",
            "stable",
            "selected_route",
            "selected_mAP50_95",
            "selected_mAP50",
            "delta_vs_names_mAP50_95",
            "delta_vs_names_mAP50",
            "selected_detector_cost_usd",
            "routing_cost_usd",
            "selected_total_cost_usd",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["datasets"])
    base.atomic_write_json(
        output_dir / "_SUCCESS.json",
        {
            "completed_at": summary["completed_at"],
            "dataset_count": len(datasets),
            "routing_call_count": len(datasets) * args.repeats,
        },
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "datasets"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
