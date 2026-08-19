#!/usr/bin/env python3
"""Route one RF20 dataset to an isolated names-only or one-shot detector.

The router never participates in detection.  After its single dataset-level
decision, scoring reuses the exact saved predictions from an established clean
class-names-only or one-reference run.  This isolates routing quality from API
generation variance and from conversational contamination of the detector.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import evaluate_qwen38_orion as base

MODEL_ID = "qwen3.8-max"
PROMPT_VERSION = "qwen3.8-max-strict-dataset-binary-router-v1"
ROUTES = {"class_names_only", "visual_references"}


def build_router_prompt(labels: list[str]) -> str:
    return (
        "Act only as a conservative dataset-level router for object detection. "
        "Decide whether these class names alone completely specify what should "
        "be detected, or whether labeled visual references are needed to define "
        "the dataset's intended visual concepts or annotation semantics. Do not "
        "predict boxes and do not reason about whether a class happens to be "
        "present in any particular target image.\n\n"
        "Choose class_names_only only when EVERY label unambiguously names a "
        "standard, visually recognizable object and examples could not "
        "materially clarify appearance, subtype, state, action, role, part-whole "
        "meaning, or annotation boundaries. If ANY label is opaque, fine-grained, "
        "domain-specific, state/action/role dependent, or annotation-convention "
        "dependent, choose visual_references. A false class_names_only decision "
        "is more costly than requesting references.\n\n"
        f"Exact class names: {json.dumps(labels, ensure_ascii=False)}\n\n"
        "Return one JSON object and nothing else using exactly this schema: "
        '{"route":"class_names_only|visual_references","confidence":0.0,'
        '"labels_requiring_visual_context":["exact class name"],'
        '"reason":"brief score-blind explanation"}. confidence must be from 0 '
        "to 1. For class_names_only, labels_requiring_visual_context must be []."
    )


def parse_router_decision(raw: str, labels: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("Router did not return valid JSON.") from error
    if not isinstance(value, dict):
        raise TypeError("Router response must be one JSON object.")
    route = str(value.get("route") or "").strip().casefold().replace("-", "_")
    aliases = {
        "names": "class_names_only",
        "zero_shot": "class_names_only",
        "one_shot": "visual_references",
        "references": "visual_references",
    }
    route = aliases.get(route, route)
    if route not in ROUTES:
        raise ValueError(f"Unknown router decision: {route!r}")
    requested = value.get("labels_requiring_visual_context", [])
    if not isinstance(requested, list) or any(not isinstance(item, str) for item in requested):
        raise ValueError("labels_requiring_visual_context must be a string list.")
    lookup = {label.casefold(): label for label in labels}
    normalized = []
    for item in requested:
        label = lookup.get(item.strip().casefold())
        if label is None:
            raise ValueError(f"Router returned an unknown label: {item!r}")
        if label not in normalized:
            normalized.append(label)
    if route == "class_names_only" and normalized:
        raise ValueError("A class_names_only decision cannot request visual context.")
    if route == "visual_references" and not normalized:
        raise ValueError("A visual_references decision must identify at least one label.")
    try:
        confidence = float(value.get("confidence"))
    except (TypeError, ValueError) as error:
        raise ValueError("Router confidence must be numeric.") from error
    if not 0 <= confidence <= 1:
        raise ValueError("Router confidence must be between 0 and 1.")
    return {
        "route": route,
        "confidence": confidence,
        "labels_requiring_visual_context": normalized,
        "reason": str(value.get("reason") or "").strip(),
    }


def score_branch(annotation_path: Path, prediction_path: Path) -> dict[str, Any]:
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise TypeError(f"Predictions are not a JSON list: {prediction_path}")
    return {
        "prediction_path": str(prediction_path.resolve()),
        "prediction_count": len(predictions),
        "metrics": base.score_coco(annotation_path, predictions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--fixed-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--base-url",
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is required.")

    dataset_dir = args.dataset_dir.resolve()
    fixed_run_dir = args.fixed_run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = dataset_dir / "test/_annotations.coco.json"
    test = base.load_coco(annotation_path)
    categories = base.categories_by_id(test)
    labels = list(categories.values())
    prompt = build_router_prompt(labels)
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

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=args.base_url.rstrip("/"),
        timeout=args.timeout_seconds,
        max_retries=0,
    )
    inference = base.stream_inference(
        client,
        [{"role": "user", "content": prompt}],
        settings,
    )
    if inference.get("finish_reason") == "length":
        raise RuntimeError("Router response was truncated.")
    decision = parse_router_decision(inference["response"], labels)

    branch_files = {
        "class_names_only": fixed_run_dir / "predictions/names_multi.json",
        "visual_references": fixed_run_dir
        / "predictions/numeric_prediction_b01_multi.json",
    }
    if any(not path.is_file() for path in branch_files.values()):
        raise FileNotFoundError("The fixed-run detector prediction branches are incomplete.")
    branches = {
        route: score_branch(annotation_path, path)
        for route, path in branch_files.items()
    }
    selected = branches[decision["route"]]
    result = {
        "completed_at": base.utc_now(),
        "prompt_version": PROMPT_VERSION,
        "dataset": dataset_dir.name,
        "test_images": len(test["images"]),
        "class_names": labels,
        "router_prompt": prompt,
        "router_raw_response": inference["response"],
        "router_decision": decision,
        "router_finish_reason": inference.get("finish_reason"),
        "router_usage": inference.get("usage"),
        "router_elapsed_seconds": inference.get("elapsed_seconds"),
        "settings": settings,
        "test_annotation_sha256": base.sha256_file(annotation_path),
        "branches": branches,
        "selected_branch": selected,
        "selected_mAP50_95": selected["metrics"]["AP"] * 100,
        "selected_mAP50": selected["metrics"]["AP50"] * 100,
    }
    base.atomic_write_json(output_dir / "result.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
