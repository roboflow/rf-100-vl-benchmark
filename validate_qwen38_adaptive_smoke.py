#!/usr/bin/env python3
"""Validate one completed adaptive no-feedback live smoke conversation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_qwen38_adaptive_no_feedback as adaptive
import evaluate_qwen38_orion as base


def validate(run_root: Path) -> dict:
    manifest = base.load_record(run_root / "run_manifest.json")
    success = base.load_record(run_root / "_SUCCESS.json")
    progress = base.load_record(run_root / "progress.json")
    if not manifest or not success or not progress:
        raise ValueError("Smoke run is missing manifest, progress, or success artifacts.")
    if manifest.get("prompt_version") != adaptive.PROMPT_VERSION:
        raise ValueError("Smoke prompt version mismatch.")
    totals = progress.get("total", {})
    if totals.get("total") != 1 or totals.get("pending") or totals.get("in_progress") or totals.get("error"):
        raise ValueError(f"Smoke progress is not terminal: {totals}")
    records = sorted((run_root / "records" / adaptive.MODE).glob("*.json"))
    if len(records) != 1:
        raise ValueError(f"Expected exactly one smoke record, found {len(records)}.")
    record = json.loads(records[0].read_text(encoding="utf-8"))
    if record.get("status") != "success":
        raise ValueError(f"Smoke record did not succeed: {record.get('status')}")
    if not record.get("rounds"):
        raise ValueError("Smoke record lacks an adaptive decision turn.")
    if not record.get("final_turn"):
        raise ValueError("Smoke record lacks a final detection turn.")
    if int((record.get("usage") or {}).get("request_count") or 0) < 2:
        raise ValueError("Smoke must contain at least one decision and one detection call.")
    maximum = int(manifest["adaptive_policy"]["max_examples_per_class"])
    selected = {
        str(key): int(value)
        for key, value in (record.get("selected_reference_counts") or {}).items()
    }
    if any(value < 0 or value > maximum for value in selected.values()):
        raise ValueError("Smoke selected-reference count exceeds the configured budget.")
    reasoning_tokens = int(
        ((record.get("usage") or {}).get("completion_tokens_details") or {}).get(
            "reasoning_tokens", 0
        )
        or 0
    )
    if reasoning_tokens:
        raise ValueError(f"Smoke used {reasoning_tokens} reasoning tokens despite disable flags.")
    return {
        "validated": True,
        "status": record["status"],
        "stop_reason": record.get("stop_reason"),
        "decision_rounds": len(record["rounds"]),
        "selected_reference_objects": sum(selected.values()),
        "request_count": record["usage"]["request_count"],
        "prompt_tokens": record["usage"]["prompt_tokens"],
        "completion_tokens": record["usage"]["completion_tokens"],
        "reasoning_tokens": reasoning_tokens,
        "prediction_count": len(record.get("predictions") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.run_root.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
