#!/usr/bin/env python3
"""Require a clean maximum-context Paper Parts all-available smoke result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODE = "numeric_prediction_all_available_multi_explicit_sparse"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def validate(run_root: Path) -> dict[str, Any]:
    manifest = _read(run_root / "run_manifest.json")
    progress = _read(run_root / "progress.json")
    success = _read(run_root / "_SUCCESS.json")
    if manifest.get("selected_test_image_ids") != [0]:
        raise ValueError("Smoke test is not locked to Paper Parts test image 0.")
    references = manifest["reference_selection"]["classes"]
    if len(references) != 19 or any(len(sequence) != 10 for sequence in references.values()):
        raise ValueError("Smoke test does not include all 190 Paper Parts references.")
    total = progress["total"]
    if total != {
        "total": 1,
        "success": 1,
        "model_failure": 0,
        "error": 0,
        "pending": 0,
    }:
        raise ValueError(f"Smoke request did not succeed cleanly: {total}")
    if success.get("request_count") != 1:
        raise ValueError("Smoke success marker has the wrong request count.")
    records = list((run_root / "records" / MODE).glob("*.json"))
    if len(records) != 1:
        raise ValueError(f"Expected one smoke record; found {len(records)}.")
    record = _read(records[0])
    if record.get("status") != "success":
        raise ValueError("Smoke record is not successful.")
    request_summary = record.get("request_summary") or {}
    if len(request_summary.get("image_sha256") or []) != 183:
        raise ValueError("Smoke request did not contain 182 reference images plus one target.")
    if not isinstance(record.get("raw_response"), str):
        raise ValueError("Smoke raw response was not retained.")
    if record.get("usage", {}).get("prompt_tokens") is None:
        raise ValueError("Smoke usage accounting is missing.")
    return {
        "smoke_passed": True,
        "test_images": 1,
        "classes": 19,
        "reference_objects": 190,
        "reference_images": 182,
        "image_attachments": 183,
        "prompt_tokens": record["usage"]["prompt_tokens"],
        "completion_tokens": record["usage"].get("completion_tokens"),
        "inference_seconds": record.get("inference_seconds"),
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
