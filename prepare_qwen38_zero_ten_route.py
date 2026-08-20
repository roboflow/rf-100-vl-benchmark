#!/usr/bin/env python3
"""Convert the locked support 0/1 gate into a frozen 0/10 RF20 route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_qwen38_orion as base
from qwen38_calibrated_counts import MODE_BY_COUNT


def load_rows(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Missing rows in {path}.")
    return rows


def prepare(pilot: Path, heldout: Path) -> dict:
    rows = []
    seen: set[str] = set()
    for phase, path in (("diagnostic", pilot), ("heldout", heldout)):
        for source in load_rows(path):
            dataset = str(source["dataset"])
            if dataset in seen:
                raise ValueError(f"Duplicate dataset: {dataset}")
            seen.add(dataset)
            selected_count = 0 if source["selected_mode"] == "names_multi" else 10
            rows.append(
                {
                    "dataset": dataset,
                    "selected_count": selected_count,
                    "selected_mode": MODE_BY_COUNT[selected_count],
                    "calibration_phase": phase,
                    "source_selected_mode": source["selected_mode"],
                    "support_delta_recall50_95": source[
                        "support_delta_recall50_95"
                    ],
                    "support_delta_recall50": source["support_delta_recall50"],
                }
            )
    if len(rows) != 20:
        raise ValueError(f"Expected 20 route rows; found {len(rows)}.")
    return {
        "created_at": base.utc_now(),
        "route": "support-calibrated-zero-or-ten-v1",
        "decision_source": "locked-support-calibrated-zero-or-one-gate-v2",
        "test_data_used_for_route": False,
        "counts": {"0": sum(row["selected_count"] == 0 for row in rows),
                   "10": sum(row["selected_count"] == 10 for row in rows)},
        "rows": sorted(rows, key=lambda row: row["dataset"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = prepare(args.pilot.resolve(), args.heldout.resolve())
    base.atomic_write_json(args.output.resolve(), value)
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
