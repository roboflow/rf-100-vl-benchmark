#!/usr/bin/env python3
"""Download the canonical RF100VL datasets using the pinned package API."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from rf100vl import download_rf100vl

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = download_rf100vl(path=str(args.output_dir))
    print(f"[data] downloader reported {len(datasets)} RF100VL datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
