#!/usr/bin/env python3
"""Write a non-secret pod exit record for final GCS upload."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--git-sha", default="unknown")
    parser.add_argument("--image-ref", default="unknown")
    args = parser.parse_args()
    args.path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.path.with_name(f".{args.path.name}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "succeeded" if args.exit_code == 0 else "failed",
                "exit_code": args.exit_code,
                "stage": args.stage,
                "benchmark_git_sha": args.git_sha,
                "image_ref": args.image_ref,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
