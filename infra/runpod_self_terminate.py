#!/usr/bin/env python3
"""Terminate the current RunPod without ever printing its API key."""

from __future__ import annotations

import os
import sys
import urllib.request


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: runpod_self_terminate.py POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("[terminate] RUNPOD_API_KEY is unavailable; pod remains active", file=sys.stderr)
        return 1
    pod_id = sys.argv[1]
    request = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods/{pod_id}",
        method="DELETE",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
    except Exception as error:
        print(f"[terminate] termination request failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print("[terminate] termination requested")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
