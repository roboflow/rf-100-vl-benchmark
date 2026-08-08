#!/usr/bin/env python3
"""Stop or terminate the current RunPod without printing its API key."""

from __future__ import annotations

import os
import sys
import urllib.request


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        raise SystemExit("usage: runpod_self_terminate.py POD_ID [stop|terminate]")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print(
            "[pod] RUNPOD_API_KEY is unavailable; pod remains active",
            file=sys.stderr,
        )
        return 1
    pod_id = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) == 3 else "terminate"
    if action not in {"stop", "terminate"}:
        raise SystemExit("action must be 'stop' or 'terminate'")
    if action == "stop":
        url = f"https://rest.runpod.io/v1/pods/{pod_id}/stop"
        method = "POST"
    else:
        url = f"https://rest.runpod.io/v1/pods/{pod_id}"
        method = "DELETE"
    request = urllib.request.Request(
        url,
        method=method,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
    except Exception as error:
        print(
            f"[pod] {action} request failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(f"[pod] {action} requested")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
