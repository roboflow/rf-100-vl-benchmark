#!/usr/bin/env python3
"""Download and validate the exact, pinned PerceptionBench release.

The benchmark embeds its images as data URIs in a 1.63 GB JSONL file.  This
preparer downloads that file without decoding/re-encoding any image and also
fetches the paper's evaluator and judge prompt at a pinned Git commit.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

HF_COMMIT = "6ba8c3135c7675ad6a5c141536a86b9460c70960"
HF_DATA_SHA256 = "f84afb46c3c150b572481ca34b351d6afb1c31b98fef0c0c46d7659333573c47"
HF_DATA_SIZE = 1_626_804_092
UPSTREAM_COMMIT = "ba032c06e9b6ee3679171ff6ba643b7a0cfebe2e"
UPSTREAM_FILES = {
    "eval/eval.py": "3512b4307853710f9034aa51e7e2335d7bc3a153e4ec503bf1fea2f770fe9908",
    "eval/judge_prompt.txt": "3736ab84d90d96a0344a25bf96a1d9ac2a0672656bbc1eeb01176f20cd3d6397",
    "README.md": "0863d20bc397e2324ea76099fe703aed419dc7f0dbb359e522f7dac07dda6529",
}
DATA_URL = (
    "https://huggingface.co/datasets/moonshotai/PerceptionBench/resolve/"
    f"{HF_COMMIT}/PerceptionBench.jsonl"
)
RAW_GITHUB = (
    "https://raw.githubusercontent.com/MoonshotAI/PerceptionBench/"
    f"{UPSTREAM_COMMIT}/"
)
IMAGE_PH = re.compile(r"<\|image_(\d+)\|>")
EXPECTED_KEYS = {
    "index",
    "answer",
    "problem",
    "image",
    "error_category",
    "source_bmk",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def download(url: str, destination: Path, expected_size: int | None = None) -> None:
    """Download to a temporary file, resuming when the server supports Range."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() and (expected_size is None or destination.stat().st_size == expected_size):
        return
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "rf100-vl-benchmark-perceptionbench/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        append = offset > 0 and getattr(response, "status", 200) == 206
        if not append:
            offset = 0
        mode = "ab" if append else "wb"
        downloaded = offset
        last_report = time.monotonic()
        with partial.open(mode) as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last_report >= 10:
                    total = f"/{expected_size}" if expected_size else ""
                    print(f"downloaded {downloaded}{total} bytes", flush=True)
                    last_report = time.monotonic()
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise ValueError(
            f"Wrong download size for {destination}: {partial.stat().st_size} != {expected_size}"
        )
    os.replace(partial, destination)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            yield row


def sniff_image_mime(payload: bytes) -> str | None:
    """Identify supported image bytes without trusting the data-URI label."""

    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return "image/webp"
    if payload.startswith(b"BM"):
        return "image/bmp"
    if payload.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if len(payload) >= 12 and payload[4:8] == b"ftyp":
        brand = payload[8:12]
        if brand in {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1"}:
            return "image/heic"
    return None


def validate_dataset(path: Path, *, inspect_image_payloads: bool = True) -> dict[str, Any]:
    indices: set[int] = set()
    categories: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    image_counts: Counter[int] = Counter()
    image_formats: Counter[str] = Counter()
    detected_image_formats: Counter[str] = Counter()
    mime_mismatch_examples: list[dict[str, Any]] = []
    mime_mismatch_count = 0
    total_image_bytes = 0
    maximum_image_bytes = 0
    maximum_data_uri_bytes = 0
    image_total = 0
    for line_number, row in enumerate(iter_jsonl(path), 1):
        missing = EXPECTED_KEYS - row.keys()
        if missing:
            raise ValueError(f"Line {line_number} lacks fields: {sorted(missing)}")
        index = row["index"]
        if not isinstance(index, int) or index in indices:
            raise ValueError(f"Invalid or duplicate index on line {line_number}: {index!r}")
        indices.add(index)
        images = row["image"]
        if not isinstance(images, list):
            raise ValueError(f"Record {index} image is not a list")
        placeholders = [int(value) for value in IMAGE_PH.findall(str(row["problem"]))]
        if any(value < 1 or value > len(images) for value in placeholders):
            raise ValueError(f"Record {index} has an out-of-range image placeholder")
        if inspect_image_payloads:
            for image_number, image in enumerate(images, 1):
                if not isinstance(image, str) or not image.startswith("data:image/"):
                    raise ValueError(f"Record {index} image {image_number} is not a data image URI")
                try:
                    metadata, payload = image.split(",", 1)
                    if ";base64" not in metadata:
                        raise ValueError("not base64")
                    # Validate without retaining decoded images or changing bytes.
                    decoded = base64.b64decode(payload, validate=True)
                    mime = metadata.removeprefix("data:").split(";", 1)[0]
                    detected_mime = sniff_image_mime(decoded)
                    if detected_mime is None:
                        raise ValueError("unrecognized or unsupported image signature")
                    if mime != detected_mime:
                        mime_mismatch_count += 1
                        if len(mime_mismatch_examples) < 20:
                            mime_mismatch_examples.append(
                                {
                                    "index": index,
                                    "image_number": image_number,
                                    "declared_mime": mime,
                                    "detected_mime": detected_mime,
                                }
                            )
                    total_image_bytes += len(decoded)
                    maximum_image_bytes = max(maximum_image_bytes, len(decoded))
                    maximum_data_uri_bytes = max(
                        maximum_data_uri_bytes, len(image.encode("utf-8"))
                    )
                    image_formats[mime] += 1
                    detected_image_formats[detected_mime] += 1
                    image_total += 1
                except Exception as error:
                    raise ValueError(f"Record {index} image {image_number} is invalid: {error}") from error
        categories[str(row["error_category"])] += 1
        sources[str(row["source_bmk"])] += 1
        image_counts[len(images)] += 1
    if len(indices) != 3000 or indices != set(range(3000)):
        raise ValueError("Expected exactly the contiguous indices 0..2999")
    if len(categories) != 10:
        raise ValueError(f"Expected 10 error categories, found {len(categories)}")
    if image_counts and max(image_counts) > 250:
        raise ValueError("A record exceeds Qwen's 250-image Base64 input limit")
    if inspect_image_payloads and maximum_image_bytes > 20_000_000:
        raise ValueError("An original image exceeds Qwen's 20 MB Base64 input limit")
    if inspect_image_payloads and maximum_data_uri_bytes > 20_000_000:
        raise ValueError("A data URI exceeds Qwen's 20 MB Base64 input limit")
    return {
        "record_count": len(indices),
        "error_categories": dict(sorted(categories.items())),
        "source_benchmarks": dict(sorted(sources.items())),
        "images_per_record": {str(k): v for k, v in sorted(image_counts.items())},
        "image_count": image_total if inspect_image_payloads else None,
        "image_formats": dict(sorted(image_formats.items())),
        "detected_image_formats": dict(sorted(detected_image_formats.items())),
        "declared_mime_mismatch_count": mime_mismatch_count,
        "declared_mime_mismatch_examples": mime_mismatch_examples,
        "decoded_image_bytes": total_image_bytes if inspect_image_payloads else None,
        "maximum_image_bytes": maximum_image_bytes if inspect_image_payloads else None,
        "maximum_data_uri_bytes": (
            maximum_data_uri_bytes if inspect_image_payloads else None
        ),
        "qwen_base64_image_limit_bytes": 20_000_000,
        "all_images_fit_qwen_limit": (
            maximum_image_bytes <= 20_000_000 if inspect_image_payloads else None
        ),
    }


def prepare(root: Path, validate_images: bool = True) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / "PerceptionBench.jsonl"
    download(DATA_URL, data_path, HF_DATA_SIZE)
    actual_data_hash = sha256_file(data_path)
    if actual_data_hash != HF_DATA_SHA256:
        raise ValueError(f"Dataset SHA-256 mismatch: {actual_data_hash}")

    for relative, expected_hash in UPSTREAM_FILES.items():
        destination = root / "upstream" / relative
        download(RAW_GITHUB + relative, destination)
        actual_hash = sha256_file(destination)
        if actual_hash != expected_hash:
            raise ValueError(f"Upstream file hash mismatch for {relative}: {actual_hash}")

    validation = validate_dataset(data_path, inspect_image_payloads=validate_images)
    manifest = {
        "dataset": {
            "repository": "moonshotai/PerceptionBench",
            "commit": HF_COMMIT,
            "path": str(data_path),
            "size": data_path.stat().st_size,
            "sha256": actual_data_hash,
        },
        "evaluator": {
            "repository": "MoonshotAI/PerceptionBench",
            "commit": UPSTREAM_COMMIT,
            "files": UPSTREAM_FILES,
        },
        "validation": validation,
    }
    atomic_json(root / "source_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("PerceptionBench"))
    parser.add_argument(
        "--skip-image-payload-validation",
        action="store_true",
        help="Validate schema but skip base64 decoding (faster, less exhaustive).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare(
        args.output_dir.resolve(),
        validate_images=not args.skip_image_payload_validation,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
