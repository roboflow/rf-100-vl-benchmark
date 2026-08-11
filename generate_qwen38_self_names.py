#!/usr/bin/env python3
"""Generate cached, label-free visual names from RF20 train exemplars only."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_orion as base

PROMPT_VERSION = "qwen3.8-max-self-name-v1"


def build_messages(
    category_id: int,
    representation: str,
    box_count: int,
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]],
    assets: dict[tuple[int, int], dict[str, Path]],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "The marked objects in these reference images are examples of one "
                "visual concept. Give that object kind one concise, distinctive name "
                "of two to eight words, based only on the visual evidence. Do not "
                "mention boxes, annotations, images, or uncertainty. Return only "
                '{"name":"your visual name"}.'
            ),
        }
    ]
    for reference in references[category_id][:box_count]:
        if representation == "numeric":
            content.extend(
                [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base.data_url(
                                assets[(category_id, reference.rank)]["source"]
                            )
                        },
                    },
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"bbox_2d": list(reference.bbox_xyxy_1000)},
                            separators=(",", ":"),
                        ),
                    },
                ]
            )
        elif representation == "drawn":
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base.data_url(
                            assets[(category_id, reference.rank)]["drawn"]
                        )
                    },
                }
            )
        else:
            raise ValueError(f"Unknown representation: {representation}")
    return [{"role": "user", "content": content}]


def parse_name(raw: str) -> str:
    stripped = raw.strip()
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1))
    object_match = re.search(r"\{.*?\}", stripped, re.DOTALL)
    if object_match:
        candidates.insert(0, object_match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("name"), str):
            value = parsed["name"].strip().strip('"\'')
            if value:
                return value[:160]
    fallback = stripped.strip('`\n \t"\'')
    if fallback and len(fallback.splitlines()) == 1:
        return fallback[:160]
    raise ValueError(f"Could not parse a visual name from: {raw[:200]!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representation", choices=("numeric", "drawn"), required=True)
    parser.add_argument("--box-count", type=int, choices=(1, 2, 3, 5, 7, 10), required=True)
    parser.add_argument("--model", default="qwen3.8-max")
    parser.add_argument(
        "--base-url",
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--allow-shared-reference-images", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.temperature < 2:
        raise ValueError("Temperature must be in [0, 2).")
    if not os.getenv("DASHSCOPE_API_KEY") and not args.prepare_only:
        raise RuntimeError("DASHSCOPE_API_KEY is required for inference.")
    dataset_directory = args.dataset_dir.resolve()
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    lock_file = (output_directory / ".run.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another process owns {output_directory}.") from error

    train_directory = dataset_directory / "train"
    test_directory = dataset_directory / "test"
    train_path = train_directory / "_annotations.coco.json"
    test_path = test_directory / "_annotations.coco.json"
    train = base.load_coco(train_path)
    test = base.load_coco(test_path)
    base.validate_split_isolation(train, test)
    categories = base.categories_by_id(test)
    if categories != base.categories_by_id(train):
        raise ValueError("Train/test category definitions differ.")
    references = box_ablation.select_reference_sequences(
        train,
        train_directory,
        required_count=args.box_count,
        distinct_images_only=not args.allow_shared_reference_images,
    )
    assets = box_ablation.prepare_reference_assets(
        train_directory, output_directory / "references", references
    )
    settings = {
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "seed": args.seed,
        "temperature": args.temperature,
        "max_completion_tokens": 256,
        "reasoning_effort": "none",
        "vl_high_resolution_images": False,
        "timeout_seconds": args.timeout_seconds,
    }
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "dataset_directory": str(dataset_directory),
        "train_annotation_sha256": base.sha256_file(train_path),
        "test_annotation_sha256": base.sha256_file(test_path),
        "representation": args.representation,
        "box_count": args.box_count,
        "semantic_class_names_exposed": False,
        "settings": settings,
        "references": {
            str(category_id): [
                {
                    **asdict(reference),
                    "source_sha256": base.sha256_file(
                        train_directory / reference.file_name
                    ),
                }
                for reference in sequence[: args.box_count]
            ]
            for category_id, sequence in references.items()
        },
    }
    existing_manifest = base.load_record(output_directory / "run_manifest.json")
    if existing_manifest:
        if {key: existing_manifest.get(key) for key in manifest} != manifest:
            raise ValueError("Existing self-name manifest does not match.")
    else:
        base.atomic_write_json(
            output_directory / "run_manifest.json",
            {**manifest, "created_at": base.utc_now()},
        )
    if args.prepare_only:
        return 0

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=settings["base_url"],
        timeout=settings["timeout_seconds"],
        max_retries=0,
    )

    def execute(category_id: int) -> dict[str, Any]:
        messages = build_messages(
            category_id, args.representation, args.box_count, references, assets
        )
        summary = base.request_summary(messages)
        task = base.Task(
            mode=f"self_name_{args.representation}_b{args.box_count:02d}",
            image_id=-1,
            file_name="train-only-references",
            width=1,
            height=1,
            category_id=category_id,
            category_name=None,
        )
        fingerprint = base.request_fingerprint(task, summary, settings)
        record_path = output_directory / "records" / f"class_{category_id}.json"
        existing = base.load_record(record_path)
        if existing and existing.get("status") == "success":
            if existing.get("request_fingerprint") != fingerprint:
                raise ValueError(f"Mismatched self-name checkpoint for class {category_id}.")
            return existing
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, args.max_retries + 2):
            try:
                inference = base.stream_inference(client, messages, settings)
                name = parse_name(inference["response"])
                return {
                    "status": "success",
                    "category_id": category_id,
                    "name": name,
                    "raw_response": inference["response"],
                    "finish_reason": inference["finish_reason"],
                    "usage": inference["usage"],
                    "inference_seconds": inference["elapsed_seconds"],
                    "request_fingerprint": fingerprint,
                    "request_summary": summary,
                    "attempts": attempts + [{"attempt": attempt, "status": "success"}],
                    "completed_at": base.utc_now(),
                }
            except Exception as error:  # noqa: BLE001
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                if attempt > args.max_retries or not base.retryable_error(error):
                    return {
                        "status": "error",
                        "category_id": category_id,
                        "error": attempts[-1]["error"],
                        "request_fingerprint": fingerprint,
                        "request_summary": summary,
                        "attempts": attempts,
                        "completed_at": base.utc_now(),
                    }
        raise AssertionError("unreachable")

    records: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(categories)) as executor:
        futures = {executor.submit(execute, category_id): category_id for category_id in categories}
        for future in concurrent.futures.as_completed(futures):
            category_id = futures[future]
            record = future.result()
            records[category_id] = record
            base.atomic_write_json(
                output_directory / "records" / f"class_{category_id}.json", record
            )
    failures = [record for record in records.values() if record.get("status") != "success"]
    base.atomic_write_json(
        output_directory / "self_names.json",
        {
            "prompt_version": PROMPT_VERSION,
            "names": {
                str(category_id): records[category_id]["name"]
                for category_id in categories
                if records[category_id].get("status") == "success"
            },
            "semantic_class_names_exposed": False,
            "completed_at": base.utc_now(),
        },
    )
    if not failures:
        base.atomic_write_json(
            output_directory / "_SUCCESS.json",
            {"completed_at": base.utc_now(), "class_count": len(categories)},
        )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
