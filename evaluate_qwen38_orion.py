#!/usr/bin/env python3
"""Compare Qwen3.8-Max object-detection prompting modes on Orion Products.

The experiment uses only RF20-VL-FSOD train images for visual references and
only the test split for evaluation. It evaluates six fixed modes:

* multi_class_names
* single_class_names
* positive_numeric
* positive_drawn
* positive_negative_numeric
* positive_negative_drawn

Each API request is checkpointed atomically before the next result is needed.
Rerunning the same command resumes successful calls, preserves raw responses,
and retries only transient failures. Single-class calls are combined before
COCO evaluation. Metrics use pycocotools and maxDets=[1, 10, 500].
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import logging
import mimetypes
import os
import random
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_cosmos import (
    CosmosResponseError,
    atomic_write_json,
    convert_detections_to_coco,
    parse_cosmos_response,
    score_coco,
)

MODEL_ID = "qwen3.8-max"
PROMPT_VERSION = "qwen3.8-max-orion-prompt-modes-v1"
MODES = (
    "multi_class_names",
    "single_class_names",
    "positive_numeric",
    "positive_drawn",
    "positive_negative_numeric",
    "positive_negative_drawn",
)
SINGLE_CLASS_MODES = frozenset(MODES[1:])
DRAWN_MODES = frozenset({"positive_drawn", "positive_negative_drawn"})
NEGATIVE_MODES = frozenset(
    {"positive_negative_numeric", "positive_negative_drawn"}
)
TERMINAL_STATUSES = frozenset({"success", "model_failure"})
NORMALIZED_COORDINATE_MAX = 1000
DEFAULT_CONCURRENCY = 256
DEFAULT_REQUESTS_PER_MINUTE = 570.0
DEFAULT_TOKENS_PER_MINUTE = 900_000.0

# Scheduling estimates measured from the completed Orion pilot, rounded upward.
# They are used only to pace requests below DashScope's account-wide TPM limit;
# provider-reported usage remains the source of truth in every saved record.
ESTIMATED_TOTAL_TOKENS = {
    "none": {
        "multi_class_names": 1325,
        "single_class_names": 750,
        "positive_numeric": 1200,
        "positive_drawn": 1125,
        "positive_negative_numeric": 1750,
        "positive_negative_drawn": 1600,
    },
    "low": {
        "multi_class_names": 2125,
        "single_class_names": 1000,
        "positive_numeric": 1550,
        "positive_drawn": 1525,
        "positive_negative_numeric": 2200,
        "positive_negative_drawn": 2050,
    },
}

# These pairs are fixed before looking at test performance. They capture the
# natural product-family confounders in the dataset; the two unpaired product
# types form the final symmetric pair.
NEGATIVE_CLASS_PAIRS = {
    "Candy Boom": "Marine Boy",
    "Marine Boy": "Candy Boom",
    "Chocopie Dark": "Chocopie Nor",
    "Chocopie Nor": "Chocopie Dark",
    "OStar Red": "OStar Yellow",
    "OStar Yellow": "OStar Red",
    "Swing Maxx": "Swing Nor",
    "Swing Nor": "Swing Maxx",
}

LOGGER = logging.getLogger("qwen38_orion")


@dataclass(frozen=True)
class Task:
    mode: str
    image_id: int
    file_name: str
    width: int
    height: int
    category_id: int | None = None
    category_name: str | None = None

    @property
    def key(self) -> str:
        suffix = "all" if self.category_id is None else str(self.category_id)
        return f"{self.mode}__image_{self.image_id}__class_{suffix}"


@dataclass(frozen=True)
class ReferenceExample:
    category_id: int
    category_name: str
    image_id: int
    file_name: str
    width: int
    height: int
    boxes_xyxy_1000: tuple[tuple[int, int, int, int], ...]


class SmoothDualRateLimiter:
    """Uniformly pace starts under both request and estimated-token quotas."""

    def __init__(
        self,
        requests_per_minute: float,
        tokens_per_minute: float,
        *,
        clock: Any = time.monotonic,
        sleeper: Any = time.sleep,
    ) -> None:
        if requests_per_minute <= 0 or tokens_per_minute <= 0:
            raise ValueError("RPM and TPM targets must be positive.")
        self.request_interval = 60.0 / requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.clock = clock
        self.sleeper = sleeper
        self.lock = threading.Lock()
        self.next_start = 0.0

    def acquire(self, estimated_tokens: int) -> None:
        if estimated_tokens <= 0:
            raise ValueError("Estimated request tokens must be positive.")
        token_interval = 60.0 * estimated_tokens / self.tokens_per_minute
        interval = max(self.request_interval, token_interval)
        with self.lock:
            now = self.clock()
            scheduled = max(now, self.next_start)
            self.next_start = scheduled + interval
        delay = scheduled - now
        if delay > 0:
            self.sleeper(delay)


def estimated_request_tokens(task: Task, reasoning_effort: str) -> int:
    by_mode = ESTIMATED_TOTAL_TOKENS.get(reasoning_effort)
    if by_mode is not None:
        # Standalone extensions may add larger multi-reference modes. Their
        # conservative fallback prevents those requests from bursting TPM.
        return by_mode.get(task.mode, 10_000)
    # Medium/xhigh can consume most of the configured completion budget. Pace
    # them conservatively until a dedicated pilot supplies measured estimates.
    return 7000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def annotation_xywh_to_normalized_xyxy(
    bbox: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int]:
    if len(bbox) != 4 or width <= 0 or height <= 0:
        raise ValueError("Invalid COCO box or image dimensions.")
    x, y, box_width, box_height = (float(value) for value in bbox)
    values = (
        round(x * NORMALIZED_COORDINATE_MAX / width),
        round(y * NORMALIZED_COORDINATE_MAX / height),
        round((x + box_width) * NORMALIZED_COORDINATE_MAX / width),
        round((y + box_height) * NORMALIZED_COORDINATE_MAX / height),
    )
    x1, y1, x2, y2 = (
        min(NORMALIZED_COORDINATE_MAX, max(0, int(value))) for value in values
    )
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate normalized box: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    for key in ("images", "annotations", "categories"):
        if not isinstance(value.get(key), list):
            raise TypeError(f"{path} has no COCO {key!r} list.")
    return value


def validate_split_isolation(train: dict[str, Any], test: dict[str, Any]) -> None:
    train_names = {image["file_name"] for image in train["images"]}
    test_names = {image["file_name"] for image in test["images"]}
    overlap = train_names & test_names
    if overlap:
        raise ValueError(f"Train/test file-name overlap: {sorted(overlap)[:5]}")


def categories_by_id(coco: dict[str, Any]) -> dict[int, str]:
    result = {int(category["id"]): str(category["name"]) for category in coco["categories"]}
    if len(result) != len(coco["categories"]):
        raise ValueError("Duplicate COCO category IDs.")
    return result


def select_reference_examples(train: dict[str, Any]) -> dict[int, ReferenceExample]:
    """Select one deterministic, large, train-only example image per class."""

    images = {int(image["id"]): image for image in train["images"]}
    categories = categories_by_id(train)
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in train["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    examples: dict[int, ReferenceExample] = {}
    for category_id, category_name in categories.items():
        candidates: list[tuple[float, str, int]] = []
        for image_id, image_annotations in annotations_by_image.items():
            image = images[image_id]
            positives = [
                annotation
                for annotation in image_annotations
                if int(annotation["category_id"]) == category_id
            ]
            if not positives:
                continue
            image_area = float(image["width"] * image["height"])
            largest_relative_area = max(
                float(annotation["bbox"][2] * annotation["bbox"][3]) / image_area
                for annotation in positives
            )
            # Descending area, then stable file-name and image-ID tie breaks.
            candidates.append((-largest_relative_area, str(image["file_name"]), image_id))
        if not candidates:
            raise ValueError(f"No train example for {category_name!r}.")
        _, _, image_id = min(candidates)
        image = images[image_id]
        positives = sorted(
            (
                annotation
                for annotation in annotations_by_image[image_id]
                if int(annotation["category_id"]) == category_id
            ),
            key=lambda annotation: int(annotation["id"]),
        )
        boxes = tuple(
            annotation_xywh_to_normalized_xyxy(
                annotation["bbox"], int(image["width"]), int(image["height"])
            )
            for annotation in positives
        )
        examples[category_id] = ReferenceExample(
            category_id=category_id,
            category_name=category_name,
            image_id=image_id,
            file_name=str(image["file_name"]),
            width=int(image["width"]),
            height=int(image["height"]),
            boxes_xyxy_1000=boxes,
        )
    return examples


def validate_negative_pairs(
    categories: dict[int, str],
    negative_class_pairs: dict[str, str] | None = None,
) -> dict[int, int]:
    negative_class_pairs = negative_class_pairs or NEGATIVE_CLASS_PAIRS
    names_to_ids = {name: category_id for category_id, name in categories.items()}
    if set(names_to_ids) != set(negative_class_pairs):
        raise ValueError(
            "The fixed negative-class map does not exactly match dataset classes: "
            f"dataset={sorted(names_to_ids)}, map={sorted(negative_class_pairs)}"
        )
    unknown_targets = set(negative_class_pairs.values()) - set(names_to_ids)
    if unknown_targets:
        raise ValueError(f"Negative-class map contains unknown targets: {sorted(unknown_targets)}")
    self_pairs = sorted(name for name, negative in negative_class_pairs.items() if name == negative)
    if self_pairs:
        raise ValueError(f"Negative-class map contains self-pairs: {self_pairs}")
    return {
        category_id: names_to_ids[negative_class_pairs[name]]
        for category_id, name in categories.items()
    }


def render_reference(
    source: Path,
    boxes_xyxy_1000: Sequence[Sequence[int]],
    destination: Path,
    positive: bool,
) -> None:
    from PIL import Image, ImageDraw

    with Image.open(source) as opened:
        image = opened.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if positive:
        outline, fill = (0, 210, 100, 255), (0, 230, 118, 34)
    else:
        outline, fill = (255, 23, 68, 255), (255, 23, 68, 34)
    stroke = max(3, round(min(image.size) / 180))
    for box in boxes_xyxy_1000:
        x1, y1, x2, y2 = box
        pixels = (
            round(x1 * image.width / NORMALIZED_COORDINATE_MAX),
            round(y1 * image.height / NORMALIZED_COORDINATE_MAX),
            round(x2 * image.width / NORMALIZED_COORDINATE_MAX),
            round(y2 * image.height / NORMALIZED_COORDINATE_MAX),
        )
        draw.rectangle(pixels, outline=outline, fill=fill, width=stroke)
    rendered = Image.alpha_composite(image, overlay).convert("RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(destination, format="JPEG", quality=95)


def prepare_reference_assets(
    train_directory: Path,
    output_directory: Path,
    examples: dict[int, ReferenceExample],
    negative_ids: dict[int, int],
) -> dict[int, dict[str, Path]]:
    assets: dict[int, dict[str, Path]] = {}
    for category_id, example in examples.items():
        positive_source = train_directory / example.file_name
        negative_example = examples[negative_ids[category_id]]
        negative_source = train_directory / negative_example.file_name
        if not positive_source.is_file() or not negative_source.is_file():
            raise FileNotFoundError("A selected train reference image is missing.")
        directory = output_directory / f"class_{category_id}"
        positive_drawn = directory / "positive.jpg"
        negative_drawn = directory / "negative.jpg"
        render_reference(
            positive_source, example.boxes_xyxy_1000, positive_drawn, positive=True
        )
        render_reference(
            negative_source,
            negative_example.boxes_xyxy_1000,
            negative_drawn,
            positive=False,
        )
        assets[category_id] = {
            "positive_source": positive_source,
            "negative_source": negative_source,
            "positive_drawn": positive_drawn,
            "negative_drawn": negative_drawn,
        }
    return assets


def output_contract(class_names: Sequence[str]) -> str:
    return (
        "Return only a JSON list exactly like "
        '[{"bbox_2d":[x1,y1,x2,y2],"label":"exact class name"}]. '
        "Use XYXY integer coordinates normalized independently from 0 to 1000 "
        "relative to the TARGET IMAGE, with the origin at top-left. Use only "
        f"these labels: {json.dumps(list(class_names), ensure_ascii=False)}. "
        "Return [] if none are present."
    )


def build_messages(
    task: Task,
    test_directory: Path,
    categories: dict[int, str],
    examples: dict[int, ReferenceExample],
    negative_ids: dict[int, int],
    assets: dict[int, dict[str, Path]],
) -> list[dict[str, Any]]:
    target = test_directory / task.file_name
    if not target.is_file():
        raise FileNotFoundError(target)

    if task.mode == "multi_class_names":
        names = list(categories.values())
        prompt = (
            "Detect every instance of the listed classes in the TARGET IMAGE. "
            + output_contract(names)
        )
        content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "TARGET IMAGE:"},
            {"type": "image_url", "image_url": {"url": data_url(target)}},
        ]
        return [{"role": "user", "content": content}]

    if task.category_id is None or task.category_name is None:
        raise ValueError(f"Single-class task lacks a category: {task}")
    category_id = task.category_id
    name = task.category_name
    prompt = f'Detect every instance of "{name}" in the TARGET IMAGE. '
    if task.mode.startswith("positive_negative"):
        prompt += (
            "Match the positive visual concept and exclude objects matching the "
            "negative visual concept. "
        )
    elif task.mode.startswith("positive"):
        prompt += "Match the positive visual concept shown in the reference. "
    prompt += output_contract([name])
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

    if task.mode in {"positive_numeric", "positive_negative_numeric"}:
        positive = examples[category_id]
        positive_boxes = [list(box) for box in positive.boxes_xyxy_1000]
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "POSITIVE REFERENCE IMAGE: the following normalized XYXY "
                        f"boxes mark positive examples of {name}: "
                        f"{json.dumps(positive_boxes)}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url(assets[category_id]["positive_source"])},
                },
            ]
        )
        if task.mode in NEGATIVE_MODES:
            negative = examples[negative_ids[category_id]]
            negative_boxes = [list(box) for box in negative.boxes_xyxy_1000]
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            "NEGATIVE REFERENCE IMAGE: the following normalized XYXY "
                            "boxes mark visually related objects that must be excluded: "
                            f"{json.dumps(negative_boxes)}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url(assets[category_id]["negative_source"])
                        },
                    },
                ]
            )
    elif task.mode in DRAWN_MODES:
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "POSITIVE REFERENCE IMAGE: green boxes mark positive examples "
                        f"of {name}."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url(assets[category_id]["positive_drawn"])},
                },
            ]
        )
        if task.mode in NEGATIVE_MODES:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            "NEGATIVE REFERENCE IMAGE: red boxes mark visually related "
                            "objects that must be excluded."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url(assets[category_id]["negative_drawn"])},
                    },
                ]
            )
    elif task.mode != "single_class_names":
        raise ValueError(f"Unknown mode: {task.mode}")

    content.extend(
        [
            {"type": "text", "text": "TARGET IMAGE:"},
            {"type": "image_url", "image_url": {"url": data_url(target)}},
        ]
    )
    return [{"role": "user", "content": content}]


def build_tasks(test: dict[str, Any], categories: dict[int, str]) -> list[Task]:
    tasks: list[Task] = []
    for mode in MODES:
        for image in sorted(test["images"], key=lambda value: int(value["id"])):
            common = {
                "mode": mode,
                "image_id": int(image["id"]),
                "file_name": str(image["file_name"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
            }
            if mode == "multi_class_names":
                tasks.append(Task(**common))
            else:
                for category_id, category_name in categories.items():
                    tasks.append(
                        Task(
                            **common,
                            category_id=category_id,
                            category_name=category_name,
                        )
                    )
    if len({task.key for task in tasks}) != len(tasks):
        raise ValueError("Task keys are not unique.")
    return tasks


def record_path(output_directory: Path, task: Task) -> Path:
    return output_directory / "records" / task.mode / f"{task.key}.json"


def load_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def request_summary(messages: list[dict[str, Any]]) -> dict[str, Any]:
    content = messages[0]["content"]
    text_parts = [part["text"] for part in content if part["type"] == "text"]
    image_digests = []
    for part in content:
        if part["type"] != "image_url":
            continue
        url = part["image_url"]["url"]
        image_digests.append(hashlib.sha256(url.encode("utf-8")).hexdigest())
    return {"text_parts": text_parts, "image_sha256": image_digests}


def request_fingerprint(task: Task, summary: dict[str, Any], settings: dict[str, Any]) -> str:
    serialized = json.dumps(
        {"task": asdict(task), "request": summary, "settings": settings},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def expected_request_fingerprint(
    task: Task,
    test_directory: Path,
    categories: dict[int, str],
    examples: dict[int, ReferenceExample],
    negative_ids: dict[int, int],
    assets: dict[int, dict[str, Path]],
    settings: dict[str, Any],
) -> str:
    messages = build_messages(
        task, test_directory, categories, examples, negative_ids, assets
    )
    return request_fingerprint(task, request_summary(messages), settings)


def retryable_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    name = type(error).__name__.casefold()
    message = str(error).casefold()
    retryable_markers = (
        "timeout",
        "timed out",
        "connection",
        "rate limit",
        "too many requests",
        "internalerror.algo",
        "<408>",
        "<409>",
        "<429>",
        "<500>",
        "<502>",
        "<503>",
        "<504>",
    )
    return any(
        marker in name or marker in message for marker in retryable_markers
    )


def stream_inference(client: Any, messages: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    stream = client.chat.completions.create(
        model=settings["model"],
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        seed=settings["seed"],
        max_completion_tokens=settings["max_completion_tokens"],
        extra_body={
            "reasoning_effort": settings["reasoning_effort"],
            "vl_high_resolution_images": settings["vl_high_resolution_images"],
        },
        # DashScope queues traffic-burst throttles for up to 30 seconds instead
        # of immediately returning a 429. Absolute RPM/TPM limits still apply.
        extra_headers={"X-DashScope-Wait-Timeout": "30"},
    )
    parts: list[str] = []
    finish_reason = None
    usage = None
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage.model_dump()
        for choice in chunk.choices:
            if choice.delta.content:
                parts.append(choice.delta.content)
            if choice.finish_reason:
                finish_reason = choice.finish_reason
    return {
        "response": "".join(parts),
        "finish_reason": finish_reason,
        "usage": usage,
        "elapsed_seconds": time.monotonic() - started,
    }


def execute_task(
    task: Task,
    client: Any,
    test_directory: Path,
    categories: dict[int, str],
    examples: dict[int, ReferenceExample],
    negative_ids: dict[int, int],
    assets: dict[int, dict[str, Path]],
    settings: dict[str, Any],
    max_retries: int,
    rate_limiter: SmoothDualRateLimiter | None = None,
    messages_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    messages = (
        messages_override
        if messages_override is not None
        else build_messages(
            task, test_directory, categories, examples, negative_ids, assets
        )
    )
    summary = request_summary(messages)
    fingerprint = request_fingerprint(task, summary, settings)
    attempts = []
    for attempt in range(1, max_retries + 2):
        try:
            if rate_limiter is not None:
                rate_limiter.acquire(
                    estimated_request_tokens(task, settings["reasoning_effort"])
                )
            inference = stream_inference(client, messages, settings)
            raw = inference["response"]
            if inference["finish_reason"] == "length":
                return {
                    "status": "model_failure",
                    "failure_type": "truncated_response",
                    "task": asdict(task),
                    "task_key": task.key,
                    "request_fingerprint": fingerprint,
                    "request_summary": summary,
                    "raw_response": raw,
                    "finish_reason": inference["finish_reason"],
                    "usage": inference["usage"],
                    "predictions": [],
                    "attempts": attempts + [{"attempt": attempt, "status": "length"}],
                    "elapsed_seconds": time.monotonic() - started,
                    "completed_at": utc_now(),
                }
            try:
                detections = parse_cosmos_response(raw)
            except CosmosResponseError as error:
                return {
                    "status": "model_failure",
                    "failure_type": "invalid_response",
                    "error": str(error),
                    "task": asdict(task),
                    "task_key": task.key,
                    "request_fingerprint": fingerprint,
                    "request_summary": summary,
                    "raw_response": raw,
                    "finish_reason": inference["finish_reason"],
                    "usage": inference["usage"],
                    "predictions": [],
                    "attempts": attempts + [{"attempt": attempt, "status": "invalid_response"}],
                    "elapsed_seconds": time.monotonic() - started,
                    "completed_at": utc_now(),
                }
            allowed_categories = (
                categories
                if task.category_id is None
                else {task.category_id: categories[task.category_id]}
            )
            predictions, diagnostics = convert_detections_to_coco(
                detections,
                task.image_id,
                task.width,
                task.height,
                allowed_categories,
            )
            return {
                "status": "success",
                "task": asdict(task),
                "task_key": task.key,
                "request_fingerprint": fingerprint,
                "request_summary": summary,
                "raw_response": raw,
                "finish_reason": inference["finish_reason"],
                "usage": inference["usage"],
                "predictions": predictions,
                "diagnostics": diagnostics,
                "attempts": attempts + [{"attempt": attempt, "status": "success"}],
                "inference_seconds": inference["elapsed_seconds"],
                "elapsed_seconds": time.monotonic() - started,
                "completed_at": utc_now(),
            }
        except Exception as error:  # noqa: BLE001 - provider/SDK errors vary
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if attempt > max_retries or not retryable_error(error):
                return {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "task": asdict(task),
                    "task_key": task.key,
                    "request_fingerprint": fingerprint,
                    "request_summary": summary,
                    "raw_response": None,
                    "predictions": [],
                    "attempts": attempts,
                    "elapsed_seconds": time.monotonic() - started,
                    "completed_at": utc_now(),
                }
            delay = min(60.0, 2 ** (attempt - 1) + random.random())
            time.sleep(delay)
    raise AssertionError("Unreachable retry loop.")


def summarize_records(tasks: Sequence[Task], output_directory: Path) -> dict[str, Any]:
    counts = {mode: {"total": 0, "success": 0, "model_failure": 0, "error": 0, "pending": 0} for mode in MODES}
    for task in tasks:
        counts[task.mode]["total"] += 1
        record = load_record(record_path(output_directory, task))
        status = record.get("status") if record else "pending"
        if status not in counts[task.mode]:
            status = "error"
        counts[task.mode][status] += 1
    total = {key: sum(value[key] for value in counts.values()) for key in ("total", "success", "model_failure", "error", "pending")}
    return {"updated_at": utc_now(), "total": total, "modes": counts}


def finalize_modes(
    tasks: Sequence[Task],
    annotation_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    final: dict[str, Any] = {}
    for mode in MODES:
        predictions = []
        statuses: dict[str, int] = {}
        mode_tasks = [task for task in tasks if task.mode == mode]
        for task in mode_tasks:
            record = load_record(record_path(output_directory, task))
            status = record.get("status", "missing") if record else "missing"
            statuses[status] = statuses.get(status, 0) + 1
            if record and status in TERMINAL_STATUSES:
                predictions.extend(record.get("predictions", []))
        predictions_path = output_directory / "predictions" / f"{mode}.json"
        atomic_write_json(predictions_path, predictions)
        complete = sum(statuses.get(value, 0) for value in TERMINAL_STATUSES) == len(mode_tasks)
        metrics = score_coco(annotation_path, predictions) if complete else None
        mode_summary = {
            "mode": mode,
            "complete": complete,
            "task_count": len(mode_tasks),
            "statuses": statuses,
            "prediction_count": len(predictions),
            "predictions_path": str(predictions_path),
            "metrics": metrics,
        }
        atomic_write_json(output_directory / "metrics" / f"{mode}.json", mode_summary)
        final[mode] = mode_summary
    aggregate = {"updated_at": utc_now(), "modes": final}
    atomic_write_json(output_directory / "aggregate_metrics.json", aggregate)
    return aggregate


def configure_logging(output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        handlers=[
            logging.FileHandler(output_directory / "experiment.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def write_or_validate_manifest(
    path: Path,
    configuration: dict[str, Any],
    examples: dict[int, ReferenceExample],
    negative_ids: dict[int, int],
) -> None:
    expected = {
        "prompt_version": PROMPT_VERSION,
        "configuration": configuration,
        "references": {str(key): asdict(value) for key, value in examples.items()},
        "negative_category_ids": {str(key): value for key, value in negative_ids.items()},
    }
    # Normalize tuples from frozen dataclasses to their JSON representation so
    # a manifest compares identically before and after serialization.
    expected = json.loads(json.dumps(expected, ensure_ascii=False))
    existing = load_record(path)
    if existing:
        comparable = {key: existing.get(key) for key in expected}
        if comparable != expected:
            raise ValueError(
                f"Existing run manifest does not match this configuration: {path}"
            )
        return
    expected["created_at"] = utc_now()
    atomic_write_json(path, expected)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("RF100VL/rf20-vl-fsod/orionproducts"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qwen38-orion-runs/orion-prompt-modes-v1"),
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--negative-pairs-file",
        type=Path,
        help="JSON object mapping every class name to a different negative class.",
    )
    parser.add_argument("--base-url", default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--requests-per-minute",
        type=float,
        default=DEFAULT_REQUESTS_PER_MINUTE,
        help="Smooth account-wide request target; Singapore qwen3.8-max limit is 600.",
    )
    parser.add_argument(
        "--tokens-per-minute",
        type=float,
        default=DEFAULT_TOKENS_PER_MINUTE,
        help="Estimated-token pacing target; Singapore qwen3.8-max limit is 1,000,000.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "xhigh"),
        default="low",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--image-ids", nargs="+", type=int, help="Run only these test image IDs.")
    parser.add_argument("--category-ids", nargs="+", type=int, help="Run only these single-class category IDs.")
    parser.add_argument("--limit", type=int, help="Run only the first N pending tasks (smoke tests).")
    parser.add_argument(
        "--limit-per-mode",
        type=int,
        help="Run only the first N pending tasks in each selected mode.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.concurrency < 1 or args.max_retries < 0:
        raise ValueError("Concurrency must be positive and retries nonnegative.")
    if args.requests_per_minute <= 0 or args.tokens_per_minute <= 0:
        raise ValueError("RPM and TPM targets must be positive.")
    if args.limit is not None and args.limit_per_mode is not None:
        raise ValueError("--limit and --limit-per-mode are mutually exclusive.")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be nonnegative.")
    if args.limit_per_mode is not None and args.limit_per_mode < 0:
        raise ValueError("--limit-per-mode must be nonnegative.")
    if not os.getenv("DASHSCOPE_API_KEY") and not args.prepare_only:
        raise RuntimeError("DASHSCOPE_API_KEY is required for inference.")

    dataset_directory = args.dataset_dir.resolve()
    output_directory = args.output_dir.resolve()
    configure_logging(output_directory)
    train_directory = dataset_directory / "train"
    test_directory = dataset_directory / "test"
    train_path = train_directory / "_annotations.coco.json"
    test_path = test_directory / "_annotations.coco.json"
    train, test = load_coco(train_path), load_coco(test_path)
    validate_split_isolation(train, test)
    categories = categories_by_id(test)
    if categories != categories_by_id(train):
        raise ValueError("Train/test categories differ.")
    negative_class_pairs = NEGATIVE_CLASS_PAIRS
    if args.negative_pairs_file is not None:
        loaded_pairs = json.loads(
            args.negative_pairs_file.resolve().read_text(encoding="utf-8")
        )
        if not isinstance(loaded_pairs, dict):
            raise ValueError("Negative-class pairs must be a JSON object.")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in loaded_pairs.items()):
            raise ValueError("Negative-class pairs must be a JSON string-to-string object.")
        negative_class_pairs = loaded_pairs
    examples = select_reference_examples(train)
    negative_ids = validate_negative_pairs(categories, negative_class_pairs)
    assets = prepare_reference_assets(
        train_directory, output_directory / "references", examples, negative_ids
    )
    all_tasks = build_tasks(test, categories)
    selected_modes = set(args.modes)
    tasks = [task for task in all_tasks if task.mode in selected_modes]
    if args.image_ids is not None:
        requested_image_ids = set(args.image_ids)
        available_image_ids = {task.image_id for task in all_tasks}
        missing_image_ids = requested_image_ids - available_image_ids
        if missing_image_ids:
            raise ValueError(f"Unknown test image IDs: {sorted(missing_image_ids)}")
        tasks = [task for task in tasks if task.image_id in requested_image_ids]
    if args.category_ids is not None:
        requested_category_ids = set(args.category_ids)
        missing_category_ids = requested_category_ids - set(categories)
        if missing_category_ids:
            raise ValueError(f"Unknown category IDs: {sorted(missing_category_ids)}")
        tasks = [
            task
            for task in tasks
            if task.category_id is None or task.category_id in requested_category_ids
        ]
    settings = {
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "seed": args.seed,
        "max_completion_tokens": args.max_completion_tokens,
        "reasoning_effort": args.reasoning_effort,
        "vl_high_resolution_images": False,
        "timeout_seconds": args.timeout_seconds,
    }
    configuration = {
        "dataset_directory": str(dataset_directory),
        "train_annotation_sha256": sha256_file(train_path),
        "test_annotation_sha256": sha256_file(test_path),
        "settings": settings,
        "modes": list(MODES),
        "negative_class_pairs": negative_class_pairs,
    }
    write_or_validate_manifest(
        output_directory / "run_manifest.json", configuration, examples, negative_ids
    )
    atomic_write_json(output_directory / "progress.json", summarize_records(all_tasks, output_directory))
    LOGGER.info(
        "Prepared %d tasks (%d selected), %d test images, %d classes.",
        len(all_tasks), len(tasks), len(test["images"]), len(categories),
    )
    if args.prepare_only:
        finalize_modes(all_tasks, test_path, output_directory)
        return 0

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=settings["base_url"],
        timeout=settings["timeout_seconds"],
        max_retries=0,
    )
    pending = []
    for task in tasks:
        existing = load_record(record_path(output_directory, task))
        if existing and existing.get("status") in TERMINAL_STATUSES:
            expected_fingerprint = expected_request_fingerprint(
                task,
                test_directory,
                categories,
                examples,
                negative_ids,
                assets,
                settings,
            )
            if (
                existing.get("task_key") != task.key
                or existing.get("request_fingerprint") != expected_fingerprint
            ):
                raise ValueError(
                    "A terminal checkpoint does not match the current request; "
                    f"use a new output directory instead of mixing runs: {task.key}"
                )
            continue
        pending.append(task)
    if args.limit is not None:
        pending = pending[: args.limit]
    elif args.limit_per_mode is not None:
        limited: list[Task] = []
        for mode in MODES:
            mode_tasks = [task for task in pending if task.mode == mode]
            limited.extend(mode_tasks[: args.limit_per_mode])
        pending = limited
    rate_limiter = SmoothDualRateLimiter(
        args.requests_per_minute,
        args.tokens_per_minute,
    )
    LOGGER.info(
        "Starting %d pending API tasks with concurrency=%d, target_rpm=%.1f, "
        "target_tpm=%.0f.",
        len(pending),
        args.concurrency,
        args.requests_per_minute,
        args.tokens_per_minute,
    )

    write_lock = threading.Lock()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                execute_task,
                task,
                client,
                test_directory,
                categories,
                examples,
                negative_ids,
                assets,
                settings,
                args.max_retries,
                rate_limiter,
            ): task
            for task in pending
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 - preserve worker failure
                record = {
                    "status": "error",
                    "error": f"WorkerFailure: {type(error).__name__}: {error}",
                    "task": asdict(task),
                    "task_key": task.key,
                    "predictions": [],
                    "completed_at": utc_now(),
                }
            with write_lock:
                atomic_write_json(record_path(output_directory, task), record)
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    progress = summarize_records(all_tasks, output_directory)
                    atomic_write_json(output_directory / "progress.json", progress)
                    LOGGER.info(
                        "Checkpoint %d/%d this invocation; overall terminal=%d/%d errors=%d.",
                        completed,
                        len(pending),
                        progress["total"]["success"] + progress["total"]["model_failure"],
                        progress["total"]["total"],
                        progress["total"]["error"],
                    )

    progress = summarize_records(all_tasks, output_directory)
    atomic_write_json(output_directory / "progress.json", progress)
    aggregate = finalize_modes(all_tasks, test_path, output_directory)
    failures = progress["total"]["error"] + progress["total"]["pending"]
    complete_modes = sum(value["complete"] for value in aggregate["modes"].values())
    LOGGER.info(
        "Invocation finished: terminal=%d/%d, complete_modes=%d/%d, unresolved=%d.",
        progress["total"]["success"] + progress["total"]["model_failure"],
        progress["total"]["total"],
        complete_modes,
        len(MODES),
        failures,
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
