#!/usr/bin/env python3
"""Zero-shot RF100-VL object detection with NVIDIA Cosmos3-Edge.

This evaluator intentionally implements only the RF100-VL "basic" setting:
each request contains one test image, the complete class list for that dataset,
and a fixed output schema. It never reads train images or dataset instructions.

The model is accessed through an OpenAI-compatible vLLM endpoint. Cosmos emits
2D boxes as normalized [x1, y1, x2, y2] coordinates in the range 0..1000;
this script converts them to COCO [x, y, width, height] boxes, checkpoints one
record per image, and computes per-dataset COCO AP with maxDets=500.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import contextlib
import hashlib
import io
import json
import logging
import math
import mimetypes
import os
from pathlib import Path, PurePosixPath
import random
import re
import statistics
import threading
import time
from typing import Any, Iterable, Sequence
from urllib.parse import quote


MODEL_ID = "nvidia/Cosmos3-Edge"
SYSTEM_PROMPT = "You are a helpful assistant."
PROMPT_VERSION = "cosmos3-edge-rf100-basic-v1"
NORMALIZED_COORDINATE_MAX = 1000.0

LOGGER = logging.getLogger("rf100_cosmos")


class CosmosResponseError(ValueError):
    """Raised when a Cosmos response cannot be used as a detection result."""


def normalize_label(value: str) -> str:
    """Normalize only casing and whitespace; do not perform fuzzy matching."""

    return " ".join(value.strip().split()).casefold()


def build_detection_prompt(category_names: Sequence[str]) -> str:
    """Build the fixed zero-shot prompt from the full dataset class list."""

    if not category_names:
        raise ValueError("The annotation file contains no categories.")
    encoded_categories = json.dumps(list(category_names), ensure_ascii=False)
    return (
        "Locate every instance of each of the following object classes in the image:\n"
        f"{encoded_categories}\n\n"
        "Return only a JSON array in this exact form:\n"
        '[{"bbox_2d":[x1,y1,x2,y2],"label":"one class name exactly"}]\n\n'
        "Use integer coordinates normalized independently to 0–1000, with the "
        "origin at the top-left. Include one entry per object instance, use only "
        "the listed class names, and return [] if none are present."
    )


def _json_candidates(response_text: str) -> Iterable[str]:
    without_thinking = re.sub(
        r"<think>.*?</think>", "", response_text, flags=re.IGNORECASE | re.DOTALL
    ).strip()
    for fenced in re.findall(
        r"```(?:json)?\s*([\s\S]*?)```", without_thinking, flags=re.IGNORECASE
    ):
        yield fenced.strip()
    yield without_thinking


def _find_json_value(text: str) -> Any:
    """Decode the first usable JSON array/object embedded in text."""

    decoder = json.JSONDecoder()
    starts = [index for index, character in enumerate(text) if character in "[{"]
    for start in starts:
        candidate = text[start:]
        for repaired in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                value, _ = decoder.raw_decode(repaired)
            except json.JSONDecodeError:
                continue
            if isinstance(value, (list, dict)):
                return value
    raise CosmosResponseError(
        "No valid JSON array or object was found in the response."
    )


def parse_cosmos_response(response_text: str) -> list[dict[str, Any]]:
    """Parse Cosmos JSON, accepting an array, one box, or a common list wrapper."""

    if not isinstance(response_text, str) or not response_text.strip():
        raise CosmosResponseError("The model returned an empty response.")

    last_error: Exception | None = None
    for candidate in _json_candidates(response_text):
        try:
            value = _find_json_value(candidate)
        except CosmosResponseError as error:
            last_error = error
            continue

        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if "bbox_2d" in value:
            return [value]
        for key in ("detections", "boxes", "objects", "annotations"):
            wrapped = value.get(key)
            if isinstance(wrapped, list):
                return [item for item in wrapped if isinstance(item, dict)]
        last_error = CosmosResponseError(
            f"JSON object has no bbox_2d or supported detection list: {sorted(value)}"
        )

    raise CosmosResponseError(str(last_error or "Could not parse the response JSON."))


def _as_finite_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean coordinates are invalid.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Coordinates must be finite.")
    return result


def convert_detections_to_coco(
    detections: Sequence[dict[str, Any]],
    image_id: int | str,
    image_width: int,
    image_height: int,
    categories_by_id: dict[int, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert strict Cosmos normalized xyxy detections to COCO predictions."""

    labels_to_ids: dict[str, int] = {}
    for category_id, category_name in categories_by_id.items():
        normalized = normalize_label(category_name)
        if normalized in labels_to_ids:
            raise ValueError(
                "Category names are not unique after case/whitespace normalization: "
                f"{category_name!r}"
            )
        labels_to_ids[normalized] = category_id

    predictions: list[dict[str, Any]] = []
    ignored_labels: list[str] = []
    invalid_boxes = 0
    duplicate_boxes = 0
    seen: set[tuple[Any, ...]] = set()

    for detection in detections:
        label = detection.get("label")
        bbox = detection.get("bbox_2d")
        if not isinstance(label, str) or normalize_label(label) not in labels_to_ids:
            ignored_labels.append(str(label))
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            invalid_boxes += 1
            continue
        try:
            x1, y1, x2, y2 = (_as_finite_number(value) for value in bbox)
        except (TypeError, ValueError):
            invalid_boxes += 1
            continue

        x1, y1, x2, y2 = (
            min(NORMALIZED_COORDINATE_MAX, max(0.0, value))
            for value in (x1, y1, x2, y2)
        )
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        x1_px = x1 * image_width / NORMALIZED_COORDINATE_MAX
        y1_px = y1 * image_height / NORMALIZED_COORDINATE_MAX
        x2_px = x2 * image_width / NORMALIZED_COORDINATE_MAX
        y2_px = y2 * image_height / NORMALIZED_COORDINATE_MAX
        width = x2_px - x1_px
        height = y2_px - y1_px
        if width <= 0 or height <= 0:
            invalid_boxes += 1
            continue

        category_id = labels_to_ids[normalize_label(label)]
        bbox_coco = [x1_px, y1_px, width, height]
        duplicate_key = (category_id, *(round(value, 6) for value in bbox_coco))
        if duplicate_key in seen:
            duplicate_boxes += 1
            continue
        seen.add(duplicate_key)
        predictions.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox_coco,
                # Cosmos does not expose detector confidences. Keep parity with
                # the repository's existing generative-VLM evaluators.
                "score": 1.0,
            }
        )

    diagnostics = {
        "parsed_detections": len(detections),
        "accepted_detections": len(predictions),
        "invalid_boxes": invalid_boxes,
        "duplicate_boxes": duplicate_boxes,
        "ignored_labels": ignored_labels,
    }
    return predictions, diagnostics


def _image_record_name(image_id: int | str) -> str:
    serialized = json.dumps(image_id, sort_keys=True, ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(serialized).hexdigest()[:24] + ".json"


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_annotation_file(test_directory: Path) -> Path:
    candidates = sorted(test_directory.glob("*_annotations.coco.json"))
    if not candidates:
        candidates = sorted(test_directory.glob("*.coco.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one COCO annotation file in {test_directory}; "
            f"found {len(candidates)}."
        )
    return candidates[0]


def resolve_image_path(test_directory: Path, file_name: str) -> Path:
    direct = test_directory / file_name
    if direct.is_file():
        return direct
    matches = list(test_directory.rglob(Path(file_name).name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not uniquely resolve {file_name!r} below {test_directory}; "
            f"found {len(matches)} matches."
        )
    return matches[0]


def _load_pillow_image(path: Path):
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required. Install requirements-cosmos.txt."
        ) from error
    return Image.open(path)


def prepare_image_reference(
    image_path: Path,
    expected_width: int,
    expected_height: int,
    transport: str,
    dataset_root: Path,
    server_media_root: str | None,
) -> str:
    """Validate image geometry and return a data URL or server-local file URL."""

    with _load_pillow_image(image_path) as image:
        if image.size != (expected_width, expected_height):
            raise ValueError(
                f"Image dimensions {image.size} do not match COCO metadata "
                f"{(expected_width, expected_height)} for {image_path}."
            )
        mode = image.mode
        image_format = image.format

        if transport == "file-url":
            if not server_media_root:
                raise ValueError(
                    "--server-media-root is required with --image-transport file-url."
                )
            if mode not in ("RGB", "RGBA"):
                raise ValueError(
                    f"{image_path} uses unsupported mode {mode!r}; use data-url "
                    "transport so the evaluator can convert it to RGB."
                )
            relative = image_path.resolve().relative_to(dataset_root.resolve())
            server_path = PurePosixPath(server_media_root) / PurePosixPath(
                relative.as_posix()
            )
            if not server_path.is_absolute():
                raise ValueError(
                    "--server-media-root must be an absolute path in the server."
                )
            return "file://" + quote(str(server_path), safe="/")

        if mode not in ("RGB", "RGBA"):
            converted = image.convert("RGB")
            buffer = io.BytesIO()
            converted.save(buffer, format="PNG")
            data = buffer.getvalue()
            mime_type = "image/png"
        else:
            data = image_path.read_bytes()
            mime_type = None
            try:
                from PIL import Image

                mime_type = Image.MIME.get(image_format)
            except (ImportError, KeyError):
                pass
            mime_type = mime_type or mimetypes.guess_type(image_path.name)[0]
            if not mime_type or not mime_type.startswith("image/"):
                raise ValueError(
                    f"Could not determine image MIME type for {image_path}."
                )

    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class CosmosInferenceClient:
    """Thread-local OpenAI clients plus deterministic retry behavior."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._thread_local = threading.local()

    def _client(self):
        client = getattr(self._thread_local, "client", None)
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    "The openai package is required. Install requirements-cosmos.txt."
                ) from error
            client = OpenAI(
                base_url=self.args.base_url,
                api_key=self.args.api_key,
                timeout=self.args.timeout,
                max_retries=0,
            )
            self._thread_local.client = client
        return client

    def infer(self, image_reference: str, prompt: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                # NVIDIA specifies media before text for Cosmos grounding.
                "content": [
                    {"type": "image_url", "image_url": {"url": image_reference}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        extra_body = {
            "chat_template_kwargs": {"enable_thinking": self.args.enable_thinking}
        }
        last_error: Exception | None = None
        for attempt in range(self.args.retries + 1):
            try:
                started = time.monotonic()
                response = self._client().chat.completions.create(
                    model=self.args.model_id,
                    messages=messages,
                    temperature=0,
                    seed=self.args.seed,
                    max_tokens=self.args.max_tokens,
                    extra_body=extra_body,
                )
                elapsed = time.monotonic() - started
                if not response.choices:
                    raise CosmosResponseError("The API returned no choices.")
                choice = response.choices[0]
                content = choice.message.content
                if not isinstance(content, str):
                    raise CosmosResponseError(
                        f"Expected string response content, received {type(content).__name__}."
                    )
                if choice.finish_reason == "length":
                    raise CosmosResponseError(
                        "The response hit max_tokens and may contain incomplete detections."
                    )
                usage = None
                if response.usage is not None:
                    usage = response.usage.model_dump()
                return {
                    "response": content,
                    "finish_reason": choice.finish_reason,
                    "usage": usage,
                    "elapsed_seconds": elapsed,
                }
            except Exception as error:  # API exception types vary by openai version.
                last_error = error
                if attempt >= self.args.retries:
                    break
                delay = min(
                    self.args.retry_max_delay,
                    self.args.retry_base_delay * (2**attempt),
                )
                delay += random.uniform(0, delay * 0.1)
                LOGGER.warning(
                    "Inference attempt %d/%d failed: %s; retrying in %.1fs",
                    attempt + 1,
                    self.args.retries + 1,
                    error,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"Inference failed after {self.args.retries + 1} attempts: {last_error}"
        ) from last_error


def process_image(
    image_info: dict[str, Any],
    test_directory: Path,
    dataset_root: Path,
    categories_by_id: dict[int, str],
    prompt: str,
    inference_client: CosmosInferenceClient,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_id = image_info["id"]
    file_name = image_info["file_name"]
    started = time.monotonic()
    raw_response: str | None = None
    try:
        image_path = resolve_image_path(test_directory, file_name)
        image_reference = prepare_image_reference(
            image_path=image_path,
            expected_width=int(image_info["width"]),
            expected_height=int(image_info["height"]),
            transport=args.image_transport,
            dataset_root=dataset_root,
            server_media_root=args.server_media_root,
        )
        inference = inference_client.infer(image_reference, prompt)
        raw_response = inference["response"]
        detections = parse_cosmos_response(raw_response)
        predictions, diagnostics = convert_detections_to_coco(
            detections=detections,
            image_id=image_id,
            image_width=int(image_info["width"]),
            image_height=int(image_info["height"]),
            categories_by_id=categories_by_id,
        )
        return {
            "status": "success",
            "image_id": image_id,
            "file_name": file_name,
            "predictions": predictions,
            "diagnostics": diagnostics,
            "raw_response": raw_response,
            "finish_reason": inference["finish_reason"],
            "usage": inference["usage"],
            "inference_seconds": inference["elapsed_seconds"],
            "total_seconds": time.monotonic() - started,
        }
    except Exception as error:
        return {
            "status": "error",
            "image_id": image_id,
            "file_name": file_name,
            "error": f"{type(error).__name__}: {error}",
            "raw_response": raw_response,
            "total_seconds": time.monotonic() - started,
        }


def load_records(record_directory: Path) -> dict[int | str, dict[str, Any]]:
    records: dict[int | str, dict[str, Any]] = {}
    if not record_directory.exists():
        return records
    for path in sorted(record_directory.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as file:
                record = json.load(file)
            if record.get("status") == "success" and "image_id" in record:
                records[record["image_id"]] = record
        except (OSError, json.JSONDecodeError) as error:
            LOGGER.warning("Ignoring unreadable checkpoint %s: %s", path, error)
    return records


def append_json_line(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def visualize_predictions(
    image_path: Path,
    predictions: Sequence[dict[str, Any]],
    categories_by_id: dict[int, str],
    output_path: Path,
) -> None:
    try:
        from PIL import ImageDraw
    except ImportError as error:
        raise RuntimeError("Pillow is required for visualization.") from error

    with _load_pillow_image(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, round(min(canvas.size) / 300))
    for prediction in predictions:
        x, y, width, height = prediction["bbox"]
        label = categories_by_id[prediction["category_id"]]
        draw.rectangle(
            (x, y, x + width, y + height), outline=(255, 45, 45), width=line_width
        )
        draw.text(
            (x + line_width, y + line_width), label, fill=(255, 45, 45), stroke_width=1
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=92)


def score_coco(
    annotation_path: Path, predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run bbox COCOeval with RF100-VL's maxDets=500 convention."""

    try:
        import numpy as np
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError(
            "pycocotools is required for scoring. Install requirements-cosmos.txt."
        ) from error

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        ground_truth = COCO(str(annotation_path))
        if predictions:
            detections = ground_truth.loadRes(predictions)
        else:
            # COCO.loadRes indexes anns[0] and cannot load an empty list.
            detections = COCO()
            detections.dataset = {
                "images": ground_truth.dataset.get("images", []),
                "categories": ground_truth.dataset.get("categories", []),
                "annotations": [],
            }
            detections.createIndex()
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.maxDets = [1, 10, 500]
        evaluator.evaluate()
        evaluator.accumulate()

    def mean_valid(values: Any) -> float:
        valid = values[values > -1]
        return float(np.mean(valid)) if valid.size else -1.0

    def area_index(label: str) -> int:
        return evaluator.params.areaRngLbl.index(label)

    def max_dets_index(max_dets: int) -> int:
        return evaluator.params.maxDets.index(max_dets)

    def mean_precision(
        iou_threshold: float | None = None,
        area: str = "all",
        max_dets: int = 500,
    ) -> float:
        # precision dimensions: [IoU, recall, category, area, maxDets]
        values = evaluator.eval["precision"]
        if iou_threshold is not None:
            indices = np.where(np.isclose(evaluator.params.iouThrs, iou_threshold))[0]
            values = values[indices]
        values = values[:, :, :, area_index(area), max_dets_index(max_dets)]
        return mean_valid(values)

    def mean_recall(area: str = "all", max_dets: int = 500) -> float:
        # recall dimensions: [IoU, category, area, maxDets]
        values = evaluator.eval["recall"][
            :, :, area_index(area), max_dets_index(max_dets)
        ]
        return mean_valid(values)

    # pycocotools.summarize() hard-codes maxDets=100 for stats[0]. Once RF100-VL
    # replaces 100 with 500, that slot incorrectly becomes -1. Compute every
    # metric from the accumulated tensors at the explicitly requested maxDets.
    metrics = {
        "AP": mean_precision(),
        "AP50": mean_precision(0.50),
        "AP75": mean_precision(0.75),
        "AP_small": mean_precision(area="small"),
        "AP_medium": mean_precision(area="medium"),
        "AP_large": mean_precision(area="large"),
        "AR_1": mean_recall(max_dets=1),
        "AR_10": mean_recall(max_dets=10),
        "AR_500": mean_recall(max_dets=500),
        "AR_small": mean_recall(area="small"),
        "AR_medium": mean_recall(area="medium"),
        "AR_large": mean_recall(area="large"),
    }
    metric_output = [
        f"{name} @ maxDets={500 if name not in ('AR_1', 'AR_10') else name.split('_')[1]}: {value:.4f}"
        for name, value in metrics.items()
    ]
    metrics.update(
        {
            "max_dets": [1, 10, 500],
            "prediction_count": len(predictions),
            "cocoeval_output": captured.getvalue() + "\n".join(metric_output) + "\n",
        }
    )
    return metrics


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "model"


def discover_datasets(root: Path, requested_names: set[str] | None) -> list[Path]:
    if (root / "test").is_dir():
        candidates = [root]
    else:
        candidates = sorted(path for path in root.iterdir() if (path / "test").is_dir())
    if requested_names is not None:
        candidates = [path for path in candidates if path.name in requested_names]
        missing = requested_names - {path.name for path in candidates}
        if missing:
            raise FileNotFoundError(
                f"Requested datasets were not found: {sorted(missing)}"
            )
    return candidates


def run_dataset(
    dataset_directory: Path,
    dataset_root: Path,
    output_root: Path,
    inference_client: CosmosInferenceClient,
    run_config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    dataset_name = dataset_directory.name
    test_directory = dataset_directory / "test"
    annotation_path = resolve_annotation_file(test_directory)
    with annotation_path.open("r", encoding="utf-8") as file:
        coco_data = json.load(file)

    categories = coco_data.get("categories", [])
    categories_by_id = {
        int(category["id"]): str(category["name"]) for category in categories
    }
    category_names = [str(category["name"]) for category in categories]
    prompt = build_detection_prompt(category_names)
    images = list(coco_data.get("images", []))
    if not images:
        raise ValueError(f"No images found in {annotation_path}.")

    annotation_hash = sha256_file(annotation_path)
    run_hash = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    result_directory = output_root / dataset_name
    record_directory = (
        result_directory / "records" / f"{run_hash}-{annotation_hash[:12]}"
    )
    visualization_directory = result_directory / "visualizations"
    record_directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        result_directory / f"run_config_{run_hash}.json",
        {
            **run_config,
            "annotation_path": str(annotation_path),
            "annotation_sha256": annotation_hash,
            "categories": categories,
            "prompt": prompt,
        },
    )

    records = load_records(record_directory)
    known_image_ids = {image["id"] for image in images}
    records = {
        image_id: record
        for image_id, record in records.items()
        if image_id in known_image_ids
    }

    selected_images = images[: args.max_images] if args.max_images else images
    pending = [image for image in selected_images if image["id"] not in records]
    LOGGER.info(
        "%s: %d images total, %d selected, %d resumed, %d pending",
        dataset_name,
        len(images),
        len(selected_images),
        len(selected_images) - len(pending),
        len(pending),
    )

    errors: list[dict[str, Any]] = []
    if pending:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            future_to_image = {
                executor.submit(
                    process_image,
                    image,
                    test_directory,
                    dataset_root,
                    categories_by_id,
                    prompt,
                    inference_client,
                    args,
                ): image
                for image in pending
            }
            try:
                from tqdm import tqdm

                completed_futures: Iterable[Any] = tqdm(
                    concurrent.futures.as_completed(future_to_image),
                    total=len(future_to_image),
                    desc=dataset_name,
                    unit="image",
                    leave=False,
                )
            except ImportError:
                completed_futures = concurrent.futures.as_completed(future_to_image)

            for future in completed_futures:
                result = future.result()
                image_id = result["image_id"]
                if result["status"] == "success":
                    atomic_write_json(
                        record_directory / _image_record_name(image_id), result
                    )
                    records[image_id] = result
                    ignored = result["diagnostics"].get("ignored_labels", [])
                    if ignored:
                        LOGGER.warning(
                            "%s/%s: ignored labels not exactly in class list: %s",
                            dataset_name,
                            result["file_name"],
                            ignored,
                        )
                else:
                    errors.append(result)
                    append_json_line(
                        result_directory / f"errors_{run_hash}.jsonl", result
                    )
                    LOGGER.error(
                        "%s/%s failed: %s",
                        dataset_name,
                        result["file_name"],
                        result["error"],
                    )

    # Visualization is derived from checkpoints so it also works on resumed runs.
    for image_info in selected_images[: args.visualize_limit]:
        record = records.get(image_info["id"])
        if not record:
            continue
        output_name = f"{image_info['id']}_{Path(image_info['file_name']).stem}.jpg"
        output_path = visualization_directory / output_name
        if output_path.exists():
            continue
        try:
            visualize_predictions(
                resolve_image_path(test_directory, image_info["file_name"]),
                record["predictions"],
                categories_by_id,
                output_path,
            )
        except Exception as error:
            LOGGER.warning(
                "%s/%s visualization failed: %s",
                dataset_name,
                image_info["file_name"],
                error,
            )

    predictions: list[dict[str, Any]] = []
    for image in images:
        record = records.get(image["id"])
        if record:
            predictions.extend(record.get("predictions", []))
    predictions_path = result_directory / "cosmos_detection_results.json"
    atomic_write_json(predictions_path, predictions)

    completed_image_count = sum(image["id"] in records for image in images)
    complete = completed_image_count == len(images)
    result: dict[str, Any] = {
        "dataset": dataset_name,
        "image_count": len(images),
        "completed_image_count": completed_image_count,
        "new_error_count": len(errors),
        "prediction_count": len(predictions),
        "complete": complete,
        "predictions_path": str(predictions_path),
    }
    if not args.skip_scoring and complete:
        metrics = score_coco(annotation_path, predictions)
        result["metrics"] = metrics
        LOGGER.info(
            "%s: AP=%.4f AP50=%.4f (%d predictions)",
            dataset_name,
            metrics["AP"],
            metrics["AP50"],
            len(predictions),
        )
    elif not args.skip_scoring:
        result["score_skipped_reason"] = (
            f"Dataset is incomplete ({completed_image_count}/{len(images)} images)."
        )
        LOGGER.warning("%s: %s", dataset_name, result["score_skipped_reason"])

    atomic_write_json(result_directory / "summary.json", result)
    return result


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "prompt_version": PROMPT_VERSION,
        "model_id": args.model_id,
        "system_prompt": SYSTEM_PROMPT,
        "temperature": 0,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "image_transport": args.image_transport,
        # Online vLLM scheduling can introduce small numerical differences, so
        # do not mix checkpoints produced with different client concurrency.
        "workers": args.workers,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Cosmos3-Edge basic zero-shot detection on RF100-VL."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("rf100-vl"),
        help="RF100-VL root, or one dataset directory containing test/.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Run only this dataset name; repeat to select several.",
    )
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Smoke-test only the first N images per dataset. Incomplete datasets are not scored.",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("COSMOS_API_KEY", "EMPTY"))
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent requests. Default 1 is the conservative reproducible setting.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--retry-max-delay", type=float, default=60.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Cosmos reasoning. Disabled by default for deterministic detection.",
    )
    parser.add_argument(
        "--image-transport",
        choices=("data-url", "file-url"),
        default="data-url",
        help="Use data URLs by default, or server-local file URLs for lower transfer overhead.",
    )
    parser.add_argument(
        "--server-media-root",
        help="Absolute path where --dataset-dir is mounted in vLLM (required for file-url).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Output root. Defaults to results_Cosmos3-Edge_basic.",
    )
    parser.add_argument("--visualize-limit", type=int, default=0)
    parser.add_argument("--skip-scoring", action="store_true")
    args = parser.parse_args(argv)

    for name in ("workers", "max_tokens", "retries"):
        if getattr(args, name) < (0 if name == "retries" else 1):
            parser.error(f"--{name.replace('_', '-')} has an invalid value.")
    for name in ("max_datasets", "max_images"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.visualize_limit < 0:
        parser.error("--visualize-limit cannot be negative.")
    if args.image_transport == "file-url" and not args.server_media_root:
        parser.error("--server-media-root is required with file-url transport.")
    return args


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_dir.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    output_root = args.save_dir or Path(
        f"results_{_safe_component(args.model_id.split('/')[-1])}_basic"
    )
    output_root = output_root.resolve()
    configure_logging(output_root / "cosmos_detection.log")

    dataset_directories = discover_datasets(
        dataset_root, set(args.datasets) if args.datasets else None
    )
    if args.max_datasets:
        dataset_directories = dataset_directories[: args.max_datasets]
    if not dataset_directories:
        raise FileNotFoundError(f"No RF100-VL datasets found below {dataset_root}.")

    LOGGER.info(
        "Found %d dataset(s); model=%s", len(dataset_directories), args.model_id
    )
    LOGGER.info(
        "Basic mode only: image + complete dataset class list; no train data or README instructions."
    )
    run_config = build_run_config(args)
    inference_client = CosmosInferenceClient(args)
    summaries: list[dict[str, Any]] = []
    for index, dataset_directory in enumerate(dataset_directories, start=1):
        LOGGER.info(
            "Dataset %d/%d: %s", index, len(dataset_directories), dataset_directory.name
        )
        try:
            summaries.append(
                run_dataset(
                    dataset_directory,
                    dataset_root,
                    output_root,
                    inference_client,
                    run_config,
                    args,
                )
            )
        except Exception as error:
            LOGGER.exception("Dataset %s failed", dataset_directory.name)
            summaries.append(
                {
                    "dataset": dataset_directory.name,
                    "fatal_error": f"{type(error).__name__}: {error}",
                }
            )

    scored = [summary for summary in summaries if "metrics" in summary]
    aggregate: dict[str, Any] = {
        "model_id": args.model_id,
        "prompt_version": PROMPT_VERSION,
        "dataset_count": len(summaries),
        "scored_dataset_count": len(scored),
        "datasets": summaries,
    }
    if scored:
        aggregate["macro_AP"] = statistics.fmean(
            summary["metrics"]["AP"] for summary in scored
        )
        aggregate["macro_AP50"] = statistics.fmean(
            summary["metrics"]["AP50"] for summary in scored
        )
        LOGGER.info(
            "Macro AP=%.4f, macro AP50=%.4f across %d datasets",
            aggregate["macro_AP"],
            aggregate["macro_AP50"],
            len(scored),
        )
    atomic_write_json(output_root / "aggregate_summary.json", aggregate)

    fatal_count = sum("fatal_error" in summary for summary in summaries)
    image_error_count = sum(summary.get("new_error_count", 0) for summary in summaries)
    incomplete_count = sum(
        not summary.get("complete", False)
        for summary in summaries
        if "fatal_error" not in summary
    )
    unexpected_incomplete_count = incomplete_count if args.max_images is None else 0
    if fatal_count or image_error_count or unexpected_incomplete_count:
        LOGGER.error(
            "Run finished with %d fatal dataset error(s), %d image error(s), "
            "and %d unexpectedly incomplete dataset(s).",
            fatal_count,
            image_error_count,
            unexpected_incomplete_count,
        )
        return 1
    if incomplete_count:
        LOGGER.info(
            "Smoke run completed successfully; %d dataset(s) are intentionally partial.",
            incomplete_count,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
