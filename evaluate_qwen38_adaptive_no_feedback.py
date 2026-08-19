#!/usr/bin/env python3
"""Evaluate prediction-blind adaptive few-shot detection with Qwen3.8-Max.

Every target starts with semantic class names and zero labeled visual examples.
Before detecting, the model repeatedly chooses either to detect or to request
one additional positive train example for specific labels.  The loop stops when
the model is ready or every requested label reaches the configured budget.  Test
annotations are used only after inference for COCO scoring.

The evaluator is resumable at conversation-turn granularity.  It records the
complete decision transcript, exact selected train annotations, final raw
response, token usage, and a prompt/settings fingerprint for matched replay.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_orion as base
import evaluate_qwen38_recipe as recipe

MODEL_ID = "qwen3.8-max"
MODE = "adaptive_no_prediction_feedback"
PROMPT_VERSION = "qwen3.8-max-adaptive-no-prediction-feedback-v1"
TERMINAL_STATUSES = base.TERMINAL_STATUSES
LOGGER = logging.getLogger("qwen38_adaptive_no_feedback")


class InvalidDecisionError(ValueError):
    """The model did not return a valid adaptive routing decision."""


def build_tasks(test: dict[str, Any]) -> list[base.Task]:
    tasks = [
        base.Task(
            mode=MODE,
            image_id=int(image["id"]),
            file_name=str(image["file_name"]),
            width=int(image["width"]),
            height=int(image["height"]),
        )
        for image in sorted(test["images"], key=lambda value: int(value["id"]))
    ]
    if len({task.key for task in tasks}) != len(tasks):
        raise ValueError("Generated adaptive task keys are not unique.")
    return tasks


def decision_contract(labels: Sequence[str], counts: dict[int, int], categories: dict[int, str]) -> str:
    count_by_label = {categories[category_id]: counts[category_id] for category_id in categories}
    return (
        "Decide whether the class names and labeled visual examples seen so far "
        "are sufficient to reliably identify and localize every requested class "
        "in the TARGET IMAGE. Do not output detections yet. If enough context is "
        "available, return only {\"action\":\"detect\",\"confidence\":0.0}. If "
        "more visual context is needed, return only "
        "{\"action\":\"request_examples\",\"labels\":[\"exact label\"],"
        "\"confidence\":0.0}. Request only genuinely uncertain labels, but you "
        "may request several labels at once. confidence must be a number from 0 "
        "to 1 for readiness to perform the final detection. Use only these exact "
        f"labels: {json.dumps(list(labels), ensure_ascii=False)}. Current labeled "
        f"example counts: {json.dumps(count_by_label, ensure_ascii=False, sort_keys=True)}."
    )


def build_initial_messages(
    task: base.Task,
    test_directory: Path,
    categories: dict[int, str],
) -> list[dict[str, Any]]:
    target = test_directory / task.file_name
    if not target.is_file():
        raise FileNotFoundError(target)
    labels = list(categories.values())
    counts = {category_id: 0 for category_id in categories}
    prompt = (
        "You control how much labeled visual context is needed before object "
        "detection. Every requested class starts with zero labeled visual "
        "examples. First inspect the semantic class names and TARGET IMAGE, then "
        "decide whether to detect or request one positive train example for "
        "specific uncertain classes. "
        + decision_contract(labels, counts, categories)
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "TARGET IMAGE:"},
                {"type": "image_url", "image_url": {"url": base.data_url(target)}},
            ],
        }
    ]


def _reference_payloads(
    added: Sequence[tuple[int, box_ablation.ReferenceBox]],
    categories: dict[int, str],
    assets: dict[tuple[int, int], dict[str, Path]],
) -> list[dict[str, Any]]:
    by_image: dict[tuple[int, str], list[tuple[int, box_ablation.ReferenceBox]]] = defaultdict(list)
    for category_id, reference in added:
        by_image[(reference.image_id, reference.file_name)].append((category_id, reference))

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Here are the additional positive reference examples you "
                "requested. The marked boxes are sparse positive exemplars. "
                "Treat every unmarked object or region as unlabeled, not as a "
                "negative example or exhaustive annotation. Reference boxes use "
                "the same normalized 0–1000 XYXY JSON format as final detections."
            ),
        }
    ]
    for image_key in sorted(by_image, key=lambda value: (value[1], value[0])):
        image_references = sorted(by_image[image_key], key=lambda value: value[0])
        first_category, first_reference = image_references[0]
        content.extend(
            [
                {"type": "text", "text": "LABELED REFERENCE IMAGE:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base.data_url(
                            assets[(first_category, first_reference.rank)]["source"]
                        )
                    },
                },
                {
                    "type": "text",
                    "text": recipe.detection_list_json(
                        [
                            (reference.bbox_xyxy_1000, categories[category_id])
                            for category_id, reference in image_references
                        ]
                    ),
                },
            ]
        )
    return content


def build_reference_message(
    added: Sequence[tuple[int, box_ablation.ReferenceBox]],
    counts: dict[int, int],
    categories: dict[int, str],
    assets: dict[tuple[int, int], dict[str, Path]],
) -> dict[str, Any]:
    content = _reference_payloads(added, categories, assets)
    content.append(
        {
            "type": "text",
            "text": decision_contract(list(categories.values()), counts, categories),
        }
    )
    return {"role": "user", "content": content}


def build_final_message(categories: dict[int, str]) -> dict[str, Any]:
    labels = list(categories.values())
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Now perform the final detection on the TARGET IMAGE from "
                    "the first turn. Detect every instance of the listed labels. "
                    + recipe._output_contract(labels)
                ),
            }
        ],
    }


def _json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    values: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def parse_decision(raw: str, categories: dict[int, str]) -> dict[str, Any]:
    values = _json_objects(raw.strip())
    if not values:
        raise InvalidDecisionError("No JSON decision object was found.")
    value = values[0]
    action = str(value.get("action") or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if action in {"detect", "ready", "stop"}:
        action = "detect"
    elif action in {"request_examples", "request_example", "request_more", "more"}:
        action = "request_examples"
    else:
        raise InvalidDecisionError(f"Unknown decision action: {value.get('action')!r}")

    confidence_value = value.get("confidence")
    confidence = None
    if confidence_value is not None:
        try:
            confidence = float(confidence_value)
        except (TypeError, ValueError) as error:
            raise InvalidDecisionError("confidence must be numeric.") from error
        if not 0 <= confidence <= 1:
            raise InvalidDecisionError("confidence must be in [0, 1].")

    labels_value = value.get("labels", [])
    if labels_value is None:
        labels_value = []
    if not isinstance(labels_value, list) or any(not isinstance(label, str) for label in labels_value):
        raise InvalidDecisionError("labels must be a JSON string list.")
    label_lookup = {name.casefold(): category_id for category_id, name in categories.items()}
    requested_ids: list[int] = []
    unknown: list[str] = []
    for label in labels_value:
        normalized = label.strip().casefold()
        category_id = label_lookup.get(normalized)
        if category_id is None:
            unknown.append(label)
        elif category_id not in requested_ids:
            requested_ids.append(category_id)
    if unknown:
        raise InvalidDecisionError(f"Unknown requested labels: {unknown}")
    if action == "detect" and requested_ids:
        raise InvalidDecisionError("detect action cannot also request labels.")
    if action == "request_examples" and not requested_ids:
        raise InvalidDecisionError("request_examples requires at least one label.")
    return {
        "action": action,
        "confidence": confidence,
        "requested_category_ids": requested_ids,
        "requested_labels": [categories[category_id] for category_id in requested_ids],
    }


def summarize_messages(messages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    turns = []
    for message in messages:
        content_summary = []
        content = message.get("content", [])
        if isinstance(content, str):
            content_summary.append({"type": "text", "text": content})
        else:
            for part in content:
                if part.get("type") == "text":
                    content_summary.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    url = str((part.get("image_url") or {}).get("url") or "")
                    content_summary.append(
                        {
                            "type": "image_url",
                            "sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
                        }
                    )
        turns.append({"role": message.get("role"), "content": content_summary})
    return {"turns": turns}


def record_path(output_directory: Path, task: base.Task) -> Path:
    return output_directory / "records" / MODE / f"{task.key}.json"


def task_fingerprint(
    task: base.Task,
    initial_messages: list[dict[str, Any]],
    settings: dict[str, Any],
    max_examples: int,
) -> str:
    return base.request_fingerprint(
        task,
        {
            "prompt_version": PROMPT_VERSION,
            "initial": summarize_messages(initial_messages),
            "max_examples_per_class": max_examples,
        },
        settings,
    )


def estimate_tokens(messages: Sequence[dict[str, Any]]) -> int:
    images = 0
    text_bytes = 0
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, str):
            text_bytes += len(content.encode("utf-8"))
            continue
        for part in content:
            if part.get("type") == "image_url":
                images += 1
            elif part.get("type") == "text":
                text_bytes += len(str(part.get("text") or "").encode("utf-8"))
    return images * 3_000 + (text_bytes + 2) // 3 + 2_500


class AdaptiveRateLimiter:
    def __init__(self, shared: base.SmoothDualRateLimiter):
        self.shared = shared

    def acquire(self, messages: Sequence[dict[str, Any]]) -> None:
        self.shared.acquire(estimate_tokens(messages))


def _call_api(
    client: Any,
    messages: list[dict[str, Any]],
    settings: dict[str, Any],
    max_retries: int,
    limiter: AdaptiveRateLimiter,
) -> dict[str, Any]:
    attempts = []
    started = time.monotonic()
    for attempt in range(1, max_retries + 2):
        try:
            limiter.acquire(messages)
            inference = base.stream_inference(client, messages, settings)
            if inference["finish_reason"] == "length":
                return {
                    "status": "model_failure",
                    "failure_type": "truncated_response",
                    "raw_response": inference["response"],
                    "finish_reason": inference["finish_reason"],
                    "usage": inference["usage"],
                    "attempts": attempts + [{"attempt": attempt, "status": "length"}],
                    "inference_seconds": inference["elapsed_seconds"],
                    "elapsed_seconds": time.monotonic() - started,
                }
            return {
                "status": "success",
                "raw_response": inference["response"],
                "finish_reason": inference["finish_reason"],
                "usage": inference["usage"],
                "attempts": attempts + [{"attempt": attempt, "status": "success"}],
                "inference_seconds": inference["elapsed_seconds"],
                "elapsed_seconds": time.monotonic() - started,
            }
        except base.GenerationDeadlineError as error:
            return {
                "status": "model_failure",
                "failure_type": "generation_timeout",
                "error": str(error),
                "raw_response": None,
                "usage": None,
                "attempts": attempts + [{"attempt": attempt, "status": "generation_timeout"}],
                "elapsed_seconds": time.monotonic() - started,
            }
        except Exception as error:  # noqa: BLE001 - SDK errors vary
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if base.terminal_provider_rejection(error):
                return {
                    "status": "model_failure",
                    "failure_type": "provider_content_rejection",
                    "error": f"{type(error).__name__}: {error}",
                    "raw_response": None,
                    "usage": None,
                    "attempts": attempts,
                    "elapsed_seconds": time.monotonic() - started,
                }
            retryable = base.retryable_error(error)
            if attempt > max_retries or not retryable:
                return {
                    "status": "model_failure" if base.provider_request_error(error) else "error",
                    "failure_type": "provider_request_failure" if base.provider_request_error(error) else None,
                    "error": f"{type(error).__name__}: {error}",
                    "raw_response": None,
                    "usage": None,
                    "attempts": attempts,
                    "retryable": retryable,
                    "retries_exhausted": attempt > max_retries,
                    "elapsed_seconds": time.monotonic() - started,
                }
            time.sleep(min(60.0, 2 ** (attempt - 1) + random.random()))
    raise AssertionError("Unreachable retry loop.")


def _added_from_round(
    round_record: dict[str, Any],
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]],
) -> list[tuple[int, box_ablation.ReferenceBox]]:
    added = []
    for value in round_record.get("references_added", []):
        category_id = int(value["category_id"])
        rank = int(value["rank"])
        reference = references[category_id][rank - 1]
        if reference.annotation_id != int(value["annotation_id"]):
            raise ValueError("Saved adaptive reference does not match locked sequence.")
        added.append((category_id, reference))
    return added


def reconstruct_messages(
    initial: list[dict[str, Any]],
    rounds: Sequence[dict[str, Any]],
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]],
    categories: dict[int, str],
    assets: dict[tuple[int, int], dict[str, Path]],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    messages = list(initial)
    counts = {category_id: 0 for category_id in categories}
    for round_record in rounds:
        messages.append({"role": "assistant", "content": round_record["raw_response"]})
        added = _added_from_round(round_record, references)
        for category_id, reference in added:
            expected_rank = counts[category_id] + 1
            if reference.rank != expected_rank:
                raise ValueError("Adaptive reference ranks are not nested and contiguous.")
            counts[category_id] = reference.rank
        if added:
            messages.append(build_reference_message(added, counts, categories, assets))
    return messages, counts


def _reference_record(category_id: int, reference: box_ablation.ReferenceBox) -> dict[str, Any]:
    return {
        "category_id": category_id,
        "category_name": reference.category_name,
        "rank": reference.rank,
        "annotation_id": reference.annotation_id,
        "image_id": reference.image_id,
        "file_name": reference.file_name,
        "bbox_xyxy_1000": list(reference.bbox_xyxy_1000),
    }


def _aggregate_usage(turns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    prompt_tokens = completion_tokens = total_tokens = cached_tokens = reasoning_tokens = 0
    inference_times = []
    for turn in turns:
        usage = turn.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)
        cached_tokens += int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        reasoning_tokens += int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        if turn.get("inference_seconds") is not None:
            inference_times.append(float(turn["inference_seconds"]))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens or prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "request_count": len(turns),
        "total_inference_seconds": sum(inference_times),
    }


def execute_adaptive_task(
    task: base.Task,
    client: Any,
    test_directory: Path,
    categories: dict[int, str],
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]],
    assets: dict[tuple[int, int], dict[str, Path]],
    settings: dict[str, Any],
    decision_settings: dict[str, Any],
    max_examples: int,
    max_retries: int,
    limiter: AdaptiveRateLimiter,
    output_path: Path,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    initial = build_initial_messages(task, test_directory, categories)
    fingerprint = task_fingerprint(task, initial, settings, max_examples)
    if (
        existing
        and existing.get("request_fingerprint") is not None
        and existing.get("request_fingerprint") != fingerprint
    ):
        raise ValueError(f"Mismatched adaptive checkpoint: {task.key}")
    rounds = list(existing.get("rounds", [])) if existing and existing.get("status") == "in_progress" else []
    messages, counts = reconstruct_messages(initial, rounds, references, categories, assets)
    stop_reason = None

    while True:
        if rounds:
            last_decision = rounds[-1]["decision"]
            if last_decision["action"] == "detect":
                stop_reason = "model_ready"
                break
            if not rounds[-1].get("references_added"):
                stop_reason = "requested_labels_exhausted"
                break

        turn = _call_api(client, messages, decision_settings, max_retries, limiter)
        turn["kind"] = "context_decision"
        turn["round_index"] = len(rounds)
        if turn["status"] != "success":
            record = {
                "status": turn["status"],
                "failure_type": turn.get("failure_type"),
                "error": turn.get("error"),
                "task": asdict(task),
                "task_key": task.key,
                "request_fingerprint": fingerprint,
                "rounds": rounds,
                "terminal_turn": turn,
                "selected_reference_counts": {str(key): value for key, value in counts.items()},
                "predictions": [],
                "usage": _aggregate_usage([*rounds, turn]),
                "elapsed_seconds": time.monotonic() - started,
                "completed_at": base.utc_now(),
            }
            base.atomic_write_json(output_path, record)
            return record
        try:
            decision = parse_decision(str(turn["raw_response"]), categories)
        except InvalidDecisionError as error:
            turn["decision_error"] = str(error)
            record = {
                "status": "model_failure",
                "failure_type": "invalid_context_decision",
                "error": str(error),
                "task": asdict(task),
                "task_key": task.key,
                "request_fingerprint": fingerprint,
                "rounds": rounds,
                "terminal_turn": turn,
                "selected_reference_counts": {str(key): value for key, value in counts.items()},
                "predictions": [],
                "usage": _aggregate_usage([*rounds, turn]),
                "elapsed_seconds": time.monotonic() - started,
                "completed_at": base.utc_now(),
            }
            base.atomic_write_json(output_path, record)
            return record

        added: list[tuple[int, box_ablation.ReferenceBox]] = []
        if decision["action"] == "request_examples":
            for category_id in decision["requested_category_ids"]:
                available = min(max_examples, len(references[category_id]))
                if counts[category_id] < available:
                    added.append((category_id, references[category_id][counts[category_id]]))
            for category_id, reference in added:
                counts[category_id] = reference.rank
        round_record = {
            **turn,
            "decision": decision,
            "references_added": [
                _reference_record(category_id, reference)
                for category_id, reference in added
            ],
            "example_counts_after": {str(key): value for key, value in counts.items()},
        }
        rounds.append(round_record)
        partial = {
            "status": "in_progress",
            "task": asdict(task),
            "task_key": task.key,
            "request_fingerprint": fingerprint,
            "rounds": rounds,
            "selected_reference_counts": {str(key): value for key, value in counts.items()},
            "predictions": [],
            "usage": _aggregate_usage(rounds),
            "updated_at": base.utc_now(),
        }
        base.atomic_write_json(output_path, partial)
        messages.append({"role": "assistant", "content": turn["raw_response"]})
        if decision["action"] == "detect":
            stop_reason = "model_ready"
            break
        if not added:
            stop_reason = "requested_labels_exhausted"
            break
        messages.append(build_reference_message(added, counts, categories, assets))

    messages.append(build_final_message(categories))
    final_turn = _call_api(client, messages, settings, max_retries, limiter)
    final_turn["kind"] = "final_detection"
    if final_turn["status"] != "success":
        predictions = []
        status = final_turn["status"]
        failure_type = final_turn.get("failure_type")
        error = final_turn.get("error")
        diagnostics = None
    else:
        try:
            detections = base.parse_cosmos_response(str(final_turn["raw_response"]))
            predictions, diagnostics = base.convert_detections_to_coco(
                detections,
                task.image_id,
                task.width,
                task.height,
                categories,
            )
            status = "success"
            failure_type = None
            error = None
        except base.CosmosResponseError as parse_error:
            status = "model_failure"
            failure_type = "invalid_response"
            error = str(parse_error)
            predictions = []
            diagnostics = None
    all_turns = [*rounds, final_turn]
    record = {
        "status": status,
        "failure_type": failure_type,
        "error": error,
        "task": asdict(task),
        "task_key": task.key,
        "request_fingerprint": fingerprint,
        "stop_reason": stop_reason,
        "rounds": rounds,
        "final_turn": final_turn,
        "selected_reference_counts": {str(key): value for key, value in counts.items()},
        "selected_references": [
            value
            for round_record in rounds
            for value in round_record.get("references_added", [])
        ],
        "raw_response": final_turn.get("raw_response"),
        "predictions": predictions,
        "diagnostics": diagnostics,
        "usage": _aggregate_usage(all_turns),
        "inference_seconds": sum(
            float(turn.get("inference_seconds") or 0) for turn in all_turns
        ),
        "elapsed_seconds": time.monotonic() - started,
        "completed_at": base.utc_now(),
    }
    base.atomic_write_json(output_path, record)
    return record


def summarize_records(tasks: Sequence[base.Task], output_directory: Path) -> dict[str, Any]:
    counts = {"total": len(tasks), "success": 0, "model_failure": 0, "error": 0, "in_progress": 0, "pending": 0}
    for task in tasks:
        record = base.load_record(record_path(output_directory, task))
        status = record.get("status") if record else "pending"
        counts[status if status in counts else "error"] += 1
    return {"updated_at": base.utc_now(), "total": counts, "modes": {MODE: counts}}


def _usage(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    prompt_tokens = completion_tokens = cached_tokens = reasoning_tokens = request_count = 0
    inference_times = []
    for record in records:
        usage = record.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        cached_tokens += int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        reasoning_tokens += int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        request_count += int(usage.get("request_count") or 0)
        if record.get("inference_seconds") is not None:
            inference_times.append(float(record["inference_seconds"]))
    return {
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "request_count": request_count,
        "mean_inference_seconds_per_image": fmean(inference_times) if inference_times else None,
        "total_inference_seconds": sum(inference_times),
    }


def finalize(
    tasks: Sequence[base.Task],
    annotation_path: Path,
    output_directory: Path,
    categories: dict[int, str],
    max_examples: int,
) -> dict[str, Any]:
    records = []
    predictions = []
    statuses: dict[str, int] = defaultdict(int)
    failure_types: dict[str, int] = defaultdict(int)
    stop_reasons: dict[str, int] = defaultdict(int)
    reference_totals = {category_id: 0 for category_id in categories}
    for task in tasks:
        record = base.load_record(record_path(output_directory, task))
        status = record.get("status", "missing") if record else "missing"
        statuses[status] += 1
        if record and status in TERMINAL_STATUSES:
            records.append(record)
            predictions.extend(record.get("predictions", []))
            if record.get("failure_type"):
                failure_types[str(record["failure_type"])] += 1
            if record.get("stop_reason"):
                stop_reasons[str(record["stop_reason"])] += 1
            for category_id, count in (record.get("selected_reference_counts") or {}).items():
                reference_totals[int(category_id)] += int(count)
    complete = sum(statuses[value] for value in TERMINAL_STATUSES) == len(tasks)
    base.atomic_write_json(output_directory / "predictions" / f"{MODE}.json", predictions)
    metrics = base.score_coco(annotation_path, predictions) if complete else None
    usage = _usage(records)
    total_references = sum(reference_totals.values())
    zero_shot_images = sum(
        not any(int(value) for value in (record.get("selected_reference_counts") or {}).values())
        for record in records
    )
    summary = {
        "mode": MODE,
        "complete": complete,
        "task_count": len(tasks),
        "statuses": dict(statuses),
        "failure_types": dict(failure_types),
        "stop_reasons": dict(stop_reasons),
        "prediction_count": len(predictions),
        "max_examples_per_class": max_examples,
        "total_selected_reference_objects": total_references,
        "mean_selected_references_per_image": total_references / len(tasks) if tasks else 0,
        "mean_selected_references_per_class_image": total_references / (len(tasks) * len(categories)) if tasks and categories else 0,
        "zero_shot_images": zero_shot_images,
        "per_class_selected_references": {
            categories[category_id]: reference_totals[category_id]
            for category_id in categories
        },
        "usage": usage,
        "metrics": metrics,
    }
    base.atomic_write_json(output_directory / "metrics" / f"{MODE}.json", summary)
    row = {
        "mode": MODE,
        "task_count": len(tasks),
        "complete": complete,
        "mAP50_95": metrics["AP"] * 100 if metrics else None,
        "mAP50": metrics["AP50"] * 100 if metrics else None,
        "model_failures": statuses.get("model_failure", 0),
        "errors": statuses.get("error", 0) + statuses.get("missing", 0),
        "zero_shot_images": zero_shot_images,
        "mean_selected_references_per_image": summary["mean_selected_references_per_image"],
        "mean_selected_references_per_class_image": summary["mean_selected_references_per_class_image"],
        **usage,
    }
    base.atomic_write_json(
        output_directory / "comparison_summary.json",
        {"updated_at": base.utc_now(), "rows": [row]},
    )
    csv_path = output_directory / "comparison_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    os.replace(temporary, csv_path)
    aggregate = {
        "updated_at": base.utc_now(),
        "prompt_version": PROMPT_VERSION,
        "image_count": len(tasks),
        "class_count": len(categories),
        "modes": {MODE: summary},
    }
    base.atomic_write_json(output_directory / "aggregate_metrics.json", aggregate)
    if complete:
        base.atomic_write_json(
            output_directory / "_SUCCESS.json",
            {
                "completed_at": base.utc_now(),
                "prompt_version": PROMPT_VERSION,
                "dataset": str(annotation_path.parents[1]),
                "image_count": len(tasks),
                "class_count": len(categories),
                "condition_count": 1,
                "request_count": usage["request_count"],
            },
        )
    return aggregate


def write_or_validate_manifest(path: Path, expected: dict[str, Any]) -> None:
    canonical = json.loads(json.dumps(expected, ensure_ascii=False))
    existing = base.load_record(path)
    if existing:
        if {key: existing.get(key) for key in canonical} != canonical:
            raise ValueError(f"Existing manifest does not match adaptive experiment: {path}")
        return
    base.atomic_write_json(path, {**canonical, "created_at": base.utc_now()})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--base-url", default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--requests-per-minute", type=float, default=13_500.0)
    parser.add_argument("--tokens-per-minute", type=float, default=1_800_000.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--max-decision-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-examples-per-class", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--image-ids", nargs="+", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.concurrency < 1 or args.max_retries < 0:
        raise ValueError("Concurrency must be positive and retries nonnegative.")
    if not 0 <= args.temperature < 2:
        raise ValueError("Temperature must be in [0, 2).")
    if not 0 <= args.max_examples_per_class <= 10:
        raise ValueError("max-examples-per-class must be between 0 and 10.")
    if not os.getenv("DASHSCOPE_API_KEY") and not args.prepare_only:
        raise RuntimeError("DASHSCOPE_API_KEY is required for inference.")

    dataset_directory = args.dataset_dir.resolve()
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        handlers=[logging.FileHandler(output_directory / "experiment.log"), logging.StreamHandler()],
        force=True,
    )
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

    evaluation_path = test_path
    selected_image_ids = None
    if args.image_ids is not None:
        requested = set(args.image_ids)
        available = {int(image["id"]) for image in test["images"]}
        if requested - available:
            raise ValueError(f"Unknown test image IDs: {sorted(requested - available)}")
        selected_image_ids = sorted(requested)
        test = {
            **test,
            "images": [image for image in test["images"] if int(image["id"]) in requested],
            "annotations": [annotation for annotation in test["annotations"] if int(annotation["image_id"]) in requested],
        }
        evaluation_path = output_directory / "ground_truth_subset.json"
        base.atomic_write_json(evaluation_path, test)

    references = box_ablation.select_reference_sequences(
        train,
        train_directory,
        required_count=args.max_examples_per_class,
        distinct_images_only=False,
        allow_fewer=True,
    ) if args.max_examples_per_class else {category_id: () for category_id in categories}
    assets = box_ablation.prepare_reference_assets(
        train_directory, output_directory / "references", references
    ) if references else {}
    tasks = build_tasks(test)
    settings = {
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "reasoning_effort": "none",
        "enable_thinking": False,
        "vl_high_resolution_images": False,
        "timeout_seconds": args.timeout_seconds,
    }
    decision_settings = {**settings, "max_completion_tokens": args.max_decision_tokens}
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "mode": MODE,
        "dataset_directory": str(dataset_directory),
        "train_annotation_sha256": base.sha256_file(train_path),
        "test_annotation_sha256": base.sha256_file(test_path),
        "selected_test_image_ids": selected_image_ids,
        "adaptive_policy": {
            "initial_labeled_examples": 0,
            "request_increment_per_label": 1,
            "max_examples_per_class": args.max_examples_per_class,
            "prediction_feedback": False,
            "test_ground_truth_visible": False,
            "final_detection_only_scored": True,
            "reference_semantics": "sparse-positive-nonexhaustive",
            "reference_box_schema": "prediction-matched-bbox_2d-label-normalized-xyxy-1000",
        },
        "common_settings": settings,
        "decision_max_completion_tokens": args.max_decision_tokens,
        "reference_selection": {
            "method": "largest-relative-area-then-greedy-crop-diversity-v1",
            "one_box_per_distinct_train_image": False,
            "classes": {
                str(category_id): [
                    {
                        **asdict(reference),
                        "source_sha256": base.sha256_file(train_directory / reference.file_name),
                    }
                    for reference in sequence
                ]
                for category_id, sequence in references.items()
            },
        },
        "concurrency": args.concurrency,
        "requests_per_minute": args.requests_per_minute,
        "tokens_per_minute": args.tokens_per_minute,
        "max_detections": 500,
    }
    write_or_validate_manifest(output_directory / "run_manifest.json", manifest)
    base.atomic_write_json(output_directory / "progress.json", summarize_records(tasks, output_directory))
    if args.prepare_only:
        finalize(tasks, evaluation_path, output_directory, categories, args.max_examples_per_class)
        return 0

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=settings["base_url"],
        timeout=settings["timeout_seconds"],
        max_retries=0,
    )
    pending = []
    existing_by_key = {}
    for task in tasks:
        path = record_path(output_directory, task)
        existing = base.load_record(path)
        initial = build_initial_messages(task, test_directory, categories)
        expected = task_fingerprint(task, initial, settings, args.max_examples_per_class)
        if (
            existing
            and existing.get("request_fingerprint") is not None
            and existing.get("request_fingerprint") != expected
        ):
            raise ValueError(f"Mismatched adaptive checkpoint: {task.key}")
        if existing and existing.get("status") in TERMINAL_STATUSES:
            continue
        pending.append(task)
        if existing and existing.get("status") == "in_progress":
            existing_by_key[task.key] = existing
    if args.limit is not None:
        pending = pending[: args.limit]

    shared_limiter = base.SmoothDualRateLimiter(args.requests_per_minute, args.tokens_per_minute)
    limiter = AdaptiveRateLimiter(shared_limiter)
    LOGGER.info("Starting %d pending of %d adaptive target conversations.", len(pending), len(tasks))
    completed = 0
    progress_lock = threading.Lock()

    def execute(task: base.Task) -> dict[str, Any]:
        return execute_adaptive_task(
            task,
            client,
            test_directory,
            categories,
            references,
            assets,
            settings,
            decision_settings,
            args.max_examples_per_class,
            args.max_retries,
            limiter,
            record_path(output_directory, task),
            existing_by_key.get(task.key),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(execute, task): task for task in pending}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                future.result()
            except Exception as error:  # noqa: BLE001
                initial = build_initial_messages(task, test_directory, categories)
                base.atomic_write_json(
                    record_path(output_directory, task),
                    {
                        "status": "error",
                        "error": f"WorkerFailure: {type(error).__name__}: {error}",
                        "task": asdict(task),
                        "task_key": task.key,
                        "request_fingerprint": task_fingerprint(
                            task,
                            initial,
                            settings,
                            args.max_examples_per_class,
                        ),
                        "predictions": [],
                        "completed_at": base.utc_now(),
                    },
                )
            with progress_lock:
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    progress = summarize_records(tasks, output_directory)
                    base.atomic_write_json(output_directory / "progress.json", progress)
                    LOGGER.info(
                        "Checkpoint %d/%d; success=%d, model_failure=%d, error=%d, pending=%d.",
                        completed,
                        len(pending),
                        progress["total"]["success"],
                        progress["total"]["model_failure"],
                        progress["total"]["error"],
                        progress["total"]["pending"] + progress["total"]["in_progress"],
                    )

    progress = summarize_records(tasks, output_directory)
    base.atomic_write_json(output_directory / "progress.json", progress)
    finalize(tasks, evaluation_path, output_directory, categories, args.max_examples_per_class)
    unresolved = progress["total"]["error"] + progress["total"]["pending"] + progress["total"]["in_progress"]
    return 0 if unresolved == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
