#!/usr/bin/env python3
"""Evaluate configurable Qwen3.8-Max FSOD recipes on an RF20-VL-FSOD dataset.

The evaluator is intentionally configuration driven.  It supports semantic class
names, anonymous Concept A/B identifiers, exemplar-only prompts, and model-made
visual names; single-class predictions are merged before standard RF100-VL COCO
scoring.  Every request is checkpointed with a prompt/settings fingerprint.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import json
import logging
import os
import re
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_exemplar_only_ablation as exemplar
import evaluate_qwen38_orion as base

MODEL_ID = "qwen3.8-max"
PROMPT_VERSION = "qwen3.8-max-configurable-fsod-recipe-v3"
FORMULATIONS = {"single", "multi"}
SEMANTICS = {
    "class_names",
    "anonymous_explicit",
    "anonymous_minimal",
    "self_name",
    "self_name_only",
}
REPRESENTATIONS = {"none", "numeric", "numeric_prediction", "drawn"}
REASONING_EFFORTS = {"none", "low", "medium"}
INSTRUCTION_MODES = {"none", "correct", "permuted"}
TERMINAL_STATUSES = base.TERMINAL_STATUSES

LOGGER = logging.getLogger("qwen38_recipe")


def detection_object(bbox: Sequence[Any], label: str) -> dict[str, Any]:
    """Build the one canonical Qwen3.8 detection/reference object."""

    return {"bbox_2d": list(bbox), "label": label}


def detection_list_json(
    detections: Sequence[tuple[Sequence[Any], str]],
) -> str:
    """Serialize references in exactly the same JSON shape as predictions."""

    return json.dumps(
        [detection_object(bbox, label) for bbox, label in detections],
        separators=(",", ":"),
        ensure_ascii=False,
    )


@dataclass(frozen=True)
class Condition:
    mode: str
    formulation: str
    semantics: str
    representation: str
    box_count: int
    reasoning_effort: str = "none"
    seed: int = 1234
    group_reference_instances_by_image: bool = False
    explicit_sparse_references: bool = False
    all_available_references: bool = False
    instruction_mode: str = "none"

    @property
    def single_class(self) -> bool:
        return self.formulation == "single"

    @property
    def uses_references(self) -> bool:
        return self.box_count > 0


def load_conditions(path: Path) -> tuple[Condition, ...]:
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    values = raw.get("conditions") if isinstance(raw, dict) else raw
    if not isinstance(values, list) or not values:
        raise ValueError("Condition config must be a nonempty list or {conditions: [...]}.")
    conditions = tuple(Condition(**value) for value in values)
    modes = [condition.mode for condition in conditions]
    if len(set(modes)) != len(modes):
        raise ValueError("Condition modes must be unique.")
    for condition in conditions:
        if condition.formulation not in FORMULATIONS:
            raise ValueError(f"Unsupported formulation in {condition.mode}.")
        if condition.semantics not in SEMANTICS:
            raise ValueError(f"Unsupported semantics in {condition.mode}.")
        if condition.representation not in REPRESENTATIONS:
            raise ValueError(f"Unsupported representation in {condition.mode}.")
        if condition.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"Unsupported reasoning effort in {condition.mode}.")
        if condition.instruction_mode not in INSTRUCTION_MODES:
            raise ValueError(f"Unsupported instruction mode in {condition.mode}.")
        if condition.box_count < 0 or condition.box_count > 10:
            raise ValueError(f"Box count must be between 0 and 10 in {condition.mode}.")
        if condition.box_count == 0 and condition.representation != "none":
            raise ValueError(f"Zero-box condition {condition.mode} must use representation=none.")
        if condition.box_count > 0 and condition.representation == "none":
            raise ValueError(f"Box condition {condition.mode} needs numeric or drawn representation.")
        if condition.semantics in {"anonymous_explicit", "anonymous_minimal"} and not condition.uses_references:
            raise ValueError(f"Anonymous condition {condition.mode} needs references.")
        if condition.semantics == "self_name_only" and condition.uses_references:
            raise ValueError(f"self_name_only condition {condition.mode} cannot attach references.")
        if condition.explicit_sparse_references and not condition.uses_references:
            raise ValueError(
                f"Explicit sparse-reference semantics in {condition.mode} require references."
            )
        if condition.all_available_references and condition.box_count != 10:
            raise ValueError(
                f"All-available condition {condition.mode} must use box_count=10."
            )
        if (
            condition.group_reference_instances_by_image
            and condition.representation not in {"numeric", "numeric_prediction"}
        ):
            raise ValueError(
                f"Grouped reference instances in {condition.mode} require a numeric representation."
            )
        if condition.instruction_mode != "none" and condition.semantics != "class_names":
            raise ValueError(
                f"Instruction condition {condition.mode} must use semantic class names."
            )
    return conditions


def condition_payload(condition: Condition) -> dict[str, Any]:
    """Serialize conditions without changing legacy default manifests."""

    payload = asdict(condition)
    if not payload["explicit_sparse_references"]:
        payload.pop("explicit_sparse_references")
    if not payload["all_available_references"]:
        payload.pop("all_available_references")
    if payload["instruction_mode"] == "none":
        payload.pop("instruction_mode")
    return payload


_OBJECT_CLASSES_HEADING = re.compile(r"(?m)^# Object Classes\s*$")
_CLASS_SECTION_HEADING = re.compile(r"(?m)^## (?!#)(.+?)\s*$")


def permute_class_instruction_sections(readme: str) -> str:
    """Rotate class-section bodies while preserving the exact README vocabulary.

    The overview and introduction are intentionally unchanged, making this a
    conservative semantic control: only the detailed class-to-guidance mapping
    is wrong. The same bodies, token content, domain, and class headings remain.
    """

    object_match = _OBJECT_CLASSES_HEADING.search(readme)
    if object_match is None:
        raise ValueError("README.dataset.txt lacks an '# Object Classes' section.")
    prefix = readme[: object_match.end()]
    remainder = readme[object_match.end() :]
    matches = list(_CLASS_SECTION_HEADING.finditer(remainder))
    if len(matches) < 2:
        raise ValueError("Permuted instructions require at least two class sections.")
    headings = [match.group(0) for match in matches]
    bodies = [
        remainder[match.end() : matches[index + 1].start() if index + 1 < len(matches) else None]
        for index, match in enumerate(matches)
    ]
    rotated = bodies[1:] + bodies[:1]
    rendered = []
    for index, (heading, body) in enumerate(zip(headings, rotated, strict=True)):
        rendered.append(heading)
        rendered.append(body)
        if index + 1 < len(headings) and body and not body[-1].isspace():
            rendered.append("\n")
    result = prefix + "".join(rendered)
    if sorted(re.findall(r"\S+", result)) != sorted(re.findall(r"\S+", readme)):
        raise AssertionError("Instruction permutation changed README token content.")
    if result == readme:
        raise AssertionError("Instruction permutation did not change the README.")
    return result


def instruction_text(condition: Condition, readme: str | None) -> str | None:
    if condition.instruction_mode == "none":
        return None
    if not readme:
        raise ValueError(f"Condition {condition.mode} requires README.dataset.txt.")
    if condition.instruction_mode == "correct":
        return readme
    if condition.instruction_mode == "permuted":
        return permute_class_instruction_sections(readme)
    raise AssertionError(condition.instruction_mode)


def concept_identifier(index: int) -> str:
    """Return stable nonsemantic identifiers: Concept A..Z, AA..AZ, etc."""

    if index < 0:
        raise ValueError("Concept index must be nonnegative.")
    letters = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        letters = chr(65 + remainder) + letters
    return f"Concept {letters}"


def load_self_names(path: Path | None, categories: dict[int, str]) -> dict[int, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    values = raw.get("names") if isinstance(raw, dict) and "names" in raw else raw
    if not isinstance(values, dict):
        raise ValueError("Self-name file must contain an ID-to-name object.")
    result = {int(key): str(value).strip() for key, value in values.items()}
    if set(result) != set(categories):
        raise ValueError("Self-name IDs must exactly match dataset category IDs.")
    if any(not value for value in result.values()):
        raise ValueError("Self-generated names cannot be empty.")
    return result


def display_labels(
    condition: Condition,
    categories: dict[int, str],
    self_names: dict[int, str],
) -> dict[int, str]:
    if condition.semantics == "class_names":
        return dict(categories)
    if condition.semantics.startswith("anonymous"):
        return {
            category_id: concept_identifier(index)
            for index, category_id in enumerate(categories)
        }
    if condition.semantics in {"self_name", "self_name_only"}:
        if not self_names:
            raise ValueError(f"Condition {condition.mode} requires self-generated names.")
        # Prefixes make multi-class output labels unambiguous even if the model
        # produced duplicate or closely related descriptions for two classes.
        return {
            category_id: f"{concept_identifier(index)} — {self_names[category_id]}"
            for index, category_id in enumerate(categories)
        }
    raise AssertionError(condition.semantics)


def build_tasks(
    test: dict[str, Any],
    categories: dict[int, str],
    conditions: Sequence[Condition],
) -> list[base.Task]:
    tasks: list[base.Task] = []
    # Keep conditions adjacent for each target image. This prevents a provider
    # change or serving-replica drift over a long dataset from being perfectly
    # confounded with condition order, while remaining deterministic/resumable.
    for image in sorted(test["images"], key=lambda value: int(value["id"])):
        for condition in conditions:
            common = {
                "mode": condition.mode,
                "image_id": int(image["id"]),
                "file_name": str(image["file_name"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
            }
            if condition.single_class:
                for category_id, category_name in categories.items():
                    tasks.append(
                        base.Task(
                            **common,
                            category_id=category_id,
                            category_name=category_name,
                        )
                    )
            else:
                tasks.append(base.Task(**common))
    if len({task.key for task in tasks}) != len(tasks):
        raise ValueError("Generated task keys are not unique.")
    return tasks


def _output_contract(labels: Sequence[str]) -> str:
    example = (
        detection_list_json([(["x1", "y1", "x2", "y2"], "exact requested label")])
        .replace('"x1"', "x1")
        .replace('"y1"', "y1")
        .replace('"x2"', "x2")
        .replace('"y2"', "y2")
    )
    return (
        f"Return only a JSON list exactly like {example}. "
        "Use XYXY integer coordinates normalized independently from 0 to 1000 "
        "relative to the TARGET IMAGE, with origin at top-left. Use only these "
        f"labels: {json.dumps(list(labels), ensure_ascii=False)}. Return [] if none."
    )


def _append_references(
    content: list[dict[str, Any]],
    condition: Condition,
    category_ids: Sequence[int],
    labels: dict[int, str],
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]],
    assets: dict[tuple[int, int], dict[str, Path]],
) -> None:
    for category_id in category_ids:
        label = labels[category_id]
        if not condition.single_class:
            content.append({"type": "text", "text": f"REFERENCE GROUP {label}:"})
        selected_references = references[category_id][: condition.box_count]
        if condition.group_reference_instances_by_image:
            references_by_image: dict[int, list[box_ablation.ReferenceBox]] = {}
            for reference in selected_references:
                references_by_image.setdefault(reference.image_id, []).append(reference)
            for image_references in references_by_image.values():
                first = image_references[0]
                reference_text = (
                    detection_list_json(
                        [
                            (reference.bbox_xyxy_1000, label)
                            for reference in image_references
                        ]
                    )
                    if condition.representation == "numeric_prediction"
                    else json.dumps(
                        [
                            {"bbox_2d": list(reference.bbox_xyxy_1000)}
                            for reference in image_references
                        ],
                        separators=(",", ":"),
                    )
                )
                content.extend(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base.data_url(
                                    assets[(category_id, first.rank)]["source"]
                                )
                            },
                        },
                        {"type": "text", "text": reference_text},
                    ]
                )
            continue
        for reference in selected_references:
            if condition.representation in {"numeric", "numeric_prediction"}:
                reference_text = (
                    detection_list_json([(reference.bbox_xyxy_1000, label)])
                    if condition.representation == "numeric_prediction"
                    else json.dumps(
                        {"bbox_2d": list(reference.bbox_xyxy_1000)},
                        separators=(",", ":"),
                    )
                )
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
                            # The experimental numeric_prediction variant is
                            # byte-shape compatible with the requested output:
                            # a list of objects containing bbox_2d and label.
                            "text": reference_text,
                        },
                    ]
                )
            elif condition.representation == "drawn":
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
                raise ValueError(f"Unknown reference representation: {condition.representation}")


def build_messages(
    task: base.Task,
    condition: Condition,
    test_directory: Path,
    categories: dict[int, str],
    self_names: dict[int, str],
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]],
    assets: dict[tuple[int, int], dict[str, Path]],
    readme: str | None = None,
) -> list[dict[str, Any]]:
    target = test_directory / task.file_name
    if not target.is_file():
        raise FileNotFoundError(target)
    labels = display_labels(condition, categories, self_names)
    requested_ids = [task.category_id] if condition.single_class else list(categories)
    if any(category_id is None for category_id in requested_ids):
        raise ValueError(f"Single-class task lacks category: {task}")
    requested_ids = [int(category_id) for category_id in requested_ids]
    requested_labels = [labels[category_id] for category_id in requested_ids]

    content: list[dict[str, Any]] = []
    if condition.semantics == "anonymous_minimal":
        if not condition.single_class:
            content.append(
                {
                    "type": "text",
                    "text": "\n".join(
                        f"{labels[category_id]}:" for category_id in requested_ids
                    ),
                }
            )
    else:
        if condition.single_class:
            if condition.semantics == "anonymous_explicit":
                task_prompt = exemplar.EXPLICIT_PROMPT
            else:
                task_prompt = f'Detect every instance of "{requested_labels[0]}" in the TARGET IMAGE.'
        else:
            if condition.semantics == "anonymous_explicit":
                task_prompt = (
                    "Each reference group defines an anonymous visual concept. Find every "
                    "object in the TARGET IMAGE that is the same kind as a marked object, "
                    "and assign its reference-group label."
                )
            else:
                task_prompt = "Detect every instance of the listed labels in the TARGET IMAGE."
        if condition.uses_references:
            if condition.all_available_references:
                task_prompt += " Use all positive reference boxes supplied for each label."
            else:
                task_prompt += (
                    f" Use the {condition.box_count} positive reference box"
                    f"{'es' if condition.box_count != 1 else ''} supplied per label."
                )
            if condition.explicit_sparse_references:
                task_prompt += (
                    " The marked boxes are sparse positive exemplars. Treat all "
                    "unmarked objects and regions in reference images as unlabeled, "
                    "not as negative examples or exhaustive annotations."
                )
        guide = instruction_text(condition, readme)
        if guide:
            prompt = (
                "Use the dataset's annotator guide as context for deciding what to "
                "detect and how to localize it.\n\n"
                "DATASET ANNOTATOR GUIDE:\n"
                f"{guide}\n"
                "END DATASET ANNOTATOR GUIDE.\n\n"
                "FINAL DETECTION REQUEST:\n"
                f"{task_prompt}"
            )
        else:
            prompt = task_prompt
        prompt += " " + _output_contract(requested_labels)
        if guide:
            prompt += (
                " Your entire response must be the JSON list. Do not explain, "
                "restate the guide, or describe the detected objects in prose."
            )
        content.append({"type": "text", "text": prompt})

    if condition.uses_references:
        _append_references(
            content,
            condition,
            requested_ids,
            labels,
            references,
            assets,
        )
    content.extend(
        [
            {"type": "text", "text": "TARGET IMAGE:"},
            {"type": "image_url", "image_url": {"url": base.data_url(target)}},
        ]
    )
    if condition.semantics == "anonymous_minimal":
        content.append(
            {
                "type": "text",
                "text": (
                    'OUTPUT(last image): [{"bbox_2d":[x1,y1,x2,y2],'
                    f'"label":"one of {json.dumps(requested_labels)}"}}] | []; '
                    "XYXY integers normalized 0..1000."
                ),
            }
        )
    return [{"role": "user", "content": content}]


def condition_settings(condition: Condition, common: dict[str, Any]) -> dict[str, Any]:
    return {
        **common,
        "seed": condition.seed,
        "reasoning_effort": condition.reasoning_effort,
        "enable_thinking": condition.reasoning_effort != "none",
        "force_single_category_labels": condition.single_class,
    }


def expected_images_per_request(
    condition: Condition,
    class_count: int,
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]] | None = None,
) -> int:
    reference_classes = 1 if condition.single_class else class_count
    if condition.group_reference_instances_by_image and references:
        counts = [
            len(
                {
                    reference.image_id
                    for reference in sequence[: condition.box_count]
                }
            )
            for sequence in references.values()
        ]
        # Multi-class requests include every class. A single-class run can
        # vary by class, so use the maximum to avoid underestimating a request.
        return 1 + (max(counts) if condition.single_class else sum(counts))
    return 1 + condition.box_count * reference_classes


def token_estimate(
    condition: Condition,
    class_count: int,
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]] | None = None,
    readme: str | None = None,
) -> int:
    guide = instruction_text(condition, readme)
    guide_tokens = (len(guide.encode("utf-8")) + 2) // 3 if guide else 0
    return (
        3_000 * expected_images_per_request(condition, class_count, references)
        + 2_500
        + guide_tokens
    )


class TaskRateLimiter:
    def __init__(self, shared: base.SmoothDualRateLimiter, estimate: int):
        self.shared = shared
        self.estimate = estimate

    def acquire(self, _unused: int) -> None:
        self.shared.acquire(self.estimate)


def record_path(output_directory: Path, task: base.Task) -> Path:
    return output_directory / "records" / task.mode / f"{task.key}.json"


def summarize_records(
    tasks: Sequence[base.Task], output_directory: Path
) -> dict[str, Any]:
    result: dict[str, dict[str, int]] = {}
    for task in tasks:
        counts = result.setdefault(
            task.mode,
            {"total": 0, "success": 0, "model_failure": 0, "error": 0, "pending": 0},
        )
        counts["total"] += 1
        record = base.load_record(record_path(output_directory, task))
        status = record.get("status") if record else "pending"
        counts[status if status in counts else "error"] += 1
    total = {
        key: sum(value[key] for value in result.values())
        for key in ("total", "success", "model_failure", "error", "pending")
    }
    return {"updated_at": base.utc_now(), "total": total, "modes": result}


def _usage(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    usage = [record.get("usage") or {} for record in records]
    times = [
        float(record["inference_seconds"])
        for record in records
        if record.get("inference_seconds") is not None
    ]
    return {
        "prompt_tokens": sum(int(value.get("prompt_tokens") or 0) for value in usage),
        "completion_tokens": sum(int(value.get("completion_tokens") or 0) for value in usage),
        "reasoning_tokens": sum(
            int((value.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
            for value in usage
        ),
        "mean_inference_seconds": fmean(times) if times else None,
        "total_inference_seconds": sum(times),
    }


def finalize(
    all_tasks: Sequence[base.Task],
    conditions: Sequence[Condition],
    annotation_path: Path,
    output_directory: Path,
    *,
    image_count: int,
    class_count: int,
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    modes: dict[str, Any] = {}
    for condition in conditions:
        tasks = [task for task in all_tasks if task.mode == condition.mode]
        records: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        statuses: dict[str, int] = {}
        for task in tasks:
            record = base.load_record(record_path(output_directory, task))
            status = record.get("status", "missing") if record else "missing"
            statuses[status] = statuses.get(status, 0) + 1
            if record and status in TERMINAL_STATUSES:
                records.append(record)
                predictions.extend(record.get("predictions", []))
        complete = sum(statuses.get(value, 0) for value in TERMINAL_STATUSES) == len(tasks)
        base.atomic_write_json(
            output_directory / "predictions" / f"{condition.mode}.json", predictions
        )
        metrics = base.score_coco(annotation_path, predictions) if complete else None
        usage = _usage(records)
        failure_types: dict[str, int] = {}
        for record in records:
            failure_type = record.get("failure_type")
            if failure_type:
                failure_types[str(failure_type)] = failure_types.get(str(failure_type), 0) + 1
        summary = {
            "condition": condition_payload(condition),
            "complete": complete,
            "task_count": len(tasks),
            "calls_per_image": class_count if condition.single_class else 1,
            "reference_images_per_request": expected_images_per_request(
                condition, class_count, references
            )
            - 1,
            "statuses": statuses,
            "failure_types": failure_types,
            "prediction_count": len(predictions),
            "usage": usage,
            "metrics": metrics,
        }
        modes[condition.mode] = summary
        base.atomic_write_json(output_directory / "metrics" / f"{condition.mode}.json", summary)
        rows.append(
            {
                "mode": condition.mode,
                "formulation": condition.formulation,
                "semantics": condition.semantics,
                "representation": condition.representation,
                "boxes_per_class": condition.box_count,
                "reasoning_effort": condition.reasoning_effort,
                "seed": condition.seed,
                "calls_per_image": summary["calls_per_image"],
                "reference_images_per_request": summary["reference_images_per_request"],
                "task_count": len(tasks),
                "complete": complete,
                "mAP50_95": metrics["AP"] * 100 if metrics else None,
                "mAP50": metrics["AP50"] * 100 if metrics else None,
                "model_failures": statuses.get("model_failure", 0),
                "errors": statuses.get("error", 0) + statuses.get("missing", 0),
                **usage,
            }
        )
    aggregate = {
        "updated_at": base.utc_now(),
        "prompt_version": PROMPT_VERSION,
        "image_count": image_count,
        "class_count": class_count,
        "modes": modes,
        "provider_failure_policy": {
            "policy": "terminal-zero-detection-after-nonretryable-or-exhausted-request-error-v1",
            "provider_failure_count": sum(
                value["failure_types"].get("provider_content_rejection", 0)
                + value["failure_types"].get("provider_request_failure", 0)
                for value in modes.values()
            ),
            "requires_review": any(
                value["failure_types"].get("provider_content_rejection", 0)
                + value["failure_types"].get("provider_request_failure", 0)
                for value in modes.values()
            ),
        },
    }
    base.atomic_write_json(output_directory / "aggregate_metrics.json", aggregate)
    base.atomic_write_json(
        output_directory / "comparison_summary.json",
        {"updated_at": base.utc_now(), "rows": rows},
    )
    csv_path = output_directory / "comparison_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    if all(value["complete"] for value in modes.values()):
        base.atomic_write_json(
            output_directory / "_SUCCESS.json",
            {
                "completed_at": base.utc_now(),
                "prompt_version": PROMPT_VERSION,
                "dataset": str(annotation_path.parents[1]),
                "image_count": image_count,
                "class_count": class_count,
                "condition_count": len(conditions),
                "request_count": sum(row["task_count"] for row in rows),
            },
        )
    return aggregate


def write_or_validate_manifest(path: Path, expected: dict[str, Any]) -> None:
    canonical = json.loads(json.dumps(expected, ensure_ascii=False))
    existing = base.load_record(path)
    if existing:
        if {key: existing.get(key) for key in canonical} != canonical:
            raise ValueError(f"Existing manifest does not match experiment: {path}")
        return
    base.atomic_write_json(path, {**canonical, "created_at": base.utc_now()})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--self-names-json", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--base-url",
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--requests-per-minute", type=float, default=570.0)
    parser.add_argument("--tokens-per-minute", type=float, default=900_000.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--image-ids", nargs="+", type=int)
    parser.add_argument("--limit-per-mode", type=int)
    parser.add_argument("--allow-shared-reference-images", action="store_true")
    parser.add_argument(
        "--reference-first-strategy",
        choices=(
            "largest-relative-area",
            "median-relative-area",
            "largest-then-seeded-random",
        ),
        default="largest-relative-area",
        help="Train-only rule for selecting the first positive object per class.",
    )
    parser.add_argument(
        "--reference-random-seed",
        type=int,
        default=1234,
        help="Stable seed used by seeded-random reference selection.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--retry-terminal-provider-failures",
        action="store_true",
        help=(
            "Re-open previously terminal provider request failures. Successful "
            "checkpoints and other model failures remain untouched."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    conditions = load_conditions(args.conditions.resolve())
    if args.modes:
        unknown = set(args.modes) - {condition.mode for condition in conditions}
        if unknown:
            raise ValueError(f"Unknown selected modes: {sorted(unknown)}")
    if args.concurrency < 1 or args.max_retries < 0:
        raise ValueError("Concurrency must be positive and retries nonnegative.")
    if not 0 <= args.temperature < 2:
        raise ValueError("Temperature must be in [0, 2).")
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
    readme_path = dataset_directory / "README.dataset.txt"
    readme = (
        readme_path.read_text(encoding="utf-8").strip()
        if readme_path.is_file()
        else None
    )
    if any(condition.instruction_mode != "none" for condition in conditions) and not readme:
        raise FileNotFoundError(readme_path)
    train_path = train_directory / "_annotations.coco.json"
    test_path = test_directory / "_annotations.coco.json"
    train = base.load_coco(train_path)
    test = base.load_coco(test_path)
    base.validate_split_isolation(train, test)
    categories = base.categories_by_id(test)
    if categories != base.categories_by_id(train):
        raise ValueError("Train/test category definitions differ.")
    self_names = load_self_names(
        args.self_names_json.resolve() if args.self_names_json else None,
        categories,
    )
    if any(condition.semantics.startswith("self_name") for condition in conditions) and not self_names:
        raise ValueError("At least one configured condition requires --self-names-json.")

    evaluation_path = test_path
    selected_image_ids: list[int] | None = None
    if args.image_ids is not None:
        requested = set(args.image_ids)
        available = {int(image["id"]) for image in test["images"]}
        if requested - available:
            raise ValueError(f"Unknown test image IDs: {sorted(requested - available)}")
        selected_image_ids = sorted(requested)
        test = {
            **test,
            "images": [
                image for image in test["images"] if int(image["id"]) in requested
            ],
            "annotations": [
                annotation
                for annotation in test["annotations"]
                if int(annotation["image_id"]) in requested
            ],
        }
        evaluation_path = output_directory / "ground_truth_subset.json"
        base.atomic_write_json(evaluation_path, test)

    max_boxes = max(condition.box_count for condition in conditions)
    references: dict[int, tuple[box_ablation.ReferenceBox, ...]] = {}
    assets: dict[tuple[int, int], dict[str, Path]] = {}
    if max_boxes:
        references = box_ablation.select_reference_sequences(
            train,
            train_directory,
            required_count=max_boxes,
            distinct_images_only=not args.allow_shared_reference_images,
            first_strategy=args.reference_first_strategy,
            random_seed=args.reference_random_seed,
            allow_fewer=any(
                condition.all_available_references for condition in conditions
            ),
        )
        assets = box_ablation.prepare_reference_assets(
            train_directory, output_directory / "references", references
        )

    all_tasks = build_tasks(test, categories, conditions)
    selected_modes = set(args.modes or [condition.mode for condition in conditions])
    tasks = [task for task in all_tasks if task.mode in selected_modes]

    common_settings = {
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": args.temperature,
        "vl_high_resolution_images": False,
        "timeout_seconds": args.timeout_seconds,
    }
    condition_by_mode = {condition.mode: condition for condition in conditions}
    token_estimates = {
        condition.mode: token_estimate(condition, len(categories), references, readme)
        for condition in conditions
    }
    reference_manifest = {
        str(category_id): [
            {
                **asdict(reference),
                "source_sha256": base.sha256_file(train_directory / reference.file_name),
            }
            for reference in sequence[:max_boxes]
        ]
        for category_id, sequence in references.items()
    }
    manifest = {
        "prompt_version": PROMPT_VERSION,
        "dataset_directory": str(dataset_directory),
        "train_annotation_sha256": base.sha256_file(train_path),
        "test_annotation_sha256": base.sha256_file(test_path),
        "dataset_readme_sha256": base.sha256_file(readme_path) if readme else None,
        "selected_test_image_ids": selected_image_ids,
        "conditions": [condition_payload(condition) for condition in conditions],
        "common_settings": common_settings,
        "self_names": self_names,
        "self_names_sha256": base.sha256_file(args.self_names_json.resolve()) if args.self_names_json else None,
        "reference_selection": {
            "method": (
                "largest-relative-area-then-greedy-crop-diversity-v1"
                if args.reference_first_strategy == "largest-relative-area"
                else (
                    "median-relative-area-then-greedy-crop-diversity-v1"
                    if args.reference_first_strategy == "median-relative-area"
                    else "largest-relative-area-then-seeded-random-object-order-v1"
                )
            ),
            **(
                {"first_reference_strategy": args.reference_first_strategy}
                if args.reference_first_strategy != "largest-relative-area"
                else {}
            ),
            **(
                {"random_seed": args.reference_random_seed}
                if args.reference_first_strategy == "largest-then-seeded-random"
                else {}
            ),
            "one_box_per_distinct_train_image": not args.allow_shared_reference_images,
            "classes": reference_manifest,
        },
        "concurrency": args.concurrency,
        "thinking_controls": {
            "policy": "reasoning_effort-plus-enable_thinking-v1",
            "none_maps_to_enable_thinking": False,
        },
        "requests_per_minute": args.requests_per_minute,
        "tokens_per_minute": args.tokens_per_minute,
        "max_detections": 500,
    }
    write_or_validate_manifest(output_directory / "run_manifest.json", manifest)
    base.atomic_write_json(output_directory / "progress.json", summarize_records(all_tasks, output_directory))
    if args.prepare_only:
        finalize(
            all_tasks,
            conditions,
            evaluation_path,
            output_directory,
            image_count=len(test["images"]),
            class_count=len(categories),
            references=references,
        )
        return 0

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=common_settings["base_url"],
        timeout=common_settings["timeout_seconds"],
        max_retries=0,
    )
    pending: list[base.Task] = []
    for task in tasks:
        condition = condition_by_mode[task.mode]
        settings = condition_settings(condition, common_settings)
        messages = build_messages(
            task,
            condition,
            test_directory,
            categories,
            self_names,
            references,
            assets,
            readme,
        )
        existing = base.load_record(record_path(output_directory, task))
        if existing:
            terminal = base.terminalize_provider_failure(existing)
            if terminal is not existing:
                expected = base.request_fingerprint(
                    task, base.request_summary(messages), settings
                )
                if (
                    existing.get("task_key") != task.key
                    or existing.get("request_fingerprint") != expected
                ):
                    raise ValueError(
                        f"Mismatched provider-rejection checkpoint: {task.key}"
                    )
                base.atomic_write_json(record_path(output_directory, task), terminal)
                existing = terminal
        if (
            existing
            and args.retry_terminal_provider_failures
            and existing.get("status") == "model_failure"
            and existing.get("failure_type")
            in {"provider_content_rejection", "provider_request_failure"}
        ):
            existing = None
        if existing and existing.get("status") in TERMINAL_STATUSES:
            expected = base.request_fingerprint(task, base.request_summary(messages), settings)
            if existing.get("task_key") != task.key or existing.get("request_fingerprint") != expected:
                raise ValueError(f"Mismatched terminal checkpoint: {task.key}")
            continue
        pending.append(task)
    if args.limit_per_mode is not None:
        pending = [
            task
            for mode in selected_modes
            for task in [value for value in pending if value.mode == mode][: args.limit_per_mode]
        ]

    limiter = base.SmoothDualRateLimiter(args.requests_per_minute, args.tokens_per_minute)
    LOGGER.info("Starting %d pending of %d configured requests.", len(pending), len(all_tasks))

    def execute(task: base.Task) -> dict[str, Any]:
        condition = condition_by_mode[task.mode]
        labels = display_labels(condition, categories, self_names)
        settings = condition_settings(condition, common_settings)
        messages = build_messages(
            task,
            condition,
            test_directory,
            categories,
            self_names,
            references,
            assets,
            readme,
        )
        return base.execute_task(
            task,
            client,
            test_directory,
            labels,
            {},
            {},
            {},
            settings,
            args.max_retries,
            TaskRateLimiter(limiter, token_estimates[task.mode]),
            messages_override=messages,
            force_single_category_labels=condition.single_class,
        )

    completed = 0
    write_lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(execute, task): task for task in pending}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001
                record = {
                    "status": "error",
                    "error": f"WorkerFailure: {type(error).__name__}: {error}",
                    "task": asdict(task),
                    "task_key": task.key,
                    "predictions": [],
                    "completed_at": base.utc_now(),
                }
            with write_lock:
                base.atomic_write_json(record_path(output_directory, task), record)
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    progress = summarize_records(all_tasks, output_directory)
                    base.atomic_write_json(output_directory / "progress.json", progress)
                    LOGGER.info(
                        "Checkpoint %d/%d; terminal=%d/%d, errors=%d.",
                        completed,
                        len(pending),
                        progress["total"]["success"] + progress["total"]["model_failure"],
                        progress["total"]["total"],
                        progress["total"]["error"],
                    )

    progress = summarize_records(all_tasks, output_directory)
    base.atomic_write_json(output_directory / "progress.json", progress)
    finalize(
        all_tasks,
        conditions,
        evaluation_path,
        output_directory,
        image_count=len(test["images"]),
        class_count=len(categories),
        references=references,
    )
    selected = summarize_records(tasks, output_directory)["total"]
    unresolved = selected["error"] + selected["pending"]
    return 0 if unresolved == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
