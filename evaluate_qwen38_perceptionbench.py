#!/usr/bin/env python3
"""Protocol-locked, resumable Qwen3.8 evaluation on PerceptionBench.

This preserves MoonshotAI's released prompt construction, generation cap,
judge prompt, strict judge parser, accuracy metric, and default concurrency.
The only model-specific addition is Qwen's official maximum reasoning switch.
Predictions and judgments are separate resumable phases so a judge outage can
never require repeating expensive multimodal inference.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from prepare_perceptionbench import (
    HF_COMMIT,
    HF_DATA_SHA256,
    UPSTREAM_COMMIT,
    UPSTREAM_FILES,
    iter_jsonl,
)

MODEL_IDS = ("qwen3.8-max", "qwen3.8-flash")
MODEL_PRICES_INTERNATIONAL = {
    "qwen3.8-max": {"input_per_million": 2.0, "output_per_million": 6.0},
    "qwen3.8-flash": {"input_per_million": 0.15, "output_per_million": 0.47},
}
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_CONCURRENCY = 16
DEFAULT_MAX_TOKENS = 65_536
REASONING_EFFORT = "xhigh"
IMAGE_PH = re.compile(r"<\|image_(\d+)\|>")
LOGGER = logging.getLogger("perceptionbench")
_THREAD_LOCAL = threading.local()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def build_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Exact semantic copy of MoonshotAI/PerceptionBench eval.build_messages."""

    images = record.get("image") or []
    images = images if isinstance(images, list) else []
    problem = str(record.get("problem", "") or "")
    content: list[dict[str, Any]] = []
    last = 0
    used: set[int] = set()
    for match in IMAGE_PH.finditer(problem):
        number = int(match.group(1))
        if 1 <= number <= len(images):
            used.add(number - 1)
            segment = problem[last : match.start()]
            if segment:
                content.append({"type": "text", "text": segment})
            content.append(
                {"type": "image_url", "image_url": {"url": images[number - 1]}}
            )
            last = match.end()
    if problem[last:]:
        content.append({"type": "text", "text": problem[last:]})
    for index, image in enumerate(images):
        if index not in used:
            content.append({"type": "image_url", "image_url": {"url": image}})
    messages: list[dict[str, Any]] = []
    system = record.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content if content else problem})
    return messages


def normalize_escape(value: str) -> str:
    """Exact semantic copy of the official evaluator helper."""

    whitelist = " 0123456789-"
    value = re.sub(rf"\\+[{whitelist}]", r"\\\\" + " ", value)
    value = re.sub(rf"\\+(?![{whitelist}])", r"\\", value)
    return re.sub(r"\\+n", "\n", value)


def decode_judge(response: str) -> tuple[bool, str]:
    """Exact strict parser used by the official evaluator."""

    response = str(response).strip()
    if "[reason]" not in response or "[judge]" not in response:
        return False, "No [reason] or [judge] in output"
    reason = response.split("[judge]")[0].split("[reason]")[-1].strip()
    verdict = response.split("[judge]")[-1].strip()
    return "true" in verdict.lower(), reason


def record_fingerprint(record: dict[str, Any]) -> str:
    """Fingerprint only information visible to the evaluated model."""

    return sha256_json(
        {
            "index": record.get("index"),
            "problem": record.get("problem"),
            "system": record.get("system"),
            "image": record.get("image"),
        }
    )


def checkpoint_path(run_dir: Path, phase: str, index: int) -> Path:
    return run_dir / phase / f"{index:04d}.json"


def response_usage(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return None


def get_client(api_key: str, base_url: str, timeout_seconds: float) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=0,
    )


def thread_client(api_key: str, base_url: str, timeout_seconds: float, name: str) -> Any:
    cache = getattr(_THREAD_LOCAL, "clients", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.clients = cache
    key = (name, base_url, timeout_seconds)
    if key not in cache:
        cache[key] = get_client(api_key, base_url, timeout_seconds)
    return cache[key]


def qwen_stream(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    """Call Qwen with the official maximum reasoning regime.

    `max_tokens`, rather than `max_completion_tokens`, intentionally mirrors
    PerceptionBench's released evaluator: it caps the final answer while Qwen's
    xhigh reasoning allowance remains controlled independently by the provider.
    """

    started = time.monotonic()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={
            "enable_thinking": True,
            "reasoning_effort": REASONING_EFFORT,
        },
        extra_headers={"X-DashScope-Wait-Timeout": "30"},
    )
    answer: list[str] = []
    finish_reason = None
    usage = None
    reasoning_characters = 0
    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage = response_usage(chunk.usage)
        for choice in getattr(chunk, "choices", []):
            delta = choice.delta
            content = getattr(delta, "content", None)
            if content:
                answer.append(content)
            # Do not persist private reasoning text. This counter is diagnostic
            # only; metered token counts come from provider usage.
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_characters += len(reasoning)
            if choice.finish_reason:
                finish_reason = choice.finish_reason
    return {
        "prediction": "".join(answer),
        "finish_reason": finish_reason,
        "usage": usage,
        "elapsed_seconds": time.monotonic() - started,
        "reasoning_characters_observed": reasoning_characters,
    }


def call_with_retry(operation: Callable[[], Any], max_retries: int) -> tuple[Any, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(max_retries + 1):
        started = time.monotonic()
        try:
            result = operation()
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "status": "ok",
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            return result, attempts
        except Exception as error:
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "status": "error",
                    "elapsed_seconds": time.monotonic() - started,
                    "error_type": type(error).__name__,
                    "error": str(error)[:1000],
                }
            )
            if attempt >= max_retries:
                raise RuntimeError(json.dumps(attempts, ensure_ascii=False)) from error
            time.sleep(min(2**attempt, 8))
    raise AssertionError("unreachable")


def predict_one(record: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    fingerprint = record_fingerprint(record)
    try:
        messages = build_messages(record)
        client = thread_client(
            settings["api_key"],
            settings["base_url"],
            settings["timeout_seconds"],
            "prediction",
        )
        result, attempts = call_with_retry(
            lambda: qwen_stream(
                client,
                model=settings["model"],
                messages=messages,
                max_tokens=settings["max_tokens"],
            ),
            settings["max_retries"],
        )
        prediction = result["prediction"]
        status = "complete" if prediction.strip() else "model_failure"
        return {
            "index": record["index"],
            "error_category": record.get("error_category"),
            "source_bmk": record.get("source_bmk"),
            "status": status,
            "prediction": prediction,
            "finish_reason": result["finish_reason"],
            "usage": result["usage"],
            "elapsed_seconds": result["elapsed_seconds"],
            "reasoning_characters_observed": result["reasoning_characters_observed"],
            "attempts": attempts,
            "record_fingerprint": fingerprint,
            "completed_at": utc_now(),
        }
    except Exception as error:
        cause = error.__cause__
        attempts = []
        if isinstance(error, RuntimeError):
            try:
                attempts = json.loads(str(error))
            except json.JSONDecodeError:
                pass
        return {
            "index": record.get("index"),
            "error_category": record.get("error_category"),
            "source_bmk": record.get("source_bmk"),
            "status": "model_failure",
            "prediction": "",
            "error_type": type(cause or error).__name__,
            "error": str(cause or error)[:2000],
            "attempts": attempts,
            "record_fingerprint": fingerprint,
            "completed_at": utc_now(),
        }


def judge_prompt(template: str, record: dict[str, Any], prediction: str) -> str:
    answer = record.get("answer")
    reference = answer.get("answer") if isinstance(answer, dict) else answer
    return (
        template.replace(
            "{problem}", normalize_escape(str(record.get("problem", ""))).strip()
        )
        .replace(
            "{reference_answer}", normalize_escape(str(reference or "")).strip()
        )
        .replace("{assistant_answer}", normalize_escape(str(prediction)).strip())
    )


def judge_one(
    record: dict[str, Any], prediction_record: dict[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    prediction = str(prediction_record.get("prediction") or "")
    if prediction_record.get("status") != "complete" or not prediction.strip():
        return {
            "index": record["index"],
            "status": "complete",
            "judge_result": 0,
            "judge_reason": "failed to obtain answer",
            "prediction_fingerprint": sha256_json(prediction_record),
            "completed_at": utc_now(),
        }
    prompt = judge_prompt(settings["judge_template"], record, prediction)
    client = thread_client(
        settings["api_key"],
        settings["base_url"],
        settings["timeout_seconds"],
        "judge",
    )

    def operation() -> Any:
        return client.chat.completions.create(
            model=settings["model"],
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )

    try:
        response, attempts = call_with_retry(operation, settings["max_retries"])
        raw = response.choices[0].message.content or ""
        correct, reason = decode_judge(raw)
        return {
            "index": record["index"],
            "status": "complete",
            "judge_result": 1 if correct else 0,
            "judge_reason": reason[:200],
            "raw_judge_response": raw,
            "usage": response_usage(getattr(response, "usage", None)),
            "attempts": attempts,
            "prediction_fingerprint": sha256_json(prediction_record),
            "completed_at": utc_now(),
        }
    except Exception as error:
        return {
            "index": record["index"],
            "status": "judge_failure",
            "judge_result": 0,
            "judge_reason": f"judge error: {error}"[:200],
            "error_type": type(error.__cause__ or error).__name__,
            "error": str(error.__cause__ or error)[:2000],
            "prediction_fingerprint": sha256_json(prediction_record),
            "completed_at": utc_now(),
        }


def iter_pending(
    dataset_path: Path,
    run_dir: Path,
    phase: str,
    *,
    limit: int | None,
    retry_failures: bool,
    indices: set[int] | None = None,
) -> Iterator[dict[str, Any]]:
    selected = 0
    for record in iter_jsonl(dataset_path):
        if indices is not None and int(record["index"]) not in indices:
            continue
        path = checkpoint_path(run_dir, phase, int(record["index"]))
        checkpoint = read_json(path)
        if checkpoint is not None:
            if phase == "predictions" and checkpoint.get(
                "record_fingerprint"
            ) != record_fingerprint(record):
                raise ValueError(
                    f"Stale prediction checkpoint for index {record['index']}"
                )
            terminal = checkpoint.get("status") == "complete"
            if phase == "predictions":
                terminal = terminal or (
                    checkpoint.get("status") == "model_failure" and not retry_failures
                )
            if terminal:
                continue
        if limit is not None and selected >= limit:
            break
        selected += 1
        yield record


def run_bounded(
    records: Iterable[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    concurrency: int,
    save: Callable[[dict[str, Any]], None],
) -> int:
    completed = 0
    iterator = iter(records)
    with futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        inflight: dict[futures.Future[dict[str, Any]], int] = {}

        def fill() -> None:
            while len(inflight) < concurrency * 2:
                try:
                    record = next(iterator)
                except StopIteration:
                    break
                future = executor.submit(worker, record)
                inflight[future] = int(record["index"])

        fill()
        while inflight:
            done, _ = futures.wait(inflight, return_when=futures.FIRST_COMPLETED)
            for future in done:
                index = inflight.pop(future)
                result = future.result()
                save(result)
                completed += 1
                if completed % 25 == 0:
                    LOGGER.info("Completed %d new records (latest index %d)", completed, index)
            fill()
    return completed


def source_manifest(data_dir: Path) -> dict[str, Any]:
    manifest = read_json(data_dir / "source_manifest.json")
    if manifest is None:
        raise FileNotFoundError(
            f"Missing {data_dir / 'source_manifest.json'}; run prepare_perceptionbench.py first"
        )
    if manifest.get("dataset", {}).get("sha256") != HF_DATA_SHA256:
        raise ValueError("Prepared dataset does not match the pinned release")
    return manifest


def write_or_check_run_manifest(run_dir: Path, expected: dict[str, Any]) -> None:
    path = run_dir / "run_manifest.json"
    existing = read_json(path)
    if existing is not None:
        comparable = {key: existing.get(key) for key in expected}
        if comparable != expected:
            raise ValueError(
                f"Run settings differ from the existing manifest at {path}; use a new run directory"
            )
        return
    atomic_json(path, {**expected, "created_at": utc_now()})


def write_or_check_judge_manifest(run_dir: Path, expected: dict[str, Any]) -> None:
    path = run_dir / "judge_manifest.json"
    existing = read_json(path)
    if existing is not None:
        comparable = {key: existing.get(key) for key in expected}
        if comparable != expected:
            raise ValueError(
                f"Judge settings differ from the existing manifest at {path}; "
                "use a new run directory or the original judge settings"
            )
        return
    atomic_json(path, {**expected, "created_at": utc_now()})


def configure_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        handlers=[logging.FileHandler(run_dir / "experiment.log"), logging.StreamHandler()],
        force=True,
    )


def command_predict(args: argparse.Namespace) -> int:
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    configure_logging(run_dir)
    manifest = source_manifest(data_dir)
    expected = {
        "benchmark": "PerceptionBench",
        "dataset_commit": HF_COMMIT,
        "dataset_sha256": manifest["dataset"]["sha256"],
        "evaluator_commit": UPSTREAM_COMMIT,
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "reasoning": {"enable_thinking": True, "reasoning_effort": REASONING_EFFORT},
        "max_tokens": args.max_tokens,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "seed": None,
        "image_resolution_override": None,
        "prompt_builder": "official-exact-v1",
    }
    write_or_check_run_manifest(run_dir, expected)
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required")
    settings = {
        "api_key": api_key,
        "base_url": args.base_url.rstrip("/"),
        "timeout_seconds": args.timeout_seconds,
        "max_retries": args.max_retries,
        "model": args.model,
        "max_tokens": args.max_tokens,
    }
    selected_indices = (
        {int(value) for value in args.indices.split(",") if value.strip()}
        if args.indices
        else None
    )
    if selected_indices is not None and any(
        value < 0 or value >= 3000 for value in selected_indices
    ):
        raise ValueError("--indices must contain only values from 0 through 2999")
    pending = iter_pending(
        data_dir / "PerceptionBench.jsonl",
        run_dir,
        "predictions",
        limit=args.limit,
        retry_failures=args.retry_failures,
        indices=selected_indices,
    )
    count = run_bounded(
        pending,
        lambda record: predict_one(record, settings),
        concurrency=args.concurrency,
        save=lambda result: atomic_json(
            checkpoint_path(run_dir, "predictions", int(result["index"])), result
        ),
    )
    LOGGER.info("Prediction phase wrote %d new checkpoints", count)
    write_status(data_dir, run_dir, args.model)
    return 0


def command_judge(args: argparse.Namespace) -> int:
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    configure_logging(run_dir)
    source_manifest(data_dir)
    run_manifest = read_json(run_dir / "run_manifest.json")
    if run_manifest is None:
        raise FileNotFoundError("Prediction run manifest is missing")
    judge_template_path = data_dir / "upstream/eval/judge_prompt.txt"
    if hashlib.sha256(judge_template_path.read_bytes()).hexdigest() != UPSTREAM_FILES[
        "eval/judge_prompt.txt"
    ]:
        raise ValueError("Judge prompt does not match the pinned upstream prompt")
    api_key = os.environ.get("PERCEPTIONBENCH_JUDGE_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    base_url = args.judge_base_url or os.environ.get("PERCEPTIONBENCH_JUDGE_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError(
            "Set PERCEPTIONBENCH_JUDGE_API_KEY (or OPENAI_API_KEY) and "
            "PERCEPTIONBENCH_JUDGE_BASE_URL for the paper's gpt-oss-120b judge"
        )
    settings = {
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "timeout_seconds": args.timeout_seconds,
        "max_retries": args.max_retries,
        "model": args.judge_model,
        "judge_template": judge_template_path.read_text(encoding="utf-8"),
    }
    write_or_check_judge_manifest(
        run_dir,
        {
            "model": args.judge_model,
            "base_url": base_url.rstrip("/"),
            "temperature": 0.3,
            "judge_prompt_sha256": UPSTREAM_FILES["eval/judge_prompt.txt"],
            "evaluator_commit": UPSTREAM_COMMIT,
        },
    )

    def records() -> Iterator[dict[str, Any]]:
        selected = 0
        for record in iter_jsonl(data_dir / "PerceptionBench.jsonl"):
            index = int(record["index"])
            prediction = read_json(checkpoint_path(run_dir, "predictions", index))
            if prediction is None:
                continue
            existing = read_json(checkpoint_path(run_dir, "judgments", index))
            expected_fingerprint = sha256_json(prediction)
            if existing is not None:
                if existing.get("prediction_fingerprint") != expected_fingerprint:
                    raise ValueError(f"Stale judgment checkpoint for index {index}")
                if existing.get("status") == "complete" or not args.retry_failures:
                    continue
            if args.limit is not None and selected >= args.limit:
                break
            selected += 1
            record["_prediction_record"] = prediction
            yield record

    count = run_bounded(
        records(),
        lambda record: judge_one(record, record.pop("_prediction_record"), settings),
        concurrency=args.concurrency,
        save=lambda result: atomic_json(
            checkpoint_path(run_dir, "judgments", int(result["index"])), result
        ),
    )
    LOGGER.info("Judge phase wrote %d new checkpoints", count)
    write_status(data_dir, run_dir, str(run_manifest["model"]))
    return 0


def usage_totals(run_dir: Path, phase: str) -> dict[str, int]:
    total: Counter[str] = Counter()
    directory = run_dir / phase
    if not directory.exists():
        return dict(total)
    for path in directory.glob("*.json"):
        record = read_json(path) or {}
        usage = record.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                total[key] += value
        completion_details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = completion_details.get("reasoning_tokens")
        if isinstance(reasoning_tokens, int):
            total["reasoning_tokens"] += reasoning_tokens
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached_tokens = prompt_details.get("cached_tokens")
        if isinstance(cached_tokens, int):
            total["cached_tokens"] += cached_tokens
    return dict(total)


def status_summary(data_dir: Path, run_dir: Path, model: str) -> dict[str, Any]:
    prediction_status: Counter[str] = Counter()
    judge_status: Counter[str] = Counter()
    for index in range(3000):
        prediction = read_json(checkpoint_path(run_dir, "predictions", index))
        judgment = read_json(checkpoint_path(run_dir, "judgments", index))
        prediction_status[
            str(prediction.get("status", "missing")) if prediction else "missing"
        ] += 1
        judge_status[str(judgment.get("status", "missing")) if judgment else "missing"] += 1
    usage = usage_totals(run_dir, "predictions")
    price = MODEL_PRICES_INTERNATIONAL.get(model)
    estimated_cost = None
    if price:
        estimated_cost = (
            usage.get("prompt_tokens", 0) * price["input_per_million"]
            + usage.get("completion_tokens", 0) * price["output_per_million"]
        ) / 1_000_000
    return {
        "updated_at": utc_now(),
        "model": model,
        "prediction_status": dict(prediction_status),
        "judgment_status": dict(judge_status),
        "prediction_usage": usage,
        "prediction_cost_usd_at_international_list_price": estimated_cost,
        "pricing": price,
        "dataset_path": str(data_dir / "PerceptionBench.jsonl"),
    }


def write_status(data_dir: Path, run_dir: Path, model: str) -> dict[str, Any]:
    summary = status_summary(data_dir, run_dir, model)
    atomic_json(run_dir / "status.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def command_status(args: argparse.Namespace) -> int:
    run_manifest = read_json(args.run_dir.resolve() / "run_manifest.json") or {}
    model = args.model or str(run_manifest.get("model") or "unknown")
    write_status(args.data_dir.resolve(), args.run_dir.resolve(), model)
    return 0


def command_score(args: argparse.Namespace) -> int:
    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    run_manifest = read_json(run_dir / "run_manifest.json")
    if run_manifest is None:
        raise FileNotFoundError("Run manifest is missing")
    judge_manifest = read_json(run_dir / "judge_manifest.json")
    if judge_manifest is None:
        raise FileNotFoundError("Judge manifest is missing")
    results: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    missing: list[int] = []
    for record in iter_jsonl(data_dir / "PerceptionBench.jsonl"):
        index = int(record["index"])
        prediction = read_json(checkpoint_path(run_dir, "predictions", index))
        judgment = read_json(checkpoint_path(run_dir, "judgments", index))
        if prediction is None or judgment is None or judgment.get("status") != "complete":
            missing.append(index)
            continue
        category = str(record.get("error_category") or "?")
        result = int(judgment.get("judge_result", 0))
        totals[category] += 1
        correct[category] += result
        results.append(
            {
                "index": index,
                "error_category": record.get("error_category"),
                "source_bmk": record.get("source_bmk"),
                "prediction": prediction.get("prediction", ""),
                "judge_result": result,
                "judge_reason": judgment.get("judge_reason", ""),
            }
        )
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"Cannot publish a benchmark score with {len(missing)} missing judgments; "
            f"first indices: {missing[:20]}"
        )
    output = run_dir / "results"
    output.mkdir(parents=True, exist_ok=True)
    model = str(run_manifest["model"])
    stem = f"PerceptionBench_{model.replace('/', '_')}"
    with (output / f"{stem}.jsonl").open("w", encoding="utf-8") as handle:
        for result in sorted(results, key=lambda value: value["index"]):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    denominator = sum(totals.values())
    scores = {
        "overall": sum(correct.values()) / max(1, denominator),
        "per_category": {
            category: correct[category] / totals[category] for category in totals
        },
        "correct": sum(correct.values()),
        "scored": denominator,
        "expected": 3000,
        "missing": missing,
        "metric": "accuracy",
        "judge_model": judge_manifest["model"],
        "judge_base_url": judge_manifest["base_url"],
        "judge_prompt_sha256": UPSTREAM_FILES["eval/judge_prompt.txt"],
        "paper_comparable": (
            denominator == 3000
            and not missing
            and judge_manifest["model"] == "gpt-oss-120b"
        ),
    }
    atomic_json(output / f"{stem}_scores.json", scores)
    print(json.dumps(scores, indent=2, ensure_ascii=False))
    return 0


def add_shared_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("PerceptionBench"))
    parser.add_argument("--run-dir", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Run/resume Qwen predictions")
    add_shared_paths(predict)
    predict.add_argument("--model", choices=MODEL_IDS, required=True)
    predict.add_argument("--base-url", default=DEFAULT_BASE_URL)
    predict.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    predict.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    predict.add_argument("--timeout-seconds", type=float, default=3600)
    predict.add_argument("--max-retries", type=int, default=3)
    predict.add_argument("--limit", type=int)
    predict.add_argument(
        "--indices",
        help="Comma-separated record indices for a smoke test; full runs omit this.",
    )
    predict.add_argument("--retry-failures", action="store_true")
    predict.set_defaults(func=command_predict)

    judge = subparsers.add_parser("judge", help="Run/resume the paper's judge")
    add_shared_paths(judge)
    judge.add_argument("--judge-model", default="gpt-oss-120b")
    judge.add_argument("--judge-base-url")
    judge.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    judge.add_argument("--timeout-seconds", type=float, default=600)
    judge.add_argument("--max-retries", type=int, default=3)
    judge.add_argument("--limit", type=int)
    judge.add_argument("--retry-failures", action="store_true")
    judge.set_defaults(func=command_judge)

    score = subparsers.add_parser("score", help="Emit official JSONL and accuracy")
    add_shared_paths(score)
    score.add_argument("--allow-incomplete", action="store_true")
    score.set_defaults(func=command_score)

    status = subparsers.add_parser("status", help="Summarize checkpoints and usage")
    add_shared_paths(status)
    status.add_argument("--model", choices=MODEL_IDS)
    status.set_defaults(func=command_status)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("concurrency", "max_retries"):
        if hasattr(args, name) and getattr(args, name) < (1 if name == "concurrency" else 0):
            raise ValueError(f"--{name.replace('_', '-')} is out of range")
    if getattr(args, "limit", None) is not None and args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
