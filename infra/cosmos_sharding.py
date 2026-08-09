#!/usr/bin/env python3
"""Fail-closed dataset sharding and aggregation for the Cosmos RF100VL run.

The original single-pod run remains the canonical source for datasets completed
before a shard plan is frozen. Every remaining dataset is assigned to exactly
one isolated GCS prefix. The final canonical success marker is written only
after all shard summaries and per-image checkpoint counts have been verified.
"""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_cosmos import MODEL_ID, PROMPT_VERSION, parse_gcs_uri
from gcs_io import client as gcs_client


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _dataset_index(preflight: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    dataset_section = preflight.get("dataset", {})
    datasets = dataset_section.get("datasets", [])
    _require(preflight.get("status") == "passed", "Preflight status is not passed.")
    _require(isinstance(datasets, list), "Preflight datasets must be a list.")
    result: dict[str, dict[str, Any]] = {}
    for item in datasets:
        _require(isinstance(item, dict), "Preflight dataset entry is not an object.")
        name = item.get("dataset")
        _require(isinstance(name, str) and name, "Preflight dataset name is invalid.")
        _require(name not in result, f"Duplicate preflight dataset: {name}")
        image_count = item.get("image_count")
        annotation_sha256 = item.get("annotation_sha256")
        _require(
            isinstance(image_count, int) and image_count > 0,
            f"Invalid image count for {name}.",
        )
        _require(
            isinstance(annotation_sha256, str) and len(annotation_sha256) == 64,
            f"Invalid annotation hash for {name}.",
        )
        result[name] = {
            "dataset": name,
            "image_count": image_count,
            "annotation_count": int(item.get("annotation_count", 0)),
            "annotation_sha256": annotation_sha256,
        }
    expected = dataset_section.get("expected_dataset_count", len(result))
    _require(len(result) == expected, "Preflight dataset count does not match its contract.")
    declared_images = dataset_section.get("image_count")
    if declared_images is not None:
        _require(
            sum(item["image_count"] for item in result.values()) == declared_images,
            "Preflight total image count is inconsistent.",
        )
    return result


def _baseline_summaries(
    baseline_aggregate: Mapping[str, Any], dataset_index: Mapping[str, Any]
) -> list[dict[str, Any]]:
    summaries = baseline_aggregate.get("datasets", [])
    _require(isinstance(summaries, list), "Baseline summaries must be a list.")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for summary in summaries:
        _require(isinstance(summary, dict), "Baseline summary is not an object.")
        name = summary.get("dataset")
        _require(name in dataset_index, f"Unknown baseline dataset: {name}")
        _require(name not in seen, f"Duplicate baseline dataset: {name}")
        _require(summary.get("complete") is True, f"Baseline dataset is incomplete: {name}")
        _require("metrics" in summary, f"Baseline dataset is unscored: {name}")
        _require(
            summary.get("completed_image_count") == dataset_index[name]["image_count"],
            f"Baseline image count mismatch for {name}.",
        )
        seen.add(name)
        result.append(dict(summary))
    return result


def _balanced_assignments(
    datasets: Sequence[dict[str, Any]], shard_count: int
) -> list[list[dict[str, Any]]]:
    _require(shard_count > 0, "Shard count must be positive.")
    _require(len(datasets) >= shard_count, "There are fewer remaining datasets than shards.")
    bins: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for dataset in sorted(
        datasets, key=lambda item: (-item["image_count"], item["dataset"])
    ):
        shard_index = min(range(shard_count), key=lambda index: (loads[index], index))
        bins[shard_index].append(dict(dataset))
        loads[shard_index] += dataset["image_count"]
    for values in bins:
        values.sort(key=lambda item: item["dataset"])
    return bins


def build_shard_plan(
    *,
    preflight: Mapping[str, Any],
    baseline_aggregate: Mapping[str, Any],
    gcs_run_uri: str,
    shard_count: int,
    image_ref: str,
    benchmark_git_sha: str,
    model_revision: str,
    plan_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build and validate a deterministic, disjoint continuation plan."""

    parse_gcs_uri(gcs_run_uri)
    _require(
        image_ref.rsplit("@sha256:", 1)[-1] != image_ref
        and len(image_ref.rsplit("@sha256:", 1)[-1]) == 64,
        "Shard image_ref must use an immutable sha256 digest.",
    )
    _require(len(benchmark_git_sha) == 40, "Benchmark git SHA must be full length.")
    _require(len(model_revision) == 40, "Model revision must be full length.")
    dataset_index = _dataset_index(preflight)
    baseline = _baseline_summaries(baseline_aggregate, dataset_index)
    baseline_names = {item["dataset"] for item in baseline}
    remaining = [
        dict(item)
        for name, item in sorted(dataset_index.items())
        if name not in baseline_names
    ]
    assignments = _balanced_assignments(remaining, shard_count)
    identity = {
        "gcs_run_uri": gcs_run_uri.rstrip("/"),
        "baseline_datasets": sorted(baseline_names),
        "remaining": [item["dataset"] for item in remaining],
        "shard_count": shard_count,
        "image_ref": image_ref,
        "benchmark_git_sha": benchmark_git_sha,
    }
    if plan_id is None:
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        plan_id = f"shard-{digest}"
    _require(
        isinstance(plan_id, str)
        and plan_id
        and all(character.isalnum() or character in "-_" for character in plan_id),
        "Plan ID contains unsafe characters.",
    )
    shard_root = f"{gcs_run_uri.rstrip('/')}/full-shards/{plan_id}"
    plan = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "plan_id": plan_id,
        "created_at": created_at or utc_now(),
        "model_id": preflight.get("model_id", MODEL_ID),
        "model_revision": model_revision,
        "prompt_version": preflight.get("prompt_version", PROMPT_VERSION),
        "image_ref": image_ref,
        "benchmark_git_sha": benchmark_git_sha,
        "gcs_run_uri": gcs_run_uri.rstrip("/"),
        "canonical_full_uri": f"{gcs_run_uri.rstrip('/')}/full",
        "expected_dataset_count": len(dataset_index),
        "expected_image_count": sum(
            item["image_count"] for item in dataset_index.values()
        ),
        "baseline": {
            "aggregate_uri": (
                f"{gcs_run_uri.rstrip('/')}/control/shards/{plan_id}/"
                "baseline_aggregate.json"
            ),
            "dataset_count": len(baseline),
            "image_count": sum(dataset_index[name]["image_count"] for name in baseline_names),
            "datasets": sorted(baseline_names),
            "aggregate_content_sha256": canonical_json_sha256(baseline_aggregate),
        },
        "datasets": [dataset_index[name] for name in sorted(dataset_index)],
        "shards": [
            {
                "shard_id": f"shard-{index:02d}",
                "gcs_uri": f"{shard_root}/shard-{index:02d}",
                "dataset_count": len(values),
                "image_count": sum(item["image_count"] for item in values),
                "datasets": values,
            }
            for index, values in enumerate(assignments)
        ],
    }
    validate_shard_plan(plan)
    return plan


def validate_shard_plan(plan: Mapping[str, Any]) -> None:
    _require(plan.get("schema_version") == SCHEMA_VERSION, "Unsupported shard schema.")
    _require(plan.get("status") == "frozen", "Shard plan is not frozen.")
    parse_gcs_uri(str(plan.get("gcs_run_uri", "")))
    parse_gcs_uri(str(plan.get("canonical_full_uri", "")))
    all_items = plan.get("datasets", [])
    _require(isinstance(all_items, list), "Plan datasets must be a list.")
    all_index = {item.get("dataset"): item for item in all_items if isinstance(item, dict)}
    _require(len(all_index) == len(all_items), "Plan datasets contain duplicates.")
    _require(
        len(all_index) == plan.get("expected_dataset_count"),
        "Plan dataset count is inconsistent.",
    )
    _require(
        sum(item.get("image_count", 0) for item in all_index.values())
        == plan.get("expected_image_count"),
        "Plan image count is inconsistent.",
    )
    baseline = plan.get("baseline", {})
    baseline_names = baseline.get("datasets", [])
    _require(
        isinstance(baseline_names, list) and len(set(baseline_names)) == len(baseline_names),
        "Baseline dataset names are invalid or duplicated.",
    )
    _require(
        set(baseline_names).issubset(all_index), "Baseline contains an unknown dataset."
    )
    _require(
        baseline.get("dataset_count") == len(baseline_names),
        "Baseline dataset count is inconsistent.",
    )
    _require(
        baseline.get("image_count")
        == sum(all_index[name]["image_count"] for name in baseline_names),
        "Baseline image count is inconsistent.",
    )
    _require(
        isinstance(baseline.get("aggregate_content_sha256"), str)
        and len(baseline["aggregate_content_sha256"]) == 64,
        "Baseline aggregate content hash is invalid.",
    )
    assigned: list[str] = []
    shard_ids: set[str] = set()
    shard_uris: set[str] = set()
    shards = plan.get("shards", [])
    _require(isinstance(shards, list) and shards, "Plan contains no shards.")
    for shard in shards:
        _require(isinstance(shard, dict), "Shard entry is not an object.")
        shard_id = shard.get("shard_id")
        gcs_uri = shard.get("gcs_uri")
        _require(isinstance(shard_id, str) and shard_id, "Shard ID is invalid.")
        _require(shard_id not in shard_ids, f"Duplicate shard ID: {shard_id}")
        parse_gcs_uri(str(gcs_uri))
        _require(gcs_uri not in shard_uris, f"Duplicate shard GCS URI: {gcs_uri}")
        shard_ids.add(shard_id)
        shard_uris.add(gcs_uri)
        items = shard.get("datasets", [])
        names = [item.get("dataset") for item in items if isinstance(item, dict)]
        _require(len(names) == len(items), f"Malformed dataset in {shard_id}.")
        _require(len(names) == len(set(names)), f"Duplicate dataset inside {shard_id}.")
        _require(
            shard.get("dataset_count") == len(items),
            f"Dataset count mismatch in {shard_id}.",
        )
        _require(
            shard.get("image_count") == sum(item.get("image_count", 0) for item in items),
            f"Image count mismatch in {shard_id}.",
        )
        for item in items:
            name = item["dataset"]
            _require(name in all_index, f"Unknown shard dataset: {name}")
            _require(item == all_index[name], f"Dataset metadata mismatch for {name}.")
        assigned.extend(names)
    remaining = set(all_index) - set(baseline_names)
    _require(len(assigned) == len(set(assigned)), "A dataset is assigned to multiple shards.")
    _require(set(assigned) == remaining, "Shard assignments do not exactly cover remaining data.")


def shard_by_id(plan: Mapping[str, Any], shard_id: str) -> dict[str, Any]:
    validate_shard_plan(plan)
    matches = [item for item in plan["shards"] if item["shard_id"] == shard_id]
    _require(len(matches) == 1, f"Unknown shard ID: {shard_id}")
    return dict(matches[0])


def verify_shard_aggregate(
    plan: Mapping[str, Any], shard_id: str, aggregate: Mapping[str, Any]
) -> None:
    shard = shard_by_id(plan, shard_id)
    expected = {item["dataset"]: item for item in shard["datasets"]}
    summaries = aggregate.get("datasets", [])
    _require(aggregate.get("status") == "complete", f"{shard_id} is not complete.")
    _require(isinstance(summaries, list), f"{shard_id} summaries are malformed.")
    actual: dict[str, Mapping[str, Any]] = {}
    for summary in summaries:
        name = summary.get("dataset") if isinstance(summary, dict) else None
        _require(name in expected, f"Unexpected dataset in {shard_id}: {name}")
        _require(name not in actual, f"Duplicate dataset in {shard_id}: {name}")
        _require(summary.get("complete") is True, f"Incomplete dataset in {shard_id}: {name}")
        _require("metrics" in summary, f"Unscored dataset in {shard_id}: {name}")
        _require(
            summary.get("completed_image_count") == expected[name]["image_count"],
            f"Completed image count mismatch for {name}.",
        )
        actual[name] = summary
    _require(set(actual) == set(expected), f"{shard_id} dataset coverage is incomplete.")
    for key in ("selected_dataset_count", "processed_dataset_count", "scored_dataset_count"):
        _require(aggregate.get(key) == len(expected), f"{shard_id} {key} is incorrect.")


def merge_aggregates(
    plan: Mapping[str, Any],
    baseline_aggregate: Mapping[str, Any],
    shard_aggregates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge summaries only after proving exact, unique RF100VL coverage."""

    validate_shard_plan(plan)
    all_index = {item["dataset"]: item for item in plan["datasets"]}
    baseline = _baseline_summaries(baseline_aggregate, all_index)
    _require(
        canonical_json_sha256(baseline_aggregate)
        == plan["baseline"]["aggregate_content_sha256"],
        "Frozen baseline aggregate content hash does not match the plan.",
    )
    _require(
        {item["dataset"] for item in baseline} == set(plan["baseline"]["datasets"]),
        "Frozen baseline aggregate no longer matches the plan.",
    )
    summaries = list(baseline)
    expected_shard_ids = {item["shard_id"] for item in plan["shards"]}
    _require(
        set(shard_aggregates) == expected_shard_ids,
        "Shard aggregate set does not match the frozen plan.",
    )
    for shard in plan["shards"]:
        shard_id = shard["shard_id"]
        aggregate = shard_aggregates[shard_id]
        verify_shard_aggregate(plan, shard_id, aggregate)
        summaries.extend(dict(item) for item in aggregate["datasets"])
    names = [item["dataset"] for item in summaries]
    _require(len(names) == len(set(names)), "Merged summaries contain duplicate datasets.")
    _require(set(names) == set(all_index), "Merged summaries do not cover RF100VL exactly.")
    summaries.sort(key=lambda item: item["dataset"])
    model_failure_types = ("timeout", "max_tokens", "invalid_response")
    merged = {
        "model_id": plan["model_id"],
        "model_revision": plan["model_revision"],
        "prompt_version": plan["prompt_version"],
        "status": "complete",
        "shard_plan_id": plan["plan_id"],
        "selected_dataset_count": len(summaries),
        "processed_dataset_count": len(summaries),
        "dataset_count": len(summaries),
        "scored_dataset_count": len(summaries),
        "expected_dataset_count": plan["expected_dataset_count"],
        "expected_image_count": plan["expected_image_count"],
        "completed_image_count": sum(
            int(item.get("completed_image_count", 0)) for item in summaries
        ),
        "model_failure_count": sum(
            int(item.get("model_failure_count", 0)) for item in summaries
        ),
        "model_failure_counts": {
            failure_type: sum(
                int(item.get("model_failure_counts", {}).get(failure_type, 0))
                for item in summaries
            )
            for failure_type in model_failure_types
        },
        "datasets": summaries,
        "macro_AP": math.fsum(item["metrics"]["AP"] for item in summaries)
        / len(summaries),
        "macro_AP50": math.fsum(item["metrics"]["AP50"] for item in summaries)
        / len(summaries),
    }
    _require(
        merged["completed_image_count"] == plan["expected_image_count"],
        "Merged completed image count is not exact.",
    )
    return merged


def _parse_uri(uri: str) -> tuple[str, str]:
    return parse_gcs_uri(uri.rstrip("/"))


def _read_json(uri: str) -> dict[str, Any]:
    bucket_name, object_name = _parse_uri(uri)
    payload = gcs_client().bucket(bucket_name).blob(object_name).download_as_bytes()
    value = json.loads(payload)
    _require(isinstance(value, dict), f"Expected JSON object at {uri}.")
    return value


def _upload_json(uri: str, value: Mapping[str, Any], *, create_only: bool = False) -> None:
    bucket_name, object_name = _parse_uri(uri)
    kwargs = {"if_generation_match": 0} if create_only else {}
    gcs_client().bucket(bucket_name).blob(object_name).upload_from_string(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        content_type="application/json",
        **kwargs,
    )


def _list_blobs(uri: str) -> list[Any]:
    bucket_name, prefix = _parse_uri(uri)
    return list(gcs_client().list_blobs(bucket_name, prefix=prefix.rstrip("/") + "/"))


def _copy_prefix(source_uri: str, destination_uri: str) -> int:
    source_bucket_name, source_prefix = _parse_uri(source_uri)
    destination_bucket_name, destination_prefix = _parse_uri(destination_uri)
    _require(source_bucket_name == destination_bucket_name, "Cross-bucket shard copy is forbidden.")
    storage = gcs_client()
    bucket = storage.bucket(source_bucket_name)
    copied = 0
    source_root = source_prefix.rstrip("/") + "/"
    destination_root = destination_prefix.rstrip("/") + "/"
    for blob in storage.list_blobs(source_bucket_name, prefix=source_root):
        suffix = blob.name[len(source_root) :]
        if not suffix:
            continue
        suffix_path = PurePosixPath(suffix)
        _require(
            not suffix_path.is_absolute() and ".." not in suffix_path.parts,
            f"Unsafe blob path: {blob.name}",
        )
        bucket.copy_blob(blob, bucket, destination_root + suffix)
        copied += 1
    return copied


def seed_existing_checkpoints(plan: Mapping[str, Any]) -> dict[str, int]:
    """Copy any partial canonical dataset checkpoints into its one assigned shard."""

    validate_shard_plan(plan)
    canonical = plan["canonical_full_uri"]
    copied: dict[str, int] = {}
    for shard in plan["shards"]:
        for dataset in shard["datasets"]:
            name = dataset["dataset"]
            count = _copy_prefix(
                f"{canonical}/{name}", f"{shard['gcs_uri']}/{name}"
            )
            if count:
                copied[name] = count
    return copied


def _record_count(dataset_uri: str) -> int:
    return sum(
        blob.name.endswith(".json") and "/records/" in blob.name
        for blob in _list_blobs(dataset_uri)
    )


def _verify_dataset_artifacts(dataset_uri: str, expected_records: int) -> int:
    blobs = _list_blobs(dataset_uri)
    _, prefix = _parse_uri(dataset_uri)
    root = prefix.rstrip("/") + "/"
    relative_names = {blob.name[len(root) :] for blob in blobs}
    required = {"summary.json", "cosmos_detection_results.json"}
    missing = required - relative_names
    _require(not missing, f"Required artifacts missing below {dataset_uri}: {sorted(missing)}")
    _require(
        any(
            len(PurePosixPath(name).parts) == 1
            and name.startswith("run_config_")
            and name.endswith(".json")
            for name in relative_names
        ),
        f"Run configuration is missing below {dataset_uri}.",
    )
    record_count = sum(
        name.endswith(".json") and "/records/" in f"/{name}"
        for name in relative_names
    )
    _require(
        record_count == expected_records,
        f"Record count for {dataset_uri} is {record_count}, expected {expected_records}.",
    )
    return record_count


def finalize_if_ready(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Idempotently publish canonical results when every shard is complete.

    Returns ``{"status": "waiting"}`` until all shard success markers exist.
    The canonical success marker is always the final write.
    """

    validate_shard_plan(plan)
    storage = gcs_client()
    shard_aggregates: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for shard in plan["shards"]:
        bucket_name, success_name = _parse_uri(f"{shard['gcs_uri']}/_SUCCESS.json")
        if not storage.bucket(bucket_name).blob(success_name).exists(client=storage):
            missing.append(shard["shard_id"])
            continue
        aggregate = _read_json(f"{shard['gcs_uri']}/aggregate_summary.json")
        verify_shard_aggregate(plan, shard["shard_id"], aggregate)
        shard_aggregates[shard["shard_id"]] = aggregate
    if missing:
        return {"status": "waiting", "missing_shards": sorted(missing)}

    baseline = _read_json(plan["baseline"]["aggregate_uri"])
    merged = merge_aggregates(plan, baseline, shard_aggregates)
    all_index = {item["dataset"]: item for item in plan["datasets"]}
    sources: dict[str, str] = {
        name: f"{plan['canonical_full_uri']}/{name}"
        for name in plan["baseline"]["datasets"]
    }
    for shard in plan["shards"]:
        for dataset in shard["datasets"]:
            sources[dataset["dataset"]] = f"{shard['gcs_uri']}/{dataset['dataset']}"

    record_counts: dict[str, int] = {}
    for name in sorted(all_index):
        count = _verify_dataset_artifacts(
            sources[name], all_index[name]["image_count"]
        )
        expected = all_index[name]["image_count"]
        record_counts[name] = count

    # Copy shard-owned dataset artifacts only after all coverage and record
    # counts pass. Baseline datasets are already canonical and never rewritten.
    copied_objects = 0
    for shard in plan["shards"]:
        for dataset in shard["datasets"]:
            name = dataset["dataset"]
            copied_objects += _copy_prefix(
                f"{shard['gcs_uri']}/{name}",
                f"{plan['canonical_full_uri']}/{name}",
            )

    aggregate_uri = f"{plan['canonical_full_uri']}/aggregate_summary.json"
    _upload_json(aggregate_uri, merged)
    verification = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "plan_id": plan["plan_id"],
        "verified_at": utc_now(),
        "dataset_count": len(record_counts),
        "image_record_count": sum(record_counts.values()),
        "record_counts": record_counts,
        "copied_objects": copied_objects,
        "aggregate_uri": aggregate_uri,
    }
    verification_uri = (
        f"{plan['gcs_run_uri']}/control/shards/{plan['plan_id']}/"
        "final_verification.json"
    )
    _upload_json(verification_uri, verification)
    success = {
        "schema_version": SCHEMA_VERSION,
        "model_id": plan["model_id"],
        "model_revision": plan["model_revision"],
        "prompt_version": plan["prompt_version"],
        "shard_plan_id": plan["plan_id"],
        "scored_dataset_count": merged["scored_dataset_count"],
        "expected_dataset_count": plan["expected_dataset_count"],
        "completed_image_count": merged["completed_image_count"],
        "expected_image_count": plan["expected_image_count"],
        "model_failure_count": merged["model_failure_count"],
        "model_failure_counts": merged["model_failure_counts"],
        "macro_AP": merged["macro_AP"],
        "macro_AP50": merged["macro_AP50"],
        "verification_uri": verification_uri,
    }
    _upload_json(f"{plan['canonical_full_uri']}/_SUCCESS.json", success)
    return verification


def load_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "Shard plan must be a JSON object.")
    validate_shard_plan(value)
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-plan")
    create.add_argument("--preflight", type=Path, required=True)
    create.add_argument("--baseline-aggregate", type=Path, required=True)
    create.add_argument("--gcs-run-uri", required=True)
    create.add_argument("--shards", type=int, required=True)
    create.add_argument("--image-ref", required=True)
    create.add_argument("--benchmark-git-sha", required=True)
    create.add_argument("--model-revision", required=True)
    create.add_argument("--plan-id")
    create.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate-plan")
    validate.add_argument("--plan", type=Path, required=True)
    seed = subparsers.add_parser("seed-checkpoints")
    seed.add_argument("--plan", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--plan", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "create-plan":
        preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
        baseline = json.loads(args.baseline_aggregate.read_text(encoding="utf-8"))
        plan = build_shard_plan(
            preflight=preflight,
            baseline_aggregate=baseline,
            gcs_run_uri=args.gcs_run_uri,
            shard_count=args.shards,
            image_ref=args.image_ref,
            benchmark_git_sha=args.benchmark_git_sha,
            model_revision=args.model_revision,
            plan_id=args.plan_id,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.replace(args.output)
        print(json.dumps({
            "status": "created",
            "plan_id": plan["plan_id"],
            "baseline_dataset_count": plan["baseline"]["dataset_count"],
            "remaining_dataset_count": sum(
                shard["dataset_count"] for shard in plan["shards"]
            ),
            "shard_image_counts": [shard["image_count"] for shard in plan["shards"]],
            "output": str(args.output),
        }, indent=2))
    elif args.command == "validate-plan":
        plan = load_plan(args.plan)
        print(json.dumps({"status": "valid", "plan_id": plan["plan_id"]}))
    elif args.command == "seed-checkpoints":
        plan = load_plan(args.plan)
        print(json.dumps({"status": "seeded", "objects": seed_existing_checkpoints(plan)}, indent=2))
    else:
        plan = load_plan(args.plan)
        print(json.dumps(finalize_if_ready(plan), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
