#!/usr/bin/env python3
"""Resolve Qwen reasoning gates and produce the final two-dataset recipe report."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

import compare_qwen38_predictions as comparison
import evaluate_qwen38_orion as base
from evaluate_qwen38_recipe import Condition, load_conditions


DATASETS = ("dreidel", "orion")
ARMS = ("top", "fast")
EFFORT_ORDER = {"none": 0, "low": 1, "medium": 2}


def read_rows(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text())
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Missing rows in {path}.")
    result = {str(row["mode"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate modes in {path}.")
    if not all(row.get("complete") for row in rows):
        raise ValueError(f"Incomplete condition in {path}.")
    return result


def condition_signature(value: Condition | dict[str, Any]) -> tuple[Any, ...]:
    if isinstance(value, Condition):
        return (
            value.formulation,
            value.semantics,
            value.representation,
            value.box_count,
        )
    return (
        str(value["formulation"]),
        str(value["semantics"]),
        str(value["representation"]),
        int(value.get("box_count", value.get("boxes_per_class", 0))),
    )


def noise_thresholds(path: Path) -> dict[str, dict[str, float]]:
    value = json.loads(path.read_text())
    result: dict[str, dict[str, float]] = {}
    for dataset in DATASETS:
        metrics = value["datasets"][dataset]["metrics"]
        result[dataset] = {
            "mAP50_95": float(metrics["AP"]["tie_threshold"]),
            "mAP50": float(metrics["AP50"]["tie_threshold"]),
        }
    return result


def reasoning_differences(
    rows_by_dataset: dict[str, dict[str, dict[str, Any]]],
    arm: str,
    candidate_effort: str,
) -> dict[str, dict[str, float]]:
    result = {}
    for dataset in DATASETS:
        rows = rows_by_dataset[dataset]
        baseline = rows[f"reasoning_{arm}_none"]
        candidate = rows[f"reasoning_{arm}_{candidate_effort}"]
        if condition_signature(baseline) != condition_signature(candidate):
            raise ValueError(f"Reasoning comparison changed recipe structure for {arm}.")
        result[dataset] = {
            "mAP50_95": float(candidate["mAP50_95"]) - float(baseline["mAP50_95"]),
            "mAP50": float(candidate["mAP50"]) - float(baseline["mAP50"]),
        }
    return result


def passes_reasoning_gate(
    differences: dict[str, dict[str, float]],
    thresholds: dict[str, dict[str, float]],
) -> bool:
    return all(
        differences[dataset]["mAP50_95"]
        > max(1.0, thresholds[dataset]["mAP50_95"])
        for dataset in DATASETS
    )


def prepare_medium(args: argparse.Namespace) -> int:
    rows_by_dataset = {
        "dreidel": read_rows(args.dreidel_summary),
        "orion": read_rows(args.orion_summary),
    }
    thresholds = noise_thresholds(args.noise_floor)
    decisions: dict[str, Any] = {}
    medium: list[Condition] = []
    seen_signatures: set[tuple[Any, ...]] = set()
    for arm in ARMS:
        for dataset in DATASETS:
            for effort in ("none", "low"):
                mode = f"reasoning_{arm}_{effort}"
                if mode not in rows_by_dataset[dataset]:
                    raise ValueError(f"Missing {mode} in {dataset} reasoning gate.")
        differences = reasoning_differences(rows_by_dataset, arm, "low")
        passed = passes_reasoning_gate(differences, thresholds)
        source = rows_by_dataset["dreidel"][f"reasoning_{arm}_low"]
        signature = condition_signature(source)
        decisions[arm] = {
            "signature": list(signature),
            "low_minus_none": differences,
            "thresholds": {
                dataset: max(1.0, thresholds[dataset]["mAP50_95"])
                for dataset in DATASETS
            },
            "low_passed_on_both_datasets": passed,
        }
        if passed and signature not in seen_signatures:
            medium.append(
                Condition(
                    mode=f"reasoning_{arm}_medium",
                    formulation=signature[0],
                    semantics=signature[1],
                    representation=signature[2],
                    box_count=int(signature[3]),
                    reasoning_effort="medium",
                    seed=int(source["seed"]),
                )
            )
            seen_signatures.add(signature)
    result = {
        "created_at": base.utc_now(),
        "rule": (
            "Low reasoning must improve mAP50-95 by more than max(1 mAP, "
            "the dataset residual-noise floor) on both datasets. Medium is "
            "evaluated only for an arm that passes."
        ),
        "decisions": decisions,
        "medium_condition_count": len(medium),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.atomic_write_json(args.output_dir / "reasoning_low_decision.json", result)
    base.atomic_write_json(
        args.output_dir / "reasoning_medium_conditions.json",
        {"conditions": [asdict(value) for value in medium]},
    )
    print(json.dumps(result, indent=2))
    return 0


def choose_reasoning_effort(
    arm: str,
    gate_rows: dict[str, dict[str, dict[str, Any]]],
    medium_rows: dict[str, dict[str, dict[str, Any]]] | None,
    thresholds: dict[str, dict[str, float]],
    low_passed: bool,
) -> tuple[str, dict[str, Any]]:
    candidates = ["none"]
    details: dict[str, Any] = {}
    if low_passed:
        candidates.append("low")
        if medium_rows is not None:
            medium_mode = f"reasoning_{arm}_medium"
            if all(medium_mode in medium_rows[dataset] for dataset in DATASETS):
                candidates.append("medium")
    scores = {}
    for effort in candidates:
        mode = f"reasoning_{arm}_{effort}"
        source = gate_rows if effort != "medium" else medium_rows
        assert source is not None
        scores[effort] = {
            dataset: {
                "mAP50_95": float(source[dataset][mode]["mAP50_95"]),
                "mAP50": float(source[dataset][mode]["mAP50"]),
            }
            for dataset in DATASETS
        }
        scores[effort]["macro_mAP50_95"] = fmean(
            scores[effort][dataset]["mAP50_95"] for dataset in DATASETS
        )
    eligible = ["none"]
    for effort in candidates:
        if effort == "none":
            continue
        improvements = {
            dataset: scores[effort][dataset]["mAP50_95"]
            - scores["none"][dataset]["mAP50_95"]
            for dataset in DATASETS
        }
        if all(
            improvements[dataset] > max(1.0, thresholds[dataset]["mAP50_95"])
            for dataset in DATASETS
        ):
            eligible.append(effort)
        details[effort] = {"improvement_over_none": improvements}
    best = max(eligible, key=lambda effort: scores[effort]["macro_mAP50_95"])
    tied = [
        effort
        for effort in eligible
        if all(
            abs(
                scores[effort][dataset]["mAP50_95"]
                - scores[best][dataset]["mAP50_95"]
            )
            <= thresholds[dataset]["mAP50_95"]
            for dataset in DATASETS
        )
    ]
    selected = min(tied, key=lambda effort: EFFORT_ORDER[effort])
    return selected, {
        "available_efforts": candidates,
        "eligible_efforts": eligible,
        "scores": scores,
        "comparisons": details,
        "selected_effort": selected,
    }


def finalize_reasoning(args: argparse.Namespace) -> int:
    gate_rows = {
        "dreidel": read_rows(args.dreidel_summary),
        "orion": read_rows(args.orion_summary),
    }
    medium_rows = None
    if args.dreidel_medium_summary and args.orion_medium_summary:
        medium_rows = {
            "dreidel": read_rows(args.dreidel_medium_summary),
            "orion": read_rows(args.orion_medium_summary),
        }
    thresholds = noise_thresholds(args.noise_floor)
    low = json.loads(args.low_decision.read_text())
    decisions = {}
    effort_by_signature: dict[tuple[Any, ...], str] = {}
    for arm in ARMS:
        selected, detail = choose_reasoning_effort(
            arm,
            gate_rows,
            medium_rows,
            thresholds,
            bool(low["decisions"][arm]["low_passed_on_both_datasets"]),
        )
        signature = tuple(low["decisions"][arm]["signature"])
        existing = effort_by_signature.get(signature)
        if existing is None or EFFORT_ORDER[selected] < EFFORT_ORDER[existing]:
            effort_by_signature[signature] = selected
        decisions[arm] = {**detail, "signature": list(signature)}

    finalists = load_conditions(args.base_finalists)
    resolved = []
    applied = []
    for condition in finalists:
        effort = effort_by_signature.get(condition_signature(condition), condition.reasoning_effort)
        resolved_condition = Condition(
            mode=condition.mode,
            formulation=condition.formulation,
            semantics=condition.semantics,
            representation=condition.representation,
            box_count=condition.box_count,
            reasoning_effort=effort,
            seed=condition.seed,
        )
        resolved.append(resolved_condition)
        if effort != condition.reasoning_effort:
            applied.append({"mode": condition.mode, "reasoning_effort": effort})
    result = {
        "created_at": base.utc_now(),
        "decisions": decisions,
        "applied_to_finalists": applied,
        "resolved_finalist_count": len(resolved),
    }
    base.atomic_write_json(args.output_dir / "reasoning_decision.json", result)
    base.atomic_write_json(
        args.output_dir / "finalist_conditions_resolved.json",
        {"conditions": [asdict(value) for value in resolved]},
    )
    print(json.dumps(result, indent=2))
    return 0


def candidate_rows(
    dreidel_run: Path,
    orion_run: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    rows_by_dataset = {
        "dreidel": read_rows(dreidel_run / "comparison_summary.json"),
        "orion": read_rows(orion_run / "comparison_summary.json"),
    }
    modes = set(rows_by_dataset["dreidel"])
    if modes != set(rows_by_dataset["orion"]):
        raise ValueError("Dreidel and Orion finalist modes differ.")
    rows = []
    for mode in sorted(modes):
        dreidel = rows_by_dataset["dreidel"][mode]
        orion = rows_by_dataset["orion"][mode]
        structural_fields = (
            "formulation",
            "semantics",
            "representation",
            "boxes_per_class",
            "reasoning_effort",
            "seed",
            "calls_per_image",
        )
        if any(dreidel[field] != orion[field] for field in structural_fields):
            raise ValueError(f"Finalist structure differs across datasets for {mode}.")
        image_counts = {
            "dreidel": json.loads(
                (dreidel_run / "aggregate_metrics.json").read_text()
            )["image_count"],
            "orion": json.loads(
                (orion_run / "aggregate_metrics.json").read_text()
            )["image_count"],
        }
        prompt_per_image = fmean(
            float(rows_by_dataset[dataset][mode].get("prompt_tokens") or 0)
            / image_counts[dataset]
            for dataset in DATASETS
        )
        completion_per_image = fmean(
            float(rows_by_dataset[dataset][mode].get("completion_tokens") or 0)
            / image_counts[dataset]
            for dataset in DATASETS
        )
        reasoning_per_image = fmean(
            float(rows_by_dataset[dataset][mode].get("reasoning_tokens") or 0)
            / image_counts[dataset]
            for dataset in DATASETS
        )
        serial_seconds = fmean(
            float(rows_by_dataset[dataset][mode].get("total_inference_seconds") or 0)
            / image_counts[dataset]
            for dataset in DATASETS
        )
        rows.append(
            {
                "mode": mode,
                **{field: dreidel[field] for field in structural_fields},
                "dreidel_mAP50_95": float(dreidel["mAP50_95"]),
                "dreidel_mAP50": float(dreidel["mAP50"]),
                "orion_mAP50_95": float(orion["mAP50_95"]),
                "orion_mAP50": float(orion["mAP50"]),
                "macro_mAP50_95": fmean(
                    (float(dreidel["mAP50_95"]), float(orion["mAP50_95"]))
                ),
                "macro_mAP50": fmean((float(dreidel["mAP50"]), float(orion["mAP50"]))),
                "prompt_tokens_per_image": prompt_per_image,
                "completion_tokens_per_image": completion_per_image,
                "reasoning_tokens_per_image": reasoning_per_image,
                "total_tokens_per_image": prompt_per_image + completion_per_image,
                "effective_serial_seconds_per_image": serial_seconds,
                "model_failures": int(dreidel["model_failures"]) + int(orion["model_failures"]),
                "errors": int(dreidel["errors"]) + int(orion["errors"]),
            }
        )
    return rows, rows_by_dataset


def efficiency_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["calls_per_image"]),
        float(row["total_tokens_per_image"]),
        float(row["effective_serial_seconds_per_image"]),
        -float(row["macro_mAP50_95"]),
    )


def tied_on_both(
    left: dict[str, Any],
    right: dict[str, Any],
    thresholds: dict[str, dict[str, float]],
) -> bool:
    return all(
        abs(float(left[f"{dataset}_mAP50_95"]) - float(right[f"{dataset}_mAP50_95"]))
        <= thresholds[dataset]["mAP50_95"]
        for dataset in DATASETS
    )


def choose_finalists(
    rows: list[dict[str, Any]],
    thresholds: dict[str, dict[str, float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_best = max(rows, key=lambda row: float(row["macro_mAP50_95"]))
    tied = [row for row in rows if tied_on_both(row, raw_best, thresholds)]
    accuracy = min(tied, key=efficiency_key)
    minimum_calls = min(int(row["calls_per_image"]) for row in rows)
    fast_rows = [row for row in rows if int(row["calls_per_image"]) == minimum_calls]
    raw_fast = max(fast_rows, key=lambda row: float(row["macro_mAP50_95"]))
    fast_tied = [row for row in fast_rows if tied_on_both(row, raw_fast, thresholds)]
    throughput = min(fast_tied, key=efficiency_key)
    return accuracy, throughput


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Qwen3.8-Max final two-dataset recipe report",
        "",
        f"Generated: {result['created_at']}",
        "",
        f"Accuracy-first: `{result['selection']['accuracy_first']}`",
        "",
        f"Throughput-first: `{result['selection']['throughput_first']}`",
        "",
        (
            "| Mode | Recipe | Calls/image | Dreidel mAP | Orion mAP | Macro mAP | "
            "Macro mAP50 | Seconds/image | Tokens/image | Failures |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["ranking"]:
        recipe = "/".join(
            (
                str(row["formulation"]),
                str(row["semantics"]),
                str(row["representation"]),
                f"b{int(row['boxes_per_class'])}",
                str(row["reasoning_effort"]),
            )
        )
        lines.append(
            "| {mode} | {recipe} | {calls_per_image} | {dreidel_mAP50_95:.2f} | "
            "{orion_mAP50_95:.2f} | {macro_mAP50_95:.2f} | {macro_mAP50:.2f} | "
            "{effective_serial_seconds_per_image:.2f} | {total_tokens_per_image:.0f} | "
            "{failures} |".format(
                **row,
                recipe=recipe,
                failures=int(row["model_failures"]) + int(row["errors"]),
            )
        )
    lines.extend(
        [
            "",
            (
                "Candidates within the residual inference-noise floor of the selected "
                "accuracy recipe receive paired-image bootstrap intervals in "
                "`final_report.json`."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def final_report(args: argparse.Namespace) -> int:
    rows, _ = candidate_rows(args.dreidel_run, args.orion_run)
    thresholds = noise_thresholds(args.noise_floor)
    accuracy, throughput = choose_finalists(rows, thresholds)
    ranking = sorted(rows, key=lambda row: float(row["macro_mAP50_95"]), reverse=True)
    macro_noise_floor = fmean(
        thresholds[dataset]["mAP50_95"] for dataset in DATASETS
    )
    close = [
        row
        for row in rows
        if row["mode"] != accuracy["mode"]
        and abs(float(row["macro_mAP50_95"]) - float(accuracy["macro_mAP50_95"]))
        <= macro_noise_floor
    ]
    bootstrap: dict[str, Any] = {}
    for candidate in close:
        comparison_key = f"{accuracy['mode']}__vs__{candidate['mode']}"
        bootstrap[comparison_key] = {}
        for dataset, run, annotations in (
            ("dreidel", args.dreidel_run, args.dreidel_annotations),
            ("orion", args.orion_run, args.orion_annotations),
        ):
            baseline = json.loads(
                (run / "predictions" / f"{accuracy['mode']}.json").read_text()
            )
            challenger = json.loads(
                (run / "predictions" / f"{candidate['mode']}.json").read_text()
            )
            bootstrap[comparison_key][dataset] = comparison.paired_bootstrap(
                annotations,
                baseline,
                challenger,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            )
    result = {
        "created_at": base.utc_now(),
        "selection_rule": (
            "Rank by equal-weight Dreidel/Orion macro mAP50-95. If recipes are "
            "within each dataset's residual inference-noise floor, prefer fewer "
            "calls, then fewer tokens, then lower effective serial latency."
        ),
        "noise_thresholds": thresholds,
        "conservative_macro_noise_floor_mAP50_95": macro_noise_floor,
        "selection": {
            "accuracy_first": accuracy["mode"],
            "throughput_first": throughput["mode"],
        },
        "ranking": ranking,
        "paired_image_bootstrap_for_noise_ties": bootstrap,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.atomic_write_json(args.output_dir / "final_report.json", result)
    with (args.output_dir / "final_ranking.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)
    write_markdown(args.output_dir / "final_report.md", result)
    base.atomic_write_json(
        args.output_dir / "_SUCCESS.json",
        {
            "completed_at": base.utc_now(),
            "accuracy_first": accuracy["mode"],
            "throughput_first": throughput["mode"],
            "candidate_count": len(rows),
            "bootstrap_comparison_count": len(close),
        },
    )
    print(json.dumps(result, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    medium = commands.add_parser("prepare-medium")
    medium.add_argument("--dreidel-summary", type=Path, required=True)
    medium.add_argument("--orion-summary", type=Path, required=True)
    medium.add_argument("--noise-floor", type=Path, required=True)
    medium.add_argument("--output-dir", type=Path, required=True)

    resolve = commands.add_parser("finalize-reasoning")
    resolve.add_argument("--dreidel-summary", type=Path, required=True)
    resolve.add_argument("--orion-summary", type=Path, required=True)
    resolve.add_argument("--dreidel-medium-summary", type=Path)
    resolve.add_argument("--orion-medium-summary", type=Path)
    resolve.add_argument("--noise-floor", type=Path, required=True)
    resolve.add_argument("--low-decision", type=Path, required=True)
    resolve.add_argument("--base-finalists", type=Path, required=True)
    resolve.add_argument("--output-dir", type=Path, required=True)

    report = commands.add_parser("final-report")
    report.add_argument("--dreidel-run", type=Path, required=True)
    report.add_argument("--orion-run", type=Path, required=True)
    report.add_argument("--dreidel-annotations", type=Path, required=True)
    report.add_argument("--orion-annotations", type=Path, required=True)
    report.add_argument("--noise-floor", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    report.add_argument("--bootstrap-iterations", type=int, default=500)
    report.add_argument("--bootstrap-seed", type=int, default=20260812)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare-medium":
        return prepare_medium(args)
    if args.command == "finalize-reasoning":
        return finalize_reasoning(args)
    return final_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
