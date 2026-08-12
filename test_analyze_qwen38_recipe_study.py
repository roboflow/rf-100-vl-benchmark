import json
from pathlib import Path
from types import SimpleNamespace

import analyze_qwen38_recipe_study as study


def row(
    mode: str,
    effort: str,
    score: float,
    *,
    formulation: str = "multi",
    semantics: str = "class_names",
    representation: str = "none",
    boxes: int = 0,
) -> dict:
    return {
        "mode": mode,
        "formulation": formulation,
        "semantics": semantics,
        "representation": representation,
        "boxes_per_class": boxes,
        "reasoning_effort": effort,
        "seed": 1234,
        "complete": True,
        "mAP50_95": score,
        "mAP50": score + 10,
    }


def write_summary(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"rows": rows}))
    return path


def write_noise(path: Path, dreidel: float = 1.0, orion: float = 1.0) -> Path:
    path.write_text(
        json.dumps(
            {
                "datasets": {
                    "dreidel": {
                        "metrics": {
                            "AP": {"tie_threshold": dreidel},
                            "AP50": {"tie_threshold": dreidel + 1},
                        }
                    },
                    "orion": {
                        "metrics": {
                            "AP": {"tie_threshold": orion},
                            "AP50": {"tie_threshold": orion + 1},
                        }
                    },
                }
            }
        )
    )
    return path


def gate_rows(top_low: float, fast_low: float) -> list[dict]:
    return [
        row(
            "reasoning_top_none",
            "none",
            20.0,
            formulation="single",
            semantics="anonymous_explicit",
            representation="numeric",
            boxes=2,
        ),
        row(
            "reasoning_top_low",
            "low",
            top_low,
            formulation="single",
            semantics="anonymous_explicit",
            representation="numeric",
            boxes=2,
        ),
        row("reasoning_fast_none", "none", 15.0),
        row("reasoning_fast_low", "low", fast_low),
    ]


def test_prepare_medium_requires_low_to_clear_both_dataset_floors(tmp_path):
    args = SimpleNamespace(
        dreidel_summary=write_summary(tmp_path / "d.json", gate_rows(22.2, 15.5)),
        orion_summary=write_summary(tmp_path / "o.json", gate_rows(21.5, 17.0)),
        noise_floor=write_noise(tmp_path / "noise.json"),
        output_dir=tmp_path / "out",
    )
    assert study.prepare_medium(args) == 0
    decision = json.loads((args.output_dir / "reasoning_low_decision.json").read_text())
    conditions = json.loads(
        (args.output_dir / "reasoning_medium_conditions.json").read_text()
    )["conditions"]
    assert decision["decisions"]["top"]["low_passed_on_both_datasets"]
    assert not decision["decisions"]["fast"]["low_passed_on_both_datasets"]
    assert [value["mode"] for value in conditions] == ["reasoning_top_medium"]
    assert conditions[0]["reasoning_effort"] == "medium"


def test_finalize_reasoning_prefers_cheaper_effort_within_noise(tmp_path):
    dreidel = write_summary(tmp_path / "d.json", gate_rows(22.2, 15.5))
    orion = write_summary(tmp_path / "o.json", gate_rows(21.5, 17.0))
    noise = write_noise(tmp_path / "noise.json")
    output = tmp_path / "out"
    prepare_args = SimpleNamespace(
        dreidel_summary=dreidel,
        orion_summary=orion,
        noise_floor=noise,
        output_dir=output,
    )
    study.prepare_medium(prepare_args)
    dreidel_medium = write_summary(
        tmp_path / "dm.json",
        [
            row(
                "reasoning_top_medium",
                "medium",
                22.6,
                formulation="single",
                semantics="anonymous_explicit",
                representation="numeric",
                boxes=2,
            )
        ],
    )
    orion_medium = write_summary(
        tmp_path / "om.json",
        [
            row(
                "reasoning_top_medium",
                "medium",
                22.0,
                formulation="single",
                semantics="anonymous_explicit",
                representation="numeric",
                boxes=2,
            )
        ],
    )
    finalists = tmp_path / "finalists.json"
    finalists.write_text(
        json.dumps(
            {
                "conditions": [
                    {
                        "mode": "final_top",
                        "formulation": "single",
                        "semantics": "anonymous_explicit",
                        "representation": "numeric",
                        "box_count": 2,
                        "reasoning_effort": "none",
                        "seed": 1234,
                    },
                    {
                        "mode": "final_fast",
                        "formulation": "multi",
                        "semantics": "class_names",
                        "representation": "none",
                        "box_count": 0,
                        "reasoning_effort": "none",
                        "seed": 1234,
                    },
                ]
            }
        )
    )
    args = SimpleNamespace(
        dreidel_summary=dreidel,
        orion_summary=orion,
        dreidel_medium_summary=dreidel_medium,
        orion_medium_summary=orion_medium,
        noise_floor=noise,
        low_decision=output / "reasoning_low_decision.json",
        base_finalists=finalists,
        output_dir=output,
    )
    assert study.finalize_reasoning(args) == 0
    resolved = json.loads(
        (output / "finalist_conditions_resolved.json").read_text()
    )["conditions"]
    assert resolved[0]["reasoning_effort"] == "low"
    assert resolved[1]["reasoning_effort"] == "none"


def test_final_selection_uses_noise_then_efficiency():
    expensive = {
        "mode": "expensive",
        "calls_per_image": 6,
        "total_tokens_per_image": 1000,
        "effective_serial_seconds_per_image": 30,
        "dreidel_mAP50_95": 30.0,
        "orion_mAP50_95": 20.0,
        "macro_mAP50_95": 25.0,
    }
    cheap_tie = {
        "mode": "cheap",
        "calls_per_image": 1,
        "total_tokens_per_image": 100,
        "effective_serial_seconds_per_image": 5,
        "dreidel_mAP50_95": 29.5,
        "orion_mAP50_95": 19.4,
        "macro_mAP50_95": 24.45,
    }
    outside = {
        "mode": "outside",
        "calls_per_image": 1,
        "total_tokens_per_image": 50,
        "effective_serial_seconds_per_image": 4,
        "dreidel_mAP50_95": 28.8,
        "orion_mAP50_95": 18.8,
        "macro_mAP50_95": 23.8,
    }
    thresholds = {
        "dreidel": {"mAP50_95": 1.0},
        "orion": {"mAP50_95": 1.0},
    }
    accuracy, throughput = study.choose_finalists(
        [expensive, cheap_tie, outside], thresholds
    )
    assert accuracy["mode"] == "cheap"
    assert throughput["mode"] == "outside"


def test_launcher_contains_two_dataset_adaptive_reasoning_and_final_report():
    launcher = Path("run_qwen38_recipe_study.sh").read_text()
    noise_conditions = json.loads(
        Path("qwen38-fsod-configs/noise-floor-multi-names.json").read_text()
    )["conditions"]
    assert len(noise_conditions) == 10
    assert "orion-reasoning-gate" in launcher
    assert "reasoning_medium_conditions.json" in launcher
    assert "finalist_conditions_resolved.json" in launcher
    assert "analyze_qwen38_recipe_study.py final-report" in launcher


def complete_finalist_row(mode: str, score: float, calls: int) -> dict:
    return {
        "mode": mode,
        "formulation": "multi" if calls == 1 else "single",
        "semantics": "class_names",
        "representation": "none",
        "boxes_per_class": 0,
        "reasoning_effort": "none",
        "seed": 1234,
        "calls_per_image": calls,
        "complete": True,
        "mAP50_95": score,
        "mAP50": score + 10,
        "prompt_tokens": 100 * calls,
        "completion_tokens": 10 * calls,
        "reasoning_tokens": 0,
        "total_inference_seconds": 5 * calls,
        "model_failures": 0,
        "errors": 0,
    }


def write_finalist_run(path: Path, scores: dict[str, tuple[float, int]]) -> Path:
    path.mkdir()
    rows = [
        complete_finalist_row(mode, score, calls)
        for mode, (score, calls) in scores.items()
    ]
    write_summary(path / "comparison_summary.json", rows)
    (path / "aggregate_metrics.json").write_text(json.dumps({"image_count": 1}))
    predictions = path / "predictions"
    predictions.mkdir()
    for mode in scores:
        (predictions / f"{mode}.json").write_text("[]")
    return path


def test_final_report_writes_machine_and_human_readable_completion(tmp_path):
    dreidel = write_finalist_run(
        tmp_path / "dreidel", {"accurate": (30.0, 6), "fast": (25.0, 1)}
    )
    orion = write_finalist_run(
        tmp_path / "orion", {"accurate": (28.0, 6), "fast": (24.0, 1)}
    )
    noise = write_noise(tmp_path / "noise.json", dreidel=0.1, orion=0.1)
    output = tmp_path / "final"
    args = SimpleNamespace(
        dreidel_run=dreidel,
        orion_run=orion,
        dreidel_annotations=tmp_path / "unused-dreidel.json",
        orion_annotations=tmp_path / "unused-orion.json",
        noise_floor=noise,
        output_dir=output,
        bootstrap_iterations=20,
        bootstrap_seed=1,
    )
    assert study.final_report(args) == 0
    report = json.loads((output / "final_report.json").read_text())
    assert report["selection"] == {
        "accuracy_first": "accurate",
        "throughput_first": "fast",
    }
    assert (output / "final_ranking.csv").is_file()
    assert (output / "final_report.md").is_file()
    assert json.loads((output / "_SUCCESS.json").read_text())["candidate_count"] == 2
