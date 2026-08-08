#!/usr/bin/env python3
"""Run the staged Cosmos3-Edge RF100VL benchmark inside one RunPod pod."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "infra"))

from evaluate_cosmos import (  # noqa: E402
    GCSArtifactStore,
    MODEL_ID,
    PROMPT_VERSION,
    atomic_write_json,
    discover_datasets,
    parse_gcs_uri,
    resolve_annotation_file,
    sha256_file,
)
from gcs_io import download, download_prefix, exists  # noqa: E402


PINNED_MODEL_REVISION = "2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2"
BASE_URL = "http://127.0.0.1:8000/v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_truthy(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class JobContract:
    stage: str
    gcs_run_uri: str
    work_dir: Path
    requested_dataset_dir: Path
    model_id: str
    model_revision: str
    expected_datasets: int
    workers: int
    smoke_dataset: str | None
    dataset_gcs_uri: str | None
    preflight_approved: bool
    image_ref: str
    benchmark_git_sha: str

    @classmethod
    def from_environment(cls) -> "JobContract":
        stage = os.getenv("COSMOS_STAGE", "").strip()
        if stage not in {"preflight", "full"}:
            raise ValueError("COSMOS_STAGE must be exactly 'preflight' or 'full'.")
        gcs_run_uri = os.getenv("COSMOS_GCS_RUN_URI", "").rstrip("/")
        parse_gcs_uri(gcs_run_uri)
        model_revision = os.getenv(
            "COSMOS_MODEL_REVISION", PINNED_MODEL_REVISION
        ).strip()
        if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
            raise ValueError("COSMOS_MODEL_REVISION must be a full 40-character commit SHA.")
        expected_datasets = int(os.getenv("COSMOS_EXPECTED_DATASETS", "100"))
        workers = int(os.getenv("COSMOS_WORKERS", "1"))
        if expected_datasets != 100:
            raise ValueError("The canonical RF100VL contract requires exactly 100 datasets.")
        if workers != 1:
            raise ValueError("The canonical Cosmos benchmark requires COSMOS_WORKERS=1.")
        preflight_approved = env_truthy("COSMOS_PREFLIGHT_APPROVED")
        if stage == "full" and not preflight_approved:
            raise ValueError(
                "The full stage requires COSMOS_PREFLIGHT_APPROVED=1 after human visual review."
            )
        dataset_gcs_uri = os.getenv("RF100VL_GCS_URI") or None
        if dataset_gcs_uri:
            parse_gcs_uri(dataset_gcs_uri)
        return cls(
            stage=stage,
            gcs_run_uri=gcs_run_uri,
            work_dir=Path(os.getenv("COSMOS_WORK_DIR", "/workspace/cosmos-runpod-work")),
            requested_dataset_dir=Path(os.getenv("RF100VL_DIR", "/workspace/rf100-vl")),
            model_id=os.getenv("COSMOS_MODEL_ID", MODEL_ID),
            model_revision=model_revision,
            expected_datasets=expected_datasets,
            workers=workers,
            smoke_dataset=os.getenv("COSMOS_SMOKE_DATASET") or None,
            dataset_gcs_uri=dataset_gcs_uri,
            preflight_approved=preflight_approved,
            image_ref=os.getenv("COSMOS_IMAGE_REF", "unknown"),
            benchmark_git_sha=os.getenv("BENCHMARK_GIT_SHA", "unknown"),
        )

    @property
    def gcs_preflight_storage_uri(self) -> str:
        return f"{self.gcs_run_uri}/preflight/storage"

    @property
    def gcs_smoke_uri(self) -> str:
        return f"{self.gcs_run_uri}/preflight/live-smoke"

    @property
    def gcs_full_uri(self) -> str:
        return f"{self.gcs_run_uri}/full"


def run_command(command: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    print("[job] running:", " ".join(command), flush=True)
    subprocess.run(list(command), cwd=ROOT, env=env, check=True)


def output_or_unknown(command: Sequence[str]) -> str:
    try:
        return subprocess.run(
            list(command), check=True, text=True, capture_output=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def vllm_command(contract: JobContract) -> list[str]:
    return [
        "/usr/local/bin/vllm",
        "serve",
        contract.model_id,
        "--revision",
        contract.model_revision,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "auto",
        "--seed",
        "0",
        "--max-model-len",
        "131072",
        "--allowed-local-media-path",
        "/",
        "--mm-processor-kwargs",
        '{"do_resize":true,"min_pixels":4096,"max_pixels":16777216}',
        "--media-io-kwargs",
        '{"video":{"num_frames":256}}',
    ]


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def write_manifest(contract: JobContract, command: Sequence[str]) -> Path:
    manifest_path = contract.work_dir / "job_manifest.json"
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "contract": {
            **asdict(contract),
            "work_dir": str(contract.work_dir),
            "requested_dataset_dir": str(contract.requested_dataset_dir),
        },
        "prompt_version": PROMPT_VERSION,
        "vllm_command": list(command),
        "precision": "bfloat16",
        "quantization": None,
        "kv_cache_dtype": "auto",
        "python": sys.version,
        "packages": {
            name: package_version(name)
            for name in ("openai", "Pillow", "pycocotools", "google-cloud-storage", "rf100vl")
        },
        "vllm_runtime": {
            "vllm": output_or_unknown(
                [
                    "/usr/bin/python3",
                    "-c",
                    "import importlib.metadata as m; print(m.version('vllm'))",
                ]
            ),
            "opencv_python_headless": output_or_unknown(
                [
                    "/usr/bin/python3",
                    "-c",
                    "import importlib.metadata as m; print(m.version('opencv-python-headless'))",
                ]
            ),
        },
        "gpu": output_or_unknown(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def find_complete_dataset_root(root: Path, expected: int) -> Path | None:
    candidates = [root, root / "rf100vl", root / "rf100-vl", root / "RF100VL"]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        try:
            datasets = discover_datasets(candidate, None)
        except Exception:
            continue
        if len(datasets) == expected:
            return candidate
    return None


def ensure_dataset(contract: JobContract) -> Path:
    existing = find_complete_dataset_root(
        contract.requested_dataset_dir, contract.expected_datasets
    )
    if existing:
        print(f"[data] found all {contract.expected_datasets} datasets at {existing}")
        return existing

    root = contract.requested_dataset_dir
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        print("[data] partial RF100VL download found; resuming with overwrite enabled")
    if contract.dataset_gcs_uri:
        print(f"[data] restoring RF100VL from {contract.dataset_gcs_uri}")
        restored = download_prefix(contract.dataset_gcs_uri, root)
        print(f"[data] restored {restored} objects")
    else:
        if not os.getenv("ROBOFLOW_API_KEY"):
            raise RuntimeError(
                "RF100VL is absent and neither RF100VL_GCS_URI nor the "
                "ROBOFLOW_API_KEY RunPod secret is available."
            )
        run_command(
            [
                sys.executable,
                "infra/download_rf100vl.py",
                "--output-dir",
                str(root),
            ]
        )

    resolved = find_complete_dataset_root(root, contract.expected_datasets)
    if not resolved:
        count = len(discover_datasets(root, None)) if root.is_dir() else 0
        raise RuntimeError(
            f"Expected {contract.expected_datasets} RF100VL test datasets after download; found {count}."
        )
    return resolved


def wait_for_server(
    process: subprocess.Popen[Any], expected_model_id: str, timeout_seconds: int = 1800
) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"{BASE_URL}/models"
    last_report = 0.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with code {process.returncode}.")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            ids = {item.get("id") for item in payload.get("data", [])}
            if expected_model_id in ids:
                print(f"[vllm] endpoint ready and advertising {expected_model_id}")
                return
        except Exception:
            pass
        now = time.monotonic()
        if now - last_report >= 30:
            print("[vllm] waiting for model server...", flush=True)
            last_report = now
        time.sleep(5)
    raise TimeoutError(f"vLLM did not become ready within {timeout_seconds} seconds.")


def select_smoke_dataset(dataset_root: Path, requested: str | None) -> str:
    datasets = discover_datasets(dataset_root, {requested} if requested else None)
    if requested:
        annotation = resolve_annotation_file(datasets[0] / "test")
        with annotation.open("r", encoding="utf-8") as file:
            image_count = len(json.load(file).get("images", []))
        if image_count < 20:
            raise ValueError(f"Requested smoke dataset {requested!r} has fewer than 20 images.")
        return requested

    candidates: list[tuple[str, int, int]] = []
    for dataset in datasets:
        with resolve_annotation_file(dataset / "test").open("r", encoding="utf-8") as file:
            coco = json.load(file)
        image_count = len(coco.get("images", []))
        category_count = len(coco.get("categories", []))
        if image_count >= 20:
            candidates.append((dataset.name, image_count, category_count))
    if not candidates:
        raise ValueError("No RF100VL dataset contains the 20 images required by the smoke test.")
    median_images = statistics.median(item[1] for item in candidates)
    candidates.sort(key=lambda item: (abs(item[1] - median_images), -item[2], item[0]))
    return candidates[0][0]


def evaluator_command(
    contract: JobContract,
    dataset_root: Path,
    save_dir: Path,
    gcs_uri: str,
    *,
    dataset: str | None = None,
    max_images: int | None = None,
    visualize_limit: int = 0,
    preflight_report: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "evaluate_cosmos.py",
        "--dataset-dir",
        str(dataset_root),
        "--base-url",
        BASE_URL,
        "--model-id",
        contract.model_id,
        "--workers",
        str(contract.workers),
        "--seed",
        "0",
        "--save-dir",
        str(save_dir),
        "--gcs-results-uri",
        gcs_uri,
    ]
    if dataset:
        command.extend(["--dataset", dataset])
    if max_images is not None:
        command.extend(["--max-images", str(max_images)])
    if visualize_limit:
        command.extend(["--visualize-limit", str(visualize_limit)])
    if preflight_report:
        command.extend(
            [
                "--expected-datasets",
                str(contract.expected_datasets),
                "--preflight-report",
                str(preflight_report),
            ]
        )
    return command


def record_hashes(save_dir: Path, dataset: str) -> dict[str, str]:
    record_root = save_dir / dataset / "records"
    return {
        str(path.relative_to(record_root)): sha256_file(path)
        for path in sorted(record_root.rglob("*.json"))
    }


def require_log_contract(log_path: Path, selected: int, resumed: int, pending: int) -> None:
    text = log_path.read_text(encoding="utf-8")
    pattern = rf"{selected} selected, {resumed} resumed, {pending} pending"
    if not re.search(pattern, text):
        raise RuntimeError(f"Resume contract {pattern!r} was not observed in {log_path}.")


def verify_one_dataset(save_dir: Path, dataset: str) -> dict[str, Any]:
    summary_path = save_dir / dataset / "summary.json"
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    if not summary.get("complete"):
        raise RuntimeError("One-dataset smoke run is not complete.")
    if summary.get("completed_image_count") != summary.get("image_count"):
        raise RuntimeError("One-dataset image counts do not match.")
    if summary.get("new_error_count") != 0:
        raise RuntimeError("One-dataset smoke run contains image errors.")
    metrics = summary.get("metrics", {})
    if not metrics or not all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in metrics.values()
    ):
        raise RuntimeError("One-dataset COCO metrics are missing or non-finite.")
    diagnostics = summary.get("diagnostics", {})
    anomaly_keys = (
        "invalid_boxes",
        "duplicate_boxes",
        "clamped_boxes",
        "reordered_axes",
        "ignored_label_count",
    )
    anomalies = {key: int(diagnostics.get(key, 0)) for key in anomaly_keys}
    if any(anomalies.values()):
        raise RuntimeError(f"One-dataset diagnostics require investigation: {anomalies}")
    if not (save_dir / "_SUCCESS.json").is_file():
        raise RuntimeError("One-dataset _SUCCESS.json is missing.")
    records = sorted((save_dir / dataset / "records").rglob("*.json"))
    if not records:
        raise RuntimeError("One-dataset raw records are missing.")
    for record_path in records[:10]:
        with record_path.open("r", encoding="utf-8") as file:
            record = json.load(file)
        if record.get("status") != "success" or not isinstance(record.get("raw_response"), str):
            raise RuntimeError(f"Invalid raw record: {record_path}")
        if record.get("finish_reason") == "length":
            raise RuntimeError(f"Truncated raw record: {record_path}")
    visualizations = sorted((save_dir / dataset / "visualizations").glob("*.jpg"))
    if len(visualizations) < min(10, int(summary["image_count"])):
        raise RuntimeError("Ten smoke visualizations were not produced.")
    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "visualization_paths": [str(path) for path in visualizations[:10]],
        "raw_record_paths": [str(path) for path in records[:10]],
    }


def run_preflight(contract: JobContract, dataset_root: Path, root_store: GCSArtifactStore) -> None:
    run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "-v",
            "test_evaluate_cosmos.py",
            "test_preflight_cosmos.py",
            "test_runpod_cosmos.py",
        ]
    )
    live_env = os.environ.copy()
    live_env["COSMOS_TEST_GCS_URI"] = contract.gcs_preflight_storage_uri
    run_command([sys.executable, "-m", "unittest", "-v", "test_gcs_live.py"], env=live_env)

    report_path = contract.work_dir / "cosmos_preflight_report.json"
    run_command(
        [
            sys.executable,
            "preflight_cosmos.py",
            "--dataset-dir",
            str(dataset_root),
            "--expected-datasets",
            str(contract.expected_datasets),
            "--base-url",
            BASE_URL,
            "--model-id",
            contract.model_id,
            "--gcs-test-uri",
            contract.gcs_preflight_storage_uri,
            "--report",
            str(report_path),
        ]
    )

    smoke_dataset = select_smoke_dataset(dataset_root, contract.smoke_dataset)
    print(f"[preflight] selected smoke dataset: {smoke_dataset}")
    first_dir = contract.work_dir / "smoke-first"
    restored_dir = contract.work_dir / "smoke-restored"
    run_command(
        evaluator_command(
            contract,
            dataset_root,
            first_dir,
            contract.gcs_smoke_uri,
            dataset=smoke_dataset,
            max_images=10,
            visualize_limit=10,
        )
    )
    first_hashes = record_hashes(first_dir, smoke_dataset)
    if len(first_hashes) != 10:
        raise RuntimeError(f"Expected 10 first-run records, found {len(first_hashes)}.")

    run_command(
        evaluator_command(
            contract,
            dataset_root,
            restored_dir,
            contract.gcs_smoke_uri,
            dataset=smoke_dataset,
            max_images=10,
            visualize_limit=10,
        )
    )
    require_log_contract(restored_dir / "cosmos_detection.log", 10, 10, 0)
    restored_hashes = record_hashes(restored_dir, smoke_dataset)
    if restored_hashes != first_hashes:
        raise RuntimeError("GCS-restored raw records do not exactly match the first run.")

    run_command(
        evaluator_command(
            contract,
            dataset_root,
            restored_dir,
            contract.gcs_smoke_uri,
            dataset=smoke_dataset,
            max_images=20,
            visualize_limit=10,
        )
    )
    require_log_contract(restored_dir / "cosmos_detection.log", 20, 10, 10)

    run_command(
        evaluator_command(
            contract,
            dataset_root,
            restored_dir,
            contract.gcs_smoke_uri,
            dataset=smoke_dataset,
            visualize_limit=10,
        )
    )
    verification = verify_one_dataset(restored_dir, smoke_dataset)
    verification["gcs_visualization_uris"] = [
        f"{contract.gcs_smoke_uri}/{Path(path).relative_to(restored_dir).as_posix()}"
        for path in verification["visualization_paths"]
    ]
    verification["gcs_raw_record_uris"] = [
        f"{contract.gcs_smoke_uri}/{Path(path).relative_to(restored_dir).as_posix()}"
        for path in verification["raw_record_paths"]
    ]
    gate_summary_path = contract.work_dir / "preflight_gate_summary.json"
    atomic_write_json(
        gate_summary_path,
        {
            "schema_version": 1,
            "status": "awaiting_human_visual_review",
            "created_at": utc_now(),
            "dataset": smoke_dataset,
            "automated_gates": {
                "offline_contracts": "passed",
                "real_gcs_round_trip": "passed",
                "all_100_dataset_validation": "passed",
                "endpoint_identity": "passed",
                "ten_image_inference": "passed",
                "gcs_only_resume_10_of_10": "passed",
                "resume_extension_10_plus_10": "passed",
                "one_complete_scored_dataset": "passed",
            },
            "human_gate": (
                "Review the ten GCS visualizations and matching raw responses for "
                "coordinate alignment, allowed labels, truncation, and thinking text."
            ),
            "gcs": {
                "preflight_report": (
                    f"{contract.gcs_preflight_storage_uri}/preflight_report.json"
                ),
                "smoke_root": contract.gcs_smoke_uri,
            },
            "verification": verification,
        },
    )
    root_store.upload_file(gate_summary_path, "control/preflight/gate_summary.json")
    print("[preflight] automated gates passed; human visual review is still required.")


def verify_full_result(contract: JobContract, save_dir: Path) -> dict[str, Any]:
    aggregate_path = save_dir / "aggregate_summary.json"
    success_path = save_dir / "_SUCCESS.json"
    with aggregate_path.open("r", encoding="utf-8") as file:
        aggregate = json.load(file)
    if aggregate.get("status") != "complete":
        raise RuntimeError("Full aggregate status is not complete.")
    for key in ("selected_dataset_count", "processed_dataset_count", "scored_dataset_count"):
        if aggregate.get(key) != contract.expected_datasets:
            raise RuntimeError(f"Full aggregate {key} is not {contract.expected_datasets}.")
    datasets = aggregate.get("datasets", [])
    if len(datasets) != contract.expected_datasets:
        raise RuntimeError("Full aggregate does not contain exactly 100 dataset summaries.")
    if any(not item.get("complete") or "metrics" not in item for item in datasets):
        raise RuntimeError("At least one RF100VL dataset is incomplete or unscored.")
    if not success_path.is_file():
        raise RuntimeError("Local full-run _SUCCESS.json is missing.")
    success_uri = f"{contract.gcs_full_uri}/_SUCCESS.json"
    if not exists(success_uri):
        raise RuntimeError("Durable GCS full-run _SUCCESS.json is missing.")
    return {
        "status": "complete",
        "verified_at": utc_now(),
        "aggregate_path": str(aggregate_path),
        "gcs_aggregate_uri": f"{contract.gcs_full_uri}/aggregate_summary.json",
        "gcs_results_root": contract.gcs_full_uri,
        "gcs_success_uri": success_uri,
        "selected_dataset_count": contract.expected_datasets,
        "scored_dataset_count": contract.expected_datasets,
        "macro_AP": aggregate.get("macro_AP"),
        "macro_AP50": aggregate.get("macro_AP50"),
    }


def run_full(contract: JobContract, dataset_root: Path, root_store: GCSArtifactStore) -> None:
    gate_summary_path = contract.work_dir / "preflight_gate_summary.json"
    download(
        f"{contract.gcs_run_uri}/control/preflight/gate_summary.json",
        gate_summary_path,
    )
    with gate_summary_path.open("r", encoding="utf-8") as file:
        gate_summary = json.load(file)
    if gate_summary.get("status") != "awaiting_human_visual_review":
        raise RuntimeError("The expected successful automated preflight summary is missing.")

    report_path = contract.work_dir / "cosmos_preflight_report.json"
    download(
        f"{contract.gcs_preflight_storage_uri}/preflight_report.json", report_path
    )
    save_dir = contract.work_dir / "full-results"
    run_command(
        evaluator_command(
            contract,
            dataset_root,
            save_dir,
            contract.gcs_full_uri,
            preflight_report=report_path,
        )
    )
    verification = verify_full_result(contract, save_dir)
    verification_path = contract.work_dir / "full_verification.json"
    atomic_write_json(verification_path, verification)
    root_store.upload_file(verification_path, "control/full/verification.json")
    print("[full] verified 100/100 datasets scored with durable GCS success marker.")


def main() -> int:
    contract = JobContract.from_environment()
    contract.work_dir.mkdir(parents=True, exist_ok=True)
    root_store = GCSArtifactStore(contract.gcs_run_uri)
    root_store.verify_access()

    command = vllm_command(contract)
    manifest_path = write_manifest(contract, command)
    root_store.upload_file(manifest_path, f"control/{contract.stage}/job_manifest.json")

    vllm_log_path = contract.work_dir / f"vllm-{contract.stage}.log"
    try:
        with vllm_log_path.open("ab") as vllm_log:
            print("[vllm] starting pinned BF16 server")
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=vllm_log, stderr=subprocess.STDOUT
            )
            try:
                # Dataset acquisition overlaps model download/startup.
                dataset_root = ensure_dataset(contract)
                wait_for_server(process, contract.model_id)
                if contract.stage == "preflight":
                    run_preflight(contract, dataset_root, root_store)
                else:
                    run_full(contract, dataset_root, root_store)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                vllm_log.flush()
    finally:
        if vllm_log_path.is_file():
            root_store.upload_file(vllm_log_path, f"control/{contract.stage}/vllm.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
