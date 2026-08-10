#!/usr/bin/env python3
"""Run a staged Cosmos3 RF100VL benchmark inside one RunPod pod."""

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
    CANONICAL_MAX_TOKENS,
    CANONICAL_TIMEOUT_SECONDS,
    GCSArtifactStore,
    MODEL_ID,
    PROMPT_VERSION,
    atomic_write_json,
    discover_datasets,
    parse_gcs_uri,
    resolve_annotation_file,
    resolve_image_path,
    sha256_file,
)
from gcs_io import download, exists  # noqa: E402
from preflight_cosmos import validate_dataset  # noqa: E402
from cosmos_sharding import (  # noqa: E402
    finalize_if_ready,
    load_plan,
    shard_by_id,
    verify_shard_aggregate,
)


PINNED_MODEL_REVISIONS = {
    "nvidia/Cosmos3-Edge": "2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2",
    "nvidia/Cosmos3-Super": "e0262be9d8f7586bc24c069a2aed2b665bdff266",
}
# Backward-compatible alias used by the Edge-specific tests and run records.
PINNED_MODEL_REVISION = PINNED_MODEL_REVISIONS["nvidia/Cosmos3-Edge"]
DEFAULT_TENSOR_PARALLEL_SIZES = {
    "nvidia/Cosmos3-Edge": 1,
    "nvidia/Cosmos3-Super": 4,
}
BASE_URL = "http://127.0.0.1:8000/v1"
GPU_MEMORY_UTILIZATION = 0.80
MAX_MODEL_LENGTH = 131_072
MAX_IMAGE_PIXELS = 16_777_216
VISION_PATCH_SIZE = 16
VISION_SPATIAL_MERGE_SIZE = 2


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
    allow_incomplete_preflight: bool
    image_ref: str
    benchmark_git_sha: str
    shard_manifest_uri: str | None = None
    shard_manifest_sha256: str | None = None
    shard_id: str | None = None
    tensor_parallel_size: int = 1

    @classmethod
    def from_environment(cls) -> "JobContract":
        stage = os.getenv("COSMOS_STAGE", "").strip()
        if stage not in {"preflight", "full", "shard"}:
            raise ValueError(
                "COSMOS_STAGE must be exactly 'preflight', 'full', or 'shard'."
            )
        gcs_run_uri = os.getenv("COSMOS_GCS_RUN_URI", "").rstrip("/")
        parse_gcs_uri(gcs_run_uri)
        model_id = os.getenv("COSMOS_MODEL_ID", MODEL_ID).strip()
        if model_id not in PINNED_MODEL_REVISIONS:
            raise ValueError(
                "COSMOS_MODEL_ID must be one of the explicitly supported Cosmos3 "
                f"checkpoints: {sorted(PINNED_MODEL_REVISIONS)}."
            )
        model_revision = os.getenv(
            "COSMOS_MODEL_REVISION", PINNED_MODEL_REVISIONS[model_id]
        ).strip()
        if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
            raise ValueError("COSMOS_MODEL_REVISION must be a full 40-character commit SHA.")
        tensor_parallel_size = int(
            os.getenv(
                "COSMOS_TENSOR_PARALLEL_SIZE",
                str(DEFAULT_TENSOR_PARALLEL_SIZES[model_id]),
            )
        )
        allowed_tensor_parallel_sizes = (
            {1} if model_id == "nvidia/Cosmos3-Edge" else {4, 8}
        )
        if tensor_parallel_size not in allowed_tensor_parallel_sizes:
            raise ValueError(
                f"{model_id} requires COSMOS_TENSOR_PARALLEL_SIZE in "
                f"{sorted(allowed_tensor_parallel_sizes)} for this BF16 benchmark."
            )
        expected_datasets = int(os.getenv("COSMOS_EXPECTED_DATASETS", "100"))
        workers = int(os.getenv("COSMOS_WORKERS", "1"))
        if expected_datasets != 100:
            raise ValueError("The canonical RF100VL contract requires exactly 100 datasets.")
        if workers != 1:
            raise ValueError("The canonical Cosmos benchmark requires COSMOS_WORKERS=1.")
        preflight_approved = env_truthy("COSMOS_PREFLIGHT_APPROVED")
        allow_incomplete_preflight = env_truthy(
            "COSMOS_ALLOW_INCOMPLETE_PREFLIGHT"
        )
        if stage in {"full", "shard"} and not preflight_approved:
            raise ValueError(
                f"The {stage} stage requires COSMOS_PREFLIGHT_APPROVED=1 after "
                "human visual review."
            )
        if allow_incomplete_preflight and (
            stage not in {"full", "shard"} or not preflight_approved
        ):
            raise ValueError(
                "COSMOS_ALLOW_INCOMPLETE_PREFLIGHT=1 requires an approved full stage "
                "or an approved shard continuation."
            )
        shard_manifest_uri = os.getenv("COSMOS_SHARD_MANIFEST_URI") or None
        shard_manifest_sha256 = os.getenv("COSMOS_SHARD_MANIFEST_SHA256") or None
        shard_id = os.getenv("COSMOS_SHARD_ID") or None
        if stage == "shard":
            if not shard_manifest_uri or not shard_manifest_sha256 or not shard_id:
                raise ValueError(
                    "Shard stage requires COSMOS_SHARD_MANIFEST_URI, "
                    "COSMOS_SHARD_MANIFEST_SHA256, and COSMOS_SHARD_ID."
                )
            parse_gcs_uri(shard_manifest_uri)
            if not re.fullmatch(r"[0-9a-f]{64}", shard_manifest_sha256):
                raise ValueError("COSMOS_SHARD_MANIFEST_SHA256 must be 64 lowercase hex characters.")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", shard_id):
                raise ValueError("COSMOS_SHARD_ID contains unsafe characters.")
        elif shard_manifest_uri or shard_manifest_sha256 or shard_id:
            raise ValueError("Shard environment variables are valid only for shard stage.")
        dataset_gcs_uri = os.getenv("RF100VL_GCS_URI") or None
        if dataset_gcs_uri:
            parse_gcs_uri(dataset_gcs_uri)
        return cls(
            stage=stage,
            gcs_run_uri=gcs_run_uri,
            work_dir=Path(os.getenv("COSMOS_WORK_DIR", "/workspace/cosmos-runpod-work")),
            requested_dataset_dir=Path(os.getenv("RF100VL_DIR", "/workspace/rf100-vl")),
            model_id=model_id,
            model_revision=model_revision,
            expected_datasets=expected_datasets,
            workers=workers,
            smoke_dataset=os.getenv("COSMOS_SMOKE_DATASET") or None,
            dataset_gcs_uri=dataset_gcs_uri,
            preflight_approved=preflight_approved,
            allow_incomplete_preflight=allow_incomplete_preflight,
            image_ref=os.getenv("COSMOS_IMAGE_REF", "unknown"),
            benchmark_git_sha=os.getenv("BENCHMARK_GIT_SHA", "unknown"),
            shard_manifest_uri=shard_manifest_uri,
            shard_manifest_sha256=shard_manifest_sha256,
            shard_id=shard_id,
            tensor_parallel_size=tensor_parallel_size,
        )

    @property
    def gcs_preflight_storage_uri(self) -> str:
        return f"{self.gcs_run_uri}/preflight/storage"

    @property
    def gcs_smoke_uri(self) -> str:
        return f"{self.gcs_run_uri}/preflight/live-smoke"

    @property
    def gcs_early_smoke_uri(self) -> str:
        return f"{self.gcs_run_uri}/preflight/early-download-smoke"

    @property
    def gcs_full_uri(self) -> str:
        return f"{self.gcs_run_uri}/full"

    @property
    def control_prefix(self) -> str:
        if self.stage == "shard":
            return f"control/shards/{self.shard_id}"
        return f"control/{self.stage}"


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
    command = [
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
        str(MAX_MODEL_LENGTH),
        "--gpu-memory-utilization",
        f"{GPU_MEMORY_UTILIZATION:.2f}",
        "--allowed-local-media-path",
        "/",
        "--mm-processor-kwargs",
        f'{{"do_resize":true,"min_pixels":4096,"max_pixels":{MAX_IMAGE_PIXELS}}}',
        "--media-io-kwargs",
        '{"video":{"num_frames":256}}',
    ]
    if contract.tensor_parallel_size > 1:
        # NVIDIA's maintained Cosmos3-Super Reasoner configuration. Tensor
        # parallelism changes placement, not the BF16 checkpoint weights.
        command.extend(
            [
                "--tensor-parallel-size",
                str(contract.tensor_parallel_size),
                "--mm-encoder-tp-mode",
                "data",
                "--async-scheduling",
            ]
        )
    return command


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
        "request_max_tokens": CANONICAL_MAX_TOKENS,
        "request_timeout_seconds": CANONICAL_TIMEOUT_SECONDS,
        "vllm_command": list(command),
        "precision": "bfloat16",
        "quantization": None,
        "kv_cache_dtype": "auto",
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "tensor_parallel_size": contract.tensor_parallel_size,
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


@dataclass
class DatasetAcquisition:
    """A resumable RF100VL acquisition that may run beside live inference."""

    requested_root: Path
    expected_datasets: int
    process: subprocess.Popen[Any] | None = None
    resolved_root: Path | None = None


def start_dataset_acquisition(contract: JobContract) -> DatasetAcquisition:
    existing = find_complete_dataset_root(
        contract.requested_dataset_dir, contract.expected_datasets
    )
    if existing:
        print(f"[data] found all {contract.expected_datasets} datasets at {existing}")
        return DatasetAcquisition(
            requested_root=contract.requested_dataset_dir,
            expected_datasets=contract.expected_datasets,
            resolved_root=existing,
        )

    root = contract.requested_dataset_dir
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        print("[data] partial RF100VL download found; resuming with overwrite enabled")
    if contract.dataset_gcs_uri:
        print(f"[data] restoring RF100VL from {contract.dataset_gcs_uri}")
        command = [
            sys.executable,
            "infra/gcs_io.py",
            "download-prefix",
            "--uri",
            contract.dataset_gcs_uri,
            "--destination",
            str(root),
        ]
    else:
        if not os.getenv("ROBOFLOW_API_KEY"):
            raise RuntimeError(
                "RF100VL is absent and neither RF100VL_GCS_URI nor the "
                "ROBOFLOW_API_KEY RunPod secret is available."
            )
        command = [
            sys.executable,
            "infra/download_rf100vl.py",
            "--output-dir",
            str(root),
        ]

    print("[data] starting asynchronous acquisition:", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=ROOT)
    return DatasetAcquisition(
        requested_root=root,
        expected_datasets=contract.expected_datasets,
        process=process,
    )


def finish_dataset_acquisition(acquisition: DatasetAcquisition) -> Path:
    if acquisition.resolved_root is not None:
        return acquisition.resolved_root
    if acquisition.process is None:
        raise RuntimeError("Dataset acquisition has no process or resolved root.")
    return_code = acquisition.process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, acquisition.process.args)

    resolved = find_complete_dataset_root(
        acquisition.requested_root, acquisition.expected_datasets
    )
    if not resolved:
        count = (
            len(discover_datasets(acquisition.requested_root, None))
            if acquisition.requested_root.is_dir()
            else 0
        )
        raise RuntimeError(
            f"Expected {acquisition.expected_datasets} RF100VL test datasets "
            f"after download; found {count}."
        )
    acquisition.resolved_root = resolved
    return resolved


def stop_dataset_acquisition(acquisition: DatasetAcquisition | None) -> None:
    """Stop a still-running download when an earlier model gate has failed."""

    if acquisition is None or acquisition.process is None:
        return
    if acquisition.process.poll() is not None:
        return
    print("[data] stopping acquisition after an earlier job failure", flush=True)
    acquisition.process.terminate()
    try:
        acquisition.process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        acquisition.process.kill()
        acquisition.process.wait(timeout=10)


def ensure_dataset(contract: JobContract) -> Path:
    """Synchronous compatibility wrapper used by tests and manual callers."""

    acquisition = start_dataset_acquisition(contract)
    try:
        return finish_dataset_acquisition(acquisition)
    except BaseException:
        stop_dataset_acquisition(acquisition)
        raise


def candidate_dataset_directories(root: Path) -> list[Path]:
    """Find test-split directories that may be complete during acquisition."""

    roots = (root, root / "rf100vl", root / "rf100-vl", root / "RF100VL")
    candidates: set[Path] = set()
    for candidate_root in roots:
        if not candidate_root.is_dir():
            continue
        if (candidate_root / "test").is_dir():
            candidates.add(candidate_root)
            continue
        candidates.update(
            path for path in candidate_root.iterdir() if (path / "test").is_dir()
        )
    return sorted(candidates)


def find_first_ready_dataset(root: Path) -> tuple[Path, dict[str, Any]] | None:
    """Return the first test split that is complete and passes full validation."""

    for dataset in candidate_dataset_directories(root):
        try:
            validation = validate_dataset(dataset)
        except Exception:
            # A directory and its annotation file can appear before the package
            # has finished writing all referenced images. It is not ready yet.
            continue
        return dataset, validation
    return None


def dataset_readiness_signature(dataset: Path) -> tuple[tuple[str, int, int], ...]:
    """Capture sizes/mtimes so inference never races files still being written."""

    test_dir = dataset / "test"
    annotation_path = resolve_annotation_file(test_dir)
    with annotation_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)
    paths = [annotation_path]
    paths.extend(
        resolve_image_path(test_dir, str(image["file_name"]))
        for image in coco.get("images", [])
    )
    signature = []
    for path in paths:
        stat = path.stat()
        signature.append(
            (str(path.relative_to(dataset)), stat.st_size, stat.st_mtime_ns)
        )
    return tuple(sorted(signature))


def wait_for_first_ready_dataset(
    acquisition: DatasetAcquisition,
    *,
    stability_seconds: float = 2.0,
) -> tuple[Path, dict[str, Any]]:
    """Wait for one stable, valid dataset without waiting for all 100."""

    while True:
        ready = find_first_ready_dataset(acquisition.requested_root)
        if ready is not None:
            dataset, first_validation = ready
            try:
                first_signature = dataset_readiness_signature(dataset)
            except Exception:
                time.sleep(1)
                continue
            if stability_seconds:
                time.sleep(stability_seconds)
            try:
                second_validation = validate_dataset(dataset)
                second_signature = dataset_readiness_signature(dataset)
            except Exception:
                # The downloader was still moving this test split into place.
                time.sleep(1)
                continue
            if (
                second_validation == first_validation
                and second_signature == first_signature
            ):
                print(
                    f"[data] first stable dataset is ready for early inference: "
                    f"{dataset.name} ({second_validation['image_count']} test images)",
                    flush=True,
                )
                return dataset, second_validation

        process = acquisition.process
        if process is None:
            raise RuntimeError("No complete RF100VL dataset is available for early smoke.")
        return_code = process.poll()
        if return_code is not None:
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, process.args)
            resolved = finish_dataset_acquisition(acquisition)
            ready = find_first_ready_dataset(resolved)
            if ready is None:
                raise RuntimeError(
                    "RF100VL acquisition completed without one valid test dataset."
                )
            return ready
        time.sleep(2)


def wait_for_server(
    process: subprocess.Popen[Any],
    expected_model_id: str,
    timeout_seconds: int = 1800,
    acquisition: DatasetAcquisition | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"{BASE_URL}/models"
    last_report = 0.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with code {process.returncode}.")
        if (
            acquisition is not None
            and acquisition.process is not None
            and acquisition.process.poll() not in (None, 0)
        ):
            raise RuntimeError(
                "RF100VL acquisition failed while vLLM was starting with code "
                f"{acquisition.process.returncode}."
            )
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
    datasets: Sequence[str] | None = None,
    expected_datasets: int | None = None,
    max_images: int | None = None,
    visualize_limit: int = 0,
    preflight_report: Path | None = None,
) -> list[str]:
    if dataset and datasets:
        raise ValueError("Use either dataset or datasets, not both.")
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
        "--max-tokens",
        str(CANONICAL_MAX_TOKENS),
        "--timeout",
        f"{CANONICAL_TIMEOUT_SECONDS:g}",
        "--save-dir",
        str(save_dir),
        "--gcs-results-uri",
        gcs_uri,
    ]
    if dataset:
        # RF100VL contains a dataset literally named "-grccs". Use argparse's
        # --option=value form so a leading hyphen cannot be mistaken for a new
        # command-line option.
        command.append(f"--dataset={dataset}")
    if datasets:
        if len(set(datasets)) != len(datasets):
            raise ValueError("Evaluator dataset selection contains duplicates.")
        command.extend(f"--dataset={name}" for name in datasets)
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
    elif expected_datasets is not None:
        command.extend(["--expected-datasets", str(expected_datasets)])
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
        if record.get("status") not in {"success", "model_failure"}:
            raise RuntimeError(f"Invalid raw record: {record_path}")
        if record.get("status") == "success" and not isinstance(
            record.get("raw_response"), str
        ):
            raise RuntimeError(f"Successful raw response is missing: {record_path}")
        if record.get("status") == "model_failure" and record.get(
            "failure_type"
        ) not in {"timeout", "max_tokens", "invalid_response"}:
            raise RuntimeError(f"Unclassified terminal model failure: {record_path}")
    visualizations = sorted((save_dir / dataset / "visualizations").glob("*.jpg"))
    if len(visualizations) < min(10, int(summary["image_count"])):
        raise RuntimeError("Ten smoke visualizations were not produced.")
    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "visualization_paths": [str(path) for path in visualizations[:10]],
        "raw_record_paths": [str(path) for path in records[:10]],
    }


def verify_early_download_smoke(save_dir: Path, dataset: str) -> dict[str, Any]:
    """Require one durable, non-truncated inference and its visualization."""

    dataset_dir = save_dir / dataset
    records = sorted((dataset_dir / "records").rglob("*.json"))
    if len(records) != 1:
        raise RuntimeError(
            f"Early download smoke expected one raw record, found {len(records)}."
        )
    with records[0].open("r", encoding="utf-8") as file:
        record = json.load(file)
    if record.get("status") != "success":
        raise RuntimeError(f"Early download smoke record failed: {records[0]}")
    if not isinstance(record.get("raw_response"), str):
        raise RuntimeError("Early download smoke did not preserve the raw response.")
    if record.get("finish_reason") == "length":
        raise RuntimeError("Early download smoke response was token-truncated.")
    diagnostics = record.get("diagnostics", {})
    anomaly_keys = (
        "invalid_boxes",
        "duplicate_boxes",
        "clamped_boxes",
        "reordered_axes",
    )
    anomalies = {key: int(diagnostics.get(key, 0)) for key in anomaly_keys}
    anomalies["ignored_label_count"] = len(diagnostics.get("ignored_labels", []))
    if any(anomalies.values()):
        raise RuntimeError(
            f"Early download smoke diagnostics require investigation: {anomalies}"
        )

    visualizations = sorted((dataset_dir / "visualizations").glob("*.jpg"))
    if len(visualizations) != 1:
        raise RuntimeError(
            f"Early download smoke expected one visualization, found {len(visualizations)}."
        )
    summary_path = dataset_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    if summary.get("completed_image_count") != 1 or summary.get("new_error_count") != 0:
        raise RuntimeError("Early download smoke summary does not report one success.")
    return {
        "dataset": dataset,
        "summary_path": str(summary_path),
        "raw_record_path": str(records[0]),
        "visualization_path": str(visualizations[0]),
        "finish_reason": record.get("finish_reason"),
        "diagnostics": anomalies,
    }


def run_early_download_smoke(
    contract: JobContract,
    dataset: Path,
    validation: dict[str, Any],
    root_store: GCSArtifactStore,
) -> None:
    """Try Cosmos as soon as one dataset is ready while acquisition continues."""

    save_dir = contract.work_dir / "early-download-smoke"
    print(
        f"[early-smoke] running one image from {dataset.name} before requiring all "
        f"{contract.expected_datasets} datasets",
        flush=True,
    )
    run_command(
        evaluator_command(
            contract,
            dataset.parent,
            save_dir,
            contract.gcs_early_smoke_uri,
            dataset=dataset.name,
            max_images=1,
            visualize_limit=1,
        )
    )
    verification = verify_early_download_smoke(save_dir, dataset.name)
    verification.update(
        {
            "schema_version": 1,
            "status": "passed",
            "created_at": utc_now(),
            "dataset_validation": validation,
            "gcs_raw_record_uri": (
                f"{contract.gcs_early_smoke_uri}/"
                f"{Path(verification['raw_record_path']).relative_to(save_dir).as_posix()}"
            ),
            "gcs_visualization_uri": (
                f"{contract.gcs_early_smoke_uri}/"
                f"{Path(verification['visualization_path']).relative_to(save_dir).as_posix()}"
            ),
        }
    )
    verification_path = contract.work_dir / "early_download_smoke.json"
    atomic_write_json(verification_path, verification)
    root_store.upload_file(
        verification_path, "control/preflight/early_download_smoke.json"
    )
    print(
        "[early-smoke] one real Cosmos inference passed and is durable in GCS; "
        "continuing acquisition",
        flush=True,
    )


def run_preflight(contract: JobContract, dataset_root: Path, root_store: GCSArtifactStore) -> None:
    early_smoke_path = contract.work_dir / "early_download_smoke.json"
    with early_smoke_path.open("r", encoding="utf-8") as file:
        early_smoke = json.load(file)
    if early_smoke.get("status") != "passed":
        raise RuntimeError("The early-download Cosmos inference gate did not pass.")

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
    verification["early_download_smoke"] = early_smoke
    gate_summary_path = contract.work_dir / "preflight_gate_summary.json"
    atomic_write_json(
        gate_summary_path,
        {
            "schema_version": 1,
            "status": "awaiting_human_visual_review",
            "created_at": utc_now(),
            "dataset": smoke_dataset,
            "automated_gates": {
                "early_download_one_image_inference": "passed",
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
                "early_download_smoke": (
                    f"{contract.gcs_run_uri}/control/preflight/"
                    "early_download_smoke.json"
                ),
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
        "model_failure_count": aggregate.get("model_failure_count", 0),
        "model_failure_counts": aggregate.get("model_failure_counts", {}),
    }


def run_full(contract: JobContract, dataset_root: Path, root_store: GCSArtifactStore) -> None:
    gate_summary_path = contract.work_dir / "preflight_gate_summary.json"
    gate_summary_uri = f"{contract.gcs_run_uri}/control/preflight/gate_summary.json"
    if exists(gate_summary_uri):
        download(gate_summary_uri, gate_summary_path)
        with gate_summary_path.open("r", encoding="utf-8") as file:
            gate_summary = json.load(file)
        if gate_summary.get("status") != "awaiting_human_visual_review":
            raise RuntimeError(
                "The expected successful automated preflight summary is missing."
            )
        if (
            gate_summary.get("automated_gates", {}).get(
                "early_download_one_image_inference"
            )
            != "passed"
        ):
            raise RuntimeError("The early-download Cosmos inference gate is missing.")
    elif contract.allow_incomplete_preflight:
        early_smoke_path = contract.work_dir / "early_download_smoke_for_full.json"
        early_smoke_uri = (
            f"{contract.gcs_run_uri}/control/preflight/early_download_smoke.json"
        )
        download(early_smoke_uri, early_smoke_path)
        with early_smoke_path.open("r", encoding="utf-8") as file:
            early_smoke = json.load(file)
        if early_smoke.get("status") != "passed":
            raise RuntimeError(
                "The explicit preflight override still requires a successful real "
                "one-image Cosmos inference."
            )
        override_path = contract.work_dir / "preflight_override.json"
        atomic_write_json(
            override_path,
            {
                "schema_version": 1,
                "status": "explicitly_approved_with_incomplete_smoke",
                "created_at": utc_now(),
                "reason": (
                    "The long smoke request exposed runaway autoregressive output. "
                    "The user explicitly approved the full benchmark with an 8192-token "
                    "ceiling, a 180-second per-image timeout, and model-side failures "
                    "stored as terminal records with any complete detections salvaged."
                ),
                "required_evidence": {
                    "early_download_smoke": early_smoke_uri,
                    "dataset_and_storage_preflight": (
                        f"{contract.gcs_preflight_storage_uri}/preflight_report.json"
                    ),
                    "partial_live_smoke": contract.gcs_smoke_uri,
                },
            },
        )
        root_store.upload_file(
            override_path, "control/full/preflight_override.json"
        )
    else:
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


def run_shard(
    contract: JobContract, dataset_root: Path, root_store: GCSArtifactStore
) -> None:
    """Run one frozen, disjoint dataset shard and attempt final aggregation."""

    if (
        not contract.shard_manifest_uri
        or not contract.shard_manifest_sha256
        or not contract.shard_id
    ):
        raise ValueError("Shard contract is missing its manifest URI or shard ID.")
    plan_path = contract.work_dir / "shard_plan.json"
    download(contract.shard_manifest_uri, plan_path)
    if sha256_file(plan_path) != contract.shard_manifest_sha256:
        raise ValueError("Downloaded shard manifest SHA-256 does not match the launch contract.")
    plan = load_plan(plan_path)
    shard = shard_by_id(plan, contract.shard_id)
    checks = {
        "gcs_run_uri": (plan["gcs_run_uri"], contract.gcs_run_uri),
        "model_id": (plan["model_id"], contract.model_id),
        "model_revision": (plan["model_revision"], contract.model_revision),
        "image_ref": (plan["image_ref"], contract.image_ref),
        "benchmark_git_sha": (
            plan["benchmark_git_sha"],
            contract.benchmark_git_sha,
        ),
        "prompt_version": (plan["prompt_version"], PROMPT_VERSION),
    }
    for name, (planned, actual) in checks.items():
        if planned != actual:
            raise ValueError(
                f"Shard plan {name} mismatch: planned {planned!r}, actual {actual!r}."
            )

    expected_by_name = {item["dataset"]: item for item in shard["datasets"]}
    selected = discover_datasets(dataset_root, set(expected_by_name))
    for dataset_directory in selected:
        validation = validate_dataset(dataset_directory)
        expected = expected_by_name[dataset_directory.name]
        for key in ("annotation_sha256", "image_count", "annotation_count"):
            if validation[key] != expected[key]:
                raise ValueError(
                    f"Shard input mismatch for {dataset_directory.name} {key}: "
                    f"planned {expected[key]!r}, actual {validation[key]!r}."
                )

    save_dir = contract.work_dir / f"results-{contract.shard_id}"
    command = evaluator_command(
        contract,
        dataset_root,
        save_dir,
        shard["gcs_uri"],
        datasets=[item["dataset"] for item in shard["datasets"]],
        expected_datasets=shard["dataset_count"],
    )
    run_command(command)
    aggregate_path = save_dir / "aggregate_summary.json"
    success_path = save_dir / "_SUCCESS.json"
    if not aggregate_path.is_file() or not success_path.is_file():
        raise RuntimeError(f"{contract.shard_id} did not produce local success artifacts.")
    with aggregate_path.open("r", encoding="utf-8") as file:
        aggregate = json.load(file)
    verify_shard_aggregate(plan, contract.shard_id, aggregate)
    if not exists(f"{shard['gcs_uri']}/_SUCCESS.json"):
        raise RuntimeError(f"{contract.shard_id} GCS success marker is missing.")

    verification = {
        "schema_version": 1,
        "status": "complete",
        "verified_at": utc_now(),
        "plan_id": plan["plan_id"],
        "shard_id": contract.shard_id,
        "gcs_uri": shard["gcs_uri"],
        "dataset_count": shard["dataset_count"],
        "image_count": shard["image_count"],
    }
    verification_path = contract.work_dir / f"{contract.shard_id}-verification.json"
    atomic_write_json(verification_path, verification)
    root_store.upload_file(
        verification_path,
        f"control/shards/{plan['plan_id']}/{contract.shard_id}/verification.json",
    )
    finalization = finalize_if_ready(plan)
    print(
        f"[shard] {contract.shard_id} verified; finalization={finalization['status']}",
        flush=True,
    )


def main() -> int:
    contract = JobContract.from_environment()
    contract.work_dir.mkdir(parents=True, exist_ok=True)
    root_store = GCSArtifactStore(contract.gcs_run_uri)
    root_store.verify_access()

    command = vllm_command(contract)
    manifest_path = write_manifest(contract, command)
    root_store.upload_file(manifest_path, f"{contract.control_prefix}/job_manifest.json")

    vllm_log_path = contract.work_dir / f"vllm-{contract.stage}.log"
    acquisition: DatasetAcquisition | None = None
    try:
        with vllm_log_path.open("ab") as vllm_log:
            print("[vllm] starting pinned BF16 server")
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=vllm_log, stderr=subprocess.STDOUT
            )
            try:
                # Dataset acquisition overlaps model download/startup. During
                # preflight, one stable test split is enough to exercise the
                # live model before acquisition of all 100 datasets completes.
                acquisition = start_dataset_acquisition(contract)
                wait_for_server(
                    process, contract.model_id, acquisition=acquisition
                )
                if contract.stage == "preflight":
                    early_dataset, validation = wait_for_first_ready_dataset(acquisition)
                    run_early_download_smoke(
                        contract, early_dataset, validation, root_store
                    )
                dataset_root = finish_dataset_acquisition(acquisition)
                if contract.stage == "preflight":
                    run_preflight(contract, dataset_root, root_store)
                elif contract.stage == "full":
                    run_full(contract, dataset_root, root_store)
                else:
                    run_shard(contract, dataset_root, root_store)
            finally:
                stop_dataset_acquisition(acquisition)
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
            root_store.upload_file(vllm_log_path, f"{contract.control_prefix}/vllm.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
