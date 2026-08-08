from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "infra"))

from gcs_io import parse_uri
import download_rf100vl
from run_cosmos_job import (
    GPU_MEMORY_UTILIZATION,
    JobContract,
    MAX_IMAGE_PIXELS,
    MAX_MODEL_LENGTH,
    PINNED_MODEL_REVISION,
    VISION_PATCH_SIZE,
    VISION_SPATIAL_MERGE_SIZE,
    evaluator_command,
    ensure_dataset,
    select_smoke_dataset,
    vllm_command,
)
import write_job_exit
import runpod_self_terminate


def contract_environment(**overrides: str) -> dict[str, str]:
    values = {
        "COSMOS_STAGE": "preflight",
        "COSMOS_GCS_RUN_URI": "gs://bucket/rf100vl/cosmos/run-1",
        "COSMOS_MODEL_REVISION": PINNED_MODEL_REVISION,
        "COSMOS_EXPECTED_DATASETS": "100",
        "COSMOS_WORKERS": "1",
    }
    values.update(overrides)
    return values


class JobContractTests(unittest.TestCase):
    def test_100k_output_budget_leaves_room_for_maximum_configured_image(self):
        from evaluate_cosmos import CANONICAL_MAX_TOKENS

        image_token_upper_bound = MAX_IMAGE_PIXELS // (
            VISION_PATCH_SIZE * VISION_SPATIAL_MERGE_SIZE
        ) ** 2
        remaining_after_image = (
            MAX_MODEL_LENGTH - CANONICAL_MAX_TOKENS - image_token_upper_bound
        )
        self.assertEqual(CANONICAL_MAX_TOKENS, 100_000)
        self.assertEqual(image_token_upper_bound, 16_384)
        # RF100VL class lists are far smaller; this reserve also covers chat
        # template and vision boundary tokens without approaching the limit.
        self.assertGreaterEqual(remaining_after_image, 14_000)

    def test_preflight_contract_pins_fair_inference_settings(self):
        with mock.patch.dict(os.environ, contract_environment(), clear=True):
            contract = JobContract.from_environment()
        command = vllm_command(contract)
        self.assertEqual(contract.workers, 1)
        self.assertEqual(contract.expected_datasets, 100)
        self.assertIn("--dtype", command)
        self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")
        self.assertEqual(command[command.index("--kv-cache-dtype") + 1], "auto")
        self.assertEqual(command[command.index("--seed") + 1], "0")
        self.assertEqual(
            command[command.index("--gpu-memory-utilization") + 1],
            f"{GPU_MEMORY_UTILIZATION:.2f}",
        )
        self.assertEqual(command[command.index("--revision") + 1], PINNED_MODEL_REVISION)
        self.assertFalse(any("quant" in value for value in command))

    def test_full_contract_requires_explicit_visual_approval(self):
        environment = contract_environment(COSMOS_STAGE="full")
        with mock.patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(ValueError, "human visual review"):
                JobContract.from_environment()
        environment["COSMOS_PREFLIGHT_APPROVED"] = "1"
        with mock.patch.dict(os.environ, environment, clear=True):
            self.assertTrue(JobContract.from_environment().preflight_approved)

    def test_noncanonical_dataset_count_or_concurrency_is_rejected(self):
        for override in (
            {"COSMOS_EXPECTED_DATASETS": "99"},
            {"COSMOS_WORKERS": "2"},
        ):
            with self.subTest(override=override):
                with mock.patch.dict(
                    os.environ, contract_environment(**override), clear=True
                ):
                    with self.assertRaises(ValueError):
                        JobContract.from_environment()

    def test_full_evaluator_command_contains_preflight_and_100_dataset_guards(self):
        with mock.patch.dict(os.environ, contract_environment(), clear=True):
            contract = JobContract.from_environment()
        command = evaluator_command(
            contract,
            Path("/data"),
            Path("/results"),
            contract.gcs_full_uri,
            preflight_report=Path("/report.json"),
        )
        self.assertEqual(command[command.index("--expected-datasets") + 1], "100")
        self.assertEqual(command[command.index("--workers") + 1], "1")
        self.assertEqual(command[command.index("--max-tokens") + 1], "100000")
        self.assertEqual(command[command.index("--timeout") + 1], "1800")
        self.assertEqual(command[command.index("--preflight-report") + 1], "/report.json")
        self.assertNotIn("--max-images", command)
        self.assertNotIn("--enable-thinking", command)


class SmokeSelectionTests(unittest.TestCase):
    @staticmethod
    def write_dataset(root: Path, name: str, image_count: int, category_count: int) -> None:
        test_dir = root / name / "test"
        test_dir.mkdir(parents=True)
        payload = {
            "images": [
                {"id": index, "file_name": f"{index}.jpg", "width": 10, "height": 10}
                for index in range(image_count)
            ],
            "annotations": [],
            "categories": [
                {"id": index + 1, "name": f"class-{index}"}
                for index in range(category_count)
            ],
        }
        (test_dir / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_selection_is_deterministic_and_requires_twenty_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_dataset(root, "small", 19, 9)
            self.write_dataset(root, "median-a", 30, 2)
            self.write_dataset(root, "median-b", 30, 4)
            self.write_dataset(root, "large", 100, 10)
            self.assertEqual(select_smoke_dataset(root, None), "median-b")
            with self.assertRaisesRegex(ValueError, "fewer than 20"):
                select_smoke_dataset(root, "small")


class GCSPathTests(unittest.TestCase):
    def test_gcs_paths_reject_bucket_only_and_parent_components(self):
        self.assertEqual(parse_uri("gs://bucket/a/b"), ("bucket", "a/b"))
        for value in ("gs://bucket", "gs://bucket/a/../b", "https://bucket/a"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_uri(value)


class PodCleanupTests(unittest.TestCase):
    def test_stop_preserves_pod_and_uses_post(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"{}"
        with (
            mock.patch.dict(
                os.environ, {"RUNPOD_API_KEY": "secret"}, clear=True
            ),
            mock.patch.object(
                sys, "argv", ["runpod_self_terminate.py", "pod-1", "stop"]
            ),
            mock.patch("urllib.request.urlopen", return_value=response) as urlopen,
        ):
            self.assertEqual(runpod_self_terminate.main(), 0)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://rest.runpod.io/v1/pods/pod-1/stop")
        self.assertEqual(request.method, "POST")

    def test_terminate_deletes_pod_only_when_requested(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"{}"
        with (
            mock.patch.dict(
                os.environ, {"RUNPOD_API_KEY": "secret"}, clear=True
            ),
            mock.patch.object(
                sys,
                "argv",
                ["runpod_self_terminate.py", "pod-1", "terminate"],
            ),
            mock.patch("urllib.request.urlopen", return_value=response) as urlopen,
        ):
            self.assertEqual(runpod_self_terminate.main(), 0)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://rest.runpod.io/v1/pods/pod-1")
        self.assertEqual(request.method, "DELETE")

    def test_entrypoint_stops_preflight_and_failures_but_terminates_full_success(self):
        cases = (
            ("preflight", 0, "stop"),
            ("preflight", 7, "stop"),
            ("full", 0, "terminate"),
            ("full", 7, "stop"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_python = root / "fake-python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "if [ \"${1:-}\" = \"infra/run_cosmos_job.py\" ]; then\n"
                "  exit \"${FAKE_JOB_RC}\"\n"
                "fi\n"
                "if [ \"${1:-}\" = \"infra/runpod_self_terminate.py\" ]; then\n"
                "  printf '%s\\n' \"$*\" >> \"${FAKE_ACTION_LOG}\"\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o700)
            for stage, job_rc, expected_action in cases:
                with self.subTest(stage=stage, job_rc=job_rc):
                    action_log = root / f"{stage}-{job_rc}.log"
                    environment = os.environ.copy()
                    environment.update(
                        {
                            "COSMOS_BENCHMARK_ROOT": str(ROOT),
                            "COSMOS_EVAL_PYTHON": str(fake_python),
                            "COSMOS_WORK_DIR": str(root / f"work-{stage}-{job_rc}"),
                            "COSMOS_STAGE": stage,
                            "COSMOS_GCS_RUN_URI": "gs://bucket/run",
                            "GOOGLE_APPLICATION_CREDENTIALS": str(root / "fake.json"),
                            "RUNPOD_POD_ID": "pod-dummy",
                            "RUNPOD_API_KEY": "dummy-key",
                            "FAKE_JOB_RC": str(job_rc),
                            "FAKE_ACTION_LOG": str(action_log),
                        }
                    )
                    result = subprocess.run(
                        ["bash", "infra/cosmos_runpod_entrypoint.sh"],
                        cwd=ROOT,
                        env=environment,
                        text=True,
                        capture_output=True,
                    )
                    self.assertEqual(result.returncode, job_rc, result.stderr)
                    self.assertEqual(
                        action_log.read_text(encoding="utf-8").strip(),
                        "infra/runpod_self_terminate.py pod-dummy "
                        + expected_action,
                    )


class LauncherDryRunTests(unittest.TestCase):
    def run_launcher(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "RUNPOD_API_KEY": "not-a-real-secret",
                "RUNPOD_REGISTRY_AUTH_ID": "registry-id",
            }
        )
        return subprocess.run(
            ["bash", "infra/runpod_cosmos_launch.sh", "launch", *arguments],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
        )

    def test_preflight_dry_run_uses_only_secret_references(self):
        result = self.run_launcher(
            "--name",
            "cosmos-preflight",
            "--image",
            "registry/image:test",
            "--stage",
            "preflight",
            "--gcs-run-uri",
            "gs://bucket/runs/run-1",
            "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("{{ RUNPOD_SECRET_GCP_SA_JSON_B64 }}", result.stdout)
        self.assertIn("{{ RUNPOD_SECRET_ROBOFLOW_API_KEY }}", result.stdout)
        self.assertNotIn("not-a-real-secret", result.stdout + result.stderr)
        self.assertIn('"gpuCount": 1', result.stdout)
        self.assertIn('"containerDiskInGb": 100', result.stdout)
        self.assertIn('"volumeInGb": 200', result.stdout)
        self.assertIn('"volumeMountPath": "/workspace"', result.stdout)
        self.assertIn('"COSMOS_WORKERS": "1"', result.stdout)
        self.assertIn('"RUNPOD_STOP_ON_EXIT": "1"', result.stdout)

    def test_undersized_disks_are_rejected(self):
        common = (
            "--name",
            "cosmos-preflight",
            "--image",
            "registry/image:test",
            "--stage",
            "preflight",
            "--gcs-run-uri",
            "gs://bucket/runs/run-1",
        )
        for size_args, message in (
            (("--disk", "99"), "at least 100 GB"),
            (("--volume-size", "199"), "at least 200 GB"),
        ):
            with self.subTest(size_args=size_args):
                result = self.run_launcher(*common, *size_args, "--dry-run")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_dataset_gcs_source_omits_roboflow_secret_reference(self):
        result = self.run_launcher(
            "--name",
            "cosmos-preflight",
            "--image",
            "registry/image:test",
            "--stage",
            "preflight",
            "--gcs-run-uri",
            "gs://bucket/runs/run-1",
            "--dataset-gcs-uri",
            "gs://bucket/datasets/rf100vl",
            "--dry-run",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("RUNPOD_SECRET_ROBOFLOW_API_KEY", result.stdout)
        self.assertIn('"RF100VL_GCS_URI": "gs://bucket/datasets/rf100vl"', result.stdout)

    def test_full_dry_run_requires_approval_and_image_digest(self):
        base = (
            "--name",
            "cosmos-full",
            "--stage",
            "full",
            "--gcs-run-uri",
            "gs://bucket/runs/run-1",
            "--dry-run",
        )
        result = self.run_launcher("--image", "registry/image:test", *base)
        self.assertNotEqual(result.returncode, 0)
        result = self.run_launcher(
            "--image",
            "registry/image@sha256:" + "a" * 64,
            *base,
            "--preflight-approved",
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class RuntimeHelperTests(unittest.TestCase):
    def test_complete_100_dataset_volume_is_reused_without_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root = Path(temporary) / "rf100-vl"
            for index in range(100):
                (dataset_root / f"dataset-{index:03d}" / "test").mkdir(
                    parents=True
                )
            contract = types.SimpleNamespace(
                requested_dataset_dir=dataset_root,
                expected_datasets=100,
                dataset_gcs_uri=None,
            )
            with (
                mock.patch("run_cosmos_job.run_command") as run_command,
                mock.patch("run_cosmos_job.download_prefix") as download_prefix,
            ):
                self.assertEqual(ensure_dataset(contract), dataset_root)
            run_command.assert_not_called()
            download_prefix.assert_not_called()

    def test_downloader_uses_package_api_and_canonical_coco_format(self):
        calls = []

        def fake_download(path):
            calls.append(path)
            return [object()] * 100

        fake_module = types.SimpleNamespace(download_rf100vl=fake_download)
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(sys.modules, {"rf100vl": fake_module}):
                with mock.patch.object(
                    sys,
                    "argv",
                    ["download_rf100vl.py", "--output-dir", temporary],
                ):
                    self.assertEqual(download_rf100vl.main(), 0)
            self.assertEqual(calls, [temporary])

    def test_exit_record_preserves_failure_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "exit.json"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "write_job_exit.py",
                    "--path",
                    str(path),
                    "--stage",
                    "preflight",
                    "--exit-code",
                    "7",
                    "--git-sha",
                    "abc",
                    "--image-ref",
                    "registry/image@sha256:digest",
                ],
            ):
                self.assertEqual(write_job_exit.main(), 0)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["exit_code"], 7)


if __name__ == "__main__":
    unittest.main()
