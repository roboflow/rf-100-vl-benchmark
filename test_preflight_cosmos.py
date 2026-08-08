import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from evaluate_cosmos import PROMPT_VERSION, validate_preflight_report
from preflight_cosmos import main, validate_dataset


def _write_dataset(root: Path, bbox=None) -> Path:
    from PIL import Image

    dataset = root / "toy-dataset"
    test_directory = dataset / "test"
    test_directory.mkdir(parents=True)
    Image.new("RGB", (20, 10), color="white").save(test_directory / "one.png")
    annotations = {
        "info": {},
        "licenses": [],
        "images": [
            {
                "id": 7,
                "file_name": "one.png",
                "width": 20,
                "height": 10,
            }
        ],
        "categories": [{"id": 3, "name": "cat", "supercategory": "object"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 7,
                "category_id": 3,
                "bbox": bbox or [1, 2, 3, 4],
                "area": 12,
                "segmentation": [],
                "iscrowd": 0,
            }
        ],
    }
    (test_directory / "_annotations.coco.json").write_text(
        json.dumps(annotations), encoding="utf-8"
    )
    return dataset


class _ModelsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/v1/models":
            self.send_error(404)
            return
        payload = json.dumps(
            {
                "object": "list",
                "data": [
                    {
                        "id": "nvidia/Cosmos3-Edge",
                        "object": "model",
                        "created": 1,
                        "owned_by": "test",
                    }
                ],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


class _DirectoryArtifactStore:
    root: Path

    def __init__(self, uri: str):
        self.uri = uri

    def verify_access(self):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "run_access_probe.json").write_text("probe", encoding="utf-8")

    def upload_file(self, local_path: Path, relative_path):
        destination = self.root.joinpath(*Path(str(relative_path)).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, destination)

    def restore_prefix(self, relative_prefix, destination: Path):
        source = self.root.joinpath(*Path(str(relative_prefix)).parts)
        if not source.is_dir():
            return 0
        count = 0
        for path in source.rglob("*"):
            if path.is_file():
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                count += 1
        return count

    def delete_if_exists(self, relative_path):
        self.root.joinpath(*Path(str(relative_path)).parts).unlink(missing_ok=True)


class DatasetValidationTests(unittest.TestCase):
    def test_validates_every_image_and_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = validate_dataset(_write_dataset(Path(temporary)))
        self.assertEqual(result["category_count"], 1)
        self.assertEqual(result["image_count"], 1)
        self.assertEqual(result["annotation_count"], 1)
        self.assertEqual(len(result["annotation_sha256"]), 64)

    def test_rejects_bbox_outside_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = _write_dataset(Path(temporary), bbox=[19, 0, 2, 2])
            with self.assertRaisesRegex(ValueError, "outside image"):
                validate_dataset(dataset)

    def test_preflight_report_is_bound_to_annotation_endpoint_prompt_and_bucket(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = _write_dataset(root)
            dataset_result = validate_dataset(dataset)
            report_path = root / "report.json"
            report = {
                "status": "passed",
                "model_id": "nvidia/Cosmos3-Edge",
                "prompt_version": PROMPT_VERSION,
                "dataset": {"dataset_count": 1, "datasets": [dataset_result]},
                "endpoint": {
                    "base_url": "http://localhost:8000/v1",
                    "expected_model_id": "nvidia/Cosmos3-Edge",
                    "advertised_model_ids": ["nvidia/Cosmos3-Edge"],
                },
                "gcs": {
                    "parent_uri": "gs://benchmark-artifacts/preflight",
                    "operations": [
                        "create",
                        "update",
                        "list",
                        "read",
                        "restore",
                        "delete",
                    ],
                },
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            args = SimpleNamespace(
                model_id="nvidia/Cosmos3-Edge",
                base_url="http://localhost:8000/v1",
                gcs_results_uri="gs://benchmark-artifacts/results/run-1",
            )
            self.assertEqual(
                validate_preflight_report(report_path, args, [dataset]),
                report,
            )

            annotation_path = dataset / "test" / "_annotations.coco.json"
            annotation_path.write_text(
                annotation_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "hash changed"):
                validate_preflight_report(report_path, args, [dataset])


class PreflightEndToEndTests(unittest.TestCase):
    def test_dataset_endpoint_gcs_and_report(self):
        try:
            import openai  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("Cosmos integration dependencies are not installed")

        server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset_root = root / "rf100"
                _write_dataset(dataset_root)
                report_path = root / "report.json"
                _DirectoryArtifactStore.root = root / "fake-gcs"
                with mock.patch(
                    "preflight_cosmos.GCSArtifactStore", _DirectoryArtifactStore
                ):
                    return_code = main(
                        [
                            "--dataset-dir",
                            str(dataset_root),
                            "--expected-datasets",
                            "1",
                            "--base-url",
                            f"http://127.0.0.1:{server.server_port}/v1",
                            "--gcs-test-uri",
                            "gs://benchmark-artifacts/preflight/test",
                            "--report",
                            str(report_path),
                        ]
                    )
                self.assertEqual(return_code, 0)
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["dataset"]["dataset_count"], 1)
                self.assertIn(
                    "nvidia/Cosmos3-Edge",
                    report["endpoint"]["advertised_model_ids"],
                )
                self.assertEqual(
                    report["gcs"]["operations"],
                    ["create", "update", "list", "read", "restore", "delete"],
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
