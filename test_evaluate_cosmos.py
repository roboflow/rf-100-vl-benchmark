import base64
import io
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

from evaluate_cosmos import (
    CANONICAL_MAX_TOKENS,
    CosmosResponseError,
    GCSArtifactError,
    build_cosmos_chat_request,
    build_cosmos_messages,
    build_detection_prompt,
    convert_detections_to_coco,
    main,
    parse_cosmos_response,
    parse_gcs_uri,
    prepare_image_reference,
    score_coco,
)


class CosmosPromptTests(unittest.TestCase):
    def test_detection_prompt_is_an_exact_golden_contract(self):
        prompt = build_detection_prompt(["red fox", 'class "quoted"'])
        self.assertEqual(
            prompt,
            "Locate every instance of each of the following object classes in the image:\n"
            '["red fox", "class \\"quoted\\""]\n\n'
            "Return only a JSON array in this exact form:\n"
            '[{"bbox_2d":[x1,y1,x2,y2],"label":"one class name exactly"}]\n\n'
            "Use integer coordinates normalized independently to 0–1000, with the "
            "origin at the top-left. Include one entry per object instance, use only "
            "the listed class names, and return [] if none are present.",
        )
        for forbidden in ("README", "few-shot", "training image", "example answer"):
            self.assertNotIn(forbidden, prompt)

    def test_message_template_matches_nvidia_media_first_example(self):
        self.assertEqual(
            build_cosmos_messages("data:image/png;base64,abc", "detect"),
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        },
                        {"type": "text", "text": "detect"},
                    ],
                }
            ],
        )

    def test_chat_request_disables_thinking_and_is_deterministic(self):
        request = build_cosmos_chat_request(
            "data:image/png;base64,abc",
            "detect",
            model_id="nvidia/Cosmos3-Edge",
            max_tokens=4096,
            seed=0,
            enable_thinking=False,
        )
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["seed"], 0)
        self.assertEqual(request["max_tokens"], 4096)
        self.assertEqual(
            request["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertEqual([message["role"] for message in request["messages"]], ["user"])


class GCSConfigurationTests(unittest.TestCase):
    def test_requires_bucket_and_run_specific_prefix(self):
        self.assertEqual(
            parse_gcs_uri("gs://benchmark-artifacts/cosmos/run-1"),
            ("benchmark-artifacts", "cosmos/run-1"),
        )
        for invalid in (
            "https://bucket/run",
            "gs://bucket",
            "gs:///run",
            "gs://bucket/run/../escape",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_gcs_uri(invalid)


class FullRunGuardTests(unittest.TestCase):
    def test_detected_full_run_requires_preflight_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_datasets = [root / f"dataset-{index:03d}" for index in range(100)]
            with mock.patch(
                "evaluate_cosmos.discover_datasets", return_value=fake_datasets
            ), self.assertRaisesRegex(ValueError, "requires --preflight-report"):
                main(
                    [
                        "--dataset-dir",
                        str(root),
                        "--expected-datasets",
                        "100",
                        "--gcs-results-uri",
                        "gs://benchmark-artifacts/cosmos/full-run",
                        "--save-dir",
                        str(root / "output"),
                    ]
                )


class CosmosParserTests(unittest.TestCase):
    def test_parses_array_after_thinking_and_fence(self):
        response = """<think>not part of the answer</think>
```json
[{"bbox_2d": [1, 2, 3, 4], "label": "cat"}]
```"""
        self.assertEqual(parse_cosmos_response(response)[0]["label"], "cat")

    def test_parses_single_official_shape(self):
        response = '{"bbox_2d":[214,141,497,722],"label":"load"}'
        self.assertEqual(len(parse_cosmos_response(response)), 1)

    def test_empty_array_is_valid(self):
        self.assertEqual(parse_cosmos_response("[]"), [])

    def test_unparseable_response_raises(self):
        with self.assertRaises(CosmosResponseError):
            parse_cosmos_response("there are no objects")


class CosmosCoordinateTests(unittest.TestCase):
    def test_full_normalized_box_becomes_full_image(self):
        predictions, diagnostics = convert_detections_to_coco(
            [{"bbox_2d": [0, 0, 1000, 1000], "label": "Cat"}],
            image_id=7,
            image_width=640,
            image_height=480,
            categories_by_id={3: "cat"},
        )
        self.assertEqual(predictions[0]["bbox"], [0.0, 0.0, 640.0, 480.0])
        self.assertEqual(predictions[0]["category_id"], 3)
        self.assertEqual(predictions[0]["score"], 1.0)
        self.assertEqual(diagnostics["accepted_detections"], 1)

    def test_non_square_image_scales_each_axis_independently(self):
        predictions, _ = convert_detections_to_coco(
            [{"bbox_2d": [250, 100, 750, 900], "label": "traffic light"}],
            image_id=11,
            image_width=640,
            image_height=480,
            categories_by_id={17: "traffic light"},
        )
        self.assertEqual(predictions[0]["bbox"], [160.0, 48.0, 320.0, 384.0])

    def test_ambiguous_normalized_category_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not unique"):
            convert_detections_to_coco(
                [],
                image_id=1,
                image_width=100,
                image_height=100,
                categories_by_id={1: "Cat", 2: " cat "},
            )

    def test_clamps_reorders_deduplicates_and_uses_exact_labels(self):
        detections = [
            {"bbox_2d": [1100, 900, -100, 100], "label": " cat "},
            {"bbox_2d": [1100, 900, -100, 100], "label": "CAT"},
            {"bbox_2d": [0, 0, 10, 10], "label": "caterpillar"},
            {"bbox_2d": [1, 1, 1, 2], "label": "cat"},
        ]
        predictions, diagnostics = convert_detections_to_coco(
            detections,
            image_id="image-a",
            image_width=200,
            image_height=100,
            categories_by_id={1: "cat"},
        )
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["bbox"], [0.0, 10.0, 200.0, 80.0])
        self.assertEqual(diagnostics["duplicate_boxes"], 1)
        self.assertEqual(diagnostics["invalid_boxes"], 1)
        self.assertEqual(diagnostics["clamped_boxes"], 2)
        self.assertEqual(diagnostics["reordered_axes"], 4)
        self.assertEqual(diagnostics["ignored_labels"], ["caterpillar"])


class CosmosImagePreparationTests(unittest.TestCase):
    def test_grayscale_input_is_converted_to_rgb_without_changing_dimensions(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "gray.png"
            Image.new("L", (20, 10), color=128).save(image_path)
            reference = prepare_image_reference(
                image_path=image_path,
                expected_width=20,
                expected_height=10,
                transport="data-url",
                dataset_root=root,
                server_media_root=None,
            )
            encoded = reference.split(",", 1)[1]
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as converted:
                self.assertEqual(converted.mode, "RGB")
                self.assertEqual(converted.size, (20, 10))

    def test_metadata_dimension_mismatch_is_rejected(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            Image.new("RGB", (20, 10), color="white").save(image_path)
            with self.assertRaisesRegex(ValueError, "do not match COCO metadata"):
                prepare_image_reference(
                    image_path=image_path,
                    expected_width=21,
                    expected_height=10,
                    transport="data-url",
                    dataset_root=root,
                    server_media_root=None,
                )


class _MockCosmosHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(content_length))
        type(self).requests.append(request)
        response = {
            "id": "mock-completion",
            "object": "chat.completion",
            "created": 1,
            "model": "nvidia/Cosmos3-Edge",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '[{"bbox_2d":[0,0,1000,1000],"label":"cat"}]',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        pass


class _LengthCosmosHandler(BaseHTTPRequestHandler):
    """Return a deterministic token-capped completion for failure testing."""

    requests = []

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(content_length))
        type(self).requests.append(request)
        response = {
            "id": "mock-length-completion",
            "object": "chat.completion",
            "created": 1,
            "model": "nvidia/Cosmos3-Edge",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '[{"bbox_2d":[0,0,1000',
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": CANONICAL_MAX_TOKENS,
                "total_tokens": 500 + CANONICAL_MAX_TOKENS,
            },
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        pass


class _TimeoutCosmosHandler(BaseHTTPRequestHandler):
    requests = 0

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        self.rfile.read(content_length)
        type(self).requests += 1
        time.sleep(0.25)
        encoded = json.dumps(
            {
                "id": "too-late",
                "object": "chat.completion",
                "created": 1,
                "model": "nvidia/Cosmos3-Edge",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "[]"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        pass


def _write_single_image_dataset(root: Path, dataset_name: str) -> Path:
    from PIL import Image

    dataset = root / dataset_name
    test_directory = dataset / "test"
    test_directory.mkdir(parents=True)
    Image.new("RGB", (32, 24), color="white").save(test_directory / "one.png")
    annotations = {
        "info": {},
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": "one.png",
                "width": 32,
                "height": 24,
            }
        ],
        "categories": [{"id": 1, "name": "cat", "supercategory": "object"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 32, 24],
                "area": 768,
                "segmentation": [],
                "iscrowd": 0,
            }
        ],
    }
    (test_directory / "_annotations.coco.json").write_text(
        json.dumps(annotations), encoding="utf-8"
    )
    return dataset


class _DirectoryArtifactStore:
    """Test double that treats a local directory as an exact GCS run root."""

    root: Path

    def __init__(self, uri: str):
        self.uri = uri

    def verify_access(self):
        self.root.mkdir(parents=True, exist_ok=True)

    def upload_file(self, local_path: Path, relative_path):
        destination = self.root.joinpath(*Path(str(relative_path)).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, destination)

    def delete_if_exists(self, relative_path):
        self.root.joinpath(*Path(str(relative_path)).parts).unlink(missing_ok=True)

    def restore_prefix(self, relative_prefix, destination: Path):
        source = self.root.joinpath(*Path(str(relative_prefix)).parts)
        if not source.is_dir():
            return 0
        count = 0
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            count += 1
        return count


class _FailingRecordArtifactStore(_DirectoryArtifactStore):
    def upload_file(self, local_path: Path, relative_path):
        if "records" in Path(str(relative_path)).parts:
            raise GCSArtifactError("intentional record upload failure")
        super().upload_file(local_path, relative_path)


class CosmosEndToEndTests(unittest.TestCase):
    def test_expensive_timeout_is_not_automatically_retried(self):
        try:
            import openai  # noqa: F401
            from PIL import Image  # noqa: F401
            import pycocotools  # noqa: F401
        except ImportError:
            self.skipTest("Cosmos integration dependencies are not installed")

        _TimeoutCosmosHandler.requests = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), _TimeoutCosmosHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset_root = root / "rf100"
                _write_single_image_dataset(dataset_root, "timeout-dataset")
                return_code = main(
                    [
                        "--dataset-dir",
                        str(dataset_root),
                        "--base-url",
                        f"http://127.0.0.1:{server.server_port}/v1",
                        "--workers",
                        "1",
                        "--timeout",
                        "0.05",
                        "--retries",
                        "3",
                        "--save-dir",
                        str(root / "output"),
                    ]
                )
                self.assertEqual(return_code, 1)
                self.assertEqual(_TimeoutCosmosHandler.requests, 1)
                error_path = next(
                    (root / "output" / "timeout-dataset").glob("errors_*.jsonl")
                )
                error = json.loads(error_path.read_text(encoding="utf-8").strip())
                self.assertIn("not automatically retried", error["error"])
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

    def test_token_capped_response_is_saved_and_never_retried(self):
        try:
            import openai  # noqa: F401
            from PIL import Image  # noqa: F401
            import pycocotools  # noqa: F401
        except ImportError:
            self.skipTest("Cosmos integration dependencies are not installed")

        _LengthCosmosHandler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _LengthCosmosHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset_root = root / "rf100"
                _write_single_image_dataset(dataset_root, "length-dataset")
                output = root / "output"
                _DirectoryArtifactStore.root = root / "remote-artifacts"
                with mock.patch(
                    "evaluate_cosmos.GCSArtifactStore", _DirectoryArtifactStore
                ):
                    return_code = main(
                        [
                            "--dataset-dir",
                            str(dataset_root),
                            "--base-url",
                            f"http://127.0.0.1:{server.server_port}/v1",
                            "--workers",
                            "1",
                            "--retries",
                            "3",
                            "--save-dir",
                            str(output),
                            "--gcs-results-uri",
                            "gs://benchmark-artifacts/cosmos/length-test",
                        ]
                    )
                self.assertEqual(return_code, 1)
                self.assertEqual(len(_LengthCosmosHandler.requests), 1)
                self.assertEqual(
                    _LengthCosmosHandler.requests[0]["max_tokens"],
                    CANONICAL_MAX_TOKENS,
                )
                error_paths = list(
                    (output / "length-dataset").glob("errors_*.jsonl")
                )
                self.assertEqual(len(error_paths), 1)
                error = json.loads(
                    error_paths[0].read_text(encoding="utf-8").strip()
                )
                self.assertEqual(error["finish_reason"], "length")
                self.assertEqual(error["raw_response"], '[{"bbox_2d":[0,0,1000')
                self.assertEqual(
                    error["usage"]["completion_tokens"], CANONICAL_MAX_TOKENS
                )
                remote_error_paths = list(
                    (_DirectoryArtifactStore.root / "length-dataset").glob(
                        "errors_*.jsonl"
                    )
                )
                self.assertEqual(len(remote_error_paths), 1)
                aggregate = json.loads(
                    (output / "aggregate_summary.json").read_text(encoding="utf-8")
                )
                self.assertEqual(aggregate["status"], "failed")
                self.assertFalse((output / "_SUCCESS.json").exists())
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

    def test_cli_inference_scoring_and_resume(self):
        try:
            from PIL import Image
            import openai  # noqa: F401
            import pycocotools  # noqa: F401
        except ImportError:
            self.skipTest("Cosmos integration dependencies are not installed")

        _MockCosmosHandler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _MockCosmosHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                test_directory = root / "rf100" / "toy-dataset" / "test"
                test_directory.mkdir(parents=True)
                Image.new("RGB", (100, 100), color="white").save(
                    test_directory / "one.png"
                )
                annotations = {
                    "info": {},
                    "licenses": [],
                    "images": [
                        {
                            "id": image_id,
                            "file_name": "one.png",
                            "width": 100,
                            "height": 100,
                        }
                        for image_id in (7, 8)
                    ],
                    "categories": [{"id": 1, "name": "cat", "supercategory": "object"}],
                    "annotations": [
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [0, 0, 100, 100],
                            "area": 10000,
                            "segmentation": [],
                            "iscrowd": 0,
                        }
                        for annotation_id, image_id in enumerate((7, 8), start=1)
                    ],
                }
                annotation_path = test_directory / "_annotations.coco.json"
                annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
                output = root / "output"
                _DirectoryArtifactStore.root = root / "remote-artifacts"
                argv = [
                    "--dataset-dir",
                    str(root / "rf100"),
                    "--base-url",
                    f"http://127.0.0.1:{server.server_port}/v1",
                    "--workers",
                    "1",
                    "--retries",
                    "0",
                    "--save-dir",
                    str(output),
                    "--gcs-results-uri",
                    "gs://benchmark-artifacts/cosmos/test-run",
                ]

                with mock.patch(
                    "evaluate_cosmos.GCSArtifactStore", _DirectoryArtifactStore
                ):
                    self.assertEqual(main([*argv, "--max-images", "1"]), 0)
                partial_summary = json.loads(
                    (output / "toy-dataset" / "summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertFalse(partial_summary["complete"])
                self.assertNotIn("metrics", partial_summary)
                remote_records = list(
                    (_DirectoryArtifactStore.root / "toy-dataset" / "records").rglob(
                        "*.json"
                    )
                )
                self.assertEqual(len(remote_records), 1)

                # Simulate pod loss: only GCS remains, and the next pod restores
                # the first raw response instead of repeating its inference.
                shutil.rmtree(output)
                with mock.patch(
                    "evaluate_cosmos.GCSArtifactStore", _DirectoryArtifactStore
                ):
                    self.assertEqual(main(argv), 0)
                    self.assertEqual(main(argv), 0)
                self.assertEqual(len(_MockCosmosHandler.requests), 2)

                request = _MockCosmosHandler.requests[0]
                self.assertEqual(len(request["messages"]), 1)
                user_content = request["messages"][0]["content"]
                self.assertEqual(
                    [part["type"] for part in user_content], ["image_url", "text"]
                )
                self.assertIn('["cat"]', user_content[1]["text"])
                self.assertEqual(request["temperature"], 0.0)
                self.assertEqual(request["max_tokens"], CANONICAL_MAX_TOKENS)
                self.assertFalse(request["chat_template_kwargs"]["enable_thinking"])

                summary = json.loads(
                    (output / "toy-dataset" / "summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertTrue(summary["complete"])
                self.assertAlmostEqual(summary["metrics"]["AP"], 1.0)
                self.assertEqual(summary["diagnostics"]["clamped_boxes"], 0)
                self.assertEqual(summary["diagnostics"]["reordered_axes"], 0)
                self.assertEqual(summary["diagnostics"]["ignored_label_count"], 0)
                predictions = json.loads(
                    (
                        output / "toy-dataset" / "cosmos_detection_results.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(predictions[0]["bbox"], [0.0, 0.0, 100.0, 100.0])
                self.assertEqual(len(predictions), 2)
                self.assertAlmostEqual(score_coco(annotation_path, [])["AP"], 0.0)
                self.assertTrue(
                    (_DirectoryArtifactStore.root / "_SUCCESS.json").is_file()
                )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

    def test_full_100_dummy_dataset_inference_scoring_and_gcs_artifacts(self):
        try:
            import openai  # noqa: F401
            from PIL import Image  # noqa: F401
            import pycocotools  # noqa: F401
            from preflight_cosmos import validate_dataset
        except ImportError:
            self.skipTest("Cosmos integration dependencies are not installed")

        _MockCosmosHandler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _MockCosmosHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset_root = root / "rf100"
                datasets = [
                    _write_single_image_dataset(
                        dataset_root, f"dataset-{index:03d}"
                    )
                    for index in range(100)
                ]
                base_url = f"http://127.0.0.1:{server.server_port}/v1"
                preflight_report = root / "preflight_report.json"
                preflight_report.write_text(
                    json.dumps(
                        {
                            "status": "passed",
                            "model_id": "nvidia/Cosmos3-Edge",
                            "prompt_version": "cosmos3-edge-rf100-basic-v2",
                            "dataset": {
                                "dataset_count": 100,
                                "datasets": [
                                    validate_dataset(dataset) for dataset in datasets
                                ],
                            },
                            "endpoint": {
                                "base_url": base_url,
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
                    ),
                    encoding="utf-8",
                )
                output = root / "output"
                _DirectoryArtifactStore.root = root / "remote-artifacts"
                with mock.patch(
                    "evaluate_cosmos.GCSArtifactStore", _DirectoryArtifactStore
                ):
                    return_code = main(
                        [
                            "--dataset-dir",
                            str(dataset_root),
                            "--base-url",
                            base_url,
                            "--workers",
                            "1",
                            "--retries",
                            "0",
                            "--expected-datasets",
                            "100",
                            "--preflight-report",
                            str(preflight_report),
                            "--save-dir",
                            str(output),
                            "--gcs-results-uri",
                            "gs://benchmark-artifacts/cosmos/dummy-full",
                        ]
                    )
                self.assertEqual(return_code, 0)
                self.assertEqual(len(_MockCosmosHandler.requests), 100)
                self.assertTrue(
                    all(
                        request["max_tokens"] == CANONICAL_MAX_TOKENS
                        for request in _MockCosmosHandler.requests
                    )
                )
                aggregate = json.loads(
                    (output / "aggregate_summary.json").read_text(encoding="utf-8")
                )
                self.assertEqual(aggregate["status"], "complete")
                self.assertEqual(aggregate["selected_dataset_count"], 100)
                self.assertEqual(aggregate["processed_dataset_count"], 100)
                self.assertEqual(aggregate["scored_dataset_count"], 100)
                self.assertEqual(len(aggregate["datasets"]), 100)
                self.assertTrue(
                    all(
                        summary["complete"]
                        and summary["new_error_count"] == 0
                        and abs(summary["metrics"]["AP"] - 1.0) < 1e-9
                        for summary in aggregate["datasets"]
                    )
                )
                self.assertTrue((output / "_SUCCESS.json").is_file())
                self.assertTrue(
                    (_DirectoryArtifactStore.root / "_SUCCESS.json").is_file()
                )
                for dataset in datasets:
                    remote_dataset = _DirectoryArtifactStore.root / dataset.name
                    self.assertTrue((remote_dataset / "summary.json").is_file())
                    self.assertTrue(
                        (remote_dataset / "cosmos_detection_results.json").is_file()
                    )
                    self.assertEqual(
                        len(list((remote_dataset / "records").rglob("*.json"))), 1
                    )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

    def test_gcs_checkpoint_failure_stops_before_next_dataset(self):
        try:
            from PIL import Image
            import openai  # noqa: F401
            import pycocotools  # noqa: F401
        except ImportError:
            self.skipTest("Cosmos integration dependencies are not installed")

        _MockCosmosHandler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _MockCosmosHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset_root = root / "rf100"
                for dataset_name in ("dataset-a", "dataset-b"):
                    test_directory = dataset_root / dataset_name / "test"
                    test_directory.mkdir(parents=True)
                    Image.new("RGB", (10, 10), color="white").save(
                        test_directory / "one.png"
                    )
                    annotations = {
                        "info": {},
                        "licenses": [],
                        "images": [
                            {
                                "id": 1,
                                "file_name": "one.png",
                                "width": 10,
                                "height": 10,
                            }
                        ],
                        "categories": [
                            {"id": 1, "name": "cat", "supercategory": "object"}
                        ],
                        "annotations": [],
                    }
                    (test_directory / "_annotations.coco.json").write_text(
                        json.dumps(annotations), encoding="utf-8"
                    )

                _FailingRecordArtifactStore.root = root / "remote-artifacts"
                with mock.patch(
                    "evaluate_cosmos.GCSArtifactStore", _FailingRecordArtifactStore
                ):
                    return_code = main(
                        [
                            "--dataset-dir",
                            str(dataset_root),
                            "--base-url",
                            f"http://127.0.0.1:{server.server_port}/v1",
                            "--retries",
                            "0",
                            "--save-dir",
                            str(root / "output"),
                            "--gcs-results-uri",
                            "gs://benchmark-artifacts/cosmos/failure-test",
                        ]
                    )
                self.assertEqual(return_code, 1)
                self.assertEqual(len(_MockCosmosHandler.requests), 1)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main()
