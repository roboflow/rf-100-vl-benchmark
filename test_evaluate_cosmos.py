import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

from evaluate_cosmos import (
    CosmosResponseError,
    build_detection_prompt,
    convert_detections_to_coco,
    main,
    parse_cosmos_response,
    score_coco,
)


class CosmosPromptTests(unittest.TestCase):
    def test_prompt_contains_all_classes_as_json(self):
        prompt = build_detection_prompt(["red fox", 'class "quoted"'])
        self.assertIn(json.dumps(["red fox", 'class "quoted"']), prompt)
        self.assertIn("return []", prompt)
        self.assertIn("0–1000", prompt)


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
        self.assertEqual(diagnostics["ignored_labels"], ["caterpillar"])


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


class CosmosEndToEndTests(unittest.TestCase):
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
                ]

                self.assertEqual(main([*argv, "--max-images", "1"]), 0)
                partial_summary = json.loads(
                    (output / "toy-dataset" / "summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertFalse(partial_summary["complete"])
                self.assertNotIn("metrics", partial_summary)

                self.assertEqual(main(argv), 0)
                self.assertEqual(main(argv), 0)
                self.assertEqual(len(_MockCosmosHandler.requests), 2)

                request = _MockCosmosHandler.requests[0]
                user_content = request["messages"][1]["content"]
                self.assertEqual(
                    [part["type"] for part in user_content], ["image_url", "text"]
                )
                self.assertIn('["cat"]', user_content[1]["text"])
                self.assertEqual(request["temperature"], 0.0)
                self.assertFalse(request["chat_template_kwargs"]["enable_thinking"])

                summary = json.loads(
                    (output / "toy-dataset" / "summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertTrue(summary["complete"])
                self.assertAlmostEqual(summary["metrics"]["AP"], 1.0)
                predictions = json.loads(
                    (
                        output / "toy-dataset" / "cosmos_detection_results.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(predictions[0]["bbox"], [0.0, 0.0, 100.0, 100.0])
                self.assertEqual(len(predictions), 2)
                self.assertAlmostEqual(score_coco(annotation_path, [])["AP"], 0.0)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main()
