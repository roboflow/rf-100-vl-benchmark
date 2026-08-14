import json
from pathlib import Path

import pytest

import evaluate_qwen38_orion as qwen

DATASET = Path("RF100VL/rf20-vl-fsod/orionproducts")


@pytest.fixture(scope="module")
def dataset():
    train = qwen.load_coco(DATASET / "train/_annotations.coco.json")
    test = qwen.load_coco(DATASET / "test/_annotations.coco.json")
    categories = qwen.categories_by_id(test)
    examples = qwen.select_reference_examples(train)
    negatives = qwen.validate_negative_pairs(categories)
    return train, test, categories, examples, negatives


def test_official_dataset_shape_and_split_isolation(dataset):
    train, test, categories, examples, _ = dataset
    qwen.validate_split_isolation(train, test)
    assert len(categories) == 8
    assert len(test["images"]) == 59
    assert len(examples) == 8
    assert all(example.boxes_xyxy_1000 for example in examples.values())


def test_task_matrix_is_complete_and_unique(dataset):
    _, test, categories, _, _ = dataset
    tasks = qwen.build_tasks(test, categories)
    assert len(tasks) == 2419
    assert len({task.key for task in tasks}) == len(tasks)
    counts = {mode: sum(task.mode == mode for task in tasks) for mode in qwen.MODES}
    assert counts["multi_class_names"] == 59
    assert all(counts[mode] == 472 for mode in qwen.SINGLE_CLASS_MODES)


def test_normalized_coordinate_round_trip():
    normalized = qwen.annotation_xywh_to_normalized_xyxy([10, 20, 30, 40], 100, 200)
    assert normalized == (100, 100, 400, 300)


def test_all_message_modes_use_train_references_and_target_last(tmp_path, dataset):
    _, test, categories, examples, negatives = dataset
    assets = qwen.prepare_reference_assets(
        DATASET / "train", tmp_path / "references", examples, negatives
    )
    test_image = min(test["images"], key=lambda value: value["id"])
    for mode in qwen.MODES:
        category_id = None if mode == "multi_class_names" else next(iter(categories))
        task = qwen.Task(
            mode=mode,
            image_id=test_image["id"],
            file_name=test_image["file_name"],
            width=test_image["width"],
            height=test_image["height"],
            category_id=category_id,
            category_name=categories.get(category_id),
        )
        messages = qwen.build_messages(
            task, DATASET / "test", categories, examples, negatives, assets
        )
        content = messages[0]["content"]
        assert messages[0]["role"] == "user"
        assert content[-2] == {"type": "text", "text": "TARGET IMAGE:"}
        assert content[-1]["type"] == "image_url"
        assert content[-1]["image_url"]["url"].startswith("data:image/")
        assert all("README.dataset" not in part.get("text", "") for part in content)
        image_count = sum(part["type"] == "image_url" for part in content)
        expected = 1
        if mode.startswith("positive"):
            expected += 1
        if mode in qwen.NEGATIVE_MODES:
            expected += 1
        assert image_count == expected


def test_drawn_reference_preserves_dimensions_and_adds_expected_colors(tmp_path, dataset):
    from PIL import Image

    _, _, _, examples, negatives = dataset
    assets = qwen.prepare_reference_assets(
        DATASET / "train", tmp_path / "references", examples, negatives
    )
    category_id = next(iter(examples))
    source = assets[category_id]["positive_source"]
    rendered = assets[category_id]["positive_drawn"]
    with Image.open(source) as before, Image.open(rendered) as after:
        assert before.size == after.size
        assert before.convert("RGB").tobytes() != after.convert("RGB").tobytes()


def test_parser_and_converter_recover_common_qwen_format(dataset):
    _, _, categories, _, _ = dataset
    category_id, name = next(iter(categories.items()))
    raw = f"```json\n[{{\"bbox_2d\":[100,200,400,500],\"label\":\"{name}\"}},]\n```"
    detections = qwen.parse_cosmos_response(raw)
    predictions, diagnostics = qwen.convert_detections_to_coco(
        detections, 7, 200, 100, {category_id: name}
    )
    assert len(predictions) == 1
    assert predictions[0]["bbox"] == pytest.approx([20, 20, 60, 30])
    assert predictions[0]["score"] == 1.0
    assert diagnostics["accepted_detections"] == 1


def test_score_uses_max_dets_500(dataset):
    _, test, _, _, _ = dataset
    image = test["images"][0]
    annotation = next(a for a in test["annotations"] if a["image_id"] == image["id"])
    prediction = {
        "image_id": image["id"],
        "category_id": annotation["category_id"],
        "bbox": annotation["bbox"],
        "score": 1.0,
    }
    metrics = qwen.score_coco(DATASET / "test/_annotations.coco.json", [prediction])
    assert metrics["max_dets"] == [1, 10, 500]
    assert metrics["prediction_count"] == 1


@pytest.mark.parametrize(
    "error",
    [
        Exception("APIError: <500> InternalError.Algo: inference engine abort"),
        Exception("429 Too Many Requests"),
        Exception("request timed out"),
        Exception("connection reset"),
    ],
)
def test_retryable_provider_errors_are_recognized(error):
    assert qwen.retryable_error(error)


def test_content_inspection_rejection_is_terminal_model_failure(monkeypatch):
    class ProviderError(Exception):
        status_code = 400

    error = ProviderError(
        "InternalError.Algo.DataInspectionFailed: Input image data may contain "
        "inappropriate content. code=data_inspection_failed"
    )
    monkeypatch.setattr(
        qwen,
        "stream_inference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    task = qwen.Task(
        mode="multi_class_names",
        image_id=1,
        file_name="unused.jpg",
        width=100,
        height=100,
        category_id=None,
        category_name=None,
    )
    record = qwen.execute_task(
        task,
        object(),
        Path("."),
        {7: "object"},
        {},
        {},
        {},
        {
            "reasoning_effort": "none",
            "seed": 1234,
            "timeout_seconds": 180.0,
        },
        max_retries=3,
        messages_override=[
            {"role": "user", "content": [{"type": "text", "text": "test"}]}
        ],
    )
    assert record["status"] == "model_failure"
    assert record["failure_type"] == "provider_content_rejection"
    assert record["predictions"] == []
    assert len(record["attempts"]) == 1


def test_only_explicit_content_inspection_rejections_are_terminalized():
    rejected = {
        "status": "error",
        "error": "BadRequestError: code=data_inspection_failed",
        "predictions": [{"should": "be removed"}],
    }
    rejected["attempts"] = [{"attempt": 1, "status": "error"}]
    terminal = qwen.terminalize_provider_failure(rejected)
    assert terminal["status"] == "model_failure"
    assert terminal["failure_type"] == "provider_content_rejection"
    assert terminal["predictions"] == []
    assert qwen.terminalize_provider_failure(
        {"status": "error", "error": "400 invalid parameter"}
    ) == {"status": "error", "error": "400 invalid parameter"}


def test_exhausted_provider_error_continues_as_auditable_zero_detection(monkeypatch):
    class ProviderError(Exception):
        status_code = 503

    monkeypatch.setattr(
        qwen,
        "stream_inference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ProviderError("service unavailable")
        ),
    )
    monkeypatch.setattr(qwen.time, "sleep", lambda _seconds: None)
    task = qwen.Task(
        mode="multi_class_names",
        image_id=1,
        file_name="unused.jpg",
        width=100,
        height=100,
        category_id=None,
        category_name=None,
    )
    record = qwen.execute_task(
        task,
        object(),
        Path("."),
        {7: "object"},
        {},
        {},
        {},
        {"reasoning_effort": "none", "seed": 1234, "timeout_seconds": 180.0},
        max_retries=2,
        messages_override=[
            {"role": "user", "content": [{"type": "text", "text": "test"}]}
        ],
    )
    assert record["status"] == "model_failure"
    assert record["failure_type"] == "provider_request_failure"
    assert record["retries_exhausted"] is True
    assert record["predictions"] == []
    assert len(record["attempts"]) == 3


def test_local_programming_error_is_not_silently_scored(monkeypatch):
    monkeypatch.setattr(
        qwen,
        "stream_inference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("local bug")),
    )
    task = qwen.Task(
        mode="multi_class_names",
        image_id=1,
        file_name="unused.jpg",
        width=100,
        height=100,
        category_id=None,
        category_name=None,
    )
    record = qwen.execute_task(
        task,
        object(),
        Path("."),
        {7: "object"},
        {},
        {},
        {},
        {"reasoning_effort": "none", "seed": 1234, "timeout_seconds": 180.0},
        max_retries=3,
        messages_override=[
            {"role": "user", "content": [{"type": "text", "text": "test"}]}
        ],
    )
    assert record["status"] == "error"


def test_authentication_and_generic_bad_requests_still_fail_closed():
    class ProviderError(Exception):
        def __init__(self, status_code, message):
            super().__init__(message)
            self.status_code = status_code

    assert not qwen.provider_request_error(ProviderError(400, "invalid parameter"))
    assert not qwen.provider_request_error(ProviderError(401, "invalid API key"))
    assert not qwen.provider_request_error(ProviderError(403, "permission denied"))


def test_no_reasoning_is_an_explicit_supported_condition():
    args = qwen.parse_args(["--reasoning-effort", "none"])
    assert args.reasoning_effort == "none"


def test_temperature_zero_is_an_explicit_supported_condition():
    args = qwen.parse_args(["--temperature", "0"])
    assert args.temperature == 0.0


def test_stream_request_sends_temperature_only_when_configured():
    class Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return iter(())

    class Client:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": Completions()})()

    settings = {
        "model": "qwen3.8-max",
        "seed": 1234,
        "max_completion_tokens": 8192,
        "reasoning_effort": "none",
        "vl_high_resolution_images": False,
    }
    client = Client()
    messages = [{"role": "user", "content": "test"}]
    qwen.stream_inference(client, messages, settings)
    assert "temperature" not in client.chat.completions.calls[0]
    assert client.chat.completions.calls[0]["extra_body"] == {
        "reasoning_effort": "none",
        "enable_thinking": False,
        "vl_high_resolution_images": False,
    }

    qwen.stream_inference(client, messages, {**settings, "temperature": 0.0})
    assert client.chat.completions.calls[1]["temperature"] == 0.0

    qwen.stream_inference(
        client,
        messages,
        {**settings, "reasoning_effort": "low"},
    )
    assert client.chat.completions.calls[2]["extra_body"]["enable_thinking"] is True


def test_streaming_generation_has_a_total_wall_clock_deadline(monkeypatch):
    class Stream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            yield type("Chunk", (), {"usage": None, "choices": []})()

        def close(self):
            self.closed = True

    stream = Stream()

    class Completions:
        def create(self, **_kwargs):
            return stream

    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": Completions()})()},
    )()
    times = iter((100.0, 281.0))
    monkeypatch.setattr(qwen.time, "monotonic", lambda: next(times))
    settings = {
        "model": "qwen3.8-max",
        "seed": 1234,
        "max_completion_tokens": 8192,
        "reasoning_effort": "none",
        "vl_high_resolution_images": False,
        "timeout_seconds": 180.0,
    }
    with pytest.raises(qwen.GenerationDeadlineError):
        qwen.stream_inference(client, [{"role": "user", "content": "test"}], settings)
    assert stream.closed


def test_generation_deadline_is_terminal_model_failure_without_retry(monkeypatch):
    monkeypatch.setattr(
        qwen,
        "stream_inference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            qwen.GenerationDeadlineError("deadline")
        ),
    )
    task = qwen.Task(
        mode="single_class_names",
        image_id=1,
        file_name="unused.jpg",
        width=100,
        height=100,
        category_id=7,
        category_name="object",
    )
    record = qwen.execute_task(
        task,
        object(),
        Path("."),
        {7: "object"},
        {},
        {},
        {},
        {
            "reasoning_effort": "none",
            "seed": 1234,
            "timeout_seconds": 180.0,
        },
        max_retries=3,
        messages_override=[
            {"role": "user", "content": [{"type": "text", "text": "test"}]}
        ],
    )
    assert record["status"] == "model_failure"
    assert record["failure_type"] == "generation_timeout"
    assert record["attempts"] == [{"attempt": 1, "status": "generation_timeout"}]


def test_negative_pair_map_can_be_supplied_for_another_fsod_dataset():
    dataset_path = Path("RF100VL/rf20-vl-fsod/lacrosse-object-detection")
    test = qwen.load_coco(dataset_path / "test/_annotations.coco.json")
    categories = qwen.categories_by_id(test)
    pairs = json.loads(
        Path(
            "qwen38-fsod-configs/lacrosse-object-detection-negative-pairs.json"
        ).read_text()
    )
    negative_ids = qwen.validate_negative_pairs(categories, pairs)
    assert set(negative_ids) == set(categories)
    assert all(category_id != negative_id for category_id, negative_id in negative_ids.items())


def test_final_defaults_can_saturate_singapore_quota_without_bursting():
    args = qwen.parse_args([])
    assert args.concurrency == 256
    assert args.requests_per_minute == 570
    assert args.tokens_per_minute == 900_000


def test_dual_rate_limiter_uses_the_binding_quota():
    now = [0.0]
    sleeps = []

    def clock():
        return now[0]

    def sleeper(delay):
        sleeps.append(delay)
        now[0] += delay

    limiter = qwen.SmoothDualRateLimiter(
        requests_per_minute=600,
        tokens_per_minute=60_000,
        clock=clock,
        sleeper=sleeper,
    )
    limiter.acquire(1000)
    limiter.acquire(1000)
    limiter.acquire(100)

    # 1,000 tokens at 60k TPM binds at one start per second. The following
    # low-token request cannot burst ahead of its already-reserved slot.
    assert sleeps == pytest.approx([1.0, 1.0])


def test_token_estimates_cover_all_modes_and_reasoning_conditions():
    for effort in ("none", "low"):
        assert set(qwen.ESTIMATED_TOTAL_TOKENS[effort]) == set(qwen.MODES)
        assert all(value > 0 for value in qwen.ESTIMATED_TOTAL_TOKENS[effort].values())


def test_manifest_fails_closed_on_configuration_change(tmp_path, dataset):
    _, _, _, examples, negatives = dataset
    path = tmp_path / "manifest.json"
    config = {"model": "qwen3.8-max"}
    qwen.write_or_validate_manifest(path, config, examples, negatives)
    qwen.write_or_validate_manifest(path, config, examples, negatives)
    with pytest.raises(ValueError):
        qwen.write_or_validate_manifest(path, {"model": "different"}, examples, negatives)


def test_request_fingerprint_changes_with_visual_payload(tmp_path, dataset):
    _, test, categories, examples, negatives = dataset
    assets = qwen.prepare_reference_assets(
        DATASET / "train", tmp_path / "references", examples, negatives
    )
    image = test["images"][0]
    category_id, category_name = next(iter(categories.items()))
    task = qwen.Task(
        mode="positive_drawn",
        image_id=image["id"],
        file_name=image["file_name"],
        width=image["width"],
        height=image["height"],
        category_id=category_id,
        category_name=category_name,
    )
    settings = {"model": "qwen3.8-max"}
    first = qwen.expected_request_fingerprint(
        task, DATASET / "test", categories, examples, negatives, assets, settings
    )
    assets[category_id]["positive_drawn"].write_bytes(b"changed")
    second = qwen.expected_request_fingerprint(
        task, DATASET / "test", categories, examples, negatives, assets, settings
    )
    assert first != second


def test_prepare_only_creates_resumable_artifacts(tmp_path):
    output = tmp_path / "run"
    assert qwen.main([
        "--dataset-dir", str(DATASET),
        "--output-dir", str(output),
        "--prepare-only",
    ]) == 0
    manifest = json.loads((output / "run_manifest.json").read_text())
    progress = json.loads((output / "progress.json").read_text())
    assert manifest["prompt_version"] == qwen.PROMPT_VERSION
    assert progress["total"]["total"] == 2419
    assert progress["total"]["pending"] == 2419
    assert (output / "aggregate_metrics.json").is_file()
