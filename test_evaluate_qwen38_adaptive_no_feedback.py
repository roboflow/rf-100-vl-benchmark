import json
from pathlib import Path

import pytest

import evaluate_qwen38_adaptive_no_feedback as adaptive
import evaluate_qwen38_box_count_ablation as box_ablation
import evaluate_qwen38_orion as base

DATASET = Path("RF100VL/rf20-vl-fsod-fresh-20260813/the-dreidel-project")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    train = base.load_coco(DATASET / "train/_annotations.coco.json")
    test = base.load_coco(DATASET / "test/_annotations.coco.json")
    categories = base.categories_by_id(test)
    references = box_ablation.select_reference_sequences(
        train,
        DATASET / "train",
        required_count=2,
        distinct_images_only=False,
        allow_fewer=True,
    )
    assets = box_ablation.prepare_reference_assets(
        DATASET / "train",
        tmp_path_factory.mktemp("adaptive-references"),
        references,
    )
    task = adaptive.build_tasks({**test, "images": test["images"][:1]})[0]
    return test, categories, references, assets, task


def test_initial_turn_is_true_class_names_only_zero_shot(prepared):
    _, categories, _, _, task = prepared
    messages = adaptive.build_initial_messages(task, DATASET / "test", categories)
    assert len(messages) == 1
    content = messages[0]["content"]
    assert sum(part["type"] == "image_url" for part in content) == 1
    text = "\n".join(part["text"] for part in content if part["type"] == "text")
    assert "zero labeled visual examples" in text
    assert "Do not output detections yet" in text
    assert all(name in text for name in categories.values())
    assert "bbox_2d" not in text


def test_decision_parser_accepts_exact_or_fenced_json_and_rejects_unknown_labels(prepared):
    _, categories, *_ = prepared
    first = next(iter(categories.values()))
    decision = adaptive.parse_decision(
        f'```json\n{{"action":"request_examples","labels":["{first}"],"confidence":0.2}}\n```',
        categories,
    )
    assert decision["action"] == "request_examples"
    assert decision["requested_labels"] == [first]
    assert decision["confidence"] == 0.2
    assert adaptive.parse_decision('{"action":"detect","confidence":1}', categories)["action"] == "detect"
    with pytest.raises(adaptive.InvalidDecisionError, match="Unknown requested labels"):
        adaptive.parse_decision(
            '{"action":"request_examples","labels":["not a real class"]}',
            categories,
        )


def test_reference_turn_uses_prediction_matched_sparse_box_schema(prepared):
    _, categories, references, assets, _ = prepared
    category_id = next(iter(categories))
    reference = references[category_id][0]
    counts = {value: 0 for value in categories}
    counts[category_id] = 1
    message = adaptive.build_reference_message(
        [(category_id, reference)], counts, categories, assets
    )
    text_parts = [part["text"] for part in message["content"] if part["type"] == "text"]
    payload = next(json.loads(text) for text in text_parts if text.startswith('[{"bbox_2d"'))
    assert payload == [
        {
            "bbox_2d": list(reference.bbox_xyxy_1000),
            "label": categories[category_id],
        }
    ]
    assert "sparse positive exemplars" in text_parts[0]
    assert "unmarked object or region as unlabeled" in text_parts[0]


def test_adaptive_conversation_requests_one_reference_then_detects(
    tmp_path, monkeypatch, prepared
):
    _, categories, references, assets, task = prepared
    category_id, label = next(iter(categories.items()))
    calls = []
    responses = iter(
        [
            f'{{"action":"request_examples","labels":[{json.dumps(label)}],"confidence":0.2}}',
            '{"action":"detect","confidence":0.9}',
            f'[{{"bbox_2d":[10,20,100,200],"label":{json.dumps(label)}}}]',
        ]
    )

    def fake_call(client, messages, settings, max_retries, limiter):
        del client, settings, max_retries, limiter
        calls.append(messages)
        raw = next(responses)
        return {
            "status": "success",
            "raw_response": raw,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 2},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
            "attempts": [{"attempt": 1, "status": "success"}],
            "inference_seconds": 0.1,
            "elapsed_seconds": 0.1,
        }

    monkeypatch.setattr(adaptive, "_call_api", fake_call)
    settings = {
        "model": "fake",
        "base_url": "https://example.invalid",
        "max_completion_tokens": 8192,
        "temperature": 0,
        "seed": 1234,
        "reasoning_effort": "none",
        "enable_thinking": False,
        "vl_high_resolution_images": False,
        "timeout_seconds": 180,
    }
    output = tmp_path / "record.json"
    record = adaptive.execute_adaptive_task(
        task,
        object(),
        DATASET / "test",
        categories,
        references,
        assets,
        settings,
        {
            **settings,
            "max_completion_tokens": 1024,
            "response_format": {"type": "json_object"},
        },
        2,
        0,
        object(),
        output,
    )
    assert record["status"] == "success"
    assert record["stop_reason"] == "model_ready"
    assert record["selected_reference_counts"][str(category_id)] == 1
    assert record["selected_references"][0]["annotation_id"] == references[category_id][0].annotation_id
    assert len(record["rounds"]) == 2
    assert record["usage"]["request_count"] == 3
    assert record["usage"]["prompt_tokens"] == 30
    assert len(record["predictions"]) == 1
    assert len(calls) == 3
    assert output.is_file()


def test_prepare_only_writes_replayable_manifest_and_does_not_score(tmp_path, prepared):
    test, categories, *_ = prepared
    image_id = int(test["images"][0]["id"])
    output = tmp_path / "prepare"
    assert adaptive.main(
        [
            "--dataset-dir",
            str(DATASET),
            "--output-dir",
            str(output),
            "--image-ids",
            str(image_id),
            "--max-examples-per-class",
            "2",
            "--prepare-only",
        ]
    ) == 0
    manifest = json.loads((output / "run_manifest.json").read_text())
    policy = manifest["adaptive_policy"]
    assert policy["initial_labeled_examples"] == 0
    assert policy["prediction_feedback"] is False
    assert policy["test_ground_truth_visible"] is False
    assert policy["reference_box_schema"].startswith("prediction-matched")
    assert manifest["decision_response_format"] == {"type": "json_object"}
    assert len(manifest["reference_selection"]["classes"]) == len(categories)
    progress = json.loads((output / "progress.json").read_text())
    assert progress["total"]["pending"] == 1
    assert not (output / "_SUCCESS.json").exists()
