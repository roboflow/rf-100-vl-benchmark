import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import evaluate_qwen38_perceptionbench as pb
from prepare_perceptionbench import sniff_image_mime, validate_dataset


def sample_record():
    return {
        "index": 7,
        "answer": "blue",
        "problem": "Before <|image_2|> between <|image_1|> after",
        "system": "  preserve this system message  ",
        "image": ["data:image/png;base64,AAA=", "data:image/jpeg;base64,BBB="],
        "error_category": "visual_attribute_error",
        "source_bmk": "fixture",
    }


def test_build_messages_exactly_interleaves_placeholders_and_preserves_bytes():
    messages = pb.build_messages(sample_record())
    assert messages[0] == {
        "role": "system",
        "content": "  preserve this system message  ",
    }
    assert messages[1] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Before "},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,BBB="},
            },
            {"type": "text", "text": " between "},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAA="},
            },
            {"type": "text", "text": " after"},
        ],
    }


def test_unused_images_are_appended_and_answer_never_enters_messages():
    record = sample_record()
    record["problem"] = "Question <|image_1|>"
    serialized = json.dumps(pb.build_messages(record))
    assert "blue" not in serialized
    assert serialized.index("AAA=") < serialized.index("BBB=")


def test_qwen_request_is_protocol_and_provider_locked():
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            usage = SimpleNamespace(
                model_dump=lambda: {
                    "prompt_tokens": 12,
                    "completion_tokens": 34,
                    "total_tokens": 46,
                }
            )
            return iter(
                [
                    SimpleNamespace(
                        usage=None,
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content=None, reasoning_content="private reasoning"
                                ),
                                finish_reason=None,
                            )
                        ],
                    ),
                    SimpleNamespace(
                        usage=None,
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="blue", reasoning_content=None),
                                finish_reason="stop",
                            )
                        ],
                    ),
                    SimpleNamespace(usage=usage, choices=[]),
                ]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    result = pb.qwen_stream(
        client,
        model="qwen3.8-max",
        messages=pb.build_messages(sample_record()),
        max_tokens=65_536,
    )
    assert captured["model"] == "qwen3.8-max"
    assert captured["max_tokens"] == 65_536
    assert "max_completion_tokens" not in captured
    assert "temperature" not in captured
    assert "top_p" not in captured
    assert "seed" not in captured
    assert captured["extra_body"] == {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
    }
    assert captured["stream"] is True
    assert result["prediction"] == "blue"
    assert result["reasoning_characters_observed"] > 0
    assert "private reasoning" not in json.dumps(result)


@pytest.mark.parametrize("model", pb.MODEL_IDS)
def test_both_requested_models_are_supported_at_xhigh(model):
    assert model in {"qwen3.8-max", "qwen3.8-flash"}
    assert pb.REASONING_EFFORT == "xhigh"


def test_strict_judge_parser_matches_official_behavior():
    assert pb.decode_judge("[reason]\nok\n[judge]\nTrue") == (True, "ok")
    assert pb.decode_judge("answer is true") == (
        False,
        "No [reason] or [judge] in output",
    )


def test_judge_prompt_substitution_uses_reference_but_prediction_prompt_does_not():
    template = "Q={problem}\nP={assistant_answer}\nA={reference_answer}"
    prompt = pb.judge_prompt(template, sample_record(), "student")
    assert "blue" in prompt
    assert "student" in prompt


def test_dataset_validator_checks_all_3000_records_and_qwen_limits(tmp_path: Path):
    # One valid 1x1 PNG, reused only in this synthetic validator fixture.
    png = base64.b64encode(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000d4944415408d763f8cfc0f01f00050001ff89993d1d0000000049454e44ae426082"
        )
    ).decode()
    path = tmp_path / "PerceptionBench.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(3000):
            row = {
                "index": index,
                "answer": "x",
                "problem": "<|image_1|>question",
                "image": [f"data:image/png;base64,{png}"],
                "error_category": f"category_{index % 10}",
                "source_bmk": "fixture",
            }
            handle.write(json.dumps(row) + "\n")
    report = validate_dataset(path)
    assert report["record_count"] == 3000
    assert report["image_count"] == 3000
    assert report["all_images_fit_qwen_limit"] is True


def test_image_magic_is_validated_without_rewriting_upstream_mime():
    png = bytes.fromhex("89504e470d0a1a0a")
    assert sniff_image_mime(png) == "image/png"
    assert sniff_image_mime(b"not an image") is None


def test_cost_summary_uses_model_specific_international_prices(tmp_path: Path):
    run_dir = tmp_path / "run"
    pb.atomic_json(
        pb.checkpoint_path(run_dir, "predictions", 0),
        {
            "status": "complete",
            "usage": {
                "prompt_tokens": 1_000_000,
                "completion_tokens": 1_000_000,
                "total_tokens": 2_000_000,
            },
        },
    )
    summary = pb.status_summary(tmp_path, run_dir, "qwen3.8-max")
    assert summary["prediction_cost_usd_at_international_list_price"] == pytest.approx(8.0)
    summary = pb.status_summary(tmp_path, run_dir, "qwen3.8-flash")
    assert summary["prediction_cost_usd_at_international_list_price"] == pytest.approx(0.62)


def test_judge_manifest_cannot_silently_change(tmp_path: Path):
    expected = {
        "model": "gpt-oss-120b",
        "base_url": "https://judge.example/v1",
        "temperature": 0.3,
    }
    pb.write_or_check_judge_manifest(tmp_path, expected)
    pb.write_or_check_judge_manifest(tmp_path, expected)
    with pytest.raises(ValueError, match="Judge settings differ"):
        pb.write_or_check_judge_manifest(
            tmp_path, {**expected, "model": "a-different-judge"}
        )
