# Qwen3.8 Max and Flash on PerceptionBench

Status: implementation and local tests complete; prediction queue is resumable.

This benchmark evaluates `qwen3.8-max` and `qwen3.8-flash` on the complete
3,000-question PerceptionBench release. The implementation deliberately keeps
MoonshotAI's released benchmark behavior and makes only the model-specific
Qwen reasoning configuration explicit.

## Locked protocol

- Dataset: `moonshotai/PerceptionBench` at commit
  `6ba8c3135c7675ad6a5c141536a86b9460c70960`.
- Evaluator and judge prompt: `MoonshotAI/PerceptionBench` at commit
  `ba032c06e9b6ee3679171ff6ba643b7a0cfebe2e`.
- Prompt: the released `problem` text and image data URIs, interleaved at the
  released `<|image_N|>` placeholders; unused images are appended exactly as
  the official evaluator does. The optional per-record `system` field is kept.
- No global instruction, answer-format hint, chain-of-thought request, image
  rewrite, resize, or high-resolution override is added.
- Qwen input: OpenAI-compatible `image_url` content blocks containing the
  released Base64 data URIs. This is Qwen's documented multimodal format.
- Reasoning: `enable_thinking=true` and `reasoning_effort=xhigh` for both
  models. No `thinking_budget` is sent, because Qwen3.8 rejects combining it
  with `reasoning_effort`. `xhigh` is Qwen3.8's maximum native tier.
- Generation: streaming, `max_tokens=65536`, concurrency 16, and no explicit
  temperature, top-p, top-k, seed, or structured-output constraint. These are
  the released PerceptionBench defaults.
- Judge: the exact released prompt, `gpt-oss-120b`, temperature 0.3, and the
  official strict `[reason]`/`[judge]` parser.
- Metric: overall accuracy and accuracy for each of the ten capability
  categories. A publishable score requires all 3,000 judgments.

Prediction and judgment are separate checkpointed phases. This prevents a
judge-provider interruption from forcing expensive multimodal inference to be
repeated. Every record stores the raw final answer, provider usage, latency,
finish reason, retry history, and an input fingerprint. Private reasoning text
is not persisted.

## Reproducibility and validation

`prepare_perceptionbench.py` downloads the 1.63 GB JSONL and upstream evaluator
at pinned commits, verifies their SHA-256 hashes, checks all 3,000 records,
decodes every Base64 payload, validates actual image signatures, and checks
Qwen's per-image and image-count limits. The source bytes are never changed.

The released dataset contains 561 data URIs whose declared MIME label does not
match the valid payload signature. The manifest records these cases; the
evaluator preserves them byte-for-byte and the API smoke includes record 10 so
acceptance is tested before full inference.

Focused tests cover official image/text interleaving, answer isolation, both
model IDs, maximum-reasoning request fields, strict judge parsing, dataset
shape and image constraints, checkpoint locking, and model-specific cost
accounting.

## Run and resume

```bash
python prepare_perceptionbench.py --output-dir PerceptionBench
python -m pytest -q test_evaluate_qwen38_perceptionbench.py
bash run_qwen38_perceptionbench_queue.sh
```

The queue first tests records `0,10,2999` on both models, then runs Flash and
Max sequentially with 16 concurrent requests. Failed requests are retried in
bounded rounds. Results are stored below:

```text
perceptionbench-runs/qwen3.8-flash-xhigh-v1/
perceptionbench-runs/qwen3.8-max-xhigh-v1/
```

The Qwen API key is read only from `DASHSCOPE_API_KEY` or the existing secure,
gitignored local loader. It is never written to manifests, commands, logs, or
Git.

Judging uses a separate OpenAI-compatible vLLM endpoint serving the paper's
exact `openai/gpt-oss-120b` weights at pinned revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`. The delayed judge queue rents one
H100 only after both prediction arms finish, applies the exact released prompt
at temperature 0.3, scores both models, and terminates only the pod it created.
The vLLM container is pinned by digest. It will not substitute another judge or
label a partial result paper-comparable.

## Canonical sources

- PerceptionBench repository and evaluator:
  <https://github.com/MoonshotAI/PerceptionBench>
- PerceptionBench paper: <https://arxiv.org/abs/2607.24957>
- Released dataset: <https://huggingface.co/datasets/moonshotai/PerceptionBench>
- Qwen OpenAI-compatible Chat API:
  <https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions>
- Qwen image and video input:
  <https://docs.modelstudio.console.alibabacloud.com/en/model-studio/vision>
- Qwen3.8 Max:
  <https://docs.modelstudio.console.alibabacloud.com/en/model-studio/qwen3-8-max>
- Qwen3.8 Flash:
  <https://docs.modelstudio.console.alibabacloud.com/en/model-studio/qwen3-8-flash>
