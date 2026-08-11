# Qwen3.8-Max Orion prompt-mode experiment

This experiment compares six object-detection prompting modes on the
`orionproducts` dataset from the official RF20-VL-FSOD download:

1. Multi-class class names only.
2. Single-class class names only, combined across classes before scoring.
3. Single-class positive reference boxes supplied as normalized text coordinates.
4. Single-class positive reference boxes drawn in green.
5. Single-class positive and negative reference boxes supplied as normalized text coordinates.
6. Single-class positive and negative reference boxes drawn in green and red.

References are selected deterministically from `train`. Targets and scoring use
only `test`. No annotator instructions, object counts, validation images, or
test annotations are included in prompts. Requests are stateless, so one test
prediction cannot influence another.

Every request is atomically checkpointed below `records/`, including the raw
response, usage, latency, parser diagnostics, and COCO predictions. Rerunning
the command resumes terminal checkpoints. Final predictions and metrics are
written per mode; scoring uses pycocotools with `maxDets=[1, 10, 500]`.

## Run

Export `DASHSCOPE_API_KEY` without writing it to the repository, then run:

```bash
uv run --with-requirements requirements-cosmos.txt \
  python evaluate_qwen38_orion.py
```

The stable default output directory is:

```text
qwen38-orion-runs/orion-prompt-modes-v1/
```

Progress is available in `progress.json` and `experiment.log`. Running the same
command again safely resumes the experiment.

To make one paid smoke-test request in every mode before the full run, add
`--limit-per-mode 1`. Those six checkpoints are reused by the full command.
`--image-ids` and `--category-ids` can target known-positive smoke cases without
changing the full run manifest.

## Full selected-strategy comparison

The completed full runs above already contain the class-name and per-class
reference strategies. To add the one-call positive numeric reference strategy
for all 59 test images under both reasoning conditions, while fingerprint-
reusing the completed 20-image records, run:

```bash
uv run --with-requirements requirements-cosmos.txt \
  python evaluate_qwen38_orion_subset.py \
  --subset full \
  --new-modes multi_class_positive_numeric \
  --reasoning-efforts none low \
  --output-dir qwen38-orion-runs/orion-full-selected-prompts-v1
```

The output comparison combines the completed original full-test predictions
with the new one-call strategy. Rerunning this command resumes only missing or
non-terminal requests.
