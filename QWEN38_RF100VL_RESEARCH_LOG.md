# Qwen3.8-Max RF100-VL/FSOD research log

Canonical status: active, append-only research record  
Last evidence snapshot: 2026-08-12 00:29 UTC  
Model/API: `qwen3.8-max` through DashScope International's OpenAI-compatible endpoint

This document is the source of truth for the Qwen3.8-Max object-detection and
few-shot prompting study in this repository. It records the research in the
order it happened, including pilots, completed evaluations, negative results,
methodology corrections, active experiments, and predeclared decision rules.

Machine-readable run artifacts remain authoritative for exact values. When
this document and an artifact disagree, verify the artifact, correct this log,
and record the correction in the change log.

## How to interpret this log

Experiment status labels:

- **Completed:** every configured request is terminal and the score is final for
  that exact manifest.
- **Screening:** complete, but run with provider-default temperature or a small
  subset; useful for selecting candidates, not a locked final conclusion.
- **Smoke:** validates execution and parsing, not accuracy.
- **Active:** currently checkpointing requests.
- **Queued:** launched through the persistent pipeline but waiting for an
  earlier stage to release the API quota.
- **Prepared only:** manifest exists, but no inference was performed.

Unless stated otherwise, every score is **mAP50-95 / mAP50**, expressed as a
percentage. All COCO evaluations use pycocotools with
`maxDets=[1, 10, 500]`. Single-class requests are merged before scoring.

Important comparison rule: a familiar label such as “positive numeric” does
not guarantee that two experiments used the same prompt or reference-selection
policy. Compare scores causally only within a shared prompt version/manifest.
Cross-phase comparisons are descriptive unless explicitly identified as
fingerprint-equivalent.

## Stable evaluation contract

- Test images come only from the RF20-VL-FSOD `test` split.
- Visual references come only from `train`.
- Prompts never contain test annotations, ground-truth object counts, annotator
  instructions, or validation examples.
- Numeric box prompting includes the original reference image plus normalized
  XYXY coordinates; it is not text-only few-shot prompting.
- Drawn prompting embeds the selected box visually in the reference image.
- Target boxes are normalized independently to `[0, 1000]`, parsed robustly,
  converted to image pixels, and scored as COCO detections.
- Terminal records include raw response, usage, latency, parser diagnostics,
  predictions, request summary, and a settings/prompt fingerprint.
- Runs resume terminal fingerprint-matching records rather than repeating them.
- Current locked inference settings are `temperature=0`, seed `1234`, reasoning
  `none`, 8,192 completion tokens, and a 180-second generation deadline.

Dataset shapes used so far:

| Dataset | Test images | Classes | Ground-truth objects |
|---|---:|---:|---:|
| Orion Products (`orionproducts`) | 59 | 8 | 555 |
| Dreidel (`the-dreidel-project`) | 54 | 6 | 171 |
| Lacrosse (`lacrosse-object-detection`) | 50 | 4 | 355 |
| Aquarium (`aquarium-combined`) | 65 | 7 | 510 |
| Wildfire smoke (`wildfire-smoke`) | 75 | 1 | 75 |

## Current evidence summary

Strong screening patterns:

1. Cross-image box references improve accuracy substantially on both complete
   Dreidel and Orion box-count screens.
2. The best count is not monotonic: additional references can help, plateau, or
   hurt depending on dataset, formulation, encoding, and semantic instruction.
3. Formulation is dataset-dependent. Matched provider-default screens favor
   multi-class on Dreidel by roughly 18–23 mAP, but single-class on Orion by
   roughly 9–11 mAP. This reversal is provisional until locked temperature-zero
   matched comparisons exceed both measured noise floors.
4. Numeric versus drawn is an interaction, not a universal ranking. The two
   encodings are nearly tied for Dreidel multi-class at two boxes; drawn is much
   stronger for Dreidel single-class at five boxes, while Orion usually peaks
   with one or two examples.
5. Low reasoning is slower and more expensive and does not provide a stable
   accuracy benefit. It remains subject to one small locked stopping-rule test.
6. Anonymous exemplar-only prompting works without exposing class names.
   Explicit instruction is stronger than a minimal output-only protocol, and
   drawn exemplars with five boxes are currently the strongest anonymous
   Dreidel screening condition.

Not yet established:

- Residual API score noise at fixed temperature and seed.
- A final temperature-zero single-versus-multi winner on each dataset.
- Whether model-generated visual names help after controlling for the same
  boxes.
- The best anonymous multi-class `Concept A/B/...` recipe.
- A final two-dataset accuracy-first and throughput-first recipe.

## Chronological experiment record

### 0. API, dataset, and evaluator establishment

Status: **Completed infrastructure validation**  
Date: 2026-08-10 to 2026-08-11

The study settled on the DashScope International OpenAI-compatible endpoint,
the `qwen3.8-max` model identifier, and the official RF20-VL-FSOD download.
The initial evaluator established train/test isolation, normalized coordinate
handling, tolerant JSON parsing, class filtering, checkpoint fingerprints,
retryable provider errors, and COCO `maxDets=500` scoring.

Initial prompt families on Orion were:

1. Multi-class class names.
2. Single-class class names, merged.
3. Positive numeric references, per class.
4. Positive drawn references, per class.
5. Positive plus negative numeric references, per class.
6. Positive plus negative drawn references, per class.

The positive/negative classes were fixed before test results. Requests were
stateless; no target prediction influenced a later target.

Implementation commits: `3a82fd1`, `8094162`, `40e48f9`.

### 1. Full Orion six-mode reasoning comparison

Status: **Completed screening**  
Completed: no reasoning at 2026-08-11 02:27 UTC; low reasoning at 03:34 UTC  
Artifacts:
`qwen38-orion-runs/orion-prompt-modes-v1-no-thinking/` and
`qwen38-orion-runs/orion-prompt-modes-v1/`

Both runs used seed 1234 and provider-default temperature. Each single-class
condition required 472 requests; multi-class names required 59.

| Prompt mode | No reasoning | Low reasoning | Low-reasoning model failures |
|---|---:|---:|---:|
| Multi-class names | 14.44 / 26.56 | 19.06 / 36.31 | 0 |
| Single-class names, merged | 11.41 / 20.94 | 15.69 / 29.44 | 0 |
| Positive numeric, per class | **23.46 / 49.50** | 20.09 / 43.82 | 1 |
| Positive drawn, per class | 21.56 / 47.18 | 17.90 / 39.22 | 0 |
| Positive + negative numeric | 22.21 / 47.33 | 20.56 / 48.25 | 0 |
| Positive + negative drawn | 16.77 / 42.29 | 18.90 / 41.99 | 3 |

What this changed:

- Reasoning was immediately mode-dependent: it helped class-name prompts but
  hurt the strongest positive-reference modes.
- Positive numeric references without reasoning became the first Orion
  accuracy-first candidate.
- Positive/negative references did not consistently beat positive-only
  references.
- Full reasoning factorials were no longer justified without a gate.

### 2. Five-image multi-reference pilot

Status: **Completed pilot; superseded by the 20-image subset**  
Completed: 2026-08-11 06:13 UTC  
Artifact: `qwen38-orion-runs/orion-five-image-single-prompt-v1/`  
Implementation commit: `07e9225`

This added one-call multi-class numeric and drawn reference prompts and rescored
the original modes on a nested five-image Orion subset.

| Prompt mode | No reasoning | Low reasoning |
|---|---:|---:|
| Multi-class names | 21.25 / 42.89 | 20.22 / 37.00 |
| Single-class names, merged | 22.10 / 37.06 | **32.69 / 54.20** |
| Positive numeric, per class | 30.70 / 60.81 | 29.73 / 60.71 |
| Positive drawn, per class | 23.45 / 53.93 | 27.33 / 47.65 |
| Positive + negative numeric | 30.75 / **72.16** | 25.88 / 55.82 |
| Positive + negative drawn | 31.43 / 70.61 | 29.18 / 58.29 |
| Multi-class positive numeric | 19.98 / 47.48 | 28.32 / 54.15 |
| Multi-class positive drawn | 16.24 / 50.38 | 15.96 / 35.40 |

The apparent low-reasoning gains were unstable. Five images were too small to
select a final recipe, which motivated a larger stratified subset.

### 3. Twenty-image stratified Orion prompt matrix

Status: **Completed screening**  
Completed: 2026-08-11 06:58 UTC  
Artifact: `qwen38-orion-runs/orion-twenty-image-single-prompt-v1/`  
Implementation commit: `5654de7`

The subset contains 20 images, all eight classes, and 188 ground-truth objects.
It nests the original five images. All 80 combined-prompt records completed;
20 were fingerprint-verified records reused from the five-image pilot.

| Prompt mode | No reasoning | Low reasoning |
|---|---:|---:|
| Multi-class names | **24.38 / 40.99** | 24.22 / 41.89 |
| Single-class names, merged | 21.84 / 34.92 | 21.20 / 39.09 |
| Positive numeric, per class | 23.84 / 48.86 | 23.65 / 45.26 |
| Positive drawn, per class | 21.57 / 44.48 | 21.66 / 41.97 |
| Positive + negative numeric | 23.42 / **49.57** | 22.76 / 47.40 |
| Positive + negative drawn | 19.77 / 48.29 | 19.58 / 39.60 |
| Multi-class positive numeric | 18.32 / 41.59 | **20.47 / 42.22** |
| Multi-class positive drawn | 18.84 / 43.25 | 14.56 / 30.32 |

Reasoning conclusions at this point:

- No reasoning won primary mAP50-95 in six of eight pairs.
- One low-reasoning win was only 0.09 mAP; the meaningful exception was the
  combined numeric prompt (+2.15 mAP).
- Across the 40 combined-prompt records, no reasoning averaged 14.40 seconds
  and produced zero reasoning tokens. Low reasoning averaged 30.47 seconds and
  used 46,090 reasoning tokens.
- The five-image subset had substantially overstated several effects.

### 4. Full 59-image Orion selected-strategy completion

Status: **Completed screening**  
Completed: 2026-08-11 07:50 UTC  
Artifact: `qwen38-orion-runs/orion-full-selected-prompts-v1/`  
Implementation/documentation commits: `c5a85df`, `bb24be9`

The new one-call numeric strategy inferred the remaining 39 images per
reasoning condition while reusing the 20 matching subset records.

| Strategy | Reasoning | Calls/image | Score |
|---|---|---:|---:|
| Multi-class names | None | 1 | 14.44 / 26.56 |
| Positive numeric, per class | None | 8 | **23.46 / 49.50** |
| Positive + negative numeric, per class | None | 8 | 22.21 / 47.33 |
| Positive drawn, per class | None | 8 | 21.56 / 47.18 |
| Multi-class positive numeric | None | 1 | 14.89 / 36.93 |
| Multi-class positive numeric | Low | 1 | 16.72 / 39.26 |

The one-call recipe remained a useful throughput option but did not match the
per-class positive numeric accuracy. Low reasoning helped this one combined
mode, but remained below the no-reasoning per-class result.

### 5. Cross-dataset validation on Lacrosse and Dreidel

Status: **Completed screening**  
Completed: 2026-08-11 08:17 UTC  
Artifacts: `qwen38-fsod-runs/lacrosse-selected-combined-v1/` and
`qwen38-fsod-runs/dreidel-selected-combined-v1/`  
Implementation/documentation commits: `7b1ff4d`, `565a910`, `b277b82`

This tested the selected Orion recipes on complete Lacrosse and Dreidel test
sets. The equal-weight macro includes Orion, Lacrosse, and Dreidel.

| Strategy | Calls/image | Orion | Lacrosse | Dreidel | 3-dataset macro |
|---|---:|---:|---:|---:|---:|
| Multi-class names, none | 1 | 14.44 / 26.56 | **33.58 / 53.57** | 42.33 / 57.02 | 30.12 / 45.71 |
| Positive numeric, per class, none | Classes | **23.46 / 49.50** | 24.92 / 39.24 | 32.00 / 45.12 | 26.80 / 44.62 |
| Positive drawn, per class, none | Classes | 21.56 / 47.18 | 24.68 / 39.33 | 31.12 / 42.86 | 25.79 / 43.13 |
| Positive + negative numeric, none | Classes | 22.21 / 47.33 | 24.54 / 38.16 | 34.98 / 50.58 | 27.24 / 45.36 |
| Multi-class positive numeric, none | 1 | 14.89 / 36.93 | 31.29 / 52.08 | **48.56 / 71.43** | **31.58 / 53.48** |
| Multi-class positive numeric, low | 1 | 16.72 / 39.26 | 25.23 / 43.13 | 36.62 / 53.64 | 26.19 / 45.34 |

New conclusions:

- Orion's accuracy-first recipe did not generalize as a universal winner.
- Lacrosse favored class names; Dreidel favored a single combined numeric
  reference prompt; Orion favored per-class numeric references.
- Combined numeric without reasoning had the best three-dataset macro while
  using one call per image.
- Low reasoning hurt the combined prompt on Lacrosse and Dreidel and produced
  five terminal model failures on Dreidel versus none without reasoning.
- Dataset dependence became a primary research question rather than a nuisance
  variable.

Aquarium note: `qwen38-fsod-runs/aquarium-selected-base-v1/` was prepared with
2,340 tasks but no inference was launched. It has no accuracy result and
must not be counted as validation evidence.

### 6. Full Dreidel named positive-box count factorial

Status: **Completed screening**  
Completed: 2026-08-11 18:56 UTC  
Artifact: `qwen38-fsod-runs/dreidel-box-count-ablation-v1/`  
Implementation commit: `b93b55e`

This experiment changed the reference policy to nested, diverse, train-only
boxes. Rank one is the largest relative-area example; later ranks use
deterministic farthest-point crop diversity. Numeric and drawn conditions use
the exact same source examples. This prompt/reference version is not identical
to the earlier selected-mode run.

Configuration: 22 conditions, 4,158 requests, counts 1/2/3/5/10,
provider-default temperature, seed 1234, reasoning none. Every request
completed with zero failures.

Multi-class results:

| Boxes/class | Class names | Numeric | Drawn |
|---:|---:|---:|---:|
| 0 | 40.93 / 54.86 | — | — |
| 1 | — | 49.61 / 70.66 | 49.37 / 72.71 |
| 2 | — | **54.78 / 77.12** | **54.06 / 78.08** |
| 3 | — | 50.96 / 75.22 | 51.11 / 72.75 |
| 5 | — | 52.02 / 78.28 | 51.47 / 76.21 |
| 10 | — | 51.29 / 76.67 | 48.38 / 71.16 |

Single-class merged results:

| Boxes/class | Class names | Numeric | Drawn |
|---:|---:|---:|---:|
| 0 | 16.88 / 19.65 | — | — |
| 1 | — | 27.99 / 39.02 | 29.02 / 38.68 |
| 2 | — | 31.63 / 47.88 | 36.10 / 53.16 |
| 3 | — | **32.75 / 46.74** | 34.04 / 48.81 |
| 5 | — | 30.56 / 49.55 | **40.08 / 58.07** |
| 10 | — | 32.73 / 49.60 | 39.70 / 58.17 |

What this revealed:

- Boxes helped strongly over the matched no-box baseline in both formulations.
- Multi-class was much stronger than single-class on Dreidel in this prompt
  version. At two numeric boxes the matched gap was 23.15 mAP; at two drawn
  boxes it was 17.96 mAP.
- Two boxes were optimal for multi-class primary mAP. More was not monotonic.
- Single-class drawn prompts continued improving through five boxes and then
  plateaued.
- Multi numeric and drawn were almost tied at two boxes; encoding preference
  depended on formulation.
- More reference images sharply increased input tokens and latency without a
  reliable accuracy gain. Multi drawn rose from 10.30 seconds/call at one box
  to 35.48 seconds at ten boxes.

### 7. Anonymous exemplar-only pilot and live API smoke

Status: **Smoke completed; full 1/2/5 experiment completed screening**  
Smoke: 2026-08-11 20:16 UTC  
Full completion: 2026-08-11 21:17 UTC  
Artifacts: `qwen38-fsod-runs/wildfire-exemplar-live-smoke-v1/` and
`qwen38-fsod-runs/dreidel-exemplar-only-box-v1/`  
Implementation commit: `183db36`

Anonymous semantics deliberately hide every ground-truth class name from the
model. Each per-class request is externally mapped back to its hidden category
after parsing.

- **Explicit:** “Find every object in the target image that is the same kind as
  the object marked in the reference image.”
- **Minimal:** reference image/box payload, target image, and only a coordinate
  output protocol; no find/detect/same-kind/class instruction.

The Wildfire run made four known-positive calls—explicit/minimal crossed with
numeric/drawn at two boxes. All four succeeded. It was an execution smoke, not
a scored accuracy experiment.

The full Dreidel matrix used 54 images, six hidden classes, 12 conditions, and
3,888 requests at counts 1/2/5.

| Instruction | Encoding | Boxes | Score | Model failures |
|---|---|---:|---:|---:|
| Explicit | Numeric | 1 | 21.67 / 30.34 | 1 |
| Explicit | Numeric | 2 | 24.03 / 34.55 | 0 |
| Explicit | Numeric | 5 | 23.84 / 34.59 | 0 |
| Explicit | Drawn | 1 | 22.96 / 31.95 | 0 |
| Explicit | Drawn | 2 | 28.39 / 41.76 | 0 |
| Explicit | Drawn | 5 | **36.82 / 58.08** | 1 |
| Minimal | Numeric | 1 | 16.55 / 21.83 | 0 |
| Minimal | Numeric | 2 | 19.11 / 27.47 | 12 |
| Minimal | Numeric | 5 | 21.14 / 31.25 | 2 |
| Minimal | Drawn | 1 | 13.94 / 21.12 | 0 |
| Minimal | Drawn | 2 | 16.66 / 24.35 | 3 |
| Minimal | Drawn | 5 | 24.70 / 36.92 | 11 |

What this revealed:

- Box-only cross-image transfer works without semantic class names.
- The explicit “same kind” instruction was consistently stronger and more
  parseable than the minimal protocol.
- Drawn boxes improved strongly with more examples; explicit drawn at five was
  the anonymous accuracy leader.
- Minimal modes had substantially more format/model failures at some counts.
- These conditions are single-class only; multi-class anonymous identifiers
  remained an open cell.

### 8. Orion replication of the named box-count factorial

Status: **Completed screening**  
Completed: 2026-08-12 00:05 UTC  
Artifact: `qwen38-fsod-runs/orion-box-count-ablation-v1/`

This replicated the same 22-condition prompt version on all 59 Orion images
and eight classes. Because RF20 is instance-based and some classes have fewer
than ten distinct train source images, the manifest explicitly allows distinct
annotations from a shared source image. All 5,841 requests completed with zero
failures.

Multi-class results:

| Boxes/class | Class names | Numeric | Drawn |
|---:|---:|---:|---:|
| 0 | 10.47 / 19.90 | — | — |
| 1 | — | **18.19 / 43.03** | 11.67 / 30.67 |
| 2 | — | 13.85 / 36.91 | 12.49 / 34.24 |
| 3 | — | 14.52 / 33.25 | **12.74 / 31.19** |
| 5 | — | 12.11 / 32.13 | 12.04 / 27.91 |
| 10 | — | 12.60 / 34.07 | 10.25 / 28.44 |

Single-class merged results:

| Boxes/class | Class names | Numeric | Drawn |
|---:|---:|---:|---:|
| 0 | 11.85 / 21.91 | — | — |
| 1 | — | 22.30 / 46.00 | **22.44 / 47.51** |
| 2 | — | **22.76 / 49.03** | 18.80 / 44.07 |
| 3 | — | 21.30 / 45.37 | 18.60 / 45.69 |
| 5 | — | 19.07 / 40.02 | 20.21 / 48.19 |
| 10 | — | 18.50 / 46.13 | 16.52 / 42.49 |

What this newly revealed:

- Boxes again helped substantially over matched no-box baselines.
- Orion reversed the Dreidel formulation ordering. Matched numeric at two boxes
  favored single-class by 8.92 mAP; matched drawn at one favored single-class
  by 10.76 mAP. This is a strong screening pattern, not yet a locked conclusion.
- Orion generally peaked at one or two examples. Five and ten often hurt.
- Best-vs-best single exceeded multi by 4.57 mAP, a less decisive comparison
  because count and encoding differ.
- Single numeric at two and single drawn at one were nearly tied on primary
  mAP, so the encoding winner requires the noise floor.
- The large Dreidel/Orion rank reversal made two-dataset final validation
  essential.

### 9. Anonymous seven/ten-box extension

Status: **Active**  
Started: 2026-08-12 00:06 UTC  
Artifact: `qwen38-fsod-runs/dreidel-exemplar-only-box-b07-b10-v1/`  
Implementation commit: `45c0974`

This adds explicit/minimal, numeric/drawn, and seven/ten boxes to the anonymous
single-class Dreidel matrix: eight conditions and 2,592 requests. It will be
combined with the completed 1/2/5 run into
`qwen38-fsod-runs/dreidel-exemplar-only-box-combined-v1/`.

Like the 1/2/5 matrix, this extension uses seed 1234, reasoning none, and
provider-default temperature.

Snapshot at 2026-08-12 00:29 UTC: 660/2,592 successful, zero model failures,
zero infrastructure errors. Explicit numeric seven and ten were complete;
explicit drawn seven was 12/324.

A read-only aggregation of all 324 completed explicit-numeric-seven records
scored **25.15 / 35.69**, with zero failures and mean inference time 6.32
seconds. It is the best explicit numeric anonymous score so far, but its
1.1–1.3 mAP gain over two/five boxes may be entirely within residual noise.
The official combined summary will replace this interim line after completion.

### 10. Explicit temperature-zero calibration

Status: **Completed diagnostic**  
Completed: 2026-08-11 23:24 UTC  
Artifacts: `qwen38-fsod-runs/dreidel-multi-names-temperature-zero-v1/` and
`qwen38-fsod-runs/final-recipe-study/temperature_calibration.json`  
Implementation commit: `be6ba6c`

The exact Dreidel multi-class-names condition was rerun on all 54 images with
reasoning none, seed 1234, and explicit temperature zero.

| Setting | Score |
|---|---:|
| Provider-default temperature | 40.93 / 54.86 |
| Explicit `temperature=0` | 41.98 / 57.40 |
| Difference | **+1.05 / +2.54** |

A 500-resample paired image bootstrap produced:

- mAP difference 95% interval: `[-2.58, +7.54]`
- mAP50 difference 95% interval: `[-4.27, +14.29]`
- Probability temperature-zero score was higher: 0.772 for mAP and 0.810 for
  mAP50.

Interpretation:

- The full 54-image result is operationally large enough to change close
  rankings, but one run per setting cannot attribute the difference solely to
  temperature.
- Bootstrap uncertainty is not inference stochasticity; it measures
  finite-image sampling uncertainty.
- Provider-default screens can nominate candidates, but locked conclusions
  require explicit temperature-zero finalists.

### 11. Generalized recipe evaluator and anonymous multi-class API smoke

Status: **Smoke completed; full run queued**  
Smoke completed: 2026-08-11 23:37 UTC  
Artifact: `qwen38-fsod-runs/final-recipe-study/anonymous-multi-api-smoke/`  
Implementation commit: `27cd344`

The generalized evaluator added:

- Single- and multi-class formulations.
- Ground-truth names, anonymous explicit/minimal concepts, self-generated
  names, and self-name-only controls.
- Numeric, drawn, or no references.
- Per-condition reasoning and seed.
- Explicit temperature zero.
- Label-free multi-class `Concept A/B/...` identifiers.
- Cached train-only self-name files.
- A hard 180-second streaming generation deadline classified as a terminal
  model failure without repeating the same deterministic request.

One request from each of the 20 anonymous multi-class cells was sent. Nineteen
parsed successfully, one reached the 8,192-token length limit, and there were
zero infrastructure errors. This validates payload structure, image ordering,
parsing, coordinate conversion, checkpointing, and failure classification; it
does not provide an accuracy score.

### 12. Noise-aware adaptive finalist study

Status: **Queued behind the active extension**  
Implementation commits: `27cd344`, `1bffd8f`  
Launcher: `run_qwen38_recipe_study.sh`  
Persistent session: `qwen-recipe-study`

The remaining stages are predeclared and run sequentially to avoid API-quota
contention:

1. Finish and combine anonymous 7/10-box records.
2. Run five identical full-test multi-class-name repetitions on Dreidel and
   five on Orion at temperature zero, seed 1234, and reasoning none.
3. Compute each dataset's fixed-test residual noise floor.
4. Run the full 20-condition anonymous multi-class count grid on Dreidel at
   temperature zero.
5. Generate cached train-only visual names.
6. Run the stratified self-name causal screen.
7. Run the small none-versus-low reasoning gate.
8. Run shortlisted finalists on full Dreidel and Orion at temperature zero.
9. Rank accuracy and efficiency using the locked decision rules below.

## Locked decision rules for future conclusions

### Residual API noise

Each dataset receives five identical complete repeats with temperature zero,
seed 1234, reasoning none, and the same prompt. For mAP and mAP50 separately:

`tie threshold = max(observed repeat range, 1.96 * sqrt(2) * sample SD)`

This is the inference/API noise floor on a fixed test set. It is separate from
paired-image bootstrap uncertainty about generalization beyond the finite test
images.

Use the noise floor as follows:

- If two locked recipes differ by no more than the relevant dataset floor,
  treat their accuracy as tied on that dataset.
- If tied and one is materially cheaper or faster, prefer it immediately.
- For the equal-weight Dreidel/Orion macro, use a conservative floor near the
  average of the two per-dataset floors.
- Use paired image bootstrap only after accounting for inference noise.
- If a complicated finalist has abnormal truncations or a conclusion lands
  exactly at the floor, repeat only the consequential pair.

### Reasoning stopping rule

Evaluate the likely accuracy winner and fast winner on the same stratified
20-image subset. Abandon reasoning unless low reasoning improves mAP50-95 by
more than `max(1 mAP, dataset noise floor)` on both datasets. Test medium only
if low passes. Do not test higher levels otherwise.

### Final recipe selection

Primary metric: equal-weight macro mAP50-95 across complete Dreidel and Orion.

Secondary evidence:

- Per-dataset mAP50-95 and mAP50.
- Calls per image.
- Mean latency and effective latency per image.
- Input/output/reasoning tokens and estimated cost.
- Model-format and timeout failure rates.
- Rank consistency across datasets.

Within measured noise, efficiency is the first tie-breaker. If accuracy remains
materially different after inference noise and finite-image uncertainty, select
the higher-accuracy recipe provided it does not collapse on either dataset.
Report separate accuracy-first and throughput-first recipes when warranted.

### Language for general trends

- A same-direction pattern across datasets, counts, and encodings may be
  described as a **suggestive trend** even when individual adjacent gaps are
  within noise.
- A method **wins** only when its locked difference exceeds the applicable
  noise threshold and is not contradicted by the other dataset.
- Mixed directions are reported as dataset-dependent behavior, not averaged
  away without qualification.
- A third dataset remains the best external-generalization check after recipe
  selection because Orion is no longer an untouched holdout.

## Open research questions and experiment queue

| Question | Current evidence | Next resolving experiment |
|---|---|---|
| Single vs multi-class | Strong provider-default reversal across Dreidel/Orion | Locked matched finalists on both datasets |
| Reasoning level | Mostly hurts or costs >2x; one combined-mode exception | Small none/low gate; medium only if gate passes |
| Number of examples | Non-monotonic; Dreidel multi peaks at 2, Orion at 1–2 | Complete anonymous 7/10 and locked shortlist |
| Numeric vs drawn | Interaction with dataset/formulation | Noise-aware finalists on both datasets |
| Explicit vs minimal anonymous | Explicit clearly stronger in Dreidel 1/2/5 | Complete 7/10 and anonymous multi-class grid |
| Anonymous single vs multi | Single matrix partly complete | Full `Concept A/B/...` multi grid |
| Self-generated names | Not yet measured | Name + boxes vs boxes-only vs name-only controls |
| Positive vs positive/negative | Dataset-dependent; no universal benefit | Not prioritized unless a locked finalist is close |
| Generalization | Rankings reverse across existing datasets | Final two-dataset macro, then optional third dataset |

Do not add a broad new factorial while the current adaptive pipeline is active.
New ideas should be added to this table with a hypothesis, smallest resolving
experiment, stopping rule, and expected call cost.

## Artifact index

| Phase | Canonical artifact |
|---|---|
| Initial Orion low reasoning | `qwen38-orion-runs/orion-prompt-modes-v1/` |
| Initial Orion no reasoning | `qwen38-orion-runs/orion-prompt-modes-v1-no-thinking/` |
| Five-image pilot | `qwen38-orion-runs/orion-five-image-single-prompt-v1/` |
| Twenty-image matrix | `qwen38-orion-runs/orion-twenty-image-single-prompt-v1/` |
| Full Orion selected modes | `qwen38-orion-runs/orion-full-selected-prompts-v1/` |
| Lacrosse validation | `qwen38-fsod-runs/lacrosse-selected-combined-v1/` |
| Dreidel selected validation | `qwen38-fsod-runs/dreidel-selected-combined-v1/` |
| Dreidel named box count | `qwen38-fsod-runs/dreidel-box-count-ablation-v1/` |
| Dreidel anonymous 1/2/5 | `qwen38-fsod-runs/dreidel-exemplar-only-box-v1/` |
| Dreidel anonymous 7/10 | `qwen38-fsod-runs/dreidel-exemplar-only-box-b07-b10-v1/` |
| Orion named box count | `qwen38-fsod-runs/orion-box-count-ablation-v1/` |
| Temperature calibration | `qwen38-fsod-runs/final-recipe-study/temperature_calibration.json` |
| Anonymous multi smoke | `qwen38-fsod-runs/final-recipe-study/anonymous-multi-api-smoke/` |
| Adaptive final study | `qwen38-fsod-runs/final-recipe-study/` |

Existing focused documents remain useful supporting notes:

- `QWEN38_ORION_EXPERIMENT.md`
- `QWEN38_ORION_SUBSET_RESULTS.md`
- `QWEN38_CROSS_DATASET_VALIDATION.md`
- `QWEN38_BOX_COUNT_ABLATION.md`

## Update protocol

When a stage completes:

1. Verify `_SUCCESS.json` or a complete selected-mode contract.
2. Read scores from `comparison_summary.json` or `aggregate_metrics.json`.
3. Verify prompt version, temperature, seed, reasoning, image count, call count,
   and failure count from the manifest/progress artifacts.
4. Replace any interim value explicitly marked in this log.
5. Append a dated chronological section; do not rewrite history to make later
   findings look inevitable.
6. Update the evidence summary, open-question table, and artifact index.
7. Commit the document together with any analysis script required to reproduce
   derived values.

## Change log

- **2026-08-12:** Created the canonical chronological log from existing
  manifests, progress files, score summaries, focused documentation, and git
  history. Recorded every completed scored experiment, non-scored smoke,
  prepared-only Aquarium run, temperature diagnostic, active anonymous
  extension, and queued noise-aware pipeline.
