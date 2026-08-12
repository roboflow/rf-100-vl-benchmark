# Qwen3.8-Max RF20-VL-FSOD recipe decision

Status: final two-dataset decision record

## Recommendation

**Accuracy-first:** one multi-class request per target image, real class names,
two positive numeric-box train examples per class, reasoning off.

**Cost-first:** one multi-class class-names-only request, reasoning off.

A numeric example contains the original train image plus its object's
normalized XYXY coordinates; it is not text-only few-shot prompting. All
classes and references are placed in one request, so no per-class merging is
needed.

| Recipe | Dreidel | Orion | Macro | Tokens/image | Seconds/image | Failures |
|---|---:|---:|---:|---:|---:|---:|
| **Names + numeric boxes x2** | **53.11 / 76.06** | 16.89 / **41.10** | **35.00 / 58.58** | 18,797 | 11.89 | 0 |
| Names + drawn boxes x2 | 51.53 / 74.30 | 10.75 / 33.07 | 31.14 / 53.69 | 18,506 | 12.71 | 0 |
| **Class names only** | 43.22 / 57.50 | **18.77 / 33.76** | 30.99 / 45.63 | **1,816** | **10.17** | 0 |
| Anonymous + numeric boxes x2 | 36.47 / 60.68 | 2.56 / 8.54 | 19.51 / 34.61 | 22,780 | 55.05 | 21 |

Scores are **mAP50-95 / mAP50**. Numeric boxes gain 4.00 macro mAP and
12.95 macro mAP50 over names only, but use 10.35x the tokens and are 17%
slower. This is why accuracy and cost recommendations are separate.

## Evidence standard

| Dataset | Test images | Classes | Objects | mAP noise threshold | mAP50 noise threshold |
|---|---:|---:|---:|---:|---:|
| Dreidel | 54 | 6 | 171 | 3.31 | 5.44 |
| Orion | 59 | 8 | 555 | 4.92 | 9.81 |

The noise thresholds come from ten identical full-test repeats at
`temperature=0`, seed `1234`, and reasoning off. No image produced identical
predictions across all ten repeats. Differences inside a threshold are treated
as ties.

Locked finalists used the complete test sets, `temperature=0`, seed `1234`,
8,192 completion tokens, and a 180-second deadline. Train supplies references;
test supplies only target images and requested class names. COCO scoring uses
pycocotools with `maxDets=[1, 10, 500]`. Terminal model failures receive no
detections rather than being omitted.

Earlier screens used provider-default temperature. They are valid for matched
within-screen comparisons but are not mixed causally with locked scores.

## Decision path

### 1. Disable reasoning

Low reasoning had to exceed each dataset's noise floor on the same 20-image
subset before medium reasoning would be tested.

| Prompt arm | Dreidel low-minus-none | Orion low-minus-none |
|---|---:|---:|
| Names + two drawn boxes | **-5.98 / -8.50** | +4.14 / +6.84 |
| Names only | **-7.56 / -10.63** | -0.92 / +2.98 |

Low reasoning materially hurt Dreidel, did not clear Orion's 4.92 mAP floor,
and more than doubled mean latency overall. It failed the gate, so medium and
higher levels were correctly skipped. **Use no reasoning.**

### 2. Use positive box examples for accuracy

Matched multi-class numeric screens:

| Boxes/class | Dreidel | Orion |
|---:|---:|---:|
| 0, names only | 40.93 / 54.86 | 10.47 / 19.90 |
| 1 | 49.61 / 70.66 | **18.19 / 43.03** |
| 2 | **54.78 / 77.12** | 13.85 / 36.91 |
| 3 | 50.96 / 75.22 | 14.52 / 33.25 |
| 5 | 52.02 / 78.28 | 12.11 / 32.13 |
| 10 | 51.29 / 76.67 | 12.60 / 34.07 |

Boxes helped both datasets, but gains were non-monotonic. Names only remain
useful because the locked prompt uses one tenth the tokens.

### 3. Use two examples per class

- Dreidel: two beat one by 5.17 mAP, above its 3.31 noise floor.
- Orion: one beat two by 4.34 mAP, inside its 4.92 floor.
- Three, five, and ten did not reliably beat two, while tokens and latency rose
  sharply; Dreidel multi-class drawn latency increased from 10.30 seconds at
  one box to 35.48 seconds at ten.

Two is therefore the smallest shared count that preserves Dreidel's material
gain without a demonstrated Orion loss. One remains plausible if optimizing
Orion alone.

### 4. Prefer numeric coordinates over drawn boxes

The locked numeric-versus-drawn comparison appears in the opening table.
Paired-image bootstrap of drawn relative to numeric found:

- Dreidel: -1.64 mAP, 95% CI `[-4.96, +1.80]`; tied.
- Orion: -6.82 mAP, 95% CI `[-10.72, -3.28]`; numeric wins.

Numeric was also 0.82 seconds faster per target, with nearly identical token
use. **Use numeric coordinates.**

### 5. Use one multi-class request for a shared recipe

Matched two-example full-set screens showed a genuine dataset reversal:

| Dataset | Encoding | Multi-class | Single-class merged | Winner |
|---|---|---:|---:|---|
| Dreidel | Numeric | **54.78 / 77.12** | 31.63 / 47.88 | Multi +23.15 mAP |
| Dreidel | Drawn | **54.06 / 78.08** | 36.10 / 53.16 | Multi +17.96 mAP |
| Orion | Numeric | 13.85 / 36.91 | **22.76 / 49.03** | Single +8.92 mAP |
| Orion | Drawn | 12.49 / 34.24 | **18.80 / 44.07** | Single +6.31 mAP |

Multi-class is not universally better. It is the shared choice because its
two-dataset numeric-screen macro was 34.31 mAP versus 27.20 for single-class,
and it uses one call instead of six on Dreidel or eight on Orion.

Limitation: single-class completed full-test screens but was not rerun as a
temperature-zero locked finalist. The evidence supports multi-class as the
shared macro/efficiency recipe, not as every dataset's winner.

### 6. Use positive-only references

Complete Orion per-class screening:

| Encoding | Positive only | Positive + negative |
|---|---:|---:|
| Numeric | **23.46 / 49.50** | 22.21 / 47.33 |
| Drawn | **21.56 / 47.18** | 16.77 / 42.29 |

Negative examples added images and complexity without improving primary mAP,
so they were not advanced.

### 7. Supply real class names

Anonymous prompts replaced names with `Concept A/B/...`. Explicit prompts said
to find objects of the same kind as the marked reference; minimal prompts gave
only references and an output schema.

| Dreidel formulation | Encoding/count | Explicit | Minimal |
|---|---|---:|---:|
| Single-class | Numeric x2 | **24.03 / 34.55** | 19.11 / 27.47 |
| Single-class | Drawn x5 | **36.82 / 58.08** | 24.70 / 36.92 |
| Multi-class | Numeric x2 | **40.34 / 62.71** | 13.77 / 18.00 |
| Multi-class | Drawn x2 | **33.94 / 57.11** | 20.02 / 31.25 |

Cross-image anonymous transfer works, but explicit same-kind wording is needed
for competitive performance. In the locked final, real names beat explicit
anonymous numeric boxes by 15.48 macro mAP, were 4.6x faster, and avoided 21
model failures. **Use real names whenever available.**

### 8. Do not self-generate class names

Matched 20-image Dreidel screen:

| Semantics/formulation | Score |
|---|---:|
| Real names + two drawn boxes, multi | **46.86 / 71.53** |
| Real names only, multi | 40.40 / 57.14 |
| Generated names only, multi | 27.37 / 40.44 |
| Generated names + boxes, multi | 26.11 / 43.40 |
| Generated names + boxes, single | 25.63 / 40.74 |

Generated labels were plausible but insufficient for fine-grained classes.
No self-name arm reached the noise-aware advancement margin.

## Cross-dataset interpretation

- **Dreidel:** box prompting gives a clear primary-mAP improvement.
- **Orion:** names only have the highest locked primary mAP, but numeric boxes
  are within the wide noise floor and have a 7.34-point higher mAP50 estimate.
- **Lacrosse supporting screen:** names only won primary mAP, while the one-call
  numeric recipe remained competitive. Across the earlier Orion/Lacrosse/
  Dreidel screen, combined numeric boxes had the strongest macro score at
  **31.58 / 53.48**. This used an earlier prompt version and is supporting, not
  locked, evidence.

The selected recipe is therefore the best shared locked finalist, not a claim
that every individual dataset prefers box prompting.

## Decision summary

| Axis | Decision | Evidence status |
|---|---|---|
| Multi vs single | Multi for shared macro/efficiency; Orion can prefer single | Complete screens; single not locked-final rerun |
| Reasoning | None | Locked two-dataset gate |
| Examples/class | Two | Full count screens + noise/cost rule |
| Numeric vs drawn | Numeric | Locked final + bootstrap |
| Names only vs boxes | Boxes for accuracy; names only for cost | Locked final |
| Positive vs positive/negative | Positive only | Complete Orion screen |
| Minimal box-only prompt | Reject; explicit same-kind wording is stronger | Complete Dreidel anonymous screens |
| Anonymous vs real names | Real names | Locked final |
| Self-generated names | Reject when real names exist | Matched causal screen |
| Sampling controls | Temperature 0, seed 1234; still stochastic | Ten full repeats/dataset |

## Limits and most useful follow-up

The result does not prove that multi-class or two examples are optimal for
every RF20/RF100 dataset, nor that the measured names-only noise floor is
identical for every prompt family.

The next major step is a locked comparison on several more preselected
RF100-VL datasets. The genuinely unsettled choices are one versus two examples
and multi-class versus single-class inference. The compact comparison is:

1. names only, multi-class;
2. names + numeric boxes x1, multi-class;
3. names + numeric boxes x2, multi-class;
4. names + numeric boxes x1, single-class merged;
5. names + numeric boxes x2, single-class merged; and
6. names + drawn boxes x2, multi-class, as a modality control.

These arms retain the cost baseline and resolve the comparisons that were
close or changed direction by dataset. The drawn-box control is included
because it tied numeric boxes within noise on Dreidel but lost clearly on
Orion. There is no current reason to reopen reasoning, self-naming, anonymous,
negative-example, or high-box-count branches.

## Sources

- Full chronology: `QWEN38_RF100VL_RESEARCH_LOG.md`
- Final scores/bootstrap:
  `qwen38-fsod-runs/final-recipe-study/final-analysis/final_report.json`
- Noise floor: `qwen38-fsod-runs/final-recipe-study/noise_floor.json`
- Reasoning gate:
  `qwen38-fsod-runs/final-recipe-study/reasoning_low_decision.json`
- Named box screens: `qwen38-fsod-runs/dreidel-box-count-ablation-v1/` and
  `qwen38-fsod-runs/orion-box-count-ablation-v1/`
- Anonymous screens:
  `qwen38-fsod-runs/dreidel-exemplar-only-box-combined-v1/` and
  `qwen38-fsod-runs/dreidel-anonymous-multi-screen-v1/`
