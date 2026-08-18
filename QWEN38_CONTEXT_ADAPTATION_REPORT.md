# Qwen3.8-Max uses visual references as dataset adaptation

> **Finding:** Qwen3.8-Max does not benefit reliably from adding more context
> everywhere. Visual references provide substantial, above-noise gains when
> class names under-specify appearance, state, or annotation semantics, but
> offer little benefit and can hurt when names already identify familiar
> objects. Instructions are understood, but visual examples generally transfer
> dataset-specific appearance more effectively.

We evaluated one multi-class request per test image on all 20 RF20-VL-FSOD
datasets: 3,970 images, reasoning off, temperature 0, fixed seed, normalized
0–1000 XYXY reference boxes, and COCO `maxDets=500` scoring.

## Shareable figures

[![Shareable Qwen3.8-Max context-adaptation summary](figures/qwen38_context_adaptation/shareable_context_adaptation_summary.png)](figures/qwen38_context_adaptation/shareable_context_adaptation_summary.png)

| Visual references helped | Visual references hurt or failed |
|---|---|
| [![Examples where visual references helped](figures/qwen38_context_adaptation/shareable_visual_references_helped.png)](figures/qwen38_context_adaptation/shareable_visual_references_helped.png) | [![Examples where visual references hurt or failed](figures/qwen38_context_adaptation/shareable_visual_references_hurt.png)](figures/qwen38_context_adaptation/shareable_visual_references_hurt.png) |

## Complete RF20 result

Scores are dataset-macro `mAP50–95 / mAP50`.

| Context given to Qwen3.8-Max | mAP50–95 | mAP50 | Gain vs. names | Estimated cost |
|---|---:|---:|---:|---:|
| Class names only | 24.37 | 43.54 | baseline | **$22.28** |
| Annotator instructions | 24.46 | 44.58 | +0.09 / +1.04 | $26.21 |
| 1 positive visual reference/class | 25.35 | 46.73 | +0.98 / +3.19 | $34.20 |
| 2 positive visual references/class | 25.14 | 46.55 | +0.77 / +3.01 | $44.43 |
| 10 positive visual references/class | **25.74** | **47.92** | **+1.37 / +4.38** | $116.79 |

Using the three repeated larger datasets as a noise proxy, instructions are
within noise, one and two references are borderline in the RF20 macro, and ten
references reach roughly the upper noise boundary only for mAP50. The global
average therefore hides the important behavior: **the direction and size of the
effect depend strongly on the dataset.**

![Visual-reference gain versus label insufficiency](figures/qwen38_context_adaptation/label_insufficiency_vs_visual_gain.png)

The score-blind class-name ratings were locked before the instruction study was
scored. Across datasets, the fraction of under-specified labels strongly
predicts visual-reference gain:

| Pre-rated dataset property | 1 reference | 2 references | 10 references |
|---|---:|---:|---:|
| Fraction of class names judged insufficient | **0.80 / 0.79** | **0.70 / 0.77** | **0.56 / 0.69** |
| Fraction requiring state or scene context | 0.48 / 0.62 | 0.43 / 0.55 | 0.49 / 0.54 |

Values are Spearman correlations with `mAP50–95 / mAP50` gain. Every
leave-one-dataset-out correlation remains positive. For one reference,
controlling for class-names-only accuracy, image count, and class count leaves
correlations of **0.83 / 0.75**.

At class level, the separation is similarly large:

| Visual context | Under-specified classes | Classes with sufficient names | Difference |
|---|---:|---:|---:|
| 1 reference | +6.68 / +12.68 | −2.22 / −1.84 | **+8.90 / +14.52** |
| 10 references | +6.57 / +13.88 | −3.89 / −4.97 | **+10.46 / +18.85** |

The mAP50 differences are outside the dataset-clustered 95% intervals. The
mAP50–95 evidence is weaker, suggesting that references help concept
recognition more consistently than precise box localization.

## Where visual adaptation helped

These cards use actual RF20 train and test images. Yellow is one of the boxed
training exemplars in the ten-reference prompt. Each target image is shown with
the same green ground truth and either names-only predictions in pink or
visual-prompt predictions in blue.

<table>
<tr>
<td><img src="figures/qwen38_context_adaptation/under_specified_helped__actions__serve.png" alt="Actions Serve visual example"></td>
<td><img src="figures/qwen38_context_adaptation/under_specified_helped__defect-detection__defective-fishplate.png" alt="Defective fishplate visual example"></td>
</tr>
<tr>
<td><img src="figures/qwen38_context_adaptation/under_specified_helped__new-defects-in-wood__knot-with-crack.png" alt="Knot with crack visual example"></td>
<td><img src="figures/qwen38_context_adaptation/under_specified_helped__orionproducts__marine-boy.png" alt="Marine Boy visual example"></td>
</tr>
<tr>
<td><img src="figures/qwen38_context_adaptation/under_specified_helped__paper-parts__table-of-contents-text.png" alt="Table of contents text visual example"></td>
<td><img src="figures/qwen38_context_adaptation/under_specified_helped__the-dreidel-project__spinning-dreidel.png" alt="Spinning Dreidel visual example"></td>
</tr>
<tr>
<td><img src="figures/qwen38_context_adaptation/under_specified_helped__wb-prova__juvenile.png" alt="Juvenile animal visual example"></td>
<td><img src="figures/qwen38_context_adaptation/under_specified_helped__wildfire-smoke__smoke.png" alt="Smoke visual example"></td>
</tr>
</table>

The largest dataset-level ten-reference gains align with dataset-specific
concepts: animal age groups **+17.51 / +24.32**, railway defects
**+11.25 / +18.85**, Dreidel states and symbols **+10.71 / +22.26**, document
components **+9.73 / +16.13**, and smoke **+5.06 / +18.90**.

## Where familiar labels were flat or hurt

<table>
<tr>
<td><img src="figures/qwen38_context_adaptation/familiar_hurt__water-meter__4.png" alt="Digit visual-reference regression"></td>
<td><img src="figures/qwen38_context_adaptation/familiar_hurt__flir-camera-objects__car.png" alt="Car visual-reference regression"></td>
</tr>
<tr>
<td><img src="figures/qwen38_context_adaptation/familiar_hurt__flir-camera-objects__dog.png" alt="Dog visual-reference regression"></td>
<td><img src="figures/qwen38_context_adaptation/familiar_hurt__aquarium-combined__fish.png" alt="Fish visual-reference regression"></td>
</tr>
<tr>
<td><img src="figures/qwen38_context_adaptation/familiar_hurt__aquarium-combined__penguin.png" alt="Penguin visual-reference regression"></td>
<td><img src="figures/qwen38_context_adaptation/familiar_hurt__gwhd2021__wheat-head.png" alt="Wheat Head visual-reference regression"></td>
</tr>
</table>

Ten references regressed on digits **−20.57 / −41.40**, thermal-camera objects
**−3.85 / −7.24**, fish and other aquarium animals **−3.32 / −0.34**, and wheat
heads **−3.12 / −4.54**. The corresponding labels already describe recognizable
object identities, leaving less useful ambiguity for an exemplar to resolve.

## The rule is predictive, not absolute

All Elements is the clearest counterexample. Its UI labels can require state or
annotation semantics, yet ten references regressed by **−12.39 / −12.77**.

<p align="center">
<img src="figures/qwen38_context_adaptation/counterexample__all-elements__checked-radio-button.png" width="58%" alt="All Elements counterexample">
</p>

Within datasets containing both simple and under-specified classes, class-name
sufficiency alone did not consistently identify the winning class. The finding
therefore supports **dataset-level routing**, not an automatic per-class rule.
More examples are also not monotonically better: ten references are only
+0.40 / +1.20 above one reference in the RF20 macro despite costing over 3× as
much.

This is strong exploratory evidence, not a prospective causal result. The
ratings themselves were assigned without viewing scores, but the hypothesis was
motivated by earlier visual-prompt results. A new semantic-control run is needed
for confirmation.

## Noise and instruction controls

The repeated larger-dataset study confirms that visual gains are real on some
datasets rather than API variance.

| Dataset | Measured names-only noise | Instructions | 1 reference after repeats | 2 references | 10 references |
|---|---:|---:|---:|---:|---:|
| Actions | 0.49 / 1.19 | inconclusive / within | **+2.22 / +4.98, outside** | **+2.55 / +5.35, outside** | **+2.02 / +4.57, outside**<sup>†</sup> |
| Paper Parts | 0.95 / 3.87 | within | **+13.43 / +20.07, outside** | **+9.89 / +16.10, outside** | **+9.73 / +16.13, outside**<sup>†</sup> |
| Defect Detection | 2.17 / 2.89 | **outside** | +2.03 / +3.60, within paired noise | **+7.40 / +10.97, outside** | **+11.25 / +18.85, outside** |

<sup>†</sup> Actions and Paper Parts were not independently repeated at exactly ten
references; those changes exceed names-only noise, while Defect Detection's
ten-reference result was directly repeated.

Instructions are not ignored. In the six-dataset semantic control, correct
instructions beat strictly class-shuffled instructions by **+4.38 / +8.25**.
The strict arm had 12 terminal model-output failures, but the gap remains
**+4.48 / +8.47** after excluding Actions, which contained 10 of them. Correct
instructions nevertheless failed as a universal accuracy addition:

| Six-dataset matched arm | mAP50–95 | mAP50 |
|---|---:|---:|
| Class names only | **22.92** | **42.74** |
| Correct instructions | 19.95 | 38.61 |
| 1 visual reference | 21.62 | 40.73 |
| Instructions + 1 visual reference | 19.33 | 37.44 |
| Conservative shuffled instructions | 19.91 | 38.77 |
| Strictly shuffled instructions | 15.57 | 30.36 |

Ten references are more selectively useful than instructions for
under-specified classes. Their advantage over instruction gains is
**+5.79 / +10.43** for insufficient names and **+7.21 / +11.28** for classes
requiring state or context, with dataset-clustered intervals above zero. The
same one-reference comparison trends positive but remains within uncertainty.

## Decision

- Keep **class names only** as the cheap, robust default.
- Use **one boxed visual reference per class** as an optional dataset-adaptation
  mode when labels are opaque, state-dependent, or visually dataset-specific.
- Do not add instructions or ten references indiscriminately.
- Treat label sufficiency as a dataset-level routing signal, not a guarantee.
- The clean prospective confirmation is a matched correct-reference versus
  class-shuffled-reference control using the already locked sufficiency ratings.

<details>
<summary><strong>All 20 per-dataset deltas versus class names only</strong></summary>

Each cell is `mAP50–95 / mAP50`.

| Dataset | Instructions | 1 visual | 2 visual | 10 visual |
|---|---:|---:|---:|---:|
| Actions | +0.83 / +0.96 | +2.11 / +3.55 | +2.55 / +5.35 | +2.02 / +4.57 |
| Aerial Airport | −3.95 / −7.69 | −0.14 / −2.01 | +2.48 / +2.04 | +3.41 / +2.06 |
| All Elements | −6.47 / −0.47 | −7.59 / −0.65 | −16.79 / −17.63 | −12.39 / −12.77 |
| Aquarium Combined | −2.93 / −2.02 | −3.14 / −0.42 | −3.60 / −2.35 | −3.32 / −0.34 |
| Defect Detection | +8.60 / +13.22 | +4.16 / +6.35 | +7.40 / +10.97 | +11.25 / +18.85 |
| DentalAI | −0.65 / −1.95 | +0.45 / +1.04 | +1.09 / +1.47 | +0.28 / +0.85 |
| FLIR Camera Objects | −1.84 / −4.71 | +0.12 / −1.68 | +0.05 / +0.09 | −3.85 / −7.24 |
| Global Wheat Head | −3.56 / −4.33 | −2.69 / −3.64 | −3.16 / −3.99 | −3.12 / −4.54 |
| Lacrosse Object Detection | +1.46 / +2.93 | +0.55 / +4.18 | +0.40 / +4.92 | −0.49 / +5.75 |
| New Defects in Wood | +1.90 / +7.35 | +1.63 / +3.84 | +1.02 / +4.46 | +1.22 / +4.54 |
| Orion Products | +0.51 / +7.88 | +3.97 / +18.25 | +3.01 / +14.51 | +1.84 / +15.11 |
| Paper Parts | −0.50 / −0.04 | +13.60 / +19.25 | +9.89 / +16.10 | +9.73 / +16.13 |
| Recode Waste | +4.09 / +7.49 | +1.22 / +3.07 | +0.86 / +3.31 | +3.96 / +6.68 |
| Soda Bottles | +0.12 / −0.35 | +0.84 / +0.70 | +1.57 / +2.06 | +2.01 / +3.37 |
| The Dreidel Project | +15.50 / +19.89 | +9.66 / +17.39 | +14.55 / +23.30 | +10.71 / +22.26 |
| Trail Camera | +0.94 / −1.00 | −1.67 / −0.27 | −1.04 / +0.95 | −0.43 / +0.91 |
| Water Meter | −15.00 / −26.63 | −16.55 / −29.58 | −19.42 / −37.85 | −20.57 / −41.40 |
| WB Prova | +1.68 / +0.62 | +12.32 / +16.12 | +11.86 / +16.63 | +17.51 / +24.32 |
| Wildfire Smoke | +0.88 / +8.90 | +0.11 / +4.69 | +0.90 / +9.05 | +5.06 / +18.90 |
| X-ray ID | +0.10 / +0.74 | +0.61 / +3.53 | +1.73 / +6.74 | +2.62 / +9.62 |

</details>

### Reproducibility

- Quantitative source: [`qwen38-fsod-runs/instruction-study-v2/analysis/rf20_per_dataset.csv`](qwen38-fsod-runs/instruction-study-v2/analysis/rf20_per_dataset.csv)
- Class-level source: [`qwen38-fsod-runs/instruction-study-v2/analysis/rf20_per_class.csv`](qwen38-fsod-runs/instruction-study-v2/analysis/rf20_per_class.csv)
- Noise study: [`QWEN38_LARGE_DATASET_NOISE_RESULT.md`](QWEN38_LARGE_DATASET_NOISE_RESULT.md)
- Figure generator: [`analysis/generate_qwen38_context_adaptation_figures.py`](analysis/generate_qwen38_context_adaptation_figures.py)
- Exact illustrated image/reference IDs: [`figures/qwen38_context_adaptation/selection_manifest.json`](figures/qwen38_context_adaptation/selection_manifest.json)

Regenerate the visual evidence with Pillow installed:

```bash
python analysis/generate_qwen38_context_adaptation_figures.py
```

The qualitative cards intentionally choose the largest per-image F1@0.5
improvement or regression within each fixed dataset/class pair so the mechanism
is visible. They are illustrations, not the statistical evidence; all reported
claims come from the complete COCO evaluations above.
