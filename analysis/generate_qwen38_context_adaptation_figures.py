#!/usr/bin/env python3
"""Generate reproducible visual evidence for the Qwen3.8 context study.

The qualitative cards deliberately select the largest per-image F1@0.5 change
for a predeclared dataset/class pair. They explain a measured behavior; aggregate
and per-class COCO metrics in the report remain the quantitative evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "RF100VL/rf20-vl-fsod-fresh-20260813"
NAMES_ROOT = REPO / "qwen38-fsod-runs/rf20-three-way-matched-v1"
VISUAL_ROOT = REPO / "qwen38-fsod-runs/rf20-all-available-explicit-sparse-v1"
CLASS_RESULTS = (
    REPO
    / "qwen38-fsod-runs/instruction-study-v2/analysis/rf20_per_class.csv"
)
DATASET_RESULTS = (
    REPO
    / "qwen38-fsod-runs/instruction-study-v2/analysis/rf20_per_dataset.csv"
)
OUTPUT_ROOT = REPO / "figures/qwen38_context_adaptation"

NAMES_MODE = "names_multi"
VISUAL_MODE = "numeric_prediction_all_available_multi_explicit_sparse"

CASES = {
    "under_specified_helped": (
        ("actions", "Serve"),
        ("defect-detection", "defective fishplate"),
        ("new-defects-in-wood", "knot with crack"),
        ("orionproducts", "Marine Boy"),
        ("paper-parts", "table of contents text"),
        ("the-dreidel-project", "Spinning Dreidel"),
        ("wb-prova", "Juvenile"),
        ("wildfire-smoke", "smoke"),
    ),
    "familiar_hurt": (
        ("water-meter", "4"),
        ("flir-camera-objects", "car"),
        ("flir-camera-objects", "dog"),
        ("aquarium-combined", "fish"),
        ("aquarium-combined", "penguin"),
        ("gwhd2021", "Wheat Head"),
    ),
    "counterexample": (
        ("all-elements", "Checked Radio button"),
    ),
}

COLORS = {
    "background": "#0f172a",
    "panel": "#111c33",
    "text": "#f8fafc",
    "muted": "#cbd5e1",
    "gt": "#00d084",
    "names": "#ff4d6d",
    "visual": "#00a8ff",
    "prompt": "#ffc857",
    "positive": "#5ee6a8",
    "negative": "#ff8ba1",
    "grid": "#475569",
}

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def xywh_to_xyxy(box: Iterable[float]) -> tuple[float, float, float, float]:
    x, y, width, height = map(float, box)
    return x, y, x + width, y + height


def iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def f1_at_50(
    ground_truth: list[tuple[float, float, float, float]],
    predictions: list[tuple[float, float, float, float]],
) -> float:
    candidates = sorted(
        (
            (iou(gt_box, pred_box), gt_index, pred_index)
            for gt_index, gt_box in enumerate(ground_truth)
            for pred_index, pred_box in enumerate(predictions)
        ),
        reverse=True,
    )
    matched_gt: set[int] = set()
    matched_predictions: set[int] = set()
    for overlap, gt_index, pred_index in candidates:
        if overlap < 0.5:
            break
        if gt_index in matched_gt or pred_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)
    true_positive = len(matched_gt)
    false_positive = len(predictions) - true_positive
    false_negative = len(ground_truth) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 1.0


def group_boxes(
    records: Iterable[dict[str, Any]], category_id: int
) -> dict[int, list[tuple[float, float, float, float]]]:
    grouped: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for record in records:
        if int(record["category_id"]) != category_id:
            continue
        grouped[int(record["image_id"])].append(xywh_to_xyxy(record["bbox"]))
    return grouped


def expanded_crop(
    boxes: list[tuple[float, float, float, float]],
    image_size: tuple[int, int],
    *,
    expansion: float,
    minimum_fraction: float,
) -> tuple[float, float, float, float]:
    width, height = image_size
    if not boxes:
        return 0.0, 0.0, float(width), float(height)
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    crop_width = max((right - left) * expansion, width * minimum_fraction)
    crop_height = max((bottom - top) * expansion, height * minimum_fraction)
    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)
    left = max(0.0, min(width - crop_width, center_x - crop_width / 2))
    top = max(0.0, min(height - crop_height, center_y - crop_height / 2))
    return left, top, left + crop_width, top + crop_height


def fit_scene(
    source: Image.Image,
    crop: tuple[float, float, float, float],
    viewport: tuple[int, int],
) -> tuple[Image.Image, float, float, float]:
    cropped = source.crop(tuple(map(round, crop)))
    contained = ImageOps.contain(cropped, viewport, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", viewport, "#020617")
    offset_x = (viewport[0] - contained.width) / 2
    offset_y = (viewport[1] - contained.height) / 2
    canvas.paste(contained, (round(offset_x), round(offset_y)))
    scale = min(viewport[0] / cropped.width, viewport[1] / cropped.height)
    return canvas, scale, offset_x, offset_y


def draw_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    label: str,
    color: str,
    *,
    size: int = 17,
) -> None:
    label_font = font(size, bold=True)
    left, top = position
    box = draw.textbbox((left, top), label, font=label_font, stroke_width=0)
    padding = 4
    canvas_width = draw._image.width  # Pillow exposes the destination image here.
    if box[2] + padding > canvas_width:
        left = max(padding, canvas_width - (box[2] - box[0]) - 2 * padding)
        box = draw.textbbox((left, top), label, font=label_font, stroke_width=0)
    draw.rectangle(
        (
            box[0] - padding,
            box[1] - padding,
            box[2] + padding,
            box[3] + padding,
        ),
        fill=color,
    )
    draw.text((left, top), label, fill="#07111f", font=label_font)


def draw_scene(
    source: Image.Image,
    crop: tuple[float, float, float, float],
    viewport: tuple[int, int],
    groups: list[tuple[str, str, list[tuple[float, float, float, float]]]],
) -> Image.Image:
    scene, scale, offset_x, offset_y = fit_scene(source, crop, viewport)
    draw = ImageDraw.Draw(scene)
    crop_left, crop_top, _, _ = crop
    line_width = max(3, round(min(viewport) / 110))
    for label, color, boxes in groups:
        first_visible = True
        for box in boxes:
            transformed = (
                (box[0] - crop_left) * scale + offset_x,
                (box[1] - crop_top) * scale + offset_y,
                (box[2] - crop_left) * scale + offset_x,
                (box[3] - crop_top) * scale + offset_y,
            )
            draw.rectangle(transformed, outline=color, width=line_width)
            if first_visible:
                label_y = max(3.0, transformed[1] - 25)
                draw_label(draw, (max(3.0, transformed[0]), label_y), label, color)
                first_visible = False
    return scene


def best_image(
    image_ids: Iterable[int],
    ground_truth: dict[int, list[tuple[float, float, float, float]]],
    names: dict[int, list[tuple[float, float, float, float]]],
    visual: dict[int, list[tuple[float, float, float, float]]],
    direction: str,
) -> tuple[int, float, float]:
    candidates = []
    for image_id in image_ids:
        gt_boxes = ground_truth.get(image_id, [])
        if not gt_boxes:
            continue
        names_f1 = f1_at_50(gt_boxes, names.get(image_id, []))
        visual_f1 = f1_at_50(gt_boxes, visual.get(image_id, []))
        # Prefer readable examples when the effect is otherwise tied.
        clutter = len(gt_boxes) + len(names.get(image_id, [])) + len(
            visual.get(image_id, [])
        )
        candidates.append((visual_f1 - names_f1, -clutter, image_id, names_f1, visual_f1))
    if not candidates:
        raise ValueError("No test image contains the selected class.")
    selected = max(candidates) if direction == "positive" else min(candidates)
    return selected[2], selected[3], selected[4]


def reference_for_class(
    manifest: dict[str, Any], category_id: int
) -> dict[str, Any]:
    references = manifest["reference_selection"]["classes"][str(category_id)]
    if not references:
        raise ValueError(f"No references for category {category_id}.")
    return references[0]


def slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-")


def pretty_dataset(value: str) -> str:
    aliases = {
        "gwhd2021": "Global Wheat Head",
        "orionproducts": "Orion Products",
        "wb-prova": "WB Prova",
    }
    return aliases.get(value, value.replace("-", " ").title())


def class_result_index() -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["dataset"], row["class_name"]): row
        for row in read_csv(CLASS_RESULTS)
    }


def generate_card(
    dataset: str,
    class_name: str,
    group: str,
    class_results: dict[tuple[str, str], dict[str, str]],
    output_root: Path,
) -> dict[str, Any]:
    annotation_path = DATA_ROOT / dataset / "test/_annotations.coco.json"
    annotations = read_json(annotation_path)
    categories = {str(item["name"]): int(item["id"]) for item in annotations["categories"]}
    if class_name not in categories:
        raise KeyError(f"{dataset} has no class {class_name!r}.")
    category_id = categories[class_name]
    images = {int(item["id"]): item for item in annotations["images"]}
    gt = group_boxes(annotations["annotations"], category_id)
    names_predictions = read_json(
        NAMES_ROOT / dataset / "predictions" / f"{NAMES_MODE}.json"
    )
    visual_predictions = read_json(
        VISUAL_ROOT / dataset / "predictions" / f"{VISUAL_MODE}.json"
    )
    names = group_boxes(names_predictions, category_id)
    visual = group_boxes(visual_predictions, category_id)
    direction = "negative" if group in {"familiar_hurt", "counterexample"} else "positive"
    image_id, names_f1, visual_f1 = best_image(images, gt, names, visual, direction)
    image_record = images[image_id]
    target_path = DATA_ROOT / dataset / "test" / image_record["file_name"]
    target = Image.open(target_path).convert("RGB")
    target_boxes = gt[image_id] + names.get(image_id, []) + visual.get(image_id, [])
    target_crop = expanded_crop(
        target_boxes,
        target.size,
        expansion=1.22,
        minimum_fraction=0.42,
    )

    manifest = read_json(VISUAL_ROOT / dataset / "run_manifest.json")
    reference = reference_for_class(manifest, category_id)
    reference_path = DATA_ROOT / dataset / "train" / reference["file_name"]
    with reference_path.open("rb") as handle:
        reference_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    if reference_sha256 != reference["source_sha256"]:
        raise ValueError(f"Reference image hash changed: {reference_path}")
    reference_image = Image.open(reference_path).convert("RGB")
    reference_box_1000 = reference["bbox_xyxy_1000"]
    reference_box = (
        reference_box_1000[0] * reference_image.width / 1000,
        reference_box_1000[1] * reference_image.height / 1000,
        reference_box_1000[2] * reference_image.width / 1000,
        reference_box_1000[3] * reference_image.height / 1000,
    )
    reference_crop = expanded_crop(
        [reference_box],
        reference_image.size,
        expansion=3.2,
        minimum_fraction=0.32,
    )

    card_width, card_height = 920, 820
    card = Image.new("RGB", (card_width, card_height), COLORS["background"])
    draw = ImageDraw.Draw(card)
    draw.text(
        (28, 20),
        f"{pretty_dataset(dataset)}  •  {class_name}",
        fill=COLORS["text"],
        font=font(29, bold=True),
    )
    draw.text(
        (28, 60),
        "One of the boxed training references used by the 10-reference prompt",
        fill=COLORS["muted"],
        font=font(17),
    )
    reference_scene = draw_scene(
        reference_image,
        reference_crop,
        (520, 205),
        [(f"prompt: {class_name}", COLORS["prompt"], [reference_box])],
    )
    card.paste(reference_scene, (200, 88))

    draw.text((40, 310), "Class names only", fill=COLORS["text"], font=font(20, bold=True))
    draw.text((505, 310), "10 visual references", fill=COLORS["text"], font=font(20, bold=True))
    names_scene = draw_scene(
        target,
        target_crop,
        (420, 315),
        [
            (f"GT: {class_name}", COLORS["gt"], gt[image_id]),
            (f"Pred: {class_name}", COLORS["names"], names.get(image_id, [])),
        ],
    )
    visual_scene = draw_scene(
        target,
        target_crop,
        (420, 315),
        [
            (f"GT: {class_name}", COLORS["gt"], gt[image_id]),
            (f"Pred: {class_name}", COLORS["visual"], visual.get(image_id, [])),
        ],
    )
    card.paste(names_scene, (30, 342))
    card.paste(visual_scene, (470, 342))

    result = class_results[(dataset, class_name)]
    names_ap50 = float(result["names_mAP50"])
    visual_ap50 = float(result["ten_references_mAP50"])
    delta = visual_ap50 - names_ap50
    names_ap = float(result["names_mAP50_95"])
    visual_ap = float(result["ten_references_mAP50_95"])
    delta_ap = visual_ap - names_ap
    delta_color = COLORS["positive"] if delta >= 0 else COLORS["negative"]
    draw.text(
        (30, 676),
        f"Class AP50: {names_ap50:.1f} → {visual_ap50:.1f}",
        fill=COLORS["text"],
        font=font(22, bold=True),
    )
    draw.text(
        (397, 676),
        f"Δ {delta:+.1f}",
        fill=delta_color,
        font=font(22, bold=True),
    )
    draw.text(
        (30, 710),
        f"Class AP50–95: {names_ap:.1f} → {visual_ap:.1f}  (Δ {delta_ap:+.1f})",
        fill=COLORS["text"],
        font=font(18, bold=True),
    )
    draw.text(
        (30, 742),
        f"Illustrated image F1@0.5: {names_f1:.2f} → {visual_f1:.2f}  •  "
        f"GT {len(gt[image_id])}, names {len(names.get(image_id, []))}, "
        f"visual {len(visual.get(image_id, []))}",
        fill=COLORS["muted"],
        font=font(17),
    )
    draw.text(
        (30, 775),
        "Green = ground truth   Pink = names-only prediction   Blue = visual-prompt prediction",
        fill=COLORS["muted"],
        font=font(15),
    )

    output_path = output_root / f"{group}__{slug(dataset)}__{slug(class_name)}.png"
    card.save(output_path, optimize=True)
    return {
        "group": group,
        "dataset": dataset,
        "class_name": class_name,
        "category_id": category_id,
        "target_image_id": image_id,
        "target_file_name": image_record["file_name"],
        "reference_image_id": int(reference["image_id"]),
        "reference_annotation_id": int(reference["annotation_id"]),
        "reference_file_name": reference["file_name"],
        "selection": f"largest per-image F1@0.5 {direction} change",
        "names_image_f1_50": names_f1,
        "visual_image_f1_50": visual_f1,
        "names_class_AP50": names_ap50,
        "visual_class_AP50": visual_ap50,
        "class_AP50_delta": delta,
        "names_class_AP50_95": names_ap,
        "visual_class_AP50_95": visual_ap,
        "class_AP50_95_delta": delta_ap,
        "figure": str(output_path.relative_to(REPO)),
    }


def draw_grouped_gain_chart(output_root: Path) -> None:
    class_rows = read_csv(CLASS_RESULTS)
    dataset_rows = read_csv(DATASET_RESULTS)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in class_rows:
        grouped[row["dataset"]].append(row)

    dataset_groups: dict[str, str] = {}
    for dataset, rows in grouped.items():
        fraction = fmean(int(row["name_alone_insufficient"]) for row in rows)
        dataset_groups[dataset] = "none" if fraction == 0 else ("all" if fraction == 1 else "some")

    group_definitions = (
        ("none", "No labels\nunder-specified"),
        ("some", "Some labels\nunder-specified"),
        ("all", "Every label\nunder-specified"),
    )
    data: dict[str, dict[str, float | int]] = {}
    for key, _ in group_definitions:
        rows = [row for row in dataset_rows if dataset_groups[row["dataset"]] == key]
        data[key] = {
            "count": len(rows),
            "one_mAP50_95": fmean(float(row["one_reference_gain_mAP50_95"]) for row in rows),
            "one_mAP50": fmean(float(row["one_reference_gain_mAP50"]) for row in rows),
            "ten_mAP50_95": fmean(float(row["ten_references_gain_mAP50_95"]) for row in rows),
            "ten_mAP50": fmean(float(row["ten_references_gain_mAP50"]) for row in rows),
        }

    canvas = Image.new("RGB", (1700, 760), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (50, 25),
        "Visual references help most when class names are under-specified",
        fill="#0f172a",
        font=font(34, bold=True),
    )
    draw.text(
        (50, 70),
        "Average dataset gain over class names only; groups were rated before these scores were analyzed",
        fill="#475569",
        font=font(20),
    )

    legend_y = 112
    for index, (label, color) in enumerate((("1 reference/class", "#0284c7"), ("10 references/class", "#7c3aed"))):
        x = 1120 + index * 280
        draw.rectangle((x, legend_y, x + 24, legend_y + 24), fill=color)
        draw.text((x + 34, legend_y - 1), label, fill="#334155", font=font(18, bold=True))

    plot_top, plot_bottom = 180, 620
    plot_width = 720
    y_min, y_max = -6.0, 22.0
    for plot_index, (metric, title) in enumerate((("mAP50_95", "mAP50–95 gain"), ("mAP50", "mAP50 gain"))):
        left = 70 + plot_index * 820
        right = left + plot_width
        draw.text((left + 250, 140), title, fill="#0f172a", font=font(24, bold=True))
        for tick in (-5, 0, 5, 10, 15, 20):
            y = plot_bottom - (tick - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
            draw.line((left, y, right, y), fill="#cbd5e1" if tick == 0 else "#e2e8f0", width=2 if tick == 0 else 1)
            draw.text((left - 46, y - 10), f"{tick:+d}", fill="#475569", font=font(16))
        cluster_width = plot_width / 3
        for group_index, (group_key, group_label) in enumerate(group_definitions):
            center = left + cluster_width * (group_index + 0.5)
            for bar_index, (mode, color) in enumerate((("one", "#0284c7"), ("ten", "#7c3aed"))):
                value = float(data[group_key][f"{mode}_{metric}"])
                zero_y = plot_bottom - (0 - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
                value_y = plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
                x0 = center - 74 + bar_index * 78
                x1 = x0 + 68
                draw.rectangle((x0, min(zero_y, value_y), x1, max(zero_y, value_y)), fill=color)
                value_text = f"{value:+.2f}"
                value_width = draw.textlength(value_text, font=font(16, bold=True))
                text_y = value_y - 25 if value >= 0 else value_y + 5
                draw.text((x0 + (68 - value_width) / 2, text_y), value_text, fill="#0f172a", font=font(16, bold=True))
            label_lines = group_label.split("\n")
            for line_index, line in enumerate(label_lines):
                line_width = draw.textlength(line, font=font(17, bold=True))
                draw.text((center - line_width / 2, 642 + line_index * 23), line, fill="#334155", font=font(17, bold=True))
            count = f"({int(data[group_key]['count'])} datasets)"
            count_width = draw.textlength(count, font=font(15))
            draw.text((center - count_width / 2, 692), count, fill="#64748b", font=font(15))
    canvas.save(output_root / "visual_gain_by_dataset_label_sufficiency.png", optimize=True)


def draw_shareable_card_sheet(
    output_root: Path,
    *,
    filename: str,
    title: str,
    subtitle: str,
    cards: tuple[str, ...],
) -> None:
    canvas = Image.new("RGB", (1920, 1850), COLORS["background"])
    draw = ImageDraw.Draw(canvas)
    draw.text((52, 38), title, fill=COLORS["text"], font=font(46, bold=True))
    draw.text((54, 102), subtitle, fill=COLORS["muted"], font=font(23))
    card_width, card_height = 880, 784
    positions = ((50, 165), (990, 165), (50, 1010), (990, 1010))
    for card_name, position in zip(cards, positions, strict=True):
        card = Image.open(output_root / card_name).convert("RGB")
        card = ImageOps.fit(card, (card_width, card_height), Image.Resampling.LANCZOS)
        canvas.paste(card, position)
    draw.text(
        (54, 1810),
        "Cards show one real boxed train exemplar and the same test image under both prediction modes.",
        fill=COLORS["muted"],
        font=font(18),
    )
    canvas.save(output_root / filename, optimize=True)


def draw_shareable_summary(output_root: Path) -> None:
    canvas = Image.new("RGB", (1920, 1250), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (54, 34),
        "Qwen3.8-Max: visual references behave like dataset adaptation",
        fill="#0f172a",
        font=font(43, bold=True),
    )
    draw.text(
        (56, 94),
        "They help most when class names under-specify appearance or state, but can hurt familiar-object tasks.",
        fill="#475569",
        font=font(22),
    )

    grouped_chart = Image.open(output_root / "visual_gain_by_dataset_label_sufficiency.png").convert("RGB")
    grouped_chart = ImageOps.fit(grouped_chart, (1220, 545), Image.Resampling.LANCZOS)
    canvas.paste(grouped_chart, (35, 160))

    panel = (1280, 160, 1880, 705)
    draw.rounded_rectangle(panel, radius=24, fill="#0f172a")
    draw.text((1320, 196), "RF20-VL-FSOD macro", fill="#ffffff", font=font(28, bold=True))
    draw.text((1320, 238), "mAP50–95 / mAP50", fill="#cbd5e1", font=font(18))
    rows = (
        ("Class names only", "24.37 / 43.54", "baseline"),
        ("Instructions", "24.46 / 44.58", "+0.09 / +1.04"),
        ("1 visual ref/class", "25.35 / 46.73", "+0.98 / +3.19"),
        ("10 visual refs/class", "25.74 / 47.92", "+1.37 / +4.38"),
    )
    for index, (label, score, delta) in enumerate(rows):
        y = 286 + index * 76
        if index:
            draw.line((1320, y - 13, 1840, y - 13), fill="#334155", width=1)
        draw.text((1320, y), label, fill="#f8fafc", font=font(20, bold=True))
        draw.text((1320, y + 28), score, fill="#cbd5e1", font=font(18))
        delta_color = "#5ee6a8" if delta != "baseline" else "#94a3b8"
        delta_width = draw.textlength(delta, font=font(18, bold=True))
        draw.text((1840 - delta_width, y + 28), delta, fill=delta_color, font=font(18, bold=True))

    evidence_boxes = (
        (
            40,
            "Under-specified classes",
            "+6.68 / +12.68 with one reference",
            "vs −2.22 / −1.84 for sufficient names",
            "#0284c7",
        ),
        (
            660,
            "Repeatability calibration",
            "Baseline noise: 0.49–2.17 / 1.19–3.87",
            "Paired-gain noise: 0.68–3.17 / 2.58–4.42",
            "#059669",
        ),
        (
            1280,
            "Three practical modes",
            "Names default • 1 ref efficient",
            "10 refs = highest observed score",
            "#7c3aed",
        ),
    )
    for x, heading, first, second, accent in evidence_boxes:
        draw.rounded_rectangle((x, 760, x + 600, 1000), radius=22, fill="#ffffff", outline="#cbd5e1", width=2)
        draw.rectangle((x, 760, x + 12, 1000), fill=accent)
        draw.text((x + 38, 798), heading, fill="#0f172a", font=font(26, bold=True))
        draw.text((x + 38, 858), first, fill="#334155", font=font(21, bold=True))
        draw.text((x + 38, 902), second, fill="#475569", font=font(19))

    draw.rounded_rectangle((40, 1050, 1880, 1195), radius=22, fill="#e2e8f0")
    takeaway = (
        "General trend, not a universal rule: visual examples transfer dataset-specific appearance "
        "better than instructions, but the effect is heterogeneous and primarily dataset-level."
    )
    wrapped = textwrap.wrap(takeaway, width=116)
    for index, line in enumerate(wrapped):
        draw.text((76, 1082 + index * 37), line, fill="#0f172a", font=font(25, bold=True))
    canvas.save(output_root / "shareable_context_adaptation_summary.png", optimize=True)


def draw_shareable_exports(output_root: Path) -> None:
    draw_shareable_summary(output_root)
    draw_shareable_card_sheet(
        output_root,
        filename="shareable_visual_references_helped.png",
        title="When boxed visual references helped",
        subtitle=(
            "Dataset-specific appearance, state, and annotation semantics • "
            "scores are class AP50 and AP50–95"
        ),
        cards=(
            "under_specified_helped__paper-parts__table-of-contents-text.png",
            "under_specified_helped__defect-detection__defective-fishplate.png",
            "under_specified_helped__the-dreidel-project__spinning-dreidel.png",
            "under_specified_helped__wb-prova__juvenile.png",
        ),
    )
    draw_shareable_card_sheet(
        output_root,
        filename="shareable_visual_references_hurt.png",
        title="When extra visual context hurt or failed",
        subtitle=(
            "Familiar labels and one important ambiguous-label counterexample • "
            "same evaluation and scoring"
        ),
        cards=(
            "familiar_hurt__water-meter__4.png",
            "familiar_hurt__flir-camera-objects__dog.png",
            "familiar_hurt__gwhd2021__wheat-head.png",
            "counterexample__all-elements__checked-radio-button.png",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for stale_path in (*args.output.glob("*.png"), args.output / "selection_manifest.json"):
        if stale_path.is_file():
            stale_path.unlink()
    class_results = class_result_index()
    manifest = []
    for group, cases in CASES.items():
        for dataset, class_name in cases:
            manifest.append(
                generate_card(dataset, class_name, group, class_results, args.output)
            )
    draw_grouped_gain_chart(args.output)
    draw_shareable_exports(args.output)
    manifest_path = args.output / "selection_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_root": str(DATA_ROOT.relative_to(REPO)),
                "names_prediction_root": str(NAMES_ROOT.relative_to(REPO)),
                "visual_prediction_root": str(VISUAL_ROOT.relative_to(REPO)),
                "selection_policy": (
                    "Predeclared dataset/class examples; within each pair choose the largest "
                    "per-image F1@0.5 improvement or regression. Aggregate COCO scores, not "
                    "these selected images, support the report conclusions."
                ),
                "cards": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(manifest)} evidence cards and one grouped-gain chart in {args.output}")


if __name__ == "__main__":
    main()
