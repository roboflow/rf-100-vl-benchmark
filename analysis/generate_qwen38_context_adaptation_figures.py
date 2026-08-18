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
import math
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


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def correlation(first: list[float], second: list[float]) -> float:
    first_mean = fmean(first)
    second_mean = fmean(second)
    centered_first = [value - first_mean for value in first]
    centered_second = [value - second_mean for value in second]
    denominator = math.sqrt(
        sum(value**2 for value in centered_first)
        * sum(value**2 for value in centered_second)
    )
    if not denominator:
        raise ValueError("Correlation is undefined for a constant input.")
    return sum(
        left * right for left, right in zip(centered_first, centered_second, strict=True)
    ) / denominator


def spearman(first: list[float], second: list[float]) -> float:
    return correlation(average_ranks(first), average_ranks(second))


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


def draw_scatter(output_root: Path) -> None:
    class_rows = read_csv(CLASS_RESULTS)
    dataset_rows = read_csv(DATASET_RESULTS)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in class_rows:
        grouped[row["dataset"]].append(row)

    data = []
    for row in dataset_rows:
        dataset = row["dataset"]
        insufficiency = fmean(
            int(item["name_alone_insufficient"]) for item in grouped[dataset]
        )
        data.append(
            {
                "dataset": dataset,
                "x": insufficiency,
                "one": float(row["one_reference_gain_mAP50"]),
                "ten": float(row["ten_references_gain_mAP50"]),
            }
        )

    correlations = {
        key: spearman(
            [row["x"] for row in data],
            [row[key] for row in data],
        )
        for key in ("one", "ten")
    }

    canvas = Image.new("RGB", (1700, 760), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (50, 25),
        "Visual-reference gain rises as class names become less sufficient",
        fill="#0f172a",
        font=font(34, bold=True),
    )
    draw.text(
        (50, 70),
        "Each point is one complete RF20-VL-FSOD dataset; y-axis is gain over class names only",
        fill="#475569",
        font=font(20),
    )

    plot_top, plot_bottom = 145, 680
    plot_width = 730
    y_min, y_max = -45.0, 30.0
    abbreviations = {
        "aerial-airport": "Airport",
        "aquarium-combined": "Aquarium",
        "defect-detection": "Defect",
        "flir-camera-objects": "FLIR",
        "gwhd2021": "Wheat",
        "lacrosse-object-detection": "Lacrosse",
        "new-defects-in-wood": "Wood",
        "orionproducts": "Orion",
        "paper-parts": "Paper",
        "recode-waste": "Waste",
        "soda-bottles": "Soda",
        "the-dreidel-project": "Dreidel",
        "trail-camera": "Trail",
        "water-meter": "Digits",
        "wb-prova": "Age groups",
        "wildfire-smoke": "Smoke",
        "x-ray-id": "X-ray",
        "all-elements": "UI elements",
        "dentalai": "Dental",
        "actions": "Actions",
    }
    for plot_index, (key, title) in enumerate(
        (("one", "One visual reference per class"), ("ten", "Ten visual references per class"))
    ):
        left = 70 + plot_index * 820
        right = left + plot_width
        draw.rectangle((left, plot_top, right, plot_bottom), outline="#94a3b8", width=2)
        zero_y = plot_bottom - (0 - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
        draw.line((left, zero_y, right, zero_y), fill="#64748b", width=2)
        for tick in (-40, -20, 0, 20):
            y = plot_bottom - (tick - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
            draw.line((left, y, right, y), fill="#e2e8f0", width=1)
            draw.text((left - 52, y - 10), f"{tick:+d}", fill="#475569", font=font(16))
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = left + tick * plot_width
            draw.line((x, plot_top, x, plot_bottom), fill="#f1f5f9", width=1)
            draw.text((x - 14, plot_bottom + 8), f"{tick:.2g}", fill="#475569", font=font(15))
        draw.text((left, 110), title, fill="#0f172a", font=font(24, bold=True))
        draw.text(
            (right - 205, 112),
            f"Spearman ρ = {correlations[key]:.2f}",
            fill="#334155",
            font=font(19, bold=True),
        )
        points = []
        for row in data:
            x = left + row["x"] * plot_width
            y = plot_bottom - (row[key] - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
            points.append((row, x, y))

        label_positions: dict[str, float] = {}
        for bucket in range(5):
            bucket_points = sorted(
                (
                    (row, x, y)
                    for row, x, y in points
                    if min(4, int(row["x"] * 5)) == bucket
                ),
                key=lambda item: item[2],
            )
            previous = plot_top - 18
            provisional = []
            for row, x, y in bucket_points:
                label_y = max(y - 9, previous + 17)
                provisional.append([row, x, y, label_y])
                previous = label_y
            if provisional and provisional[-1][3] > plot_bottom - 18:
                shift = provisional[-1][3] - (plot_bottom - 18)
                for item in provisional:
                    item[3] -= shift
            for row, _, _, label_y in provisional:
                label_positions[row["dataset"]] = label_y

        for row, x, y in points:
            color = "#0ea5e9" if row[key] >= 0 else "#f43f5e"
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="#ffffff", width=2)
            label = abbreviations.get(row["dataset"], row["dataset"])
            label_width = draw.textlength(label, font=font(13))
            label_x = x + 8 if row["x"] < 0.82 else x - label_width - 9
            label_y = label_positions[row["dataset"]]
            if abs(label_y - (y - 9)) > 12:
                line_end_x = label_x - 2 if row["x"] < 0.82 else label_x + label_width + 2
                draw.line((x, y, line_end_x, label_y + 8), fill="#94a3b8", width=1)
            draw.text((label_x, label_y), label, fill="#334155", font=font(13))
        draw.text(
            (left + 205, 720),
            "Fraction of classes whose names are insufficient",
            fill="#334155",
            font=font(18),
        )
    canvas.save(output_root / "label_insufficiency_vs_visual_gain.png", optimize=True)


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
    draw_scatter(args.output)
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
    print(f"Generated {len(manifest)} evidence cards and one scatter plot in {args.output}")


if __name__ == "__main__":
    main()
