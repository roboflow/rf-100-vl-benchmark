#!/usr/bin/env python3
"""Render the RF100-VL class-names-only category comparison figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_STEM = Path(__file__).resolve().parents[1] / "figures" / (
    "rf100vl_class_names_only_category_comparison"
)

CATEGORIES = [
    "Aerial",
    "Document",
    "Flora & Fauna",
    "Industrial",
    "Medical",
    "Sports",
    "Other",
    "All",
]
DATASET_COUNTS = [11, 10, 23, 22, 13, 6, 15, 100]

# RF100-VL paper, Table 2: multi-class, class-names-only, pycocotools mAP50-95.
QWEN_MAP = [4.6, 3.9, 10.4, 4.1, 1.6, 6.0, 5.6, 5.6]
GEMINI_MAP = [8.7, 11.8, 18.3, 8.6, 5.3, 6.5, 15.4, 11.6]

# Final Cosmos3 RF100-VL runs. Category values are unweighted means over the
# constituent datasets; "All" is the unweighted mean over all 100 datasets.
# Super is derived from the four verified shard aggregates in plan
# shard-40cebbcfa951; categories use RF100-VL's dataset_name_to_category.json.
COSMOS_EDGE_MAP = [
    3.927299667606183,
    4.537677135391117,
    10.815587228343441,
    3.161170903173489,
    0.5435460534788226,
    3.36668183416285,
    5.7959048636913965,
    5.210860964748679,
]
COSMOS_EDGE_MAP50 = [
    10.72823690043041,
    10.272001840891562,
    18.245396803079747,
    5.434403084878932,
    1.1511294581802275,
    8.211624596238543,
    10.039048730834864,
    9.74751780148118,
]
COSMOS_SUPER_MAP = [
    6.210200932023468,
    13.467486771164285,
    19.55898097764258,
    7.987275979982661,
    2.1287380461789036,
    11.3879819564823,
    10.281268017363786,
    10.787842186089748,
]
COSMOS_SUPER_MAP50 = [
    15.300258271535261,
    23.763737928617132,
    34.02479704770679,
    14.16998024943469,
    4.893253593105744,
    24.90327388557606,
    19.433653426821497,
    20.047868592840313,
]

COLORS = {
    "qwen": "#F2A900",
    "gemini": "#4C78A8",
    "edge": "#16A085",
    "edge_light": "#61CDBB",
    "super": "#7A5195",
    "super_light": "#B89AC8",
    "ink": "#16212B",
    "muted": "#65727E",
    "grid": "#DCE3E9",
    "paper": "#F7F9FC",
    "all_band": "#EAF0F5",
}


def style_axis(axis: plt.Axes, *, ymax: float, all_position: float) -> None:
    axis.set_facecolor(COLORS["paper"])
    axis.set_ylim(0, ymax)
    axis.set_ylabel("Average precision (%)", color=COLORS["ink"], fontweight="bold")
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.9, linestyle=(0, (2, 3)))
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(COLORS["grid"])
    axis.tick_params(axis="y", colors=COLORS["muted"], length=0)
    axis.tick_params(axis="x", colors=COLORS["ink"], length=0, pad=10)
    axis.axvspan(
        all_position - 0.52,
        all_position + 0.52,
        color=COLORS["all_band"],
        zorder=0,
    )


def label_bars(axis: plt.Axes, bars, *, decimals: int = 1) -> None:
    for bar in bars:
        height = bar.get_height()
        axis.annotate(
            f"{height:.{decimals}f}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=COLORS["ink"],
            fontweight="bold",
        )


def validate_data() -> None:
    series = (
        QWEN_MAP,
        GEMINI_MAP,
        COSMOS_EDGE_MAP,
        COSMOS_EDGE_MAP50,
        COSMOS_SUPER_MAP,
        COSMOS_SUPER_MAP50,
    )
    if any(len(values) != len(CATEGORIES) for values in series):
        raise ValueError("Every metric series must cover all categories plus All.")

    for values in (
        COSMOS_EDGE_MAP,
        COSMOS_EDGE_MAP50,
        COSMOS_SUPER_MAP,
        COSMOS_SUPER_MAP50,
    ):
        category_macro = np.average(values[:-1], weights=DATASET_COUNTS[:-1])
        if not np.isclose(category_macro, values[-1], rtol=0, atol=1e-10):
            raise ValueError(
                "Cosmos category values do not reproduce the 100-dataset macro."
            )


def main() -> None:
    validate_data()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
        }
    )

    figure, (comparison_axis, cosmos_axis) = plt.subplots(
        2,
        1,
        figsize=(18, 13.5),
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.72},
        facecolor=COLORS["paper"],
    )
    figure.subplots_adjust(left=0.065, right=0.985, top=0.85, bottom=0.12)

    figure.suptitle(
        "RF100-VL · Class Names Only",
        x=0.065,
        y=0.965,
        ha="left",
        fontsize=23,
        fontweight="bold",
        color=COLORS["ink"],
    )
    figure.text(
        0.065,
        0.925,
        "Category-level object detection accuracy · macro average over datasets · higher is better",
        ha="left",
        fontsize=12,
        color=COLORS["muted"],
    )

    x_positions = np.arange(len(CATEGORIES)) * 1.18
    tick_labels = [
        f"{name}\n(n={count})"
        for name, count in zip(CATEGORIES, DATASET_COUNTS)
    ]

    style_axis(comparison_axis, ymax=22.0, all_position=x_positions[-1])
    comparison_axis.set_title(
        "A. Model comparison · mAP50–95",
        loc="left",
        color=COLORS["ink"],
        y=1.17,
        pad=0,
    )
    comparison_axis.text(
        1.0,
        1.17,
        "Comparable metric reported for all four models",
        transform=comparison_axis.transAxes,
        ha="right",
        color=COLORS["muted"],
        fontsize=9.5,
    )
    comparison_width = 0.19
    qwen_bars = comparison_axis.bar(
        x_positions - 1.5 * comparison_width,
        QWEN_MAP,
        comparison_width,
        label="Qwen 2.5-VL 72B",
        color=COLORS["qwen"],
        edgecolor="none",
        zorder=3,
    )
    gemini_bars = comparison_axis.bar(
        x_positions - 0.5 * comparison_width,
        GEMINI_MAP,
        comparison_width,
        label="Gemini 2.5 Pro",
        color=COLORS["gemini"],
        edgecolor="none",
        zorder=3,
    )
    edge_bars = comparison_axis.bar(
        x_positions + 0.5 * comparison_width,
        COSMOS_EDGE_MAP,
        comparison_width,
        label="Cosmos3-Edge",
        color=COLORS["edge"],
        edgecolor="none",
        zorder=3,
    )
    super_bars = comparison_axis.bar(
        x_positions + 1.5 * comparison_width,
        COSMOS_SUPER_MAP,
        comparison_width,
        label="Cosmos3-Super",
        color=COLORS["super"],
        edgecolor="none",
        zorder=3,
    )
    label_bars(comparison_axis, qwen_bars)
    label_bars(comparison_axis, gemini_bars)
    label_bars(comparison_axis, edge_bars)
    label_bars(comparison_axis, super_bars)
    comparison_axis.set_xticks(x_positions, tick_labels)
    comparison_axis.legend(
        loc="lower left",
        ncols=4,
        frameon=False,
        bbox_to_anchor=(0.0, 1.025),
        borderaxespad=0,
        columnspacing=1.8,
        handlelength=1.3,
    )

    style_axis(cosmos_axis, ymax=38.5, all_position=x_positions[-1])
    cosmos_axis.set_title(
        "B. Cosmos3 family detail · mAP50 versus mAP50–95",
        loc="left",
        color=COLORS["ink"],
        y=1.17,
        pad=0,
    )
    cosmos_axis.text(
        1.0,
        1.17,
        "Same prompts and scoring for Edge and Super",
        transform=cosmos_axis.transAxes,
        ha="right",
        color=COLORS["muted"],
        fontsize=9.5,
    )
    cosmos_width = 0.19
    edge_map50_bars = cosmos_axis.bar(
        x_positions - 1.5 * cosmos_width,
        COSMOS_EDGE_MAP50,
        cosmos_width,
        label="Edge · mAP50",
        color=COLORS["edge_light"],
        edgecolor="none",
        zorder=3,
    )
    edge_map_bars = cosmos_axis.bar(
        x_positions - 0.5 * cosmos_width,
        COSMOS_EDGE_MAP,
        cosmos_width,
        label="Edge · mAP50–95",
        color=COLORS["edge"],
        edgecolor="none",
        zorder=3,
    )
    super_map50_bars = cosmos_axis.bar(
        x_positions + 0.5 * cosmos_width,
        COSMOS_SUPER_MAP50,
        cosmos_width,
        label="Super · mAP50",
        color=COLORS["super_light"],
        edgecolor="none",
        zorder=3,
    )
    super_map_bars = cosmos_axis.bar(
        x_positions + 1.5 * cosmos_width,
        COSMOS_SUPER_MAP,
        cosmos_width,
        label="Super · mAP50–95",
        color=COLORS["super"],
        edgecolor="none",
        zorder=3,
    )
    label_bars(cosmos_axis, edge_map50_bars)
    label_bars(cosmos_axis, edge_map_bars)
    label_bars(cosmos_axis, super_map50_bars)
    label_bars(cosmos_axis, super_map_bars)
    cosmos_axis.set_xticks(x_positions, tick_labels)
    cosmos_axis.legend(
        loc="lower left",
        ncols=4,
        frameon=False,
        bbox_to_anchor=(0.0, 1.025),
        borderaxespad=0,
        columnspacing=1.8,
        handlelength=1.3,
    )

    figure.text(
        0.5,
        0.052,
        "Evaluation: pycocotools · maxDets=500 · multi-class prompts · no few-shot examples or annotator instructions",
        ha="center",
        fontsize=9.5,
        color=COLORS["muted"],
    )
    figure.text(
        0.5,
        0.027,
        "Qwen/Gemini: RF100-VL paper, Table 2  ·  Cosmos3: verified 100-dataset runs",
        ha="center",
        fontsize=9.5,
        color=COLORS["muted"],
    )

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_STEM.with_suffix(".png")
    svg_path = OUTPUT_STEM.with_suffix(".svg")
    figure.savefig(png_path, dpi=240, facecolor=figure.get_facecolor())
    figure.savefig(svg_path, facecolor=figure.get_facecolor())
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
