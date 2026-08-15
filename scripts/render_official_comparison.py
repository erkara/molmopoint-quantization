#!/usr/bin/env python3
"""Render the official-protocol F1-versus-memory comparison figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANTS = ("bf16", "int4", "int8")
LABELS = {"bf16": "BF16", "int4": "Final INT4", "int8": "Final INT8"}
COLORS = {"bf16": "#2878B5", "int4": "#E07A1F", "int8": "#2A9D72"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results/official/pointing-eval-v2",
        help="Directory containing <variant>-summary.json files",
    )
    parser.add_argument("--output", default="results/figures/official-f1-memory.png")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import matplotlib.pyplot as plt

    summaries = {
        variant: json.loads(
            (Path(args.results_dir) / f"{variant}-summary.json").read_text(encoding="utf-8")
        )
        for variant in VARIANTS
    }
    f1 = [100 * summaries[variant]["metrics"]["f1"] for variant in VARIANTS]
    memory = [summaries[variant]["performance"]["max_peak_vram_gib"] for variant in VARIANTS]
    colors = [COLORS[variant] for variant in VARIANTS]
    labels = [LABELS[variant] for variant in VARIANTS]

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 6.0))
    figure.subplots_adjust(left=0.18, right=0.95, top=0.82, bottom=0.12, hspace=0.50)
    figure.suptitle(
        "MolmoPoint quantization: quality retained, memory reduced", fontsize=14, y=0.97
    )
    figure.text(
        0.5,
        0.915,
        "AllenAI official PointingEval scorer · same 231/436 publicly recoverable examples",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    for axis, values, annotations, title, limit in (
        (
            axes[0],
            f1,
            ("84.40 · baseline", "84.06 · -0.34 pt", "84.15 · -0.25 pt"),
            "Pointing F1 (%) · higher is better",
            100,
        ),
        (
            axes[1],
            memory,
            ("18.24 GiB · baseline", "8.69 GiB · -52%", "11.66 GiB · -36%"),
            "Peak allocated GPU memory (GiB) · lower is better",
            21,
        ),
    ):
        bars = axis.barh(labels, values, color=colors, height=0.58)
        axis.set_xlim(0, limit)
        axis.set_title(title, loc="left", fontsize=11)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.invert_yaxis()
        for bar, value, annotation in zip(bars, values, annotations):
            axis.text(
                value + limit * 0.012,
                bar.get_y() + bar.get_height() / 2,
                annotation,
                va="center",
                fontweight="bold",
                fontsize=9,
            )
    figure.text(
        0.01,
        0.025,
        "Partial public recovery; not the full 436-example paper result. Full PointBench: "
        "BF16 70.50, INT4 70.69, INT8 70.49.",
        fontsize=8,
        color="#555555",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
