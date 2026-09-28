#!/usr/bin/env python3
"""Render the publication figure from full PointBench and PixMo-Points runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VARIANTS = ("bf16", "int4", "int8")
LABELS = {"bf16": "BF16", "int4": "INT4", "int8": "INT8"}
COLORS = {"bf16": "#2878B5", "int4": "#E67E22", "int8": "#2A9D72"}


def build_parser() -> argparse.ArgumentParser:
    """Create the standalone publication-figure command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pointbench-dir",
        required=True,
        type=Path,
        help="Directory containing full PointBench <variant>.jsonl files",
    )
    parser.add_argument(
        "--pixmo-dir",
        required=True,
        type=Path,
        help="Directory containing recovered PixMo-Points <variant>.jsonl files",
    )
    parser.add_argument("--output", default=PROJECT_ROOT / "runs/full_comparison.png", type=Path)
    return parser


def load_records(directory: Path, variant: str) -> list[dict]:
    """Load the latest successful attempt for every source example.

    Evaluation files are append-only so interrupted runs can resume. Mapping by
    source index makes a later retry supersede its earlier record.
    """
    path = directory / f"{variant}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    latest = {int(record["source_index"]): record for record in records}
    failures = [record for record in latest.values() if record.get("status") != "passed"]
    if not latest or failures:
        raise ValueError(
            f"{path}: expected unique passed examples, found "
            f"{len(latest)} unique and {len(failures)} failures"
        )
    return list(latest.values())


def load_results(pointbench_dir: Path, pixmo_dir: Path) -> tuple[dict, dict]:
    """Validate aligned runs and aggregate quality and peak-memory metrics."""
    pointbench = {variant: load_records(pointbench_dir, variant) for variant in VARIANTS}
    pixmo = {variant: load_records(pixmo_dir, variant) for variant in VARIANTS}
    # Comparisons are meaningful only when every precision saw the exact same
    # examples. Fail instead of silently aggregating mismatched resumptions.
    for name, datasets in (("PointBench", pointbench), ("PixMo-Points", pixmo)):
        source_sets = [
            {int(record["source_index"]) for record in datasets[variant]} for variant in VARIANTS
        ]
        if any(sources != source_sets[0] for sources in source_sets[1:]):
            raise ValueError(f"{name} variants do not contain the same source examples")

    results = {}
    for variant in VARIANTS:
        category_scores: dict[str, list[float]] = {}
        for record in pointbench[variant]:
            category_scores.setdefault(str(record["category"]), []).append(
                float(record["metrics"]["accuracy"])
            )
        # PointBench defines its headline score as an unweighted average of
        # category means, rather than a mean over all examples.
        pointbench_average = statistics.fmean(
            statistics.fmean(values) for values in category_scores.values()
        )
        pixmo_f1 = statistics.fmean(float(record["metrics"]["f1"]) for record in pixmo[variant])
        results[variant] = {
            "pointbench": 100 * pointbench_average,
            "pixmo-points": 100 * pixmo_f1,
            "memory": max(
                float(record["peak_vram_gib"]) for record in pointbench[variant] + pixmo[variant]
            ),
        }
    counts = {"pointbench": len(pointbench[VARIANTS[0]]), "pixmo-points": len(pixmo[VARIANTS[0]])}
    return results, counts


def add_bars(axis, values, annotations, title, xlabel, limit):
    """Draw one consistently styled horizontal-bar panel."""

    labels = [LABELS[variant] for variant in VARIANTS]
    colors = [COLORS[variant] for variant in VARIANTS]
    bars = axis.barh(labels, values, color=colors, height=0.56)
    axis.set_xlim(0, limit)
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
    axis.set_xlabel(xlabel, color="#555555", labelpad=6)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.invert_yaxis()
    for bar, value, annotation in zip(bars, values, annotations, strict=True):
        axis.text(
            value + limit * 0.015,
            bar.get_y() + bar.get_height() / 2,
            annotation,
            va="center",
            fontweight="bold",
            fontsize=10,
        )


def render(results: dict, counts: dict, output: Path) -> None:
    """Render the three-panel quality-versus-memory publication figure."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor="white")
    pointbench_axis, pixmo_axis, memory_axis = axes
    figure.subplots_adjust(left=0.10, right=0.95, top=0.75, bottom=0.24, wspace=0.56)

    figure.suptitle(
        "MolmoPoint: quality vs memory",
        fontsize=18,
        y=0.94,
    )

    pointbench = [results[variant]["pointbench"] for variant in VARIANTS]
    pointbench_annotations = [f"{value:.2f}" for value in pointbench]
    add_bars(
        pointbench_axis,
        pointbench,
        pointbench_annotations,
        "PointBench",
        "Average accuracy (%)",
        100,
    )

    pixmo = [results[variant]["pixmo-points"] for variant in VARIANTS]
    pixmo_annotations = [f"{value:.2f}" for value in pixmo]
    add_bars(
        pixmo_axis,
        pixmo,
        pixmo_annotations,
        "PixMo-Points",
        "F1 (%)",
        100,
    )

    memory = [results[variant]["memory"] for variant in VARIANTS]
    memory_annotations = [f"{value:.2f}" for value in memory]
    add_bars(
        memory_axis,
        memory,
        memory_annotations,
        "Peak VRAM",
        "GiB",
        21,
    )

    figure.text(
        0.5,
        0.07,
        f"Official evaluation · PointBench n={counts['pointbench']} · "
        f"PixMo public recovery n={counts['pixmo-points']}/436",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    """Load both full runs and save their combined comparison figure."""

    args = build_parser().parse_args()
    results, counts = load_results(args.pointbench_dir.resolve(), args.pixmo_dir.resolve())
    for variant in VARIANTS:
        print(f"{variant}: {results[variant]}")
    render(results, counts, args.output.resolve())
    print(f"Saved {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
