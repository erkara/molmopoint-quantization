#!/usr/bin/env python3
"""Render one paired BF16/INT8/INT4 pointing example as a 1x3 PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from molmo_quant.pixmo_dataset import load_pixmo_dataset, read_manifest


COLORS = {
    "bf16": "#2563EB",
    "int8": "#059669",
    "int4": "#DC2626",
}
TITLES = {
    "bf16": "BF16 baseline",
    "int8": "INT8 quantized",
    "int4": "INT4 quantized",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=103)
    parser.add_argument("--dataset-arrow")
    parser.add_argument("--manifest", default="data/pixmo-points-eval/manifest.json")
    parser.add_argument("--bf16", default="results/pixmo/full/bf16.jsonl")
    parser.add_argument("--int8", default="results/pixmo/full/int8.jsonl")
    parser.add_argument("--int4", default="results/pixmo/full/int4.jsonl")
    parser.add_argument(
        "--output",
        default="results/figures/problematic-example-103.png",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = {
        variant: read_record(path, args.index)
        for variant, path in (("bf16", args.bf16), ("int8", args.int8), ("int4", args.int4))
    }
    dataset = load_pixmo_dataset(dataset_arrow=args.dataset_arrow)
    example = dataset[args.index]
    manifest = read_manifest(args.manifest)
    image_path = manifest[args.index]["image_path"]
    if not image_path:
        raise RuntimeError(f"Dataset index {args.index} has no verified image")

    with Image.open(image_path) as opened:
        source = opened.convert("RGB")
    figure = render_comparison(source, example, records, args.index)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.save(destination, format="PNG", optimize=True)
    print(destination.resolve())
    return 0


def read_record(path: str | Path, index: int) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if int(record["index"]) == index:
                return record
    raise KeyError(f"Index {index} not found in {path}")


def render_comparison(
    source: Image.Image,
    example: dict,
    records: dict[str, dict],
    index: int,
) -> Image.Image:
    panel_width = 500
    image_height = round(panel_width * source.height / source.width)
    margin = 42
    gap = 28
    header_height = 126
    panel_header_height = 78
    footer_height = 112
    canvas_width = margin * 2 + panel_width * 3 + gap * 2
    canvas_height = header_height + panel_header_height + image_height + footer_height + 34
    canvas = Image.new("RGB", (canvas_width, canvas_height), "#FFFFFF")
    draw = ImageDraw.Draw(canvas)

    fonts = fonts_for_figure()
    label = str(example["label"])
    draw.text(
        (margin, 28),
        f'One deliberately difficult example: “{label}”',
        fill="#111827",
        font=fonts["title"],
    )
    draw.text(
        (margin, 76),
        f"PixMo Points Eval index {index} · yellow boxes are target regions · numbered circles are model predictions",
        fill="#4B5563",
        font=fonts["subtitle"],
    )

    resized = source.resize((panel_width, image_height), Image.Resampling.LANCZOS)
    masks = np.asarray(example["masks"], dtype=bool)
    variants = ("bf16", "int8", "int4")
    for panel_index, variant in enumerate(variants):
        panel_x = margin + panel_index * (panel_width + gap)
        image_y = header_height + panel_header_height
        record = records[variant]
        color = COLORS[variant]

        draw.text(
            (panel_x, header_height + 4),
            TITLES[variant],
            fill=color,
            font=fonts["panel_title"],
        )
        draw.text(
            (panel_x, header_height + 42),
            f"{record['prediction_point_count']} predicted point(s)  ·  F1 {record['metrics']['f1']:.2f}",
            fill="#374151",
            font=fonts["panel_meta"],
        )

        panel_image = resized.copy()
        panel_image = overlay_target_regions(panel_image, masks, source.size)
        canvas.paste(panel_image, (panel_x, image_y))
        draw.rectangle(
            (panel_x, image_y, panel_x + panel_width - 1, image_y + image_height - 1),
            outline="#D1D5DB",
            width=2,
        )

        scale_x = panel_width / source.width
        scale_y = image_height / source.height
        for point_index, point in enumerate(record["points"], start=1):
            x = panel_x + float(point["x"]) * scale_x
            y = image_y + float(point["y"]) * scale_y
            draw.ellipse((x - 17, y - 17, x + 17, y + 17), outline="#FFFFFF", width=8)
            draw.ellipse((x - 14, y - 14, x + 14, y + 14), outline=color, width=7)
            draw.line((x - 20, y, x + 20, y), fill=color, width=4)
            draw.line((x, y - 20, x, y + 20), fill=color, width=4)
            draw.ellipse((x + 10, y - 26, x + 36, y), fill="#FFFFFF", outline=color, width=3)
            number_box = draw.textbbox((0, 0), str(point_index), font=fonts["marker"])
            number_width = number_box[2] - number_box[0]
            draw.text(
                (x + 23 - number_width / 2, y - 25),
                str(point_index),
                fill=color,
                font=fonts["marker"],
            )

        coordinates = ", ".join(
            f"({point['x']:.0f}, {point['y']:.0f})" for point in record["points"]
        ) or "none"
        footer_y = image_y + image_height + 18
        draw.text(
            (panel_x, footer_y),
            f"Predicted coordinates: {coordinates}",
            fill="#111827",
            font=fonts["coordinates"],
        )

    return canvas


def overlay_target_regions(
    image: Image.Image, masks: np.ndarray, source_size: tuple[int, int]
) -> Image.Image:
    panel_width, panel_height = image.size
    source_width, source_height = source_size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for mask in masks:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        x0 = int(xs.min() * panel_width / source_width)
        x1 = int((xs.max() + 1) * panel_width / source_width)
        y0 = int(ys.min() * panel_height / source_height)
        y1 = int((ys.max() + 1) * panel_height / source_height)
        padding = 5
        box = (x0 - padding, y0 - padding, x1 + padding, y1 + padding)
        overlay_draw.rectangle(box, fill=(245, 158, 11, 45), outline=(217, 119, 6, 255), width=5)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def fonts_for_figure() -> dict[str, ImageFont.FreeTypeFont]:
    regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return {
        "title": ImageFont.truetype(bold, 34),
        "subtitle": ImageFont.truetype(regular, 21),
        "panel_title": ImageFont.truetype(bold, 27),
        "panel_meta": ImageFont.truetype(regular, 19),
        "coordinates": ImageFont.truetype(regular, 18),
        "marker": ImageFont.truetype(bold, 16),
    }


if __name__ == "__main__":
    raise SystemExit(main())
