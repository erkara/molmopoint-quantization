"""Run one model variant on PixMo Points Eval with resumable JSONL output."""

from __future__ import annotations

import argparse
import json
import statistics
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from molmo_quant.config import Variant
from molmo_quant.inference import run_pil_pointing
from molmo_quant.loading import load_saved_model
from molmo_quant.pixmo_dataset import load_pixmo_dataset, read_manifest
from molmo_quant.pointing_metrics import absolute_gt_points, score_pointing_prediction
from molmo_quant.provenance import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--processor")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--variant", choices=[variant.value for variant in Variant], required=True)
    parser.add_argument("--dataset", default="allenai/pixmo-points-eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-arrow")
    parser.add_argument("--manifest", default="data/pixmo-points-eval/manifest.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--prompt-template", default="Point to the {label}.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument(
        "--indices",
        help="Comma-separated dataset indices for a targeted paired pilot",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_pixmo_dataset(args.dataset, args.split, args.dataset_arrow)
    manifest = read_manifest(args.manifest)
    if args.indices and args.max_examples is not None:
        raise ValueError("Choose either --indices or --max-examples, not both")
    if args.indices:
        selected = sorted({int(value.strip()) for value in args.indices.split(",")})
        invalid = [index for index in selected if index < 0 or index >= len(dataset)]
        if invalid:
            raise ValueError(f"Dataset indices out of range: {invalid}")
    else:
        limit = min(len(dataset), args.max_examples) if args.max_examples is not None else len(dataset)
        selected = list(range(limit))
    ready = [index for index in selected if manifest.get(index, {}).get("status") == "ready"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(output_path)
    latest = latest_by_index(existing)
    completed = {
        index for index, record in latest.items() if record.get("status") == "passed"
    }
    pending = [index for index in ready if index not in completed]
    print(
        f"[eval] {args.variant}: selected={len(selected)}, ready={len(ready)}, "
        f"completed={len(completed & set(ready))}, pending={len(pending)}"
    )

    if pending:
        model, processor = load_saved_model(
            args.model,
            processor_id_or_path=args.processor,
            revision=args.revision,
        )
        model.eval()
        with output_path.open("a", encoding="utf-8", buffering=1) as output_handle:
            for position, index in enumerate(pending, start=1):
                record = evaluate_one(
                    dataset[index],
                    index=index,
                    image_path=manifest[index]["image_path"],
                    model=model,
                    processor=processor,
                    model_name=args.model,
                    processor_name=args.processor or args.model,
                    variant=args.variant,
                    prompt_template=args.prompt_template,
                    max_new_tokens=args.max_new_tokens,
                )
                output_handle.write(json.dumps(record, sort_keys=True) + "\n")
                if position == 1 or position % 10 == 0 or position == len(pending):
                    print(
                        f"[eval] {args.variant}: {position}/{len(pending)} new; "
                        f"index={index}; status={record['status']}"
                    )

    records = read_jsonl(output_path)
    summary = summarize_variant(
        records,
        variant=args.variant,
        model=args.model,
        dataset=args.dataset,
        split=args.split,
        selected_indices=selected,
        ready_count=len(ready),
        unavailable_count=len(selected) - len(ready),
        prompt_template=args.prompt_template,
    )
    write_json(args.summary, summary)
    print(
        f"[eval] {args.variant}: passed={summary['passed_examples']}, "
        f"F1={summary['metrics'].get('f1')}; wrote {args.summary}"
    )
    return 0 if summary["failed_examples"] == 0 and summary["passed_examples"] else 1


def evaluate_one(
    example: dict[str, Any],
    *,
    index: int,
    image_path: str,
    model,
    processor,
    model_name: str,
    processor_name: str,
    variant: str,
    prompt_template: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    label = str(example["label"])
    prompt = prompt_template.format(label=label)
    try:
        masks = np.asarray(example["masks"], dtype=bool)
        if masks.ndim != 3:
            raise ValueError(f"Expected masks with shape [N,H,W], got {masks.shape}")
        height, width = masks.shape[1:]
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        if image.size != (width, height):
            raise ValueError(
                f"Image/mask size mismatch: image={image.size}, masks={(width, height)}"
            )
        result = run_pil_pointing(
            model,
            processor,
            model_name=model_name,
            processor_name=processor_name,
            variant=variant,
            image=image,
            image_reference=image_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        ).to_dict()
        points = result["points"]
        ground_truth = absolute_gt_points(example["points"], width, height)
        metrics = score_pointing_prediction(points, ground_truth, masks)
        return {
            "schema_version": 1,
            "status": "passed",
            "index": index,
            "variant": variant,
            "label": label,
            "prompt": prompt,
            "image_sha256": str(example["image_sha256"]),
            "image_width": width,
            "image_height": height,
            "is_negative": len(example["points"]) == 0,
            "ground_truth_point_count": len(example["points"]),
            "prediction_point_count": len(points),
            "generated_text": result["generated_text"],
            "generated_token_ids": result["generated_token_ids"],
            "points": points,
            "metrics": metrics,
            "inference_seconds": result["inference_seconds"],
            "peak_vram_gib": result["peak_vram_gib"],
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "status": "error",
            "index": index,
            "variant": variant,
            "label": label,
            "prompt": prompt,
            "image_sha256": str(example["image_sha256"]),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def latest_by_index(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Keep the newest attempt so a transient per-example error can be retried."""
    return {
        int(record["index"]): record
        for record in records
        if "index" in record
    }


def summarize_variant(
    records: list[dict[str, Any]],
    *,
    variant: str,
    model: str,
    dataset: str,
    split: str,
    selected_indices: list[int],
    ready_count: int,
    unavailable_count: int,
    prompt_template: str,
) -> dict[str, Any]:
    selected_set = set(selected_indices)
    in_scope = [
        record
        for index, record in latest_by_index(records).items()
        if index in selected_set
    ]
    passed = [record for record in in_scope if record.get("status") == "passed"]
    failed = [record for record in in_scope if record.get("status") == "error"]
    metrics = {
        name: round(statistics.fmean(record["metrics"][name] for record in passed), 6)
        for name in ("precision", "recall", "f1")
    } if passed else {}
    latencies = [float(record["inference_seconds"]) for record in passed]
    return {
        "schema_version": 1,
        "variant": variant,
        "model": model,
        "dataset": dataset,
        "split": split,
        "prompt_template": prompt_template,
        "selected_examples": len(selected_indices),
        "ready_examples": ready_count,
        "unavailable_examples": unavailable_count,
        "passed_examples": len(passed),
        "failed_examples": len(failed),
        "positive_examples": sum(not record["is_negative"] for record in passed),
        "negative_examples": sum(record["is_negative"] for record in passed),
        "metrics": metrics,
        "performance": {
            "mean_inference_seconds": round(statistics.fmean(latencies), 6) if latencies else None,
            "median_inference_seconds": (
                round(statistics.median(latencies), 6) if latencies else None
            ),
            "max_peak_vram_gib": max(
                (float(record["peak_vram_gib"]) for record in passed), default=None
            ),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
