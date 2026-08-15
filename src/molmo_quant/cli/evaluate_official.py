"""Evaluate a Hugging Face MolmoPoint checkpoint with AllenAI's official protocol."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

from molmo_quant.inference import run_pil_pointing
from molmo_quant.loading import load_saved_model
from molmo_quant.official_eval import (
    OFFICIAL_TASKS,
    configure_official_source,
    evaluate_with_official_metric,
    load_official_dataset,
    official_prompt,
    summarize_records,
    verify_official_commit,
)
from molmo_quant.provenance import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", required=True)
    parser.add_argument("--data-root", default="data/official")
    parser.add_argument("--task", choices=OFFICIAL_TASKS, required=True)
    parser.add_argument("--pixmo-path")
    parser.add_argument("--model", required=True)
    parser.add_argument("--processor")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--variant", choices=("bf16", "int4", "int8"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--indices", help="Comma-separated positions in the available dataset")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--expected-examples", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_official_source(args.official_source, args.data_root)
    official_commit = verify_official_commit(args.official_source)
    dataset = load_official_dataset(args.task, pixmo_path=args.pixmo_path)
    positions = _select_positions(len(dataset), args.indices, args.max_examples)
    defaults = {"point_bench": 256, "pointing_eval_v2": 192}
    max_new_tokens = args.max_new_tokens or defaults[args.task]
    expected_examples = args.expected_examples or len(dataset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latest = _latest_by_source_index(_read_jsonl(output_path))
    pending = []
    for position in positions:
        source_index = _source_index(dataset, position)
        if latest.get(source_index, {}).get("status") != "passed":
            pending.append(position)
    print(
        f"[official-eval] {args.task}/{args.variant}: available={len(dataset)}, "
        f"selected={len(positions)}, pending={len(pending)}"
    )

    if pending:
        model, processor = load_saved_model(
            args.model,
            processor_id_or_path=args.processor,
            revision=args.revision,
        )
        model.eval()
        with output_path.open("a", encoding="utf-8", buffering=1) as handle:
            for number, position in enumerate(pending, start=1):
                record = _evaluate_one(
                    dataset,
                    position=position,
                    task=args.task,
                    model=model,
                    processor=processor,
                    model_name=args.model,
                    processor_name=args.processor or args.model,
                    variant=args.variant,
                    max_new_tokens=max_new_tokens,
                )
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                if number == 1 or number % 10 == 0 or number == len(pending):
                    print(
                        f"[official-eval] {args.task}/{args.variant}: "
                        f"{number}/{len(pending)} new; source_index={record['source_index']}; "
                        f"status={record['status']}"
                    )

    records = _read_jsonl(output_path)
    selected_sources = {_source_index(dataset, position) for position in positions}
    selected_records = [
        record
        for source_index, record in _latest_by_source_index(records).items()
        if source_index in selected_sources
    ]
    summary = summarize_records(
        selected_records,
        task=args.task,
        variant=args.variant,
        model=args.model,
        official_commit=official_commit,
        expected_examples=expected_examples,
        public_source_examples=len(dataset),
        selected_examples=len(positions),
        max_new_tokens=max_new_tokens,
    )
    write_json(args.summary, summary)
    print(f"[official-eval] wrote {args.summary}: {summary['metrics']}")
    return 0 if summary["failed_examples"] == 0 and summary["passed_examples"] else 1


def _evaluate_one(
    dataset,
    *,
    position: int,
    task: str,
    model,
    processor,
    model_name: str,
    processor_name: str,
    variant: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    example = dataset[position]
    source_index = _source_index(dataset, position)
    prompt = official_prompt(example, source_index)
    category = example.get("metadata", {}).get("category")
    label = example.get("label") or example.get("question")
    try:
        image_reference = str(example["image"])
        with Image.open(image_reference) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        result = run_pil_pointing(
            model,
            processor,
            model_name=model_name,
            processor_name=processor_name,
            variant=variant,
            image=image,
            image_reference=image_reference,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        ).to_dict()
        metadata = dict(example.get("metadata", {}))
        metadata["image_size"] = (width, height)
        if label is not None:
            metadata.setdefault("label", str(label))
        metrics = evaluate_with_official_metric(
            task,
            metadata=metadata,
            prompt=prompt,
            generated_text=result["generated_text"],
            points=result["points"],
            tokenizer=processor.tokenizer,
        )
        return {
            "schema_version": 1,
            "status": "passed",
            "task": task,
            "variant": variant,
            "position": position,
            "source_index": source_index,
            "category": category,
            "label": label,
            "prompt": prompt,
            "image": image_reference,
            "image_width": width,
            "image_height": height,
            "generated_text": result["generated_text"],
            "generated_token_ids": result["generated_token_ids"],
            "points": result["points"],
            "metrics": metrics,
            "inference_seconds": result["inference_seconds"],
            "peak_vram_gib": result["peak_vram_gib"],
        }
    except Exception as error:
        # A single unusually large image should not poison the remaining
        # resumable run after an out-of-memory exception.
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "schema_version": 1,
            "status": "error",
            "task": task,
            "variant": variant,
            "position": position,
            "source_index": source_index,
            "category": category,
            "label": label,
            "prompt": prompt,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


def _source_index(dataset, position: int) -> int:
    example = dataset[position]
    if "source_index" in example:
        return int(example["source_index"])
    raw_dataset = getattr(dataset, "data", None)
    if raw_dataset is not None and "source_index" in getattr(raw_dataset, "column_names", ()):
        return int(raw_dataset[position]["source_index"])
    return position


def _select_positions(length: int, indices: str | None, max_examples: int | None) -> list[int]:
    if indices and max_examples is not None:
        raise ValueError("Choose either --indices or --max-examples")
    if indices:
        positions = sorted({int(value.strip()) for value in indices.split(",")})
    else:
        limit = min(length, max_examples) if max_examples is not None else length
        positions = list(range(limit))
    invalid = [position for position in positions if position < 0 or position >= length]
    if invalid:
        raise ValueError(f"Dataset positions out of range: {invalid}")
    return positions


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _latest_by_source_index(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(record["source_index"]): record for record in records}


if __name__ == "__main__":
    raise SystemExit(main())
