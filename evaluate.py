#!/usr/bin/env python3
"""Evaluate BF16 and promoted INT4/INT8 models with AllenAI's official protocol."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from molmo_common import load_model, read_jsonl, run_pil_image, write_json


PROJECT_ROOT = Path(__file__).resolve().parent
OFFICIAL_MOLMO2_COMMIT = "f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a"
BASE_REVISION = "188130f961c8e0888a34e11121a1423c461a01ba"
PROMPT_TEMPLATES = "uber_model_v2"
DATASETS = {
    "pointbench": {"task": "point_bench", "expected": 982, "tokens": 256},
    "pixmo-points": {"task": "pointing_eval_v2", "expected": 436, "tokens": 192},
}
MODEL_DEFAULTS = {
    "bf16": "allenai/MolmoPoint-8B",
    "int4": str(PROJECT_ROOT / "artifacts/MolmoPoint-8B-bnb-nf4-vision-bf16"),
    "int8": str(PROJECT_ROOT / "artifacts/MolmoPoint-8B-bnb-int8-native"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="all", help="all or comma-separated bf16,int4,int8")
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    sample = parser.add_mutually_exclusive_group()
    sample.add_argument("--fraction", type=float, help="Deterministic fraction, e.g. 0.05")
    sample.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name")
    parser.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data/official"))
    parser.add_argument(
        "--official-source",
        default=str(PROJECT_ROOT / "data/molmo2-official"),
        help="Pinned AllenAI Molmo2 checkout",
    )
    parser.add_argument("--pixmo-path")
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--bf16-model", default=MODEL_DEFAULTS["bf16"])
    parser.add_argument("--int4-model", default=MODEL_DEFAULTS["int4"])
    parser.add_argument("--int8-model", default=MODEL_DEFAULTS["int8"])
    parser.add_argument("--revision", default=BASE_REVISION)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_variant", choices=("bf16", "int4", "int8"), help=argparse.SUPPRESS)
    parser.add_argument("--_selection", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = prepare_official_source(args.official_source, args.download_if_missing)
    commit = configure_official_source(source, args.data_dir)
    dataset = load_dataset(
        args.dataset,
        data_dir=args.data_dir,
        pixmo_path=args.pixmo_path,
        download_if_missing=args.download_if_missing,
        download_workers=args.download_workers,
    )
    if args._worker:
        return run_worker(args, dataset, commit)
    return run_comparison(args, dataset, commit)


def run_comparison(args: argparse.Namespace, dataset, official_commit: str) -> int:
    variants = parse_variants(args.variants)
    positions = select_positions(
        len(dataset), fraction=args.fraction, max_examples=args.max_examples, seed=args.seed
    )
    run_name = args.run_name or default_run_name(
        args.dataset, fraction=args.fraction, max_examples=args.max_examples, seed=args.seed
    )
    run_dir = Path(args.runs_dir).resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    selection_path = run_dir / "selection.json"
    selection = {
        "schema_version": 1,
        "dataset": args.dataset,
        "official_molmo2_commit": official_commit,
        "available_examples": len(dataset),
        "expected_examples": DATASETS[args.dataset]["expected"],
        "fraction": args.fraction,
        "max_examples": args.max_examples,
        "seed": args.seed,
        "models": {
            variant: getattr(args, f"{variant}_model")
            for variant in ("bf16", "int4", "int8")
        },
        "base_revision": args.revision,
        "positions": positions,
        "source_indices": [_source_index(dataset, position) for position in positions],
    }
    if selection_path.exists():
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing != selection:
            raise RuntimeError(
                f"Run directory {run_dir} already has a different selection or model config; "
                "choose a new --run-name"
            )
    write_json(selection_path, selection)
    print(
        f"[evaluate] {args.dataset}: selected {len(positions)}/{len(dataset)} examples; "
        f"saved {selection_path}"
    )

    for variant in variants:
        command = worker_command(args, variant, selection_path)
        print(f"[evaluate] Starting isolated {variant} worker")
        subprocess.run(command, check=True)

    summaries = {
        variant: json.loads((run_dir / f"{variant}-summary.json").read_text(encoding="utf-8"))
        for variant in variants
    }
    aggregate = {
        "schema_version": 1,
        "run_name": run_name,
        "dataset": args.dataset,
        "selection": selection,
        "variants": summaries,
    }
    write_json(run_dir / "summary.json", aggregate)
    print_summary(summaries)
    print(f"[evaluate] Saved resumable outputs and summary under {run_dir}")
    return 0


def run_worker(args: argparse.Namespace, dataset, official_commit: str) -> int:
    if not args._variant or not args._selection:
        raise ValueError("Internal worker requires --_variant and --_selection")
    variant = args._variant
    selection_path = Path(args._selection).resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    positions = [int(value) for value in selection["positions"]]
    run_dir = selection_path.parent
    output_path = run_dir / f"{variant}.jsonl"
    summary_path = run_dir / f"{variant}-summary.json"
    latest = latest_by_source_index(read_jsonl(output_path))
    pending = [
        position
        for position in positions
        if latest.get(_source_index(dataset, position), {}).get("status") != "passed"
    ]
    model_name = getattr(args, f"{variant}_model")
    revision = args.revision if variant == "bf16" else "main"
    max_new_tokens = args.max_new_tokens or DATASETS[args.dataset]["tokens"]
    print(f"[evaluate] {args.dataset}/{variant}: selected={len(positions)}, pending={len(pending)}")

    if pending:
        model, processor = load_model(model_name, revision=revision)
        model.eval()
        with output_path.open("a", encoding="utf-8", buffering=1) as handle:
            for number, position in enumerate(pending, start=1):
                record = evaluate_one(
                    dataset,
                    position=position,
                    task=DATASETS[args.dataset]["task"],
                    model=model,
                    processor=processor,
                    model_name=model_name,
                    variant=variant,
                    max_new_tokens=max_new_tokens,
                )
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                if number == 1 or number % 10 == 0 or number == len(pending):
                    print(
                        f"[evaluate] {args.dataset}/{variant}: {number}/{len(pending)} new; "
                        f"source_index={record['source_index']}; status={record['status']}"
                    )

    selected_sources = {_source_index(dataset, position) for position in positions}
    records = [
        record
        for source_index, record in latest_by_source_index(read_jsonl(output_path)).items()
        if source_index in selected_sources
    ]
    summary = summarize_records(
        records,
        task=DATASETS[args.dataset]["task"],
        variant=variant,
        model=model_name,
        official_commit=official_commit,
        expected_examples=DATASETS[args.dataset]["expected"],
        public_source_examples=len(dataset),
        selected_examples=len(positions),
        max_new_tokens=max_new_tokens,
    )
    write_json(summary_path, summary)
    print(f"[evaluate] wrote {summary_path}: {summary['metrics']}")
    return 0 if summary["evaluation_complete"] and not summary["failed_examples"] else 1


def worker_command(
    args: argparse.Namespace, variant: str, selection_path: Path
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset",
        args.dataset,
        "--variants",
        variant,
        "--official-source",
        str(Path(args.official_source).resolve()),
        "--data-dir",
        str(Path(args.data_dir).resolve()),
        "--runs-dir",
        str(Path(args.runs_dir).resolve()),
        "--revision",
        args.revision,
        "--bf16-model",
        args.bf16_model,
        "--int4-model",
        args.int4_model,
        "--int8-model",
        args.int8_model,
        "--_worker",
        "--_variant",
        variant,
        "--_selection",
        str(selection_path),
    ]
    if args.pixmo_path:
        command.extend(["--pixmo-path", args.pixmo_path])
    if args.max_new_tokens:
        command.extend(["--max-new-tokens", str(args.max_new_tokens)])
    return command


def parse_variants(raw: str) -> list[str]:
    if raw == "all":
        return ["bf16", "int4", "int8"]
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    invalid = sorted(set(values).difference({"bf16", "int4", "int8"}))
    if invalid or not values:
        raise ValueError(f"Invalid variants: {invalid or values}")
    return list(dict.fromkeys(values))


def select_positions(
    length: int,
    *,
    fraction: float | None = None,
    max_examples: int | None = None,
    seed: int = 0,
) -> list[int]:
    if length < 1:
        raise ValueError("Dataset is empty")
    if fraction is not None:
        if not 0 < fraction <= 1:
            raise ValueError("--fraction must be in (0, 1]")
        count = max(1, min(length, round(length * fraction)))
    elif max_examples is not None:
        if max_examples < 1:
            raise ValueError("--max-examples must be positive")
        count = min(length, max_examples)
    else:
        count = length
    if count == length:
        return list(range(length))
    return sorted(random.Random(seed).sample(range(length), count))


def default_run_name(
    dataset: str,
    *,
    fraction: float | None,
    max_examples: int | None,
    seed: int,
) -> str:
    if fraction is not None:
        scope = f"{fraction * 100:g}pct"
    elif max_examples is not None:
        scope = f"n{max_examples}"
    else:
        scope = "full"
    return f"{dataset}-{scope}-seed{seed}"


def prepare_official_source(path: str | Path, download_if_missing: bool) -> Path:
    source = Path(path).resolve()
    if (source / "olmo/eval/evaluators.py").is_file():
        return source
    if not download_if_missing:
        raise FileNotFoundError(
            f"Pinned AllenAI Molmo2 source not found at {source}. "
            "Pass --download-if-missing or --official-source /path/to/molmo2."
        )
    source.parent.mkdir(parents=True, exist_ok=True)
    if source.exists() and any(source.iterdir()):
        raise FileExistsError(f"Cannot initialize non-empty official source directory: {source}")
    source.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=source, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/allenai/molmo2.git"],
        cwd=source,
        check=True,
    )
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", OFFICIAL_MOLMO2_COMMIT],
        cwd=source,
        check=True,
    )
    subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=source, check=True)
    return source


def configure_official_source(source: str | Path, data_dir: str | Path) -> str:
    source = Path(source).resolve()
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    os.environ["MOLMO_DATA_DIR"] = str(Path(data_dir).resolve())
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if commit != OFFICIAL_MOLMO2_COMMIT:
        raise RuntimeError(f"Expected Molmo2 {OFFICIAL_MOLMO2_COMMIT}, found {commit}")
    return commit


def load_dataset(
    name: str,
    *,
    data_dir: str | Path,
    pixmo_path: str | Path | None,
    download_if_missing: bool,
    download_workers: int,
):
    data_dir = Path(data_dir).resolve()
    if name == "pointbench":
        from huggingface_hub import hf_hub_download
        from olmo.data.academic_image_datasets import PointBench

        pointbench_root = data_dir / "torch_datasets/point_arena"
        data_json = pointbench_root / "data.json"
        ready = (
            data_json.is_file()
            and (pointbench_root / "selected_images").is_dir()
            and (pointbench_root / "selected_masks").is_dir()
        )
        if not ready:
            if not download_if_missing:
                raise FileNotFoundError(
                    f"PointBench is not fully prepared at {pointbench_root}; "
                    "pass --download-if-missing"
                )
            pointbench_root.mkdir(parents=True, exist_ok=True)
            if not data_json.is_file():
                cached = hf_hub_download(
                    repo_id="PointArena/pointarena-data",
                    filename="data.json",
                    repo_type="dataset",
                )
                shutil.copy2(cached, data_json)
            PointBench.download(n_procs=download_workers)
        return PointBench()

    from olmo.data.pixmo_datasets import PixMoPointsEval

    path = Path(
        pixmo_path
        or data_dir / "torch_datasets/pixmo_datasets/pixmo-points-eval-public-union"
    ).resolve()
    PixMoPointsEval.path = str(path)
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"PixMo-Points is not prepared at {path}; pass --download-if-missing"
            )
        PixMoPointsEval.download(n_procs=download_workers, check_sha=True)
    return PixMoPointsEval()


def official_prompt(example: dict[str, Any], source_index: int) -> str:
    from olmo.data.utils import make_random_state
    from olmo.models.molmo_point.molmo_point_data_formatter import MolmoPointDataFormatter

    prompt_example = dict(example)
    prompt_example.pop("points", None)
    formatter = MolmoPointDataFormatter(
        prompt_templates=PROMPT_TEMPLATES,
        message_format="none",
        system_prompt=None,
    )
    messages, _ = formatter(
        prompt_example,
        is_training=False,
        for_inference=True,
        rng=make_random_state(source_index, 0),
        points_to_indices=None,
    )
    if not messages or not messages[0].text:
        raise ValueError("Official formatter returned an empty prompt")
    return messages[0].text


def evaluate_one(
    dataset,
    *,
    position: int,
    task: str,
    model,
    processor,
    model_name: str,
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
        result = run_pil_image(
            model,
            processor,
            model_name=model_name,
            processor_name=model_name,
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
        metrics = official_metrics(
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
        gc.collect()
        import torch

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


def official_metrics(
    task: str,
    *,
    metadata: dict[str, Any],
    prompt: str,
    generated_text: str,
    points: list[dict[str, Any]],
    tokenizer,
) -> dict[str, float]:
    predictions: dict[str, Any] = {
        "predictions_text": [generated_text],
        "prompts_text": [prompt],
    }
    if task == "point_bench":
        from olmo.eval.evaluators import PointBenchEval

        predictions["points"] = [
            np.asarray(
                [[point["object_id"], point["x"], point["y"]] for point in points],
                dtype=np.float64,
            ).reshape(-1, 3)
        ]
        output = PointBenchEval(n_to_log=None)([metadata], predictions, tokenizer)
        return {"accuracy": metric_value(output[str(metadata["category"])])}

    from olmo.eval.evaluators import PointingEval
    from olmo.preprocessing.point_formatter import UnifiedPointFormatter

    width, height = metadata["image_size"]
    coordinates = np.asarray(
        [[point["x"], point["y"]] for point in points], dtype=np.float64
    ).reshape(-1, 2)
    predictions["predictions_text"] = [
        UnifiedPointFormatter.build_for_format("html-v2").format_image_points(
            coordinates,
            [width, height],
            str(metadata["label"]),
            mode="pointing",
        )
        if len(coordinates)
        else "There are none."
    ]
    output = PointingEval(n_to_log=None)([metadata], predictions, tokenizer)
    return {name: metric_value(output[name]) for name in ("precision", "recall", "f1")}


def summarize_records(
    records: list[dict[str, Any]],
    *,
    task: str,
    variant: str,
    model: str,
    official_commit: str,
    expected_examples: int,
    public_source_examples: int,
    selected_examples: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    latest = latest_by_source_index(records)
    passed = [record for record in latest.values() if record.get("status") == "passed"]
    failed = [record for record in latest.values() if record.get("status") == "error"]
    latencies = [float(record["inference_seconds"]) for record in passed]
    if task == "point_bench":
        category_scores = {}
        for category in ("affordable", "counting", "reasoning", "spatial", "steerable"):
            values = [
                record["metrics"]["accuracy"]
                for record in passed
                if record["category"] == category
            ]
            if values:
                category_scores[category] = statistics.fmean(values)
        metrics = {
            "categories": {key: round(value, 6) for key, value in category_scores.items()},
            "average": (
                round(statistics.fmean(category_scores.values()), 6)
                if category_scores
                else None
            ),
        }
    else:
        metrics = (
            {
                name: round(statistics.fmean(record["metrics"][name] for record in passed), 6)
                for name in ("precision", "recall", "f1")
            }
            if passed
            else {}
        )
    return {
        "schema_version": 1,
        "protocol": "allenai-molmo2-official-evaluator",
        "official_molmo2_commit": official_commit,
        "task": task,
        "variant": variant,
        "model": model,
        "expected_examples": expected_examples,
        "public_source_examples": public_source_examples,
        "coverage_fraction": round(public_source_examples / expected_examples, 6),
        "protocol_coverage": (
            "full" if public_source_examples == expected_examples else "partial-public-recovery"
        ),
        "selected_examples": selected_examples,
        "passed_examples": len(passed),
        "failed_examples": len(failed),
        "evaluation_complete": len(passed) + len(failed) == selected_examples,
        "settings": {
            "prompt_templates": PROMPT_TEMPLATES,
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "batch_size": 1,
        },
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


def latest_by_source_index(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(record["source_index"]): record for record in records}


def metric_value(metric) -> float:
    value = metric.compute()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _source_index(dataset, position: int) -> int:
    example = dataset[position]
    if "source_index" in example:
        return int(example["source_index"])
    raw_dataset = getattr(dataset, "data", None)
    if raw_dataset is not None and "source_index" in getattr(raw_dataset, "column_names", ()):
        return int(raw_dataset[position]["source_index"])
    return position


def print_summary(summaries: dict[str, dict[str, Any]]) -> None:
    print("\nvariant  quality     peak VRAM")
    print("-------  ----------  ---------")
    for variant, summary in summaries.items():
        metrics = summary["metrics"]
        quality = metrics.get("average", metrics.get("f1"))
        peak = summary["performance"]["max_peak_vram_gib"]
        print(f"{variant:<7}  {quality:<10.4f}  {peak:.2f} GiB")


if __name__ == "__main__":
    raise SystemExit(main())
