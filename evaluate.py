"""Evaluate BF16 and promoted quantizations with AllenAI's official protocol.

The parent process fixes a shared deterministic sample and launches one worker
per precision. Workers append one JSON record per example, making runs safe to
resume, while process isolation prevents model and CUDA state from leaking from
one variant into another.
"""

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
import textwrap
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from molmo_common import load_model, read_jsonl, run_pil_image, write_json

PROJECT_ROOT = Path(__file__).resolve().parent

# Immutable upstream inputs used to format prompts and score predictions. These
# pins keep future upstream changes from silently altering published results.
OFFICIAL_MOLMO2_COMMIT = "f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a"
BASE_MODEL_REVISION = "188130f961c8e0888a34e11121a1423c461a01ba"
PROMPT_TEMPLATES = "uber_model_v2"
DATASETS = {
    "pointbench": {"task": "point_bench", "expected": 982, "tokens": 256},
    "pixmo-points": {"task": "pointing_eval_v2", "expected": 436, "tokens": 192},
}
MODEL_DEFAULTS = {
    "bf16": "allenai/MolmoPoint-8B",
    "int4": "artifacts/MolmoPoint-8B-bnb-nf4-vision-bf16",
    "int8": "artifacts/MolmoPoint-8B-bnb-int8-native",
}


# --- Command-line entry point ----------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Create the public CLI plus private arguments used by worker processes."""

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
    parser.add_argument(
        "--revision",
        default=BASE_MODEL_REVISION,
        help="Pinned allenai/MolmoPoint-8B revision used for the BF16 baseline",
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_variant", choices=("bf16", "int4", "int8"), help=argparse.SUPPRESS)
    parser.add_argument("--_selection", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Prepare pinned dependencies and dispatch parent or worker execution."""

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


# --- Run orchestration and process isolation -------------------------------


def run_comparison(args: argparse.Namespace, dataset, official_commit: str) -> int:
    """Evaluate requested variants against one immutable, resumable selection."""
    variants = parse_variants(args.variants)
    positions = select_positions(
        len(dataset), fraction=args.fraction, max_examples=args.max_examples, seed=args.seed
    )
    run_name = args.run_name or default_run_name(
        args.dataset, fraction=args.fraction, max_examples=args.max_examples, seed=args.seed
    )
    run_dir = Path(args.runs_dir).resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    json_dir = run_json_directory(run_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    selection_path = json_dir / "selection.json"
    # Persist every input that affects example selection or model identity. A
    # resume is rejected if any of them differs from the original invocation.
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
            variant: getattr(args, f"{variant}_model") for variant in ("bf16", "int4", "int8")
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

    # A fresh process per model releases all CUDA allocations before the next
    # variant loads and makes peak-memory comparisons reproducible.
    for variant in variants:
        command = worker_command(args, variant, selection_path)
        print(f"[evaluate] Starting isolated {variant} worker")
        subprocess.run(command, check=True)

    summaries = {
        variant: json.loads((json_dir / f"{variant}-summary.json").read_text(encoding="utf-8"))
        for variant in variants
    }
    aggregate = {
        "schema_version": 1,
        "run_name": run_name,
        "dataset": args.dataset,
        "selection": selection,
        "variants": summaries,
    }
    write_json(json_dir / "summary.json", aggregate)
    comparison_path = run_dir / "full_comparison.png"
    render_run_comparison(summaries, dataset=args.dataset, output=comparison_path)
    samples_path = run_dir / "sample_predictions.png"
    render_sample_predictions(
        dataset,
        positions=positions,
        variants=variants,
        json_dir=json_dir,
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        seed=args.seed,
        output=samples_path,
    )
    print_summary(summaries)
    print(f"[evaluate] Saved comparison plot to {comparison_path}")
    print(f"[evaluate] Saved sample predictions to {samples_path}")
    print(f"[evaluate] Saved resumable outputs and summary under {run_dir}")
    return 0


def run_worker(args: argparse.Namespace, dataset, official_commit: str) -> int:
    """Evaluate or resume one model variant inside an isolated process."""
    if not args._variant or not args._selection:
        raise ValueError("Internal worker requires --_variant and --_selection")
    variant = args._variant
    selection_path = Path(args._selection).resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    positions = [int(value) for value in selection["positions"]]
    run_dir = selection_path.parent
    output_path = run_dir / f"{variant}.jsonl"
    summary_path = run_dir / f"{variant}-summary.json"
    # JSONL is append-only. A passed latest attempt is complete; failed or
    # missing examples remain pending and can be retried safely.
    latest = latest_by_source_index(read_jsonl(output_path))
    pending = [
        position
        for position in positions
        if latest.get(_source_index(dataset, position), {}).get("status") != "passed"
    ]
    model_name = getattr(args, f"{variant}_model")
    model_source = resolve_model_source(model_name)
    revision = args.revision if variant == "bf16" else "main"
    max_new_tokens = args.max_new_tokens or DATASETS[args.dataset]["tokens"]
    completed = len(positions) - len(pending)
    with tqdm(
        total=len(positions),
        initial=completed,
        desc=f"{args.dataset}/{variant}",
        unit="example",
        dynamic_ncols=True,
        file=sys.stdout,
    ) as progress:
        if not pending:
            progress.set_postfix(status="already complete")
        else:
            progress.set_postfix(stage="loading model")
            model, processor = load_model(model_source, revision=revision)
            model.eval()
            progress.set_postfix(stage="evaluating")
            with output_path.open("a", encoding="utf-8", buffering=1) as handle:
                for position in pending:
                    record = evaluate_one(
                        dataset,
                        position=position,
                        task=DATASETS[args.dataset]["task"],
                        data_dir=args.data_dir,
                        model=model,
                        processor=processor,
                        model_name=model_name,
                        variant=variant,
                        max_new_tokens=max_new_tokens,
                    )
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    progress.update()
                    progress.set_postfix(
                        source_index=record["source_index"],
                        status=record["status"],
                        refresh=False,
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


def worker_command(args: argparse.Namespace, variant: str, selection_path: Path) -> list[str]:
    """Build the internal subprocess command for one model variant."""

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


# --- Selection and official evaluator setup --------------------------------


def parse_variants(raw: str) -> list[str]:
    """Normalize variants and ensure every candidate has a BF16 baseline."""
    if raw == "all":
        return ["bf16", "int4", "int8"]
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    invalid = sorted(set(values).difference({"bf16", "int4", "int8"}))
    if invalid or not values:
        raise ValueError(f"Invalid variants: {invalid or values}")
    requested = list(dict.fromkeys(values))
    if "bf16" not in requested:
        requested.insert(0, "bf16")
    return [variant for variant in ("bf16", "int4", "int8") if variant in requested]


def resolve_model_source(model_name: str, project_dir: str | Path = PROJECT_ROOT) -> str:
    """Resolve a local model path for loading without changing its saved name."""
    path = Path(model_name).expanduser()
    if path.is_absolute():
        return str(path.resolve())

    project_path = Path(project_dir).resolve() / path
    if project_path.exists():
        return str(project_path.resolve())

    if path.exists():
        return str(path.resolve())

    # A non-existent path can be a Hugging Face repository id.
    return model_name


def select_positions(
    length: int,
    *,
    fraction: float | None = None,
    max_examples: int | None = None,
    seed: int = 0,
) -> list[int]:
    """Select deterministic positions while preserving dataset order."""
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
    """Describe the dataset, sample size, and seed in a stable directory name."""

    if fraction is not None:
        scope = f"{fraction * 100:g}pct"
    elif max_examples is not None:
        scope = f"n{max_examples}"
    else:
        scope = "full"
    return f"{dataset}-{scope}-seed{seed}"


def prepare_official_source(path: str | Path, download_if_missing: bool) -> Path:
    """Locate or minimally fetch the pinned AllenAI Molmo2 source tree."""
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
    # Fetch only the pinned commit; a complete Molmo2 history is unnecessary for
    # evaluation and would make clean-environment setup much larger.
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
    """Import the official evaluator, configure its cache, and verify its pin."""
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
    """Load PointBench or the publicly recoverable PixMo-Points subset.

    PointBench can be reconstructed completely from its public repository.
    PixMo-Points stores external image URLs, so its prepared directory may
    contain only images whose bytes remain available and SHA-verified.
    """
    data_dir = Path(data_dir).resolve()
    if name == "pointbench":
        from huggingface_hub import hf_hub_download
        from olmo.data.academic_image_datasets import PointBench

        pointbench_root = data_dir / "torch_datasets/point_arena"
        data_json = pointbench_root / "data.json"
        # ``data.json`` alone is insufficient: the official loader also expects
        # materialized images and masks.
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
        pixmo_path or data_dir / "torch_datasets/pixmo_datasets/pixmo-points-eval-public-union"
    ).resolve()
    # The official dataset class exposes its location as a class attribute.
    # Point it at the verified public union before constructing the dataset.
    PixMoPointsEval.path = str(path)
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"PixMo-Points is not prepared at {path}; pass --download-if-missing"
            )
        PixMoPointsEval.download(n_procs=download_workers, check_sha=True)
    return PixMoPointsEval()


# --- One-example inference and official scoring ----------------------------


def official_prompt(example: dict, source_index: int) -> str:
    """Reproduce AllenAI's deterministic inference prompt for one example."""
    from olmo.data.utils import make_random_state
    from olmo.models.molmo_point.molmo_point_data_formatter import MolmoPointDataFormatter

    # Ground-truth points must never enter the inference prompt.
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
    data_dir: str | Path,
    model,
    processor,
    model_name: str,
    variant: str,
    max_new_tokens: int,
) -> dict:
    """Run and officially score one example, returning a durable record.

    Errors become JSONL records instead of aborting the entire multi-hour run.
    A subsequent invocation retries them because only passed records count as
    complete.
    """
    example = dataset[position]
    source_index = _source_index(dataset, position)
    prompt = official_prompt(example, source_index)
    category = example.get("metadata", {}).get("category")
    label = example.get("label") or example.get("question")
    try:
        image_path = resolve_image_path(example["image"], data_dir=data_dir)
        image_reference = portable_image_reference(image_path, data_dir=data_dir)
        with Image.open(image_path) as opened:
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
        )
        metadata = dict(example.get("metadata", {}))
        metadata["image_size"] = (width, height)
        if label is not None:
            metadata.setdefault("label", str(label))
        # Use the exact upstream evaluator rather than a local approximation of
        # its coordinate conversion or matching rules.
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
    # Example-level failures are intentionally broad so one corrupt image or
    # transient model error does not discard all prior work. The traceback is
    # retained in the record and the worker exits nonzero if failures remain.
    except Exception as error:  # noqa: BLE001
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


def resolve_image_path(
    image_reference: str | Path,
    *,
    data_dir: str | Path,
    project_dir: str | Path = PROJECT_ROOT,
) -> Path:
    """Resolve an image across current and legacy dataset-cache layouts."""
    original = Path(image_reference).expanduser()
    data_root = Path(data_dir).expanduser().resolve()
    project_root = Path(project_dir).expanduser().resolve()
    candidates = [original]

    # Published JSONL files can contain absolute paths from the machine that ran
    # the evaluation. Rebase the suffix following ``data/`` for portability.
    if original.is_absolute():
        parts = original.parts
        if "data" in parts:
            data_index = parts.index("data")
            relative_to_data = Path(*parts[data_index + 1 :])
            candidates.extend(
                [
                    data_root / relative_to_data,
                    project_root / "data" / relative_to_data,
                ]
            )
    else:
        candidates.extend([data_root / original, project_root / original])

    attempted: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in attempted:
            continue
        attempted.append(candidate)
        if candidate.is_file():
            return candidate

    locations = "\n".join(f"  - {candidate}" for candidate in attempted)
    raise FileNotFoundError(
        f"Could not resolve dataset image {image_reference!s}. Attempted locations:\n{locations}"
    )


def portable_image_reference(
    image_path: str | Path,
    *,
    data_dir: str | Path,
    project_dir: str | Path = PROJECT_ROOT,
) -> str:
    """Store a portable image reference while inference uses the resolved path."""
    image_path = Path(image_path).resolve()
    project_root = Path(project_dir).resolve()
    data_root = Path(data_dir).resolve()

    try:
        return image_path.relative_to(project_root).as_posix()
    except ValueError:
        pass

    try:
        return image_path.relative_to(data_root).as_posix()
    except ValueError:
        # Custom datasets outside either configured root cannot be made relative
        # without losing the information needed to locate them again.
        return str(image_path)


def official_metrics(
    task: str,
    *,
    metadata: dict,
    prompt: str,
    generated_text: str,
    points: list[dict],
    tokenizer,
) -> dict:
    """Adapt decoded points to the selected official evaluator's input schema."""
    predictions = {
        "predictions_text": [generated_text],
        "prompts_text": [prompt],
    }
    if task == "point_bench":
        from olmo.eval.evaluators import PointBenchEval

        # PointBench consumes object id plus absolute x/y coordinates directly.
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
    # PointingEval expects the model's HTML-v2 representation rather than the
    # raw coordinate arrays accepted by PointBench.
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


# --- Summaries and visual outputs ------------------------------------------


def summarize_records(
    records: list[dict],
    *,
    task: str,
    variant: str,
    model: str,
    official_commit: str,
    expected_examples: int,
    public_source_examples: int,
    selected_examples: int,
    max_new_tokens: int,
) -> dict:
    """Aggregate quality, coverage, latency, and peak-memory metadata."""
    latest = latest_by_source_index(records)
    passed = [record for record in latest.values() if record.get("status") == "passed"]
    failed = [record for record in latest.values() if record.get("status") == "error"]
    latencies = [float(record["inference_seconds"]) for record in passed]
    if task == "point_bench":
        # Match the official headline metric: mean within each category, then
        # an unweighted mean across the five categories.
        category_scores = {}
        for category in ("affordable", "counting", "reasoning", "spatial", "steerable"):
            values = [
                record["metrics"]["accuracy"] for record in passed if record["category"] == category
            ]
            if values:
                category_scores[category] = statistics.fmean(values)
        metrics = {
            "categories": {key: round(value, 6) for key, value in category_scores.items()},
            "average": (
                round(statistics.fmean(category_scores.values()), 6) if category_scores else None
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
        "evaluation_complete": len(passed) == selected_examples and not failed,
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


def latest_by_source_index(records: list[dict]) -> dict:
    """Collapse append-only attempts so the latest record wins."""

    return {int(record["source_index"]): record for record in records}


def run_json_directory(run_dir: str | Path) -> Path:
    """Choose the current JSON subdirectory while supporting legacy resumes."""
    root = Path(run_dir)
    return root if (root / "selection.json").is_file() else root / "json_summaries"


def render_run_comparison(summaries: dict, *, dataset: str, output: str | Path) -> None:
    """Render a compact quality-and-memory comparison for one dataset run."""
    import matplotlib.pyplot as plt

    variants = list(summaries)
    labels = {"bf16": "BF16", "int4": "INT4", "int8": "INT8"}
    colors = {"bf16": "#2878B5", "int4": "#E67E22", "int8": "#2A9D72"}
    quality_key = "average" if dataset == "pointbench" else "f1"
    quality_label = "Average accuracy (%)" if dataset == "pointbench" else "F1 (%)"
    dataset_label = "PointBench" if dataset == "pointbench" else "PixMo-Points"
    quality = [100 * float(summaries[variant]["metrics"][quality_key]) for variant in variants]
    memory = [float(summaries[variant]["performance"]["max_peak_vram_gib"]) for variant in variants]

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.4), facecolor="white")
    figure.subplots_adjust(left=0.14, right=0.93, top=0.73, bottom=0.25, wspace=0.52)
    figure.suptitle(f"{dataset_label}: quality vs memory", fontsize=17, y=0.94)

    bar_labels = [labels[variant] for variant in variants]
    bar_colors = [colors[variant] for variant in variants]
    for axis, values, title, xlabel, limit in (
        (axes[0], quality, "Quality", quality_label, 100),
        (axes[1], memory, "Peak VRAM", "GiB", max(memory) * 1.15),
    ):
        bars = axis.barh(bar_labels, values, color=bar_colors, height=0.56)
        axis.set_xlim(0, limit)
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
        axis.set_xlabel(xlabel, color="#555555", labelpad=6)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        if len(variants) == 1:
            axis.set_ylim(0.8, -0.8)
        else:
            axis.invert_yaxis()
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                value + limit * 0.018,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                fontweight="bold",
            )

    first = summaries[variants[0]]
    selected = int(first["selected_examples"])
    public = int(first["public_source_examples"])
    expected = int(first["expected_examples"])
    if dataset == "pointbench":
        coverage = f"selected n={selected}/{public}" if selected != public else f"n={selected}"
    elif selected == public:
        coverage = f"public recovery n={public}/{expected}"
    else:
        coverage = f"selected n={selected}/{public} public · recovery {public}/{expected}"
    figure.text(
        0.5,
        0.07,
        f"Official evaluation · {coverage}",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_sample_predictions(
    dataset,
    *,
    positions: list[int],
    variants: list[str],
    json_dir: str | Path,
    data_dir: str | Path,
    dataset_name: str,
    seed: int,
    output: str | Path,
) -> None:
    """Render deterministic shared examples with each variant's predictions."""
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt

    json_root = Path(json_dir)
    records_by_variant = {
        variant: latest_by_source_index(read_jsonl(json_root / f"{variant}.jsonl"))
        for variant in variants
    }
    position_by_source = {_source_index(dataset, position): position for position in positions}
    # Only show examples successfully evaluated by every requested model; each
    # row then compares identical inputs across all columns.
    common_sources = [
        source_index
        for source_index in position_by_source
        if all(
            records_by_variant[variant].get(source_index, {}).get("status") == "passed"
            for variant in variants
        )
    ]
    if not common_sources:
        raise ValueError("No passed examples are shared by every requested variant")
    sample_count = min(3, len(common_sources))
    sampled_sources = sorted(random.Random(seed).sample(common_sources, sample_count))

    labels = {"bf16": "BF16", "int4": "INT4", "int8": "INT8"}
    colors = {"bf16": "#2878B5", "int4": "#E67E22", "int8": "#2A9D72"}
    figure = plt.figure(figsize=(4.0 * len(variants), 3.35 * sample_count + 1.1), facecolor="white")
    grid = figure.add_gridspec(
        sample_count * 2,
        len(variants),
        height_ratios=[0.22, 1.0] * sample_count,
        hspace=0.12,
        wspace=0.05,
    )
    figure.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.05)
    dataset_label = "PointBench" if dataset_name == "pointbench" else "PixMo-Points"
    figure.suptitle(f"{dataset_label}: sample predictions · seed {seed}", fontsize=18, y=0.97)

    for row, source_index in enumerate(sampled_sources):
        position = position_by_source[source_index]
        example = dataset[position]
        first_record = records_by_variant[variants[0]][source_index]
        prompt = textwrap.fill(str(first_record["prompt"]), width=115, max_lines=2, placeholder="…")
        prompt_axis = figure.add_subplot(grid[row * 2, :])
        prompt_axis.axis("off")
        prompt_axis.text(
            0.0,
            0.5,
            f"Example {source_index} · {prompt}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        image_path = resolve_image_path(example["image"], data_dir=data_dir)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        for column, variant in enumerate(variants):
            axis = figure.add_subplot(grid[row * 2 + 1, column])
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines[:].set_visible(False)
            if row == 0:
                axis.set_title(labels[variant], fontsize=12, fontweight="bold", pad=8)
            points = records_by_variant[variant][source_index].get("points", [])
            for number, point in enumerate(points, start=1):
                x, y = float(point["x"]), float(point["y"])
                axis.scatter(
                    [x],
                    [y],
                    s=120,
                    facecolors="none",
                    edgecolors="white",
                    linewidths=4.5,
                )
                axis.scatter(
                    [x],
                    [y],
                    s=120,
                    facecolors="none",
                    edgecolors=colors[variant],
                    linewidths=2.5,
                )
                label = axis.text(
                    x,
                    y,
                    str(number),
                    color=colors[variant],
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )
                label.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="white")])

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


# --- Small compatibility adapters ------------------------------------------


def metric_value(metric) -> float:
    """Normalize official metric wrappers and scalar tensors to a Python float."""

    value = metric.compute()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _source_index(dataset, position: int) -> int:
    """Recover a stable source id, falling back to the dataset position."""

    example = dataset[position]
    if "source_index" in example:
        return int(example["source_index"])
    raw_dataset = getattr(dataset, "data", None)
    if raw_dataset is not None and "source_index" in getattr(raw_dataset, "column_names", ()):
        return int(raw_dataset[position]["source_index"])
    return position


def print_summary(summaries: dict) -> None:
    """Print the headline quality and peak-memory metrics as a small table."""

    print("\nvariant  quality     peak VRAM")
    print("-------  ----------  ---------")
    for variant, summary in summaries.items():
        metrics = summary["metrics"]
        quality = metrics.get("average", metrics.get("f1"))
        peak = summary["performance"]["max_peak_vram_gib"]
        print(f"{variant:<7}  {quality:<10.4f}  {peak:.2f} GiB")


if __name__ == "__main__":
    raise SystemExit(main())
