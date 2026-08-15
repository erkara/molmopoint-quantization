"""Adapter between native Hugging Face checkpoints and AllenAI's evaluators."""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np


OFFICIAL_MOLMO2_COMMIT = "f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a"
OFFICIAL_PROMPT_TEMPLATES = "uber_model_v2"
OFFICIAL_TASKS = ("point_bench", "pointing_eval_v2")


def configure_official_source(source: str | Path, data_root: str | Path) -> Path:
    """Expose a pinned Molmo2 checkout and configure its public data root."""

    source = Path(source).resolve()
    if not (source / "olmo" / "eval" / "evaluators.py").is_file():
        raise FileNotFoundError(f"Not an AllenAI Molmo2 checkout: {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    os.environ["MOLMO_DATA_DIR"] = str(Path(data_root).resolve())
    return source


def verify_official_commit(source: str | Path) -> str:
    """Return the checkout commit and reject a protocol-code mismatch."""

    import subprocess

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(source),
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if commit != OFFICIAL_MOLMO2_COMMIT:
        raise RuntimeError(
            f"Expected AllenAI molmo2 commit {OFFICIAL_MOLMO2_COMMIT}, found {commit}"
        )
    return commit


def load_official_dataset(task: str, *, pixmo_path: str | Path | None = None):
    """Load an official PointBench or PointingEval dataset implementation."""

    if task == "point_bench":
        from olmo.data.academic_image_datasets import PointBench

        return PointBench()
    if task == "pointing_eval_v2":
        if pixmo_path is None:
            raise ValueError("--pixmo-path is required for pointing_eval_v2")
        from olmo.data.pixmo_datasets import PixMoPointsEval

        # The pinned public commit contains an Ai2-internal absolute path. Override
        # only the storage location; dataset parsing remains the official class.
        PixMoPointsEval.path = str(Path(pixmo_path).resolve())
        return PixMoPointsEval()
    raise ValueError(f"Unsupported official task {task!r}")


def official_prompt(example: dict[str, Any], source_index: int) -> str:
    """Apply MolmoPoint's deterministic official evaluation prompt templating."""

    from olmo.data.utils import make_random_state
    from olmo.models.molmo_point.molmo_point_data_formatter import MolmoPointDataFormatter

    prompt_example = dict(example)
    # Ground-truth points are evaluator metadata, never inference input. Removing
    # them also avoids constructing an unused target sequence.
    prompt_example.pop("points", None)
    formatter = MolmoPointDataFormatter(
        prompt_templates=OFFICIAL_PROMPT_TEMPLATES,
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
    if not messages:
        raise ValueError("Official formatter returned no messages")
    prompt = messages[0].text
    if not prompt:
        raise ValueError("Official formatter returned an empty prompt")
    return prompt


def evaluate_with_official_metric(
    task: str,
    *,
    metadata: dict[str, Any],
    prompt: str,
    generated_text: str,
    points: list[dict[str, Any]],
    tokenizer,
) -> dict[str, float]:
    """Score one prediction by calling AllenAI's official evaluator class."""

    predictions: dict[str, Any] = {
        "predictions_text": [generated_text],
        "prompts_text": [prompt],
    }
    if task == "point_bench":
        from olmo.eval.evaluators import PointBenchEval

        # This is the same [object_id, x, y] representation produced by
        # MolmoPointConfig.token_ids_to_coordinates in AllenAI's inference loop.
        predictions["points"] = [
            np.asarray(
                [[point["object_id"], point["x"], point["y"]] for point in points],
                dtype=np.float64,
            ).reshape(-1, 3)
        ]
        output = PointBenchEval(n_to_log=None)([metadata], predictions, tokenizer)
        category = str(metadata["category"])
        return {"accuracy": _metric_value(output[category])}

    if task == "pointing_eval_v2":
        from olmo.eval.evaluators import PointingEval
        from olmo.preprocessing.point_formatter import UnifiedPointFormatter

        width, height = metadata["image_size"]
        coordinates = np.asarray(
            [[point["x"], point["y"]] for point in points], dtype=np.float64
        ).reshape(-1, 2)
        # AllenAI's current PointingEval reads canonical coordinate text, while
        # grounding-token generation returns coordinates out-of-band. Serialize
        # those coordinates with AllenAI's own html-v2 formatter before scoring.
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
        return {name: _metric_value(output[name]) for name in ("precision", "recall", "f1")}

    raise ValueError(f"Unsupported official task {task!r}")


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
    latest = {int(record["source_index"]): record for record in records}
    passed = [record for record in latest.values() if record.get("status") == "passed"]
    failed = [record for record in latest.values() if record.get("status") == "error"]
    latencies = [float(record["inference_seconds"]) for record in passed]

    if task == "point_bench":
        categories = ["affordable", "counting", "reasoning", "spatial", "steerable"]
        category_scores = {}
        for category in categories:
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
        metrics = {
            name: round(statistics.fmean(record["metrics"][name] for record in passed), 6)
            for name in ("precision", "recall", "f1")
        } if passed else {}

    full_coverage = public_source_examples == expected_examples
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
        "protocol_coverage": "full" if full_coverage else "partial-public-recovery",
        "selected_examples": selected_examples,
        "passed_examples": len(passed),
        "failed_examples": len(failed),
        "evaluation_complete": len(passed) + len(failed) == selected_examples,
        "settings": {
            "prompt_templates": OFFICIAL_PROMPT_TEMPLATES,
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "batch_size": 1,
        },
        "metrics": metrics,
        "performance": {
            "mean_inference_seconds": round(statistics.fmean(latencies), 6) if latencies else None,
            "median_inference_seconds": round(statistics.median(latencies), 6) if latencies else None,
            "max_peak_vram_gib": max(
                (float(record["peak_vram_gib"]) for record in passed), default=None
            ),
        },
    }


def _metric_value(metric) -> float:
    value = metric.compute()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)
