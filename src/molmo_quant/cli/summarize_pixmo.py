"""Compare PixMo evaluation JSONL files and write publication-ready summaries."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from molmo_quant.cli.evaluate_pixmo import read_jsonl
from molmo_quant.pointing_metrics import matched_prediction_distances
from molmo_quant.provenance import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", required=True)
    parser.add_argument("--int4", required=True)
    parser.add_argument("--int8", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--base-revision")
    parser.add_argument("--hardware")
    parser.add_argument("--run-date")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    variants = {
        "bf16": passed_by_index(read_jsonl(args.bf16)),
        "int4": passed_by_index(read_jsonl(args.int4)),
        "int8": passed_by_index(read_jsonl(args.int8)),
    }
    manifest = (
        json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        if args.manifest else None
    )
    report = build_report(
        variants,
        manifest=manifest,
        environment=environment_metadata(
            base_revision=args.base_revision,
            hardware=args.hardware,
            run_date=args.run_date,
        ),
    )
    write_json(args.output, report)
    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"[summary] Wrote {args.output}")
    print(f"[summary] Wrote {args.markdown}")
    return 0


def passed_by_index(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {
        int(record["index"]): record
        for record in records
        if record.get("status") == "passed"
    }


def build_report(
    variants: dict[str, dict[int, dict[str, Any]]],
    *,
    manifest: list[dict[str, Any]] | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quality = {name: quality_summary(records.values()) for name, records in variants.items()}
    report = {
        "schema_version": 1,
        "dataset": "allenai/pixmo-points-eval",
        "quality": quality,
        "agreement_with_bf16": {
            candidate: agreement_summary(variants["bf16"], variants[candidate])
            for candidate in ("int4", "int8")
        },
    }
    if manifest is not None:
        failure_reasons = Counter(
            _manifest_failure_reason(record.get("error"))
            for record in manifest
            if record.get("status") != "ready"
        )
        report["coverage"] = {
            "dataset_rows": len(manifest),
            "sha256_verified_examples": sum(
                record.get("status") == "ready" for record in manifest
            ),
            "unavailable_images": sum(
                record.get("status") != "ready" for record in manifest
            ),
            "unavailable_reasons": dict(sorted(failure_reasons.items())),
        }
    if environment is not None:
        report["environment"] = environment
    return report


def quality_summary(records) -> dict[str, Any]:
    records = list(records)
    return {
        "examples": len(records),
        "precision": _mean(records, "precision"),
        "recall": _mean(records, "recall"),
        "f1": _mean(records, "f1"),
        "mean_inference_seconds": round(
            statistics.fmean(float(record["inference_seconds"]) for record in records), 6
        ) if records else None,
        "max_peak_vram_gib": max(
            (float(record["peak_vram_gib"]) for record in records), default=None
        ),
    }


def agreement_summary(
    baseline: dict[int, dict[str, Any]], candidate: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    common = sorted(set(baseline).intersection(candidate))
    token_matches = 0
    count_matches = 0
    coordinate_distances: list[float] = []
    identical_points = 0
    for index in common:
        left = baseline[index]
        right = candidate[index]
        token_matches += left["generated_token_ids"] == right["generated_token_ids"]
        count_matches += len(left["points"]) == len(right["points"])
        distances = matched_prediction_distances(left["points"], right["points"])
        coordinate_distances.extend(distances)
        identical_points += (
            len(left["points"]) == len(right["points"])
            and all(distance == 0 for distance in distances)
        )
    count = len(common)
    return {
        "common_examples": count,
        "exact_token_matches": token_matches,
        "exact_token_agreement": round(token_matches / count, 6) if count else None,
        "point_count_matches": count_matches,
        "point_count_agreement": round(count_matches / count, 6) if count else None,
        "exact_point_set_matches": identical_points,
        "exact_point_set_agreement": round(identical_points / count, 6) if count else None,
        "matched_points": len(coordinate_distances),
        "mean_matched_point_distance_px": round(statistics.fmean(coordinate_distances), 6)
        if coordinate_distances else None,
        "max_matched_point_distance_px": round(max(coordinate_distances), 6)
        if coordinate_distances else None,
        "macro_f1_delta_vs_bf16": round(
            statistics.fmean(
                candidate[index]["metrics"]["f1"] - baseline[index]["metrics"]["f1"]
                for index in common
            ),
            6,
        ) if common else None,
        "examples_better_than_bf16": sum(
            candidate[index]["metrics"]["f1"] > baseline[index]["metrics"]["f1"]
            for index in common
        ),
        "examples_equal_to_bf16": sum(
            candidate[index]["metrics"]["f1"] == baseline[index]["metrics"]["f1"]
            for index in common
        ),
        "examples_worse_than_bf16": sum(
            candidate[index]["metrics"]["f1"] < baseline[index]["metrics"]["f1"]
            for index in common
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    quality = report["quality"]
    agreement = report["agreement_with_bf16"]
    lines = [
        "# PixMo Points Eval results",
        "",
        "Quality uses AllenAI's mask-based one-to-one pointing metric. Agreement compares each "
        "quantized model with BF16 on the same successfully evaluated examples.",
    ]
    if "coverage" in report:
        coverage = report["coverage"]
        reasons = coverage["unavailable_reasons"]
        lines.extend([
            "",
            "## Coverage",
            "",
            f"The dataset contains {coverage['dataset_rows']} rows. We evaluated "
            f"{coverage['sha256_verified_examples']} examples whose image bytes matched the "
            "dataset's "
            f"published SHA256. The remaining {coverage['unavailable_images']} rows were excluded: "
            f"{reasons.get('sha256_mismatch', 0)} source URLs returned changed bytes, "
            f"{reasons.get('http_error', 0)} returned HTTP errors, and "
            f"{reasons.get('network_error', 0)} failed due to URL/timeout errors.",
        ])
    if "environment" in report:
        environment = report["environment"]
        lines.extend([
            "",
            "## Environment",
            "",
            f"Run date: {environment.get('run_date') or 'not recorded'}. Hardware: "
            f"{environment.get('hardware') or 'not recorded'}. Base revision: "
            f"`{environment.get('base_revision') or 'not recorded'}`. PyTorch "
            f"{environment['torch']}; Transformers {environment['transformers']}; "
            f"bitsandbytes {environment['bitsandbytes']}.",
        ])
    lines.extend([
        "",
        "## Quality",
        "",
        "| Variant | Examples | Precision | Recall | F1 | Mean latency (s) | Peak VRAM (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for variant in ("bf16", "int4", "int8"):
        values = quality[variant]
        lines.append(
            f"| {variant.upper()} | {values['examples']} | {_fmt(values['precision'])} | "
            f"{_fmt(values['recall'])} | {_fmt(values['f1'])} | "
            f"{_fmt(values['mean_inference_seconds'])} | {_fmt(values['max_peak_vram_gib'])} |"
        )
    lines.extend([
        "",
        "## Agreement with BF16",
        "",
        "| Variant | Common examples | Exact tokens | Point count | Exact point set | "
        "Mean point distance (px) | Max point distance (px) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for variant in ("int4", "int8"):
        values = agreement[variant]
        lines.append(
            f"| {variant.upper()} | {values['common_examples']} | "
            f"{_pct(values['exact_token_agreement'])} | {_pct(values['point_count_agreement'])} | "
            f"{_pct(values['exact_point_set_agreement'])} | "
            f"{_fmt(values['mean_matched_point_distance_px'])} | "
            f"{_fmt(values['max_matched_point_distance_px'])} |"
        )
    lines.extend([
        "",
        "Paired task-F1 deltas versus BF16: "
        f"INT4 {_signed(agreement['int4']['macro_f1_delta_vs_bf16'])}; "
        f"INT8 {_signed(agreement['int8']['macro_f1_delta_vs_bf16'])}.",
        "",
        "The benchmark uses a fixed prompt, `Point to the {label}.`, for every variant. Images "
        "are accepted only when their bytes match the SHA256 published in the dataset.",
        "",
    ])
    return "\n".join(lines)


def _mean(records: list[dict[str, Any]], metric: str) -> float | None:
    return (
        round(statistics.fmean(record["metrics"][metric] for record in records), 6)
        if records else None
    )


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _signed(value: float | None) -> str:
    return "—" if value is None else f"{value:+.4f}"


def _manifest_failure_reason(error: str | None) -> str:
    error = error or ""
    if "SHA256 mismatch" in error:
        return "sha256_mismatch"
    if error.startswith("HTTPError"):
        return "http_error"
    return "network_error"


def environment_metadata(
    *, base_revision: str | None, hardware: str | None, run_date: str | None
) -> dict[str, Any]:
    import bitsandbytes
    import torch
    import transformers

    return {
        "run_date": run_date,
        "hardware": hardware,
        "base_revision": base_revision,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "cuda": torch.version.cuda,
    }


if __name__ == "__main__":
    raise SystemExit(main())
