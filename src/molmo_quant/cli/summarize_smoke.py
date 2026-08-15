"""Summarize completed BF16, INT4, and INT8 smoke result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from molmo_quant.metrics import compare_to_baseline
from molmo_quant.provenance import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", required=True, help="Successful BF16 smoke JSON")
    parser.add_argument("--int4", required=True, help="Successful INT4 smoke JSON")
    parser.add_argument("--int8", required=True, help="Successful INT8 smoke JSON")
    parser.add_argument(
        "--int8-native-gate",
        help="Optional native INT8 smoke JSON, including a structured failure",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = {
        "bf16": _read_success(args.bf16, "bf16"),
        "int4": _read_success(args.int4, "int4"),
        "int8": _read_success(args.int8, "int8"),
    }
    baseline = results["bf16"]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "image": baseline.get("image"),
        "prompt": baseline.get("prompt"),
        "runs": {
            variant: {
                "status": result.get("status", "passed"),
                "parse_success": result.get("parse_success"),
                "point_count": len(result.get("points", [])),
                "inference_seconds": result.get("inference_seconds"),
                "peak_vram_gib": result.get("peak_vram_gib"),
                "int8_compat_patch_applied": result.get(
                    "int8_compat_patch_applied", False
                ),
            }
            for variant, result in results.items()
        },
        "comparisons_to_bf16": [
            compare_to_baseline(baseline, results["int4"]),
            compare_to_baseline(baseline, results["int8"]),
        ],
    }
    if args.int8_native_gate:
        gate = _read_json(args.int8_native_gate)
        summary["int8_native_gate"] = {
            "status": gate.get("status", "unknown"),
            "int8_compat_patch_applied": gate.get("int8_compat_patch_applied", False),
            "error_type": gate.get("error_type"),
            "error": gate.get("error"),
        }

    write_json(args.output, summary)
    print(f"[summarize] Wrote {args.output}")
    return 0


def _read_success(path: str, expected_variant: str) -> dict[str, Any]:
    result = _read_json(path)
    status = result.get("status")
    if status == "error" or not result.get("parse_success"):
        raise ValueError(f"Expected a successful {expected_variant} result in {path}")
    if result.get("variant") != expected_variant:
        raise ValueError(
            f"Expected variant {expected_variant!r} in {path}, got {result.get('variant')!r}"
        )
    return result


def _read_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
