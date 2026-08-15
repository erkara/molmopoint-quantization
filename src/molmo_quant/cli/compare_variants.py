"""Run BF16, INT4, and INT8 in isolated processes and compare outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from molmo_quant.metrics import compare_to_baseline
from molmo_quant.provenance import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Point to the center of the colored spot or band.")
    parser.add_argument("--base-model", default="allenai/MolmoPoint-8B")
    parser.add_argument("--int4-model", required=True)
    parser.add_argument("--int8-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        "bf16": (args.base_model, True),
        "int4": (args.int4_model, False),
        "int8": (args.int8_model, False),
    }
    results: dict[str, dict] = {}
    for variant, (model, source_load) in runs.items():
        destination = output_dir / f"{variant}.json"
        command = [
            sys.executable,
            "-m",
            "molmo_quant.cli.smoke_test",
            "--model",
            model,
            "--variant",
            variant,
            "--image",
            args.image,
            "--prompt",
            args.prompt,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--output",
            str(destination),
        ]
        if source_load:
            command.append("--source-load")
        print(f"[compare] Starting isolated {variant} run")
        subprocess.run(command, check=True)
        results[variant] = json.loads(destination.read_text(encoding="utf-8"))

    summary = {
        "image": str(Path(args.image).resolve()),
        "prompt": args.prompt,
        "comparisons": [
            compare_to_baseline(results["bf16"], results["int4"]),
            compare_to_baseline(results["bf16"], results["int8"]),
        ],
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[compare] Wrote {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

