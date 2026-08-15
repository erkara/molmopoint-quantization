"""Build and package a standalone quantized MolmoPoint checkpoint."""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from molmo_quant.config import BuildConfig, Variant
from molmo_quant.loading import load_source_model
from molmo_quant.provenance import (
    collect_provenance,
    resolve_hub_revision,
    sync_processor_assets,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an INT4 or INT8 YAML config")
    parser.add_argument(
        "--smoke-image",
        help="Optional image used for a native fresh-process reload after saving",
    )
    parser.add_argument(
        "--skip-fresh-process-smoke",
        action="store_true",
        help="Skip the reload gate even when --smoke-image is supplied",
    )
    parser.add_argument(
        "--smoke-output",
        help="JSON path for the fresh-process result (defaults under results/build)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BuildConfig.from_yaml(args.config)
    config.validate()
    if config.variant is Variant.BF16:
        raise ValueError("The build command is for INT4/INT8 packed checkpoints")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_revision = resolve_hub_revision(config.model_id, config.revision)
    resolved_config = replace(config, revision=resolved_revision)
    print(
        f"[build] Loading {config.model_id}@{resolved_revision} as {config.variant.value}"
    )
    started_at = time.perf_counter()
    model, processor = load_source_model(resolved_config)
    load_seconds = time.perf_counter() - started_at

    print(f"[build] Saving packed checkpoint to {config.output_dir}")
    started_at = time.perf_counter()
    model.save_pretrained(config.output_dir, safe_serialization=True)
    sync_processor_assets(config.model_id, resolved_revision, config.output_dir)
    save_seconds = time.perf_counter() - started_at

    provenance = collect_provenance(
        model_id=config.model_id,
        requested_revision=config.revision,
        resolved_revision=resolved_revision,
        variant=config.variant.value,
        quantization_config=config.quantization,
        load_seconds=load_seconds,
        save_seconds=save_seconds,
    )
    write_json(config.output_dir / "quantization_provenance.json", provenance)
    print(f"[build] Wrote {config.output_dir / 'quantization_provenance.json'}")

    if args.smoke_image and not args.skip_fresh_process_smoke:
        del model, processor
        gc.collect()
        import torch

        torch.cuda.empty_cache()
        smoke_output = Path(args.smoke_output) if args.smoke_output else (
            Path.cwd() / "results" / "build" / f"{config.output_dir.name}.fresh-process.json"
        )
        command = [
            sys.executable,
            "-m",
            "molmo_quant.cli.smoke_test",
            "--model",
            str(config.output_dir),
            "--variant",
            config.variant.value,
            "--image",
            args.smoke_image,
            "--output",
            str(smoke_output),
        ]
        print("[build] Starting native reload in a fresh process")
        subprocess.run(command, check=True)
    elif not args.smoke_image:
        print("[build] No --smoke-image supplied; fresh-process reload remains unverified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
