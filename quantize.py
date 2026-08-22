#!/usr/bin/env python3
"""Build the validated INT4 or INT8 MolmoPoint-8B release checkpoint."""

from __future__ import annotations

import argparse
import platform
import shutil
import time
from dataclasses import dataclass, replace
from enum import StrEnum
from fnmatch import fnmatch
from pathlib import Path

import yaml

from molmo_common import torch_dtype, write_json

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIGS = {"int4": PROJECT_ROOT / "configs/int4.yaml", "int8": PROJECT_ROOT / "configs/int8.yaml"}
PROCESSOR_PATTERNS = [
    "added_tokens.json",
    "chat_template.jinja",
    "image_processing_*.py",
    "merges.txt",
    "preprocessor_config.json",
    "processing_*.py",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "video_processing_*.py",
    "vocab.json",
]
INT4_BF16_MODULES = [
    "model.vit",
    "model.connector",
    "model.build_vit_embedding",
    "model.point_predictor",
]


class Variant(StrEnum):
    INT4 = "int4"
    INT8 = "int8"


@dataclass(frozen=True)
class BuildConfig:
    model_id: str
    revision: str
    variant: Variant
    output_dir: Path
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    device_map: str = "auto"
    quantization: dict | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> BuildConfig:
        """Load a release recipe from YAML."""
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping in {config_path}")
        required = {"model_id", "revision", "variant", "output_dir"}
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"Missing required keys in {config_path}: {sorted(missing)}")
        output_dir = Path(raw["output_dir"])
        if not output_dir.is_absolute():
            output_dir = (PROJECT_ROOT / output_dir).resolve()
        return cls(
            model_id=str(raw["model_id"]),
            revision=str(raw["revision"]),
            variant=Variant(str(raw["variant"])),
            output_dir=output_dir,
            dtype=str(raw.get("dtype", "bfloat16")),
            trust_remote_code=bool(raw.get("trust_remote_code", True)),
            device_map=str(raw.get("device_map", "auto")),
            quantization=raw.get("quantization"),
        )

    def validate(self) -> None:
        """Reject changes to the validated quantization recipe."""
        actual = self.quantization or {}
        if self.variant is Variant.INT4:
            expected = {
                "method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": True,
                "llm_int8_skip_modules": INT4_BF16_MODULES,
            }
        else:
            expected = {
                "method": "bitsandbytes",
                "load_in_8bit": True,
                "llm_int8_threshold": 6.0,
                "llm_int8_skip_modules": ["model.point_predictor"],
            }
        differences = {
            key: {"expected": value, "actual": actual.get(key)}
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if differences:
            raise ValueError(
                f"{self.variant.value} settings do not match the validated recipe: {differences}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("int4", "int8"), required=True)
    parser.add_argument("--config", help="Override the final config selected by --variant")
    parser.add_argument("--output-dir", help="Override the checkpoint destination")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BuildConfig.from_yaml(args.config or CONFIGS[args.variant])
    config.validate()
    if args.output_dir:
        config = replace(config, output_dir=Path(args.output_dir).resolve())
    if config.variant.value != args.variant:
        raise ValueError(f"Config variant {config.variant.value!r} does not match {args.variant!r}")
    if config.output_dir.exists() and any(config.output_dir.glob("*.safetensors")):
        if not args.overwrite:
            raise FileExistsError(
                f"Checkpoint already exists at {config.output_dir}; pass --overwrite explicitly"
            )

    resolved_revision = _resolve_revision(config.model_id, config.revision)
    resolved = replace(config, revision=resolved_revision)
    print(f"[quantize] Loading {config.model_id}@{resolved_revision} as {config.variant.value}")
    started_at = time.perf_counter()
    model = _load_source_model(resolved)
    load_seconds = time.perf_counter() - started_at

    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[quantize] Saving {config.output_dir}")
    started_at = time.perf_counter()
    model.save_pretrained(config.output_dir, safe_serialization=True)
    _sync_processor_assets(config.model_id, resolved_revision, config.output_dir)
    save_seconds = time.perf_counter() - started_at
    provenance = _collect_provenance(
        config,
        resolved_revision=resolved_revision,
        load_seconds=load_seconds,
        save_seconds=save_seconds,
    )
    write_json(config.output_dir / "quantization_provenance.json", provenance)

    return 0


def _load_source_model(config: BuildConfig):
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    from molmo_common import require_cuda

    require_cuda()
    settings = config.quantization or {}
    if config.variant is Variant.INT4:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype("bfloat16"),
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=INT4_BF16_MODULES,
        )
    else:
        quantization = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=settings["llm_int8_threshold"],
            llm_int8_skip_modules=settings["llm_int8_skip_modules"],
        )
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        dtype=torch_dtype(config.dtype),
        quantization_config=quantization,
        device_map=config.device_map,
    )
    return model


def _resolve_revision(model_id: str, revision: str) -> str:
    """Resolve a remote revision to an immutable commit."""
    path = Path(model_id)
    if path.exists():
        return str(path.resolve())
    from huggingface_hub import HfApi

    return HfApi().model_info(model_id, revision=revision).sha


def _sync_processor_assets(model_id: str, revision: str, output_dir: Path) -> None:
    """Copy the upstream processor files needed for native loading."""
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(model_id, revision=revision)
    selected = sorted(
        filename
        for filename in files
        if any(fnmatch(filename, pattern) for pattern in PROCESSOR_PATTERNS)
    )
    if not selected:
        raise RuntimeError(f"No processor assets matched in {model_id}@{revision}")
    for filename in selected:
        source = Path(hf_hub_download(model_id, filename=filename, revision=revision))
        destination = output_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _collect_provenance(
    config: BuildConfig,
    *,
    resolved_revision: str,
    load_seconds: float,
    save_seconds: float,
) -> dict:
    """Record the build recipe and environment."""
    import bitsandbytes
    import torch
    import transformers

    return {
        "schema_version": 1,
        "base_model": config.model_id,
        "requested_revision": config.revision,
        "resolved_revision": resolved_revision,
        "variant": config.variant.value,
        "quantization_config": config.quantization,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "bitsandbytes_version": bitsandbytes.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "load_seconds": round(load_seconds, 3),
        "save_seconds": round(save_seconds, 3),
    }


if __name__ == "__main__":
    raise SystemExit(main())
