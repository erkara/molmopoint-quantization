"""Build a validated INT4 or INT8 MolmoPoint-8B release checkpoint.

The release recipes are deliberately narrow: this module accepts only the two
quantization layouts that survived evaluation, resolves the upstream model to
an immutable revision, and records enough provenance to reproduce the build.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import time
from pathlib import Path

import yaml

from molmo_common import torch_dtype, write_json

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIGS = {"int4": PROJECT_ROOT / "configs/int4.yaml", "int8": PROJECT_ROOT / "configs/int8.yaml"}
# ``save_pretrained`` writes the model, but the custom MolmoPoint processor also
# needs these upstream files before the checkpoint can load on its own.
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
# The ablation study showed that quantizing the visual path caused the largest
# pointing regressions. These modules therefore remain BF16 in the INT4 release.
INT4_BF16_MODULES = [
    "model.vit",
    "model.connector",
    "model.build_vit_embedding",
    "model.point_predictor",
]


# --- Release recipe ---------------------------------------------------------


def load_config(path: str | Path) -> dict:
    """Read a YAML recipe and fill in the few supported defaults."""
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Expected a YAML mapping in {config_path}")

    required = {"model_id", "revision", "variant", "output_dir"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing required keys in {config_path}: {sorted(missing)}")

    config = dict(config)
    config["model_id"] = str(config["model_id"])
    config["revision"] = str(config["revision"])
    config["variant"] = str(config["variant"])
    config["dtype"] = str(config.get("dtype", "bfloat16"))
    config["trust_remote_code"] = bool(config.get("trust_remote_code", True))
    config["device_map"] = str(config.get("device_map", "auto"))
    config["quantization"] = config.get("quantization") or {}

    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    config["output_dir"] = output_dir.resolve()
    return config


def validate_config(config: dict) -> None:
    """Reject recipe changes that were not covered by the final evaluation."""
    if config["variant"] == "int4":
        expected = {
            "method": "bitsandbytes",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
            "llm_int8_skip_modules": INT4_BF16_MODULES,
        }
    elif config["variant"] == "int8":
        expected = {
            "method": "bitsandbytes",
            "load_in_8bit": True,
            "llm_int8_threshold": 6.0,
            "llm_int8_skip_modules": ["model.point_predictor"],
        }
    else:
        raise ValueError(f"Unsupported variant: {config['variant']!r}")

    actual = config["quantization"]
    differences = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if differences:
        raise ValueError(
            f"{config['variant']} settings do not match the validated recipe: {differences}"
        )


# --- Top-level build workflow ----------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Create the checkpoint-building command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("int4", "int8"), required=True)
    parser.add_argument("--config", help="Override the final config selected by --variant")
    parser.add_argument("--output-dir", help="Override the checkpoint destination")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate the recipe, build the checkpoint, and write its provenance."""

    args = build_parser().parse_args(argv)
    config = load_config(args.config or CONFIGS[args.variant])
    validate_config(config)
    if args.output_dir:
        config["output_dir"] = Path(args.output_dir).resolve()
    if config["variant"] != args.variant:
        raise ValueError(f"Config variant {config['variant']!r} does not match {args.variant!r}")
    # Refuse accidental multi-gigabyte overwrites unless the caller opts in.
    if (
        config["output_dir"].exists()
        and any(config["output_dir"].glob("*.safetensors"))
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Checkpoint already exists at {config['output_dir']}; pass --overwrite explicitly"
        )

    # Branch names such as ``main`` can move; provenance must point to the exact
    # upstream commit whose weights were quantized.
    resolved_revision = resolve_revision(config["model_id"], config["revision"])
    print(f"[quantize] Loading {config['model_id']}@{resolved_revision} as {config['variant']}")
    started_at = time.perf_counter()
    model = load_quantized_model(config, resolved_revision)
    load_seconds = time.perf_counter() - started_at

    config["output_dir"].mkdir(parents=True, exist_ok=True)
    print(f"[quantize] Saving {config['output_dir']}")
    started_at = time.perf_counter()
    model.save_pretrained(config["output_dir"], safe_serialization=True)
    copy_processor_files(config["model_id"], resolved_revision, config["output_dir"])
    save_seconds = time.perf_counter() - started_at
    provenance = collect_provenance(
        config,
        resolved_revision=resolved_revision,
        load_seconds=load_seconds,
        save_seconds=save_seconds,
    )
    write_json(config["output_dir"] / "quantization_provenance.json", provenance)

    return 0


# --- Model loading and checkpoint packaging --------------------------------


def load_quantized_model(config: dict, resolved_revision: str):
    """Load the base model while bitsandbytes replaces eligible linear layers."""

    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    from molmo_common import require_cuda

    require_cuda()
    settings = config["quantization"]
    if config["variant"] == "int4":
        # Keep this explicit rather than forwarding arbitrary YAML options: the
        # validated release recipe is part of the public quality claim.
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
        config["model_id"],
        revision=resolved_revision,
        trust_remote_code=config["trust_remote_code"],
        dtype=torch_dtype(config["dtype"]),
        quantization_config=quantization,
        device_map=config["device_map"],
    )
    return model


def resolve_revision(model_id: str, revision: str) -> str:
    """Resolve a Hub revision to a commit SHA, preserving local model paths."""
    path = Path(model_id)
    if path.exists():
        return str(path.resolve())
    from huggingface_hub import HfApi

    return HfApi().model_info(model_id, revision=revision).sha


def copy_processor_files(model_id: str, revision: str, output_dir: Path) -> None:
    """Copy custom processor/tokenizer files required for standalone loading."""
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(model_id, revision=revision)
    selected = sorted(
        filename
        for filename in files
        if any(Path(filename).match(pattern) for pattern in PROCESSOR_PATTERNS)
    )
    if not selected:
        raise RuntimeError(f"No processor assets matched in {model_id}@{revision}")
    for filename in selected:
        source = Path(hf_hub_download(model_id, filename=filename, revision=revision))
        destination = output_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def collect_provenance(
    config: dict,
    *,
    resolved_revision: str,
    load_seconds: float,
    save_seconds: float,
) -> dict:
    """Describe the exact recipe, software stack, hardware, and build timing."""
    import bitsandbytes
    import torch
    import transformers

    return {
        "schema_version": 1,
        "base_model": config["model_id"],
        "requested_revision": config["revision"],
        "resolved_revision": resolved_revision,
        "variant": config["variant"],
        "quantization_config": config["quantization"],
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
