"""Checkpoint provenance and processor-asset packaging."""

from __future__ import annotations

import json
import platform
import shutil
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Any


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


def resolve_hub_revision(model_id: str, revision: str) -> str:
    path = Path(model_id)
    if path.exists():
        return str(path.resolve())

    from huggingface_hub import HfApi

    return HfApi().model_info(model_id, revision=revision).sha


def sync_processor_assets(model_id: str, revision: str, output_dir: str | Path) -> None:
    """Copy immutable upstream processor/tokenizer assets into the checkpoint."""

    from huggingface_hub import HfApi, hf_hub_download

    destination_root = Path(output_dir)
    repo_files = HfApi().list_repo_files(model_id, revision=revision)
    selected_files = sorted(
        filename
        for filename in repo_files
        if any(fnmatch(filename, pattern) for pattern in PROCESSOR_PATTERNS)
    )
    if not selected_files:
        raise RuntimeError(f"No processor assets matched in {model_id}@{revision}")

    for filename in selected_files:
        source = Path(hf_hub_download(model_id, filename=filename, revision=revision))
        destination = destination_root / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def collect_provenance(
    *,
    model_id: str,
    requested_revision: str,
    resolved_revision: str,
    variant: str,
    quantization_config: dict[str, Any] | None,
    load_seconds: float,
    save_seconds: float,
) -> dict[str, Any]:
    import bitsandbytes
    import torch
    import transformers

    return {
        "schema_version": 1,
        "base_model": model_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "variant": variant,
        "quantization_config": quantization_config,
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


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
