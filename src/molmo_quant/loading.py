"""Single source of truth for BF16, INT4, and INT8 model loading."""

from __future__ import annotations

import logging
from pathlib import Path

from .compatibility import patch_bnb_linear8bitlt_high_rank
from .config import BuildConfig, Variant


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("MolmoPoint-8B quantization and smoke inference require a CUDA GPU")


def torch_dtype(name: str):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {sorted(mapping)}") from error


def quantization_config(config: BuildConfig):
    """Construct the bitsandbytes recipe recorded in a validated build config."""

    import torch
    from transformers import BitsAndBytesConfig

    settings = config.quantization or {}
    if config.variant is Variant.INT4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=settings["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=torch_dtype(settings["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=settings["bnb_4bit_use_double_quant"],
            llm_int8_skip_modules=settings.get("llm_int8_skip_modules"),
        )
    if config.variant is Variant.INT8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=settings["llm_int8_threshold"],
            llm_int8_skip_modules=settings["llm_int8_skip_modules"],
        )
    if config.variant is Variant.BF16:
        return None
    raise ValueError(f"Unsupported variant: {config.variant}")


def load_source_model(config: BuildConfig):
    """Load the upstream model and quantize it on the fly when requested."""

    require_cuda()
    config.validate()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    _quiet_bitsandbytes_cast_logs()
    common = {
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
        "dtype": torch_dtype(config.dtype),
    }
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        padding_side="left",
    )

    if config.variant is Variant.BF16:
        model = AutoModelForImageTextToText.from_pretrained(config.model_id, **common).to("cuda")
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            config.model_id,
            quantization_config=quantization_config(config),
            device_map=config.device_map,
            **common,
        )
    return model, processor


def load_saved_model(
    model_id_or_path: str | Path,
    *,
    processor_id_or_path: str | Path | None = None,
    revision: str = "main",
    dtype: str = "bfloat16",
    apply_int8_compat_patch: bool = False,
):
    """Load an already-packed checkpoint without a quantization config.

    By default this is a native Transformers load. The optional INT8 patch is
    exposed for diagnosis, but results must record when it was required.
    """

    require_cuda()
    _quiet_bitsandbytes_cast_logs()
    if apply_int8_compat_patch:
        patch_bnb_linear8bitlt_high_rank()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_source = str(model_id_or_path)
    processor_source = str(processor_id_or_path or model_id_or_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_source,
        revision=revision,
        trust_remote_code=True,
        dtype=torch_dtype(dtype),
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        processor_source,
        revision=revision,
        trust_remote_code=True,
        padding_side="left",
    )
    return model, processor


def _quiet_bitsandbytes_cast_logs() -> None:
    """Hide per-layer dtype-cast warnings while preserving real errors."""

    logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
    logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)
