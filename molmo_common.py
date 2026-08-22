"""Shared native loading and deterministic inference for MolmoPoint checkpoints."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class Point:
    object_id: int | float
    image_id: int | float
    x: float
    y: float


@dataclass(frozen=True)
class InferenceResult:
    model: str
    processor: str
    variant: str
    prompt: str
    image: str
    generated_text: str
    generated_token_ids: list[int]
    points: list[Point]
    parse_success: bool
    inference_seconds: float
    peak_vram_gib: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["status"] = "passed"
        return payload


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("MolmoPoint-8B quantization and inference require a CUDA GPU")


def torch_dtype(name: str):
    import torch

    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }
    try:
        return dtypes[name]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {sorted(dtypes)}") from error


def load_model(
    model_id_or_path: str | Path,
    *,
    processor_id_or_path: str | Path | None = None,
    revision: str = "main",
    dtype: str = "bfloat16",
):
    """Load an upstream BF16 model or an already-packed checkpoint natively."""

    require_cuda()
    logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
    logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)

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


def run_image(
    model,
    processor,
    *,
    model_name: str,
    processor_name: str,
    variant: str,
    image_path: str | Path,
    prompt: str,
    max_new_tokens: int = 200,
) -> InferenceResult:
    """Open an image and run deterministic pointing inference."""
    image_path = Path(image_path).resolve()
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    return run_pil_image(
        model,
        processor,
        model_name=model_name,
        processor_name=processor_name,
        variant=variant,
        image=image,
        image_reference=str(image_path),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )


def run_pil_image(
    model,
    processor,
    *,
    model_name: str,
    processor_name: str,
    variant: str,
    image: Image.Image,
    image_reference: str,
    prompt: str,
    max_new_tokens: int = 200,
) -> InferenceResult:
    """Run greedy pointing inference and return absolute image coordinates."""

    import torch

    image = image.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
        return_pointing_metadata=True,
    )
    metadata = inputs.pop("metadata")
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(
            **inputs,
            logits_processor=model.build_logit_processor_from_inputs(inputs),
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started_at

    generated = output[:, prompt_length:]
    token_ids = [int(token_id) for token_id in generated[0].detach().cpu().tolist()]
    generated_text = processor.post_process_image_text_to_text(
        generated,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    raw_points = model.extract_image_points(
        generated_text,
        metadata["token_pooling"],
        metadata["subpatch_mapping"],
        metadata["image_sizes"],
    )
    points = [
        Point(
            object_id=_number(point[0]),
            image_id=_number(point[1]),
            x=float(point[2]),
            y=float(point[3]),
        )
        for point in (raw_points or [])
    ]
    return InferenceResult(
        model=model_name,
        processor=processor_name,
        variant=variant,
        prompt=prompt,
        image=image_reference,
        generated_text=generated_text,
        generated_token_ids=token_ids,
        points=points,
        parse_success=bool(points),
        inference_seconds=round(inference_seconds, 6),
        peak_vram_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 4),
    )


def write_json(path: str | Path, payload: dict) -> None:
    """Write JSON atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def read_jsonl(path: str | Path) -> list[dict]:
    """Read JSONL records, returning an empty list when absent."""
    source = Path(path)
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]


def _number(value) -> int | float:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric
