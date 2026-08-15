"""Deterministic MolmoPoint image inference with machine-readable output."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class Point:
    object_id: int | float
    image_id: int | float
    x: float
    y: float


@dataclass(frozen=True)
class SmokeResult:
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
    int8_compat_patch_applied: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = "passed"
        return payload


def run_image_pointing(
    model,
    processor,
    *,
    model_name: str,
    processor_name: str,
    variant: str,
    image_path: str | Path,
    prompt: str,
    max_new_tokens: int = 200,
    int8_compat_patch_applied: bool = False,
) -> SmokeResult:
    image_path = Path(image_path).resolve()
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    return run_pil_pointing(
        model,
        processor,
        model_name=model_name,
        processor_name=processor_name,
        variant=variant,
        image=image,
        image_reference=str(image_path),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        int8_compat_patch_applied=int8_compat_patch_applied,
    )


def run_pil_pointing(
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
    int8_compat_patch_applied: bool = False,
) -> SmokeResult:
    """Run inference on an already-decoded RGB image.

    Dataset evaluation uses this entry point so cached images are decoded once
    and no temporary files are needed. Coordinates returned by MolmoPoint are
    absolute pixels in the original image coordinate system.
    """
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

    return SmokeResult(
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
        int8_compat_patch_applied=int8_compat_patch_applied,
    )


def _number(value) -> int | float:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric
