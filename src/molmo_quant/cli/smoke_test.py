"""Run one deterministic image through a model and write JSON evidence."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from molmo_quant.config import INT4_BF16_MODULES, BuildConfig, Variant
from molmo_quant.inference import run_image_pointing
from molmo_quant.loading import load_saved_model, load_source_model
from molmo_quant.provenance import write_json


DEFAULT_PROMPT = "Point to the center of the colored spot or band."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="allenai/MolmoPoint-8B")
    parser.add_argument("--processor", default=None)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--variant", choices=[variant.value for variant in Variant], required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source-load",
        action="store_true",
        help="Load the upstream model and quantize on the fly instead of loading saved weights",
    )
    parser.add_argument(
        "--apply-int8-compat-patch",
        action="store_true",
        help="Diagnostic only: patch bitsandbytes before a saved INT8 load",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        failure = {
            "status": "error",
            "model": args.model,
            "processor": args.processor or args.model,
            "variant": args.variant,
            "prompt": args.prompt,
            "image": str(Path(args.image).resolve()),
            "int8_compat_patch_applied": args.apply_int8_compat_patch,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(args.output, failure)
        print(f"[smoke] ERROR: {type(error).__name__}: {error}")
        print(f"[smoke] Wrote failure evidence to {args.output}")
        return 1


def run(args: argparse.Namespace) -> int:
    variant = Variant(args.variant)

    if args.source_load:
        config = BuildConfig(
            model_id=args.model,
            revision=args.revision,
            variant=variant,
            output_dir=Path("."),
            quantization=(
                None
                if variant is Variant.BF16
                else {
                    "method": "bitsandbytes",
                    **(
                        {
                            "load_in_4bit": True,
                            "bnb_4bit_quant_type": "nf4",
                            "bnb_4bit_compute_dtype": "bfloat16",
                            "bnb_4bit_use_double_quant": True,
                            "llm_int8_skip_modules": INT4_BF16_MODULES,
                        }
                        if variant is Variant.INT4
                        else {
                            "load_in_8bit": True,
                            "llm_int8_threshold": 6.0,
                            "llm_int8_skip_modules": ["model.point_predictor"],
                        }
                    ),
                }
            ),
        )
        model, processor = load_source_model(config)
        processor_name = args.processor or args.model
        patch_applied = False
    else:
        model, processor = load_saved_model(
            args.model,
            processor_id_or_path=args.processor,
            revision=args.revision,
            apply_int8_compat_patch=args.apply_int8_compat_patch,
        )
        processor_name = args.processor or args.model
        patch_applied = args.apply_int8_compat_patch

    result = run_image_pointing(
        model,
        processor,
        model_name=args.model,
        processor_name=processor_name,
        variant=variant.value,
        image_path=args.image,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        int8_compat_patch_applied=patch_applied,
    )
    write_json(args.output, result.to_dict())
    print(
        f"[smoke] {variant.value}: parse_success={result.parse_success}, "
        f"points={len(result.points)}, inference={result.inference_seconds:.3f}s, "
        f"peak_vram={result.peak_vram_gib:.3f} GiB"
    )
    print(f"[smoke] Wrote {args.output}")
    return 0 if result.parse_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
