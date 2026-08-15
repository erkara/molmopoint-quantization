from pathlib import Path

import pytest

from quantize import (
    CONFIGS,
    INT4_BF16_MODULES,
    PROCESSOR_PATTERNS,
    BuildConfig,
    Variant,
)


def test_final_configs_match_validated_recipes():
    int4 = BuildConfig.from_yaml(CONFIGS["int4"])
    int8 = BuildConfig.from_yaml(CONFIGS["int8"])
    int4.validate()
    int8.validate()

    assert int4.variant is Variant.INT4
    assert int4.quantization["llm_int8_skip_modules"] == INT4_BF16_MODULES
    assert int4.output_dir.name == "MolmoPoint-8B-bnb-nf4-vision-bf16"
    assert int8.variant is Variant.INT8
    assert int8.quantization["llm_int8_skip_modules"] == ["model.point_predictor"]
    assert int8.output_dir.name == "MolmoPoint-8B-bnb-int8-native"


def test_recipe_drift_is_rejected():
    config = BuildConfig(
        model_id="allenai/MolmoPoint-8B",
        revision="main",
        variant=Variant.INT4,
        output_dir=Path("artifacts/test"),
        quantization={"method": "bitsandbytes", "load_in_4bit": True},
    )
    with pytest.raises(ValueError, match="validated recipe"):
        config.validate()


def test_processor_assets_cover_native_loading_files():
    required = {
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert required.issubset(PROCESSOR_PATTERNS)
