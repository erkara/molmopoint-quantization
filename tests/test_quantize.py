import pytest

from quantize import (
    CONFIGS,
    INT4_BF16_MODULES,
    PROCESSOR_PATTERNS,
    load_config,
    validate_config,
)


def test_final_configs_match_validated_recipes():
    int4 = load_config(CONFIGS["int4"])
    int8 = load_config(CONFIGS["int8"])
    validate_config(int4)
    validate_config(int8)

    assert int4["variant"] == "int4"
    assert int4["quantization"]["llm_int8_skip_modules"] == INT4_BF16_MODULES
    assert int4["output_dir"].name == "MolmoPoint-8B-bnb-nf4-vision-bf16"
    assert int8["variant"] == "int8"
    assert int8["quantization"]["llm_int8_skip_modules"] == ["model.point_predictor"]
    assert int8["output_dir"].name == "MolmoPoint-8B-bnb-int8-native"


def test_recipe_drift_is_rejected():
    config = {
        "variant": "int4",
        "quantization": {"method": "bitsandbytes", "load_in_4bit": True},
    }
    with pytest.raises(ValueError, match="validated recipe"):
        validate_config(config)


def test_processor_assets_cover_native_loading_files():
    required = {
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert required.issubset(PROCESSOR_PATTERNS)
