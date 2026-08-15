from pathlib import Path
import unittest

from molmo_quant.config import INT4_BF16_MODULES, BuildConfig, Variant


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BuildConfigTests(unittest.TestCase):
    def test_release_configs_match_tested_recipe(self):
        for filename, variant in [
            ("int4.yaml", Variant.INT4),
            ("int8.yaml", Variant.INT8),
        ]:
            with self.subTest(filename=filename):
                config = BuildConfig.from_yaml(PROJECT_ROOT / "configs" / filename)
                config.validate()
                self.assertIs(config.variant, variant)
                self.assertEqual(config.model_id, "allenai/MolmoPoint-8B")
                self.assertTrue(config.output_dir.is_absolute())
                if variant is Variant.INT4:
                    self.assertEqual(
                        config.quantization["llm_int8_skip_modules"],
                        INT4_BF16_MODULES,
                    )
                    self.assertEqual(
                        config.output_dir.name,
                        "MolmoPoint-8B-bnb-nf4-vision-bf16",
                    )
                else:
                    self.assertEqual(
                        config.quantization["llm_int8_skip_modules"],
                        ["model.point_predictor"],
                    )

    def test_int4_config_rejects_recipe_drift(self):
        config = BuildConfig(
            model_id="allenai/MolmoPoint-8B",
            revision="main",
            variant=Variant.INT4,
            output_dir=Path("artifacts/test"),
            quantization={"method": "bitsandbytes", "load_in_4bit": True},
        )
        with self.assertRaisesRegex(ValueError, "tested recipe"):
            config.validate()

    def test_int4_ablation_configs_are_explicit_and_valid(self):
        hybrid = BuildConfig.from_yaml(
            PROJECT_ROOT / "configs" / "ablations" / "int4-point-head-bf16.yaml"
        )
        hybrid.validate()
        self.assertEqual(
            hybrid.quantization["llm_int8_skip_modules"],
            ["model.point_predictor"],
        )

        no_double = BuildConfig.from_yaml(
            PROJECT_ROOT / "configs" / "ablations" / "int4-no-double-quant.yaml"
        )
        no_double.validate()
        self.assertFalse(no_double.quantization["bnb_4bit_use_double_quant"])



if __name__ == "__main__":
    unittest.main()
