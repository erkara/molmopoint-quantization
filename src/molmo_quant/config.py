"""Typed configuration for reproducible checkpoint builds."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class Variant(str, Enum):
    BF16 = "bf16"
    INT4 = "int4"
    INT8 = "int8"


INT4_BF16_MODULES = [
    "model.vit",
    "model.connector",
    "model.build_vit_embedding",
    "model.point_predictor",
]


@dataclass(frozen=True)
class BuildConfig:
    model_id: str
    revision: str
    variant: Variant
    output_dir: Path
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    device_map: str = "auto"
    quantization: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BuildConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping in {config_path}")

        required = {"model_id", "revision", "variant", "output_dir"}
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"Missing required keys in {config_path}: {sorted(missing)}")

        output_dir = Path(raw["output_dir"])
        if not output_dir.is_absolute():
            resolved_config_path = config_path.resolve()
            project_root = next(
                (
                    parent
                    for parent in resolved_config_path.parents
                    if (parent / "pyproject.toml").is_file()
                ),
                resolved_config_path.parent,
            )
            output_dir = (project_root / output_dir).resolve()

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
        if self.variant is Variant.BF16 and self.quantization:
            raise ValueError("BF16 builds must not define a quantization section")
        if self.variant is Variant.INT4:
            invariant = {
                "method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
            }
            self._require_quantization_values(invariant)
            actual = self.quantization or {}
            if not isinstance(actual.get("bnb_4bit_use_double_quant"), bool):
                raise ValueError(
                    "INT4 quantization must explicitly set bnb_4bit_use_double_quant"
                )
            skip_modules = actual.get("llm_int8_skip_modules")
            if skip_modules is not None and (
                not isinstance(skip_modules, list)
                or not all(isinstance(module, str) for module in skip_modules)
            ):
                raise ValueError("llm_int8_skip_modules must be a list of module names")
            allowed = set(invariant).union(
                {"bnb_4bit_use_double_quant", "llm_int8_skip_modules"}
            )
            unexpected = set(actual).difference(allowed)
            if unexpected:
                raise ValueError(f"Unexpected INT4 quantization settings: {sorted(unexpected)}")
        if self.variant is Variant.INT8:
            expected = {
                "method": "bitsandbytes",
                "load_in_8bit": True,
                "llm_int8_threshold": 6.0,
                "llm_int8_skip_modules": ["model.point_predictor"],
            }
            self._require_quantization_values(expected)

    def _require_quantization_values(self, expected: dict[str, Any]) -> None:
        actual = self.quantization or {}
        differences = {
            key: {"expected": value, "actual": actual.get(key)}
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if differences:
            raise ValueError(
                f"{self.variant.value} quantization settings do not match the tested recipe: "
                f"{differences}"
            )
