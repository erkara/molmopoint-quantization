# Results

The final INT4 and INT8 checkpoints preserve MolmoPoint-8B pointing quality
while substantially reducing peak allocated GPU memory.

![Official PointingEval F1 and peak GPU memory](assets/comparison.png)

## Final release candidates

- **INT4:** NF4 language transformer with BF16 compute and double
  quantization; vision encoder, multimodal connector, visual embedding, and
  pointing head remain BF16.
- **INT8:** bitsandbytes LLM.int8; the 7.5M-parameter pointing head remains BF16
  so the checkpoint loads natively without a runtime patch.

The evaluated local artifacts were:

```text
artifacts/MolmoPoint-8B-bnb-nf4-vision-bf16
artifacts/MolmoPoint-8B-bnb-int8-native
```

## Official PointBench — full 982 examples

AllenAI's official `PointBenchEval` and `uber_model_v2` prompt formatting were
used for all variants. Values are percentages; average is the unweighted mean
of the five official categories.

| Variant | Affordable | Counting | Reasoning | Spatial | Steerable | Average | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 86.36 | 72.45 | 77.72 | 77.95 | 38.00 | 70.50 | 18.24 GiB |
| **Final INT4** | **86.87** | **75.51** | **74.09** | **78.46** | **38.50** | **70.69** | **8.69 GiB** |
| **Final INT8** | **86.36** | **72.96** | **76.68** | **76.92** | **39.50** | **70.49** | **11.66 GiB** |

The reproduced BF16 average, 70.50%, is close to the 70.7 reported for
MolmoPoint-8B. INT4 is +0.19 percentage points and INT8 is -0.01 points versus
the reproduced BF16 run. The small positive INT4 difference reflects discrete
prediction variation; it is not evidence that quantization improves the base
model.

## Official PointingEval — 231/436 publicly recoverable examples

PixMo-Points exposes metadata, masks, hashes, and external image URLs rather
than redistributing all image bytes. AllenAI's downloader plus an earlier
verified cache recovered 231 unique rows. The other 205 images had changed or
unavailable source URLs.

| Variant | Examples | Precision | Recall | F1 | F1 delta | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 231/436 | 86.43 | 83.93 | 84.40 | baseline | 18.24 GiB |
| **Final INT4** | **231/436** | **86.12** | **83.70** | **84.06** | **-0.34 pt** | **8.69 GiB** |
| **Final INT8** | **231/436** | **86.21** | **83.61** | **84.15** | **-0.25 pt** | **11.66 GiB** |

INT4 reduces peak allocated GPU memory by 52.3% and remains within 0.34 F1
points of BF16. INT8 reduces memory by 36.1% and remains within 0.25 points.
This is an official-scorer result with 52.98% public-data coverage, not a full
436-example reproduction and not directly comparable with the paper's
full-dataset result.

## How the INT4 recipe was selected

The first all-layer NF4 checkpoint produced large failures on sensitive
examples. A 14-example diagnostic hard slice was used to compare candidate
recipes quickly; the strongest candidate was then confirmed on a broader
225-example paired evaluation.

| Recipe | Stored size | Hard-slice F1 | Full paired F1 | Exact point sets vs BF16 |
|---|---:|---:|---:|---:|
| BF16 baseline | — | 0.5278 | 0.8360 | 100% |
| Standard all-layer NF4 | 6.37 GB | 0.5278 | 0.8315 | 48.89% |
| NF4 without double quantization | 6.74 GB | 0.5278 | Not promoted | — |
| NF4 + BF16 pointing head | 6.37 GB | 0.6071 | 0.8315 | 48.89% |
| NF4 + BF16 bridge/head | 6.51 GB | 0.5519 | Not promoted | — |
| **NF4 + BF16 visual path/head** | **7.08 GB** | **0.6945** | **0.8355** | **69.78%** |

This experiment showed that vision-encoder quantization caused most severe
INT4 drift. Keeping only the pointing head or connector in BF16 did not repair
those failures. The final recipe therefore quantizes the 7.6B-parameter
language transformer while retaining the full visual path in BF16.

The hard slice was deliberately difficult and was used for model selection,
not as an unbiased quality estimate. Full PointBench is the strongest
independent final evaluation because it was not used to select the recipe.

## Earlier paired 225-example agreement study

The earlier study used the fixed prompt `Point to the {label}.` rather than the
official prompt templates. It remains useful for output-agreement diagnostics
but is separate from the official-protocol tables above.

| Variant | Precision | Recall | F1 | Exact tokens vs BF16 | Exact point sets vs BF16 |
|---|---:|---:|---:|---:|---:|
| BF16 | 85.85 | 83.07 | 83.60 | baseline | baseline |
| Final INT4 | 85.96 | 83.06 | 83.55 | 63.56% | 69.78% |
| Final INT8 | 85.11 | 82.70 | 83.16 | 79.56% | 83.56% |

## Reproducibility

- Upstream model revision:
  `188130f961c8e0888a34e11121a1423c461a01ba`
- Official Molmo2 evaluator commit:
  `f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a`
- Hardware: NVIDIA GeForce RTX 5090 Laptop GPU
- PyTorch: `2.10.0.dev20250916+cu130`
- Transformers: `4.57.1`
- bitsandbytes: `0.50.0`

Rejected checkpoint weights were removed after selection. Their compact build
metadata remains in [`assets/rejected-provenance.json`](assets/rejected-provenance.json).
The original modular implementation and detailed intermediate reports remain
available on the `codex/modular-archive` branch.

## Interpretation

These results support “high agreement on tested examples.” They do not support
claims that quantization is lossless or that every generation is identical to
BF16. Peak VRAM and latency are environment-specific measurements rather than
universal hardware requirements.
