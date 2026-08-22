# Results

The final INT4 and INT8 checkpoints preserve MolmoPoint-8B pointing quality
while substantially reducing peak allocated GPU memory.

![Official PointBench, PixMo-Points, and peak GPU memory comparison](runs/full_comparison.png)

## Release checkpoints

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
| BF16 | 86.36 | 73.47 | 77.72 | 77.95 | 39.00 | 70.90 | 18.26 GiB |
| **INT4** | **86.36** | **77.04** | **73.58** | **77.95** | **38.50** | **70.69** | **8.72 GiB** |
| **INT8** | **86.87** | **75.51** | **77.72** | **76.92** | **38.00** | **71.00** | **11.68 GiB** |

All three variants passed all 982 selected examples with zero failures. The
reproduced BF16 average, 70.90%, is close to the 70.7 reported for
MolmoPoint-8B. INT4 is -0.21 percentage points and INT8 is +0.10 points versus
the reproduced BF16 run. Small differences in either direction reflect
discrete prediction variation; they are not evidence that quantization improves
the base model.

## Official PointingEval — 231/436 publicly recoverable examples

PixMo-Points exposes metadata, masks, hashes, and external image URLs rather
than redistributing all image bytes. AllenAI's downloader plus an earlier
verified cache recovered 231 unique rows. The other 205 images had changed or
unavailable source URLs.

| Variant | Examples | Precision | Recall | F1 | F1 delta | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 231/436 | 86.09 | 83.50 | 84.02 | baseline | 18.26 GiB |
| **INT4** | **231/436** | **86.12** | **83.77** | **84.11** | **+0.08 pt** | **8.72 GiB** |
| **INT8** | **231/436** | **86.30** | **84.17** | **84.49** | **+0.47 pt** | **11.68 GiB** |

All three variants passed all 231 recoverable examples with zero failures. INT4
reduces peak allocated GPU memory by 52.3% and is +0.08 F1 points versus BF16;
INT8 reduces memory by 36.0% and is +0.47 points. These are official-scorer
results with 52.98% public-data coverage, not a full 436-example reproduction
and not directly comparable with the paper's full-dataset result.

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

- Fresh evaluation date: 2026-08-16
- PointBench run: `runs/pointbench-full-rerun-20260816/`
- PixMo-Points run: `runs/pixmo-points-full-rerun-20260816/`
- Upstream model revision:
  `188130f961c8e0888a34e11121a1423c461a01ba`
- Official Molmo2 evaluator commit:
  `f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a`
- Hardware: NVIDIA GeForce RTX 5090 Laptop GPU
- PyTorch: `2.10.0.dev20250916+cu130`
- Transformers: `4.57.1`
- bitsandbytes: `0.50.0`

Rejected checkpoint weights were removed after selection. The original modular
implementation and detailed intermediate reports remain available on the
`codex/modular-archive` branch.

## Interpretation

These results support “high agreement on tested examples.” They do not support
claims that quantization is lossless or that every generation is identical to
BF16. Peak VRAM and latency are environment-specific measurements rather than
universal hardware requirements.
