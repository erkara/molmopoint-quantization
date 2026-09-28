# Results

The final INT4 and INT8 checkpoints remain close to MolmoPoint-8B's BF16
pointing quality while substantially reducing peak allocated GPU memory.

![Official PointBench, PixMo-Points, and peak GPU memory comparison](runs/full_comparison.png)

## Release checkpoints

- **[INT4 checkpoint](https://huggingface.co/erdi28/MolmoPoint-8B-bnb-nf4-vision-bf16):**
  NF4 language transformer with BF16 compute and double
  quantization; vision encoder, multimodal connector, visual embedding, and
  pointing head remain BF16.
- **[INT8 checkpoint](https://huggingface.co/erdi28/MolmoPoint-8B-bnb-int8-native):**
  bitsandbytes LLM.int8; the 7.5M-parameter pointing head remains BF16 so the
  checkpoint loads without a runtime patch.

## Official PointBench — full 982 examples

AllenAI's official `PointBenchEval` and `uber_model_v2` prompt formatting were
used for all variants. Values are percentages; average is the unweighted mean
of the five official categories.

| Variant | Affordable | Counting | Reasoning | Spatial | Steerable | Average | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 86.36 | 73.47 | 77.72 | 77.95 | 39.00 | 70.90 | 18.26 GiB |
| **INT4** | 86.36 | 77.04 | 73.58 | 77.95 | 38.50 | 70.69 | 8.72 GiB |
| **INT8** | 86.87 | 75.51 | 77.72 | 76.92 | 38.00 | 71.00 | 11.68 GiB |

All three variants passed all 982 selected examples with zero failures. The
reproduced BF16 average, 70.90%, is close to the 70.7 reported for
MolmoPoint-8B. INT4 is -0.21 percentage points and INT8 is +0.10 points versus
the reproduced BF16 run. Small differences in either direction reflect
discrete prediction variation; they are not evidence that quantization improves
the base model.

## PixMo-Points — official PointingEval on 231/436 available examples

PixMo-Points publishes metadata, masks, hashes, and external image URLs rather
than redistributing the image files. We downloaded and hash-verified 231 images.
The other 205 URLs were unavailable or returned files that no longer matched
the published hashes.

| Variant | Examples | Precision | Recall | F1 | F1 delta | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 231/436 | 86.09 | 83.50 | 84.02 | baseline | 18.26 GiB |
| **INT4** | 231/436 | 86.12 | 83.77 | 84.11 | +0.08 pt | 8.72 GiB |
| **INT8** | 231/436 | 86.30 | 84.17 | 84.49 | +0.47 pt | 11.68 GiB |

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

Keeping only the pointing head, or the connector and pointing head, in BF16
did not adequately repair the observed failures. Disabling double quantization
also did not improve the diagnostic results enough to select that recipe.
Keeping the complete visual path and pointing head in BF16 gave the strongest
candidate in this comparison.

These experiments support retaining the visual path at higher precision, but
do not isolate the effect of every component. The final recipe therefore
quantizes eligible linear layers in the language transformer while retaining
the vision encoder, connector, visual embedding, and pointing head in BF16.

The hard slice was deliberately difficult and was used for model selection,
not as an unbiased quality estimate. Full PointBench is the strongest
independent final evaluation because it was not used to select the recipe.

## Reproducibility

- Final evaluation date: 2026-08-16
- [PointBench run](runs/pointbench-full-rerun-20260816/)
- [PixMo-Points run](runs/pixmo-points-full-rerun-20260816/)
- Upstream model revision:
  `188130f961c8e0888a34e11121a1423c461a01ba`
- Official Molmo2 evaluator commit:
  `f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a`
- Hardware: NVIDIA GeForce RTX 5090 Laptop GPU
- PyTorch: `2.10.0.dev20250916+cu130`
- Transformers: `4.57.1`
- bitsandbytes: `0.50.0`

## Interpretation

These results support “high agreement on tested examples.” They do not support
claims that quantization is lossless or that every generation is identical to
BF16. Peak VRAM is an environment-specific measurement rather than a universal
hardware requirement.
