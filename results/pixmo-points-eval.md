# PixMo Points Eval results

Quality uses AllenAI's mask-based one-to-one pointing metric. Agreement compares each quantized model with BF16 on the same successfully evaluated examples.

## Coverage

The dataset contains 436 rows. We evaluated 225 examples whose image bytes matched the dataset's published SHA256. The remaining 211 rows were excluded: 151 source URLs returned changed bytes, 45 returned HTTP errors, and 15 failed due to URL/timeout errors.

## Environment

Run date: 2026-08-14. Hardware: NVIDIA GeForce RTX 5090 Laptop GPU. Base revision: `188130f961c8e0888a34e11121a1423c461a01ba`. PyTorch 2.10.0.dev20250916+cu130; Transformers 4.57.1; bitsandbytes 0.50.0.

## Quality

| Variant | Examples | Precision | Recall | F1 | Mean latency (s) | Peak VRAM (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 225 | 0.8585 | 0.8307 | 0.8360 | 0.9959 | 18.2383 |
| INT4 | 225 | 0.8596 | 0.8306 | 0.8355 | 0.9049 | 8.6934 |
| INT8 | 225 | 0.8511 | 0.8270 | 0.8316 | 1.3776 | 11.6552 |

## Agreement with BF16

| Variant | Common examples | Exact tokens | Point count | Exact point set | Mean point distance (px) | Max point distance (px) |
|---|---:|---:|---:|---:|---:|---:|
| INT4 | 225 | 63.56% | 92.00% | 69.78% | 8.3239 | 298.0462 |
| INT8 | 225 | 79.56% | 96.44% | 83.56% | 6.0979 | 516.4717 |

Paired task-F1 deltas versus BF16: INT4 -0.0005; INT8 -0.0044.

The benchmark uses a fixed prompt, `Point to the {label}.`, for every variant. Images are accepted only when their bytes match the SHA256 published in the dataset.
