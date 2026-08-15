---
base_model: allenai/MolmoPoint-8B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
license: apache-2.0
datasets:
  - allenai/pixmo-points-eval
tags:
  - molmo
  - molmopoint
  - bitsandbytes
  - nf4
  - mixed-precision
  - quantized
---

# MolmoPoint-8B — NF4 language model with BF16 vision

> Draft model card. Replace the repository identifier before publication.

This is a mixed-precision derivative of `allenai/MolmoPoint-8B`. The
7.6B-parameter language transformer uses bitsandbytes NF4 with BF16 compute and
double quantization. The vision encoder, multimodal connector, visual embedding
projection, and pointing head remain BF16 to preserve pointing behavior.

## Usage

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

repo_id = "YOUR_USERNAME/MolmoPoint-8B-bnb-nf4"
processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    repo_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
)
```

## Evaluation

AllenAI's official Molmo2 formatter and evaluator code was pinned to commit
`f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a`. The full 982-example PointBench
score is 70.69 for this checkpoint versus 70.50 for reproduced BF16.

PixMo-Points was evaluated with AllenAI's official PointingEval scorer on all
231/436 examples whose source image bytes could currently be recovered and
verified. The remaining external source URLs returned changed bytes or were
inaccessible, so this is a partial public-data result rather than a reproduction
of the paper's full-dataset score.

| Model | Precision | Recall | F1 | Mean latency | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| BF16 base | 0.8643 | 0.8393 | 0.8440 | 1.0171 s | 18.2380 GiB |
| Language NF4 + BF16 vision | 0.8612 | 0.8370 | 0.8406 | 0.9794 s | 8.6932 GiB |

The official-protocol F1 difference is -0.0034 and peak allocated VRAM is 52.3%
lower. In the separate earlier fixed-prompt study, INT4 matched the exact generated token
sequence on 63.56%, the point count on 92.00%, and the exact point set on
69.78%. Task F1 was equal on 203 examples, better on 10, and worse on 12. Its
paired macro-F1 difference was -0.000466. The checkpoint occupies 7.08 GB
(about 6.59 GiB) locally. Measurements used an NVIDIA GeForce RTX 5090 Laptop
GPU, Transformers 4.57.1, and bitsandbytes 0.50.0.

## Limitations

This release inherits the upstream model's limitations and requires a CUDA
environment supported by bitsandbytes. The PointingEval result uses the official
protocol but only 231/436 publicly recoverable images; it should not be read as
a full-dataset score or proof that individual generations are identical to
BF16. This is not an entirely 4-bit model: the visual path and pointing head
intentionally remain BF16.
