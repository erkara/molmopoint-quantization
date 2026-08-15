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
  - int8
  - quantized
---

# MolmoPoint-8B — bitsandbytes LLM.int8

> Draft model card. Replace the repository identifier before publication.

This is a bitsandbytes LLM.int8 derivative of `allenai/MolmoPoint-8B`. The
7.5M-parameter pointing head is retained in BF16 because bitsandbytes 0.50.0's
INT8 kernel does not natively accept its high-rank input tensors. This is under
0.1% of the model and avoids requiring users to install a runtime monkey patch.

## Usage

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

repo_id = "YOUR_USERNAME/MolmoPoint-8B-bnb-int8"
processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    repo_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
)
```

## Quantization details

- Quantized: all eligible linear modules outside `model.point_predictor`
- Retained in BF16: `model.point_predictor` (7.5M parameters)
- bitsandbytes outlier threshold: 6.0

## Evaluation

AllenAI's official Molmo2 formatter and evaluator code was pinned to commit
`f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a`. The full 982-example PointBench
score is 70.49 for this checkpoint versus 70.50 for reproduced BF16.

PixMo-Points was evaluated with AllenAI's official PointingEval scorer on all
231/436 examples whose source image bytes could currently be recovered and
verified. The remaining external source URLs returned changed bytes or were
inaccessible, so this is a partial public-data result rather than a reproduction
of the paper's full-dataset score.

| Model | Precision | Recall | F1 | Mean latency | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| BF16 base | 0.8643 | 0.8393 | 0.8440 | 1.0171 s | 18.2380 GiB |
| LLM.int8 | 0.8621 | 0.8361 | 0.8415 | 1.4637 s | 11.6551 GiB |

The official-protocol F1 difference is -0.0025 and peak allocated VRAM is 36.1%
lower. In the separate earlier fixed-prompt study, INT8 matched the exact generated token
sequence on 79.56%, the point count on 96.44%, and the task F1 on 216 examples.
Its paired macro-F1 difference was -0.0044. The checkpoint occupies about 9.3
GiB locally. Measurements used an NVIDIA GeForce RTX 5090 Laptop GPU,
Transformers 4.57.1, and bitsandbytes 0.50.0.

The saved checkpoint also passed a native fresh-process Transformers load and
inference test without the diagnostic compatibility patch.

## Limitations

This release inherits the upstream model's limitations and requires a CUDA
environment supported by bitsandbytes. The PointingEval result uses the official
protocol but only 231/436 publicly recoverable images; it should not be read as
a full-dataset score or proof that individual generations are identical to
BF16. INT8 was slower than BF16 on the benchmark hardware; its primary benefit
here is reduced memory and storage.
