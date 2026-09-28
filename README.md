# MolmoPoint-8B Quantization

This repository contains the scripts and fixed recipes used to build and
evaluate two lower-memory versions of
[`allenai/MolmoPoint-8B`](https://huggingface.co/allenai/MolmoPoint-8B): INT4
and INT8.

Both versions remain close to the BF16 model on the tested pointing benchmarks
while using substantially less GPU memory.

| Model | PointBench accuracy | PixMo-Points F1 | Peak VRAM |
| --- | ---: | ---: | ---: |
| [BF16](https://huggingface.co/allenai/MolmoPoint-8B) | 70.90% | 84.02% | 18.26 GiB |
| [INT4](https://huggingface.co/erdi28/MolmoPoint-8B-bnb-nf4-vision-bf16) | 70.69% | 84.11% | 8.72 GiB |
| [INT8](https://huggingface.co/erdi28/MolmoPoint-8B-bnb-int8-native) | 71.00% | 84.49% | 11.68 GiB |

PixMo-Points results cover 231 of the dataset's 436 examples: the images that
could still be downloaded and verified. See [`RESULTS.md`](RESULTS.md) for the
full protocol, coverage, hardware details, and interpretation.

[![MolmoPoint quality and memory comparison](runs/full_comparison.png)](RESULTS.md)

Two scripts cover the complete workflow:

1. `quantize.py` builds Hugging Face-compatible INT4 and INT8 checkpoints.
2. `evaluate.py` compares BF16, INT4, and INT8 using AllenAI's official
   PointBench or PointingEval scorer.

INT4 quantizes the language transformer with NF4 while keeping the vision path
and pointing head in BF16. INT8 uses LLM.int8 for all supported linear layers
except the pointing head, so it loads without a runtime patch.

## Install

Python 3.11 or 3.12, CUDA, and a bitsandbytes-supported NVIDIA GPU are required
for model building and inference.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

AllenAI's official evaluator is used from a separately pinned Molmo2 checkout.
Passing `--download-if-missing` lets `evaluate.py` create that checkout and
prepare missing public evaluation data.

## Build the models

```bash
python quantize.py --variant int4
python quantize.py --variant int8
```

The two fixed recipes used for the releases are:

- [`configs/int4.yaml`](configs/int4.yaml)
- [`configs/int8.yaml`](configs/int8.yaml)

By default, the checkpoints are saved locally as:

```text
artifacts/MolmoPoint-8B-bnb-nf4-vision-bf16/
artifacts/MolmoPoint-8B-bnb-int8-native/
```

Each artifact contains quantized model weights, upstream model and processor
files, and a `quantization_provenance.json`. The build refuses to overwrite an
existing checkpoint unless `--overwrite` is supplied. Use `evaluate.py` to load
and validate the resulting checkpoints with real benchmark examples.

Checkpoint weights are not stored in Git. This repository contains the recipes
needed to build them locally, along with model cards under
[`huggingface/`](huggingface/).

## Evaluate the models

The evaluator supports two datasets. Choose one dataset per command;
`--variants all` evaluates BF16, INT4, and INT8 on that dataset.

| CLI value | Evaluation task | Public-data status | Validation |
| --- | --- | --- | --- |
| `pointbench` | PointBench with AllenAI's official scorer | Complete: 982/982 examples | 982 examples processed, zero execution errors |
| `pixmo-points` | PixMo-Points with AllenAI's PointingEval protocol | Partial: 231/436 public images were available and SHA-verified | 231 available examples processed, zero execution errors |

Run BF16, INT4, and INT8 on the same deterministic 5% sample:

```bash
python evaluate.py \
  --variants all \
  --dataset pointbench \
  --fraction 0.05 \
  --seed 0 \
  --download-if-missing
```

Use `--dataset pixmo-points` for PixMo-Points. Omit `--fraction` (or
`--max-examples`) to evaluate every available example:

```bash
python evaluate.py --variants all --dataset pointbench --download-if-missing
```

Every model in a command uses the same seeded example selection.

Useful options:

```text
--variants bf16,int4,int8   One or more variants; "all" is the default
--fraction 0.05             Deterministic fraction shared by every variant
--max-examples 50           Deterministic fixed-size sample
--seed 0                    Sample seed
--run-name NAME             Optional override for the generated run name
--data-dir PATH             Reusable dataset cache
--official-source PATH      Existing pinned AllenAI Molmo2 checkout
--download-if-missing       Permit source/data downloads
```

`--fraction` and `--max-examples` are mutually exclusive. Models run in
separate processes so GPU memory is reset between models.

## Saved evaluation output

Every run is resumable and saved under `runs/<run-name>/`:

```text
json_summaries/
  selection.json        Exact shared positions and original source indices
  bf16.jsonl            Raw per-example BF16 predictions
  int4.jsonl            Raw per-example INT4 predictions
  int8.jsonl            Raw per-example INT8 predictions
  bf16-summary.json     Compact per-model result
  int4-summary.json
  int8-summary.json
  summary.json          Combined comparison
full_comparison.png     Quality-versus-VRAM plot for the compared models
sample_predictions.png  Three deterministic qualitative examples
```

BF16 is included automatically when INT4 or INT8 is requested so candidate
results always have a baseline.

The sample command above writes to `runs/pointbench-5pct-seed0/` by default.

Local datasets, checkpoints, and newly generated evaluation runs are excluded
from Git. The two complete runs used for the published results remain tracked
under `runs/`, together with the combined figure at `runs/full_comparison.png`.

## Repository map

```text
quantize.py           Build/package final INT4 or INT8
evaluate.py           Download data and run official evaluation
molmo_common.py       Shared native loading and deterministic inference
render_comparison.py  Rebuild the combined publication figure
configs/              Two final release recipes
tests/                Lightweight CPU-only checks
runs/                 Published full evaluation records and figures
huggingface/          Model cards and demo assets for the Hugging Face releases
RESULTS.md            Selection evidence and final results
LICENSE               Apache License 2.0 for the original code and documentation
```

## Limitations

Aggregate scores are close, but individual generations can differ. Results
should be interpreted as agreement on tested examples, not as lossless or
identical behavior across model variants.

## License and attribution

Except where otherwise noted, the original code and documentation in this
repository are licensed under the [Apache License 2.0](LICENSE).

Quantized checkpoints are derived from `allenai/MolmoPoint-8B`. Third-party
components and assets retain their respective licenses and terms.
