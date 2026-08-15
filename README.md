# MolmoPoint-8B Quantization

Reproducible INT4 and INT8 releases for
[`allenai/MolmoPoint-8B`](https://huggingface.co/allenai/MolmoPoint-8B), with
official-protocol quality evaluation against the BF16 model.

The repository has two jobs:

1. `quantize.py` builds the two validated Hugging Face-compatible checkpoints.
2. `evaluate.py` evaluates BF16, INT4, and INT8 with AllenAI's official
   PointBench or PointingEval protocol.

The final INT4 model quantizes the language transformer to NF4 while retaining
the vision path and pointing head in BF16. The final INT8 model uses LLM.int8
while retaining its pointing head in BF16 for native compatibility.

## Install

Python 3.11 or 3.12, CUDA, and a bitsandbytes-supported NVIDIA GPU are required
for model building and inference.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[official,dev]"
```

AllenAI's official evaluator is used from a separately pinned Molmo2 checkout.
Passing `--download-if-missing` lets `evaluate.py` create that checkout and
prepare missing public evaluation data.

## Build the final models

```bash
python quantize.py --variant int4 --smoke-image path/to/image.png
python quantize.py --variant int8 --smoke-image path/to/image.png
```

The two immutable release recipes are:

- [`configs/int4.yaml`](configs/int4.yaml)
- [`configs/int8.yaml`](configs/int8.yaml)

By default, the checkpoints are saved locally as:

```text
artifacts/MolmoPoint-8B-bnb-nf4-vision-bf16/
artifacts/MolmoPoint-8B-bnb-int8-native/
```

Each artifact contains packed weights, upstream model/processor assets, and a
`quantization_provenance.json`. The build refuses to overwrite an existing
checkpoint unless `--overwrite` is supplied. When a smoke image is provided,
the saved checkpoint must reload and run through native Transformers in a fresh
process.

## Evaluate the promoted models

Run BF16, INT4, and INT8 on the same deterministic 5% PointBench sample:

```bash
python evaluate.py \
  --variants all \
  --dataset pointbench \
  --fraction 0.05 \
  --seed 0 \
  --download-if-missing
```

Run the complete public PointBench task:

```bash
python evaluate.py --variants all --dataset pointbench --download-if-missing
```

Run all currently recoverable PixMo-Points examples:

```bash
python evaluate.py --variants all --dataset pixmo-points --download-if-missing
```

Useful options:

```text
--variants bf16,int4,int8   One or more variants; "all" is the default
--fraction 0.05             Deterministic fraction shared by every variant
--max-examples 50           Deterministic fixed-size sample
--seed 0                    Sample seed
--run-name NAME             Explicit output directory name
--data-dir PATH             Reusable dataset cache
--official-source PATH      Existing pinned AllenAI Molmo2 checkout
--download-if-missing       Permit source/data downloads
```

`--fraction` and `--max-examples` are mutually exclusive. Models run in
separate processes so CUDA state cannot leak between variants.

## Saved evaluation output

Every run is resumable and saved under `runs/<run-name>/`:

```text
selection.json          Exact shared positions and original source indices
bf16.jsonl              Raw per-example BF16 predictions
int4.jsonl              Raw per-example INT4 predictions
int8.jsonl              Raw per-example INT8 predictions
bf16-summary.json       Compact per-model result
int4-summary.json
int8-summary.json
summary.json            Combined comparison
```

`runs/`, `data/`, and `artifacts/` are intentionally excluded from Git. The
reviewed release results and figure are retained in [`RESULTS.md`](RESULTS.md).

## Dataset availability

- PointBench is fully public: 982/982 examples were reproduced.
- PixMo-Points publishes 436 metadata rows but stores images at external URLs.
  At evaluation time, only 231 images could be recovered and SHA-verified.

The evaluator records both selected-example count and public-data coverage; it
never labels a partial PixMo run as full.

## Repository map

```text
quantize.py       Build/package final INT4 or INT8
evaluate.py       Download data and run official evaluation
molmo_common.py   Shared native loading and deterministic inference
configs/          Two final release recipes
tests/            Lightweight CPU-only integration tests
assets/           Main figure and rejected-checkpoint provenance
huggingface/      Draft INT4 and INT8 model cards
RESULTS.md        Selection evidence and final results
```

The earlier fully modular research implementation is preserved on the
`codex/modular-archive` branch.

## Accuracy language

Use “agreement on tested examples,” not “lossless,” “identical,” or “no accuracy
loss.” Aggregate scores are close, but individual generations can differ.

## Attribution and publication

The checkpoints are derivatives of AllenAI's Apache-2.0-licensed
MolmoPoint-8B. They must retain upstream attribution and responsible-use
guidance. A source-code license for this repository must be selected before the
GitHub repository is made public.
