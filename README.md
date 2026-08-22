# MolmoPoint-8B Quantization

Reproducible INT4 and INT8 releases for
[`allenai/MolmoPoint-8B`](https://huggingface.co/allenai/MolmoPoint-8B), with
official-protocol quality evaluation against the BF16 model.

[![MolmoPoint quality and memory comparison](runs/full_comparison.png)](RESULTS.md)

This repository provides two reproducible workflows:

1. `quantize.py` builds Hugging Face-compatible INT4 and INT8 checkpoints.
2. `evaluate.py` evaluates BF16, INT4, and INT8 with AllenAI's official
   PointBench or PointingEval protocol.

INT4 uses NF4 for the language transformer while retaining the vision path and
pointing head in BF16. INT8 uses LLM.int8 while retaining the pointing head in
BF16 for native compatibility.

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

## Build the models

```bash
python quantize.py --variant int4
python quantize.py --variant int8
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
checkpoint unless `--overwrite` is supplied. Use `evaluate.py` to load and
validate the resulting checkpoints with real benchmark examples.

Checkpoint weights are not stored in Git. This repository contains the recipes
needed to build them locally, along with model cards under [`huggingface/`](huggingface/).

## Evaluate the models

The evaluator supports exactly two datasets. One dataset is evaluated per
command; `--variants all` means all three model variants, not all datasets.

| CLI value | Evaluation task | Public-data status | Validation |
| --- | --- | --- | --- |
| `pointbench` | PointBench with AllenAI's official scorer | Complete: 982/982 examples | Full GPU run: 982/982 passed, zero failures |
| `pixmo-points` | PixMo-Points with AllenAI's PointingEval protocol | Partial public recovery: 231/436 images were available and SHA-verified | Full available-data GPU run: 231/231 passed, zero failures |

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

One dataset is evaluated per command. Every variant in a command uses the same
deterministic example selection.

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
separate processes so CUDA state cannot leak between variants.

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
full_comparison.png     Quality-versus-VRAM plot for the effective variants
sample_predictions.png  Three deterministic qualitative examples
```

BF16 is included automatically when INT4 or INT8 is requested so candidate
results always have a baseline.

The sample command above writes to `runs/pointbench-5pct-seed0/` by default.

Raw run outputs, `data/`, and `artifacts/` are intentionally excluded from Git.
The combined publication figure at `runs/full_comparison.png` is retained along
with the documented results in [`RESULTS.md`](RESULTS.md).

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
runs/             Published comparison figure; local run outputs are ignored
huggingface/      INT4 and INT8 model cards
RESULTS.md        Selection evidence and final results
```

## Limitations

Aggregate scores are close, but individual generations can differ. Results
should be interpreted as agreement on tested examples, not as lossless or
identical behavior across model variants.

## Attribution and publication

The checkpoints are derivatives of AllenAI's Apache-2.0-licensed
MolmoPoint-8B. They must retain upstream attribution and responsible-use
guidance. A source-code license for this repository must be selected before the
GitHub repository is made public.
