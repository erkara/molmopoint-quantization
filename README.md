# MolmoPoint-8B Quantization

Reproducible bitsandbytes quantization, clean-process loading checks, and
agreement evaluation for [`allenai/MolmoPoint-8B`](https://huggingface.co/allenai/MolmoPoint-8B).

This repository will produce two independently loadable Hugging Face model
repositories:

- **NF4 4-bit language transformer** with BF16 compute and double
  quantization; the visual path and pointing head remain BF16 for accuracy.
- **LLM.int8** using the bitsandbytes 8-bit format, with the 7.5M-parameter
  pointing head retained in BF16 for native compatibility.

The project is intentionally separate from the application where the original
prototype was developed. Public code here must not depend on that application.

## Status

The standalone build, clean-process compatibility tests, and public-data
evaluation are complete. Both release candidates load through native
Transformers calls without a project runtime patch. On the full 982-example
PointBench task, reproduced BF16/final INT4/final INT8 scores are
70.50/70.69/70.49. AllenAI's official PointingEval scorer on all 231/436
currently recoverable PixMo-Points rows gives F1 84.40/84.06/84.15, while peak
GPU memory is 18.24/8.69/11.66 GiB. The PixMo result is explicitly partial
because the dataset publishes external image URLs rather than the image bytes.
See [results/official-evaluation.md](results/official-evaluation.md) for the new
official-protocol results and
[results/pixmo-points-eval.md](results/pixmo-points-eval.md) for the separate
earlier 225-example agreement study. Model weights are not published yet.

## Environment

MolmoPoint currently requires `transformers==4.57.1` and custom model code.
The quantized variants additionally require CUDA and bitsandbytes.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,eval]"
```

The lightweight tests require no model weights, GPU, or inference run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## 1. Build a checkpoint

The build command loads the upstream model, applies the selected quantization,
saves the packed weights, copies the upstream processor/tokenizer assets, and
writes provenance metadata.

```bash
molmo-quant-build --config configs/int4.yaml --smoke-image path/to/image.jpg
molmo-quant-build --config configs/int8.yaml --smoke-image path/to/image.jpg
```

By default, artifacts are written under `artifacts/` and ignored by Git.

## 2. Run a one-image smoke test

Test the upstream BF16 baseline:

```bash
molmo-quant-smoke \
  --model allenai/MolmoPoint-8B \
  --processor allenai/MolmoPoint-8B \
  --variant bf16 \
  --image path/to/image.jpg \
  --output results/bf16.json
```

Test a saved checkpoint using a native Transformers load. This mode does not
apply the project compatibility patch, by design:

```bash
molmo-quant-smoke \
  --model artifacts/MolmoPoint-8B-bnb-nf4-vision-bf16 \
  --variant int4 \
  --image path/to/image.jpg \
  --output results/int4.json
```

For diagnosis of the original full-INT8 prototype only,
`--apply-int8-compat-patch` can be supplied when loading a saved checkpoint.
The release recipe instead leaves `model.point_predictor` in BF16 so users do
not need a process-wide workaround.

## 3. Compare all three variants

The comparison runner starts each model in a separate subprocess, preventing
CUDA state and monkey patches from leaking between variants:

```bash
molmo-quant-compare \
  --image path/to/image.jpg \
  --int4-model artifacts/MolmoPoint-8B-bnb-nf4-vision-bf16 \
  --int8-model artifacts/MolmoPoint-8B-bnb-int8-native \
  --output-dir results/smoke
```

It reports exact generated-token agreement, point-count agreement, and matched
point distances relative to BF16. Dataset-scale evaluation uses the same
inference and comparison conventions.

Existing smoke JSON files can be summarized without reloading the models:

```bash
molmo-quant-summarize \
  --bf16 results/build/MolmoPoint-8B-bf16.json \
  --int4 results/build/MolmoPoint-8B-bnb-nf4-vision-bf16.fresh-process.json \
  --int8 results/build/MolmoPoint-8B-bnb-int8-native.fresh-process.json \
  --output results/smoke-summary.json
```

See [results/README.md](results/README.md) for the reviewed preliminary result
and its limitations.

## 4. Evaluate on PixMo Points Eval

First download each source image and accept it only when its bytes match the
SHA256 published by the dataset:

```bash
molmo-quant-prepare-pixmo \
  --cache-dir data/pixmo-points-eval/images \
  --manifest data/pixmo-points-eval/manifest.json
```

Run one variant per process. Each JSONL is flushed per example and resumes
automatically from successful rows:

```bash
molmo-quant-eval-pixmo \
  --model allenai/MolmoPoint-8B \
  --processor allenai/MolmoPoint-8B \
  --revision 188130f961c8e0888a34e11121a1423c461a01ba \
  --variant bf16 \
  --output results/pixmo/full/bf16.jsonl \
  --summary results/pixmo/full/bf16-summary.json

molmo-quant-eval-pixmo \
  --model artifacts/MolmoPoint-8B-bnb-nf4-vision-bf16 \
  --variant int4 \
  --output results/pixmo/full/int4.jsonl \
  --summary results/pixmo/full/int4-summary.json

molmo-quant-eval-pixmo \
  --model artifacts/MolmoPoint-8B-bnb-int8-native \
  --variant int8 \
  --output results/pixmo/full/int8.jsonl \
  --summary results/pixmo/full/int8-summary.json
```

The evaluator follows AllenAI's one-to-one Hungarian matching and mask-membership
metric. It uses the fixed prompt `Point to the {label}.` for all variants, which
makes the paired comparison reproducible but is not directly comparable with an
official score produced using different prompt templating.

See [results/pixmo-points-eval.md](results/pixmo-points-eval.md) for the reviewed
results and coverage caveat. Of 436 dataset rows, 225 examples had source images that matched
their published hashes; changed or inaccessible source URLs were excluded.

## 5. Reproduce the official-protocol evaluation

The official adapter calls AllenAI's dataset formatter and evaluator classes
from a checkout pinned to Molmo2 commit
`f3cb1085fbb97c4a4d7fdcadd77cf871bf37a88a`. PointBench can be reproduced in
full; PixMo-Points requires a prepared SHA-verified dataset directory because
its image bytes are hosted at external source URLs.

```bash
molmo-quant-eval-official \
  --official-source /path/to/pinned/molmo2 \
  --data-root data/official \
  --task point_bench \
  --model artifacts/MolmoPoint-8B-bnb-nf4-vision-bf16 \
  --variant int4 \
  --output results/official/point-bench/int4.jsonl \
  --summary results/official/point-bench/int4-summary.json
```

All release results, exact settings, artifact identifiers, and the PixMo image
availability limitation are recorded in
[results/official-evaluation.md](results/official-evaluation.md).

### INT4 design decision

Controlled ablations showed that quantizing the vision encoder caused most of
the severe INT4 drift. Keeping only the pointing head or connector in BF16 did
not repair those failures, and disabling double quantization was larger and
worse. The production `configs/int4.yaml` therefore excludes `model.vit`,
`model.connector`, `model.build_vit_embedding`, and `model.point_predictor`
from quantization. See [results/int4-ablation.md](results/int4-ablation.md) for
the experimental evidence.

## Repository layout

See [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md) for the file-by-file responsibility,
dependency, ownership, and generated-artifact map.

```text
configs/       Reproducible build and evaluation settings
model_cards/   Hugging Face model-card drafts
scripts/       Direct source-tree entry points
src/           Installable molmo_quant package
tests/         Tests that do not download model weights
results/       Reviewed result summaries (large/raw outputs stay local)
```

## Accuracy language

Use “agreement on tested examples,” not “lossless,” “identical,” or “no accuracy
loss.” The aggregate F1 difference is small, but individual generations do
differ. On-the-fly quantization versus a saved quantized checkpoint establishes
serialization equivalence; it does not establish agreement with the BF16 base
model.

## License and attribution

The upstream MolmoPoint-8B weights are Apache-2.0 licensed and subject to Ai2's
documented responsible-use guidance. The license for this source repository
will be selected before publication. Each model card must retain upstream
attribution and describe the quantization as a derivative release.
