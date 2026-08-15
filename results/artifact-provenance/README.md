# Artifact provenance archive

This directory preserves the small `quantization_provenance.json` records from
rejected or superseded local checkpoints whose multi-gigabyte weight directories
were removed after the final release candidates passed official evaluation.

## Retained release artifacts

The following complete checkpoint directories remain under `artifacts/`:

- `MolmoPoint-8B-bnb-nf4-vision-bf16` — final INT4 release candidate.
- `MolmoPoint-8B-bnb-int8-native` — final INT8 release candidate.

## Removed weight directories

| Archived provenance | Former artifact | Status |
|---|---|---|
| `rejected/MolmoPoint-8B-bnb-nf4.json` | Original all-layer NF4 prototype | Superseded by final mixed-precision INT4 |
| `rejected/MolmoPoint-8B-bnb-nf4-no-double-quant.json` | NF4 without double quantization | Rejected ablation |
| `rejected/MolmoPoint-8B-bnb-nf4-point-head-bf16.json` | NF4 with BF16 pointing head | Rejected ablation |
| `rejected/MolmoPoint-8B-bnb-nf4-vision-bridge-bf16.json` | NF4 with BF16 bridge/head but quantized vision encoder | Rejected ablation |
| `rejected/MolmoPoint-8B-bnb-int8.json` | Fully INT8 prototype | Rejected because native inference required a runtime patch |

The recorded experimental results remain in
[`../int4-ablation.md`](../int4-ablation.md). Reproducible INT4 ablation recipes
remain under `configs/ablations/`. The rejected fully INT8 checkpoint is retained
only as provenance because it is not a release target.

These JSON files were copied byte-for-byte from their original checkpoint
directories before removal. They record the upstream revision, environment,
tool versions, variant, and—where available—the quantization configuration.
