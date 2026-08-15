# INT4 ablation results

We tested whether MolmoPoint-8B's INT4 behavior could be improved by keeping
small, potentially sensitive components in BF16. The strongest result is a
mixed-precision checkpoint that keeps the complete visual path and pointing
head in BF16 while quantizing the 7.6B-parameter language transformer to NF4.

All runs use upstream commit
`188130f961c8e0888a34e11121a1423c461a01ba`, deterministic generation, and the
same fixed prompt, `Point to the {label}.`. Full results cover the same 225
SHA256-verified PixMo Points Eval examples used in the main benchmark.

## Results

| Recipe | Stored size | Hard-slice F1 (14) | Full F1 (225) | Exact tokens vs BF16 | Exact point sets vs BF16 | Mean point distance vs BF16 |
|---|---:|---:|---:|---:|---:|---:|
| BF16 baseline | — | 0.5278 | 0.8360 | 100% | 100% | 0 px |
| Standard NF4 | 6.36 GB | 0.6071 | 0.8315 | 36.44% | 41.78% | 19.67 px |
| NF4 + pointing head BF16 | 6.37 GB | 0.6071 | 0.8315 | 43.11% | 48.89% | 19.99 px |
| NF4 without double quantization | 6.70 GB | 0.4847 | Not promoted | — | — | — |
| NF4 + bridge/head BF16 | 6.51 GB | 0.5519 | Not promoted | — | — | — |
| **NF4 + visual path/head BF16** | **7.08 GB** | **0.6945** | **0.8355** | **63.56%** | **69.78%** | **8.32 px** |

The 14-example slice was deliberately selected from disagreements and severe
failures. Its average F1 is useful for screening recipes, but it is not an
unbiased estimate of model quality and can exceed the BF16 score. Candidates
were promoted only when inspection showed a real qualitative repair, followed
by confirmation on the paired 225-example benchmark.

## What changed

The best recipe leaves these modules in BF16:

- `model.vit`
- `model.connector`
- `model.build_vit_embedding`
- `model.point_predictor`

The language transformer remains NF4 with BF16 compute, NF4 quantization, and
double quantization. Inspection of the saved checkpoint found 144 NF4
quantization-state entries under `model.transformer` and none under the four
excluded visual/pointing modules. The artifact also passed a fresh-process,
native Transformers reload and inference smoke test without a compatibility
patch.

On the full paired evaluation, this recipe differs from BF16 by only -0.000466
macro F1. It is better than BF16 on 10 examples, equal on 203, and worse on 12.
Point-count agreement is 92.0%. Compared with standard NF4, it raises macro F1
by 0.004046 and reduces the largest BF16-relative coordinate deviation from
805.9 px to 298.0 px.

The deliberately difficult red-cross example illustrates the practical effect:
standard NF4 predicts one point in the wrong corner (F1 0), while the visual-BF16
recipe predicts both target crosses (F1 1.0), matching the BF16 task outcome.

## Interpretation

Keeping only the 7.5M-parameter pointing head in BF16 changes coordinates and
increases exact-output agreement, but it leaves every task-F1 decision unchanged
on all 225 examples. Keeping the connector and pointing head in BF16 also fails
the hard-slice screen. Disabling double quantization is larger and worse.

The evidence therefore points to NF4 quantization of the vision encoder as the
main source of the severe INT4 drift. The residual gap shows that language-side
NF4 can still change generation, so the mixed checkpoint should be described as
high agreement on the tested examples, not as lossless or identical.

## Recommendation

The **visual-path-BF16 / language-NF4** recipe has been promoted to the sole
release recipe in `configs/int4.yaml`. It adds about 0.72 GB to the stored
checkpoint (7.08 GB versus 6.36 GB) but produces a materially stronger
behavioral match. Its one-image native-load smoke test used 7.82 GiB peak VRAM;
the comparable rejected all-layer-NF4 experiment used 7.15 GiB.

The smaller all-eligible-layers NF4 checkpoint is retained only as historical
ablation evidence and is not part of the documented production workflow.
