# Supplementary Tables

All fusion experiments reported here were performed with the same text normalization pipeline and reference transcripts used in the main paper (Whisper English normalizer; figshare-OSCE reference with injected apostrophes per Methods). Source scripts and per-file results are available in the project repository under `fusion_depth/` (depth-study) and `scripts/round_robin_fusion.py` (round-robin).

## Table S1. Fusion algorithm comparison on the strongest two-model pair.

PriMock57 (57 files), parakeet-tdt-0.6b-v2 + whisper-distil-v3.5. Δ is against the best solo model (parakeet-tdt-0.6b-v2, 13.76% WER). "Uses token log-probs" indicates whether the method consumes per-token probability information from the ASR decoder, as exposed by the API proposed in sherpa-onnx PR #2897. Oracle is the lowest WER achievable by any word-picking voting scheme given the two hypotheses. CNC = Confusion Network Combination (Mangu et al., 2000); epsilon parameters control the cost of null arcs for handling insertions and deletions.

Data: `supplementary_table_S1_fusion_methods.csv`.

| Method | WER (%) | Δ (pp) | Uses token log-probs |
|---|---:|---:|:---:|
| parakeet-tdt-0.6b-v2 (solo) | 13.76 | — | no |
| whisper-distil-v3.5 (solo) | 15.27 | +1.51 | no |
| Naive ROVER (Fiscus 1997) | 13.41 | −0.35 | no |
| Confidence-weighted ROVER | 13.36 | −0.40 | yes |
| Shannon entropy-weighted | 13.30 | −0.46 | yes |
| Tsallis entropy-weighted | 13.41 | −0.35 | yes |
| Margin-weighted | 13.39 | −0.37 | yes |
| Asymmetric defaulting | 14.02 | +0.26 | yes |
| CNC default (Mangu 2000) | 13.80 | +0.04 | no |
| **CNC epsilon-tuned (del 0.6 / ins 0.7)** | **13.05** | **−0.71** | **no** |
| CNC 3-model (+ sensevoice-no-itn) | 15.54 | +1.78 | no |
| Learned per-word GBM (5-fold group CV) | 13.14 | −0.62 | yes |
| Oracle upper bound | 9.07 | −4.69 | n/a |

## Table S2. Round-robin pairwise ROVER WER, all 36 pairs × 2 datasets.

Naive equal-weight ROVER on the 9-model on-device pool. Rows sorted by dataset then WER ascending. figshare-OSCE is not included because its 272-file size made exhaustive search infeasible in our pure-Python implementation; the main paper reports only the targeted parakeet-tdt-0.6b-v2 + sensevoice fusion for that dataset.

Data: `supplementary_table_S2_rover_pairs.csv` (72 rows).

## Table S3. Round-robin three-model ROVER WER.

Naive equal-weight progressive pairwise ROVER. PriMock57 contains all 84 triples from the 9-model pool. The psychiatric dataset contains 12 of 84 triples — the exhaustive search was terminated due to runtime and no deliberate selection was applied to the completed subset; all 12 happen to include parakeet-tdt-0.6b-v2 as an artifact of parallel job ordering. figshare-OSCE is not included for the same runtime reason as Table S2.

Data: `supplementary_table_S3_rover_triples.csv` (96 rows).
