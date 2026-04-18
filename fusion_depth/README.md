# Fusion Depth Study

Supporting data and code for the fusion depth-study reported in the main paper (Methods, Discussion) and Supplementary Table S1. This directory is the content formerly hosted as the separate `malleus` repository, collapsed into stapes so that all paper data lives in one place.

## Scope

All experiments here were performed on **PriMock57** (57 simulated primary care consultations, ≈12.6 hours) using the two best-performing solo on-device models identified in the main paper: `parakeet-tdt-0.6b-v2` and `whisper-distil-v3.5`. The goal was to test whether fusion algorithms more sophisticated than naive equal-weight ROVER — confusion network combination with parameter tuning, per-token-probability-weighted voting variants, and a learned per-word classifier — could meaningfully exceed naive ROVER's WER on the strongest pair. They did not; see Supplementary Table S1.

## Layout

- `scripts/` — Python scripts for each fusion variant. Entry points:
  - `rover_variants.py` — naive, confidence, Shannon, Tsallis, margin, asymmetric; oracle upper bound
  - `cnc_fusion.py` — confusion network combination with epsilon tuning
  - `run_eps_grid.sh` — epsilon-deletion × epsilon-insertion grid driver for `cnc_fusion.py`
  - `train_selector.py`, `learned_fusion.py` — learned per-word keep classifier (5-fold group CV by file)
  - `evaluate_with_logprobs.py`, `run_benchmark_with_logprobs.py` — per-file benchmark + token log-probability capture
  - `extract_disagreement_slots.py`, `empty_space_stats.py`, `empty_space_patterns.py`, `prob_diagnostic.py`, `headroom_analysis.py` — diagnostics
- `results/primock57/` — per-file outputs for every fusion variant, plus aggregate JSONs. Token log-probabilities are committed alongside hypotheses for full reproducibility.
- `lib/` — Dart implementations (`rover_fusion.dart`, `confidence_calibrator.dart`) for the on-device fusion prototype; not used in the paper evaluations.
- `research/tsallis_entropy.md` — notes on Tsallis entropy as a hallucination-detection signal.

## Reproducing Supplementary Table S1

Per-file hypothesis and token log-probability JSONLs for the two solo models are at `results/primock57/parakeet-tdt-0.6b-v2.jsonl` and `results/primock57/whisper-distil-v3.5.jsonl`. The CNC 3-model comparison additionally uses `sensevoice-no-itn.jsonl`. Each fusion variant's aggregate WER is in the corresponding result JSON; the epsilon grid is summarized in `results/primock57/eps_grid_sweep.tsv` and the learned-classifier sweep in `results/primock57/learned_sweep.tsv`.

## License

MIT (see `LICENSE`).
