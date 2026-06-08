#!/usr/bin/env python3
"""BCa bootstrap 95% CIs for per-model clinical term recall (CTR) over files.

Reads results/per_file_ctr.tsv (produced by compute_per_file_ctr.py) and for
each (dataset, model) cell:
    - sorts the cell's files by file_id (so resample slot i is the same file
      across every model on that dataset),
    - resamples files with replacement 10,000 times,
    - computes aggregate CTR = 1 - sum(errors) / sum(totals) per resample,
    - reports the bias-corrected-and-accelerated (BCa) 95% interval.

Seeding — common random numbers per dataset:
    Each cell draws its resamples from a generator seeded *only* from the
    dataset name plus a recorded BASE_SEED. Every model on a given dataset
    therefore draws the IDENTICAL set of resampled files (common random
    numbers), which (a) leaves each model's marginal CI unbiased, (b) makes
    model-vs-model differences directly comparable on matched resamples, and
    (c) makes the computation order-independent and future-proof: adding a new
    model later re-seeds from the same dataset seed, reproduces the identical
    resamples, and does not perturb any existing cell. To recompute one cell in
    the future, only that cell needs to run — never the whole table.

Writes results/ctr_bootstrap_ci.tsv incrementally (one row per cell).

Run with the stapes lens venv python (numpy + scipy + pandas):
    /home/grey/dev/graiai/lens/venv/bin/python scripts/bootstrap_ctr.py
"""

import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

SCRIPT_DIR = Path(__file__).resolve().parent
STAPES_DIR = SCRIPT_DIR.parent
IN_PATH = STAPES_DIR / 'results' / 'per_file_ctr.tsv'
OUT_PATH = STAPES_DIR / 'results' / 'ctr_bootstrap_ci.tsv'

N_RESAMPLES = 10000
# Recorded base seed. Combined with the dataset name to seed each cell so that
# results are reproducible and one cell can be recomputed without rerunning the
# rest of the table. Do not change without regenerating the whole table.
BASE_SEED = 20260419
CONFIDENCE_LEVEL = 0.95


def dataset_rng(dataset: str) -> np.random.Generator:
    """Return a fresh generator seeded from BASE_SEED + the dataset name.

    Same dataset -> same generator -> identical resampled file slots across
    every model on that dataset (common random numbers).
    """
    h = int.from_bytes(hashlib.sha256(dataset.encode()).digest()[:8], 'big')
    return np.random.default_rng([BASE_SEED, h])


def ctr_statistic(errors: np.ndarray, totals: np.ndarray, axis: int = -1) -> np.ndarray:
    """Aggregate clinical term recall = 1 - sum(errors) / sum(totals)."""
    return 1.0 - errors.sum(axis=axis) / totals.sum(axis=axis)


def cell_ci(errors: np.ndarray, totals: np.ndarray,
            rng: np.random.Generator) -> tuple[float, float, float, str]:
    """Return (ctr_pct, ci_lo_pct, ci_hi_pct, method) for one cell.

    Uses BCa; falls back to the percentile method only for degenerate cells
    where BCa is undefined (e.g. a single file or zero resample variance).
    """
    total = totals.sum()
    ctr = 1.0 - errors.sum() / total if total else float('nan')

    def run(method: str) -> tuple[float, float]:
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # turn degenerate-distribution warnings into errors
            res = bootstrap(
                (errors, totals),
                ctr_statistic,
                n_resamples=N_RESAMPLES,
                method=method,
                paired=True,
                vectorized=True,
                confidence_level=CONFIDENCE_LEVEL,
                random_state=rng,
            )
        return res.confidence_interval.low, res.confidence_interval.high

    try:
        lo, hi = run('BCa')
        method = 'BCa'
    except Exception:  # noqa: BLE001 — degenerate BCa; fall back to percentile
        # Re-seed so the fallback is deterministic too.
        lo, hi = run('percentile')
        method = 'percentile'

    if not np.isfinite(lo) or not np.isfinite(hi):
        method = 'percentile'
        lo, hi = run('percentile')

    return 100 * ctr, 100 * lo, 100 * hi, method


def main() -> int:
    if not IN_PATH.exists():
        print(f'error: {IN_PATH} does not exist; run compute_per_file_ctr.py first', flush=True)
        return 1
    df = pd.read_csv(IN_PATH, sep='\t')
    print(f'[loaded] {len(df)} rows from {IN_PATH}', flush=True)

    fh = open(OUT_PATH, 'w')
    fh.write('dataset\tmodel\tn_files\tctr_pct\tci_lo_pct\tci_hi_pct\tmethod\n')
    fh.flush()

    for (dataset, model), group in df.groupby(['dataset', 'model']):
        # Sort by file_id so resample slot i is the same file across models.
        group = group.sort_values('file_id')
        errors = group['medical_errors'].to_numpy()
        totals = group['medical_total'].to_numpy()
        n_files = len(group)

        # Fresh per-dataset generator → common random numbers across models.
        rng = dataset_rng(dataset)
        ctr, lo, hi, method = cell_ci(errors, totals, rng)

        fh.write(f'{dataset}\t{model}\t{n_files}\t{ctr:.2f}\t{lo:.2f}\t{hi:.2f}\t{method}\n')
        fh.flush()
        print(
            f'[{dataset}] {model}: CTR = {ctr:.2f}% [{lo:.2f}, {hi:.2f}] '
            f'({method}, n_files={n_files})',
            flush=True,
        )

    fh.close()
    print(f'[done] wrote {OUT_PATH}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
