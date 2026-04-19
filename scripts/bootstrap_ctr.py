#!/usr/bin/env python3
"""Bootstrap 95% CIs for per-model clinical term recall (CTR) over files.

Reads results/per_file_ctr.tsv (produced by compute_per_file_ctr.py) and for
each (dataset, model) cell:
    - resamples files with replacement 1000 times
    - computes aggregate CTR = 1 - sum(errors) / sum(totals) per resample
    - reports the 2.5th and 97.5th percentiles as a 95% CI

Writes results/ctr_bootstrap_ci.tsv incrementally (one row per cell).

Run with the stapes lens venv python (numpy + pandas only):
    /home/grey/dev/graiai/lens/venv/bin/python scripts/bootstrap_ctr.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STAPES_DIR = SCRIPT_DIR.parent
IN_PATH = STAPES_DIR / 'results' / 'per_file_ctr.tsv'
OUT_PATH = STAPES_DIR / 'results' / 'ctr_bootstrap_ci.tsv'

N_RESAMPLES = 1000
RNG_SEED = 20260419


def bootstrap_ctr_ci(
    errors: np.ndarray,
    totals: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return (ctr_pct, ci_lo_pct, ci_hi_pct) — percentile bootstrap."""
    n = len(errors)
    # Point estimate.
    total = totals.sum()
    ctr = 1.0 - errors.sum() / total if total else float('nan')

    # Resample file indices with replacement.
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_err = errors[idx].sum(axis=1)
    resampled_tot = totals[idx].sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ctr_resamples = 1.0 - (resampled_err / resampled_tot)
    valid = resampled_tot > 0
    ctr_resamples = ctr_resamples[valid]
    lo = float(np.percentile(ctr_resamples, 2.5))
    hi = float(np.percentile(ctr_resamples, 97.5))
    return 100 * ctr, 100 * lo, 100 * hi


def main() -> int:
    if not IN_PATH.exists():
        print(f'error: {IN_PATH} does not exist; run compute_per_file_ctr.py first', flush=True)
        return 1
    df = pd.read_csv(IN_PATH, sep='\t')
    print(f'[loaded] {len(df)} rows from {IN_PATH}', flush=True)

    rng = np.random.default_rng(RNG_SEED)

    fh = open(OUT_PATH, 'w')
    fh.write('dataset\tmodel\tn_files\tctr_pct\tci_lo_pct\tci_hi_pct\n')
    fh.flush()

    for (dataset, model), group in df.groupby(['dataset', 'model']):
        errors = group['medical_errors'].to_numpy()
        totals = group['medical_total'].to_numpy()
        n_files = len(group)
        ctr, lo, hi = bootstrap_ctr_ci(errors, totals, N_RESAMPLES, rng)
        fh.write(f'{dataset}\t{model}\t{n_files}\t{ctr:.2f}\t{lo:.2f}\t{hi:.2f}\n')
        fh.flush()
        print(
            f'[{dataset}] {model}: CTR = {ctr:.2f}% [{lo:.2f}, {hi:.2f}]  (n_files={n_files})',
            flush=True,
        )

    fh.close()
    print(f'[done] wrote {OUT_PATH}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
