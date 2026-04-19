#!/usr/bin/env python3
"""Rewrite paper/table2_ctr.csv to include bootstrap 95% CIs per cell.

Output format per (model, dataset) cell: "ctr [lo, hi]" (all in %, rounded
to 1 decimal), e.g. "93.1 [92.0, 94.2]".

Reads bootstrap CIs from results/ctr_bootstrap_ci.tsv; writes the updated
table incrementally row-by-row to paper/table2_ctr.csv.

Run with the stapes lens venv python:
    /home/grey/dev/graiai/lens/venv/bin/python scripts/update_table2_ctr_with_ci.py
"""

import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STAPES_DIR = SCRIPT_DIR.parent
CI_PATH = STAPES_DIR / 'results' / 'ctr_bootstrap_ci.tsv'
OUT_PATH = STAPES_DIR / 'paper' / 'table2_ctr.csv'

# Table-row order (matches the existing table2_ctr.csv) and display names.
ROWS = [
    ('whisper-distil-v3.5', 'Whisper distil-v3.5', 'On-device'),
    ('whisper-turbo', 'Whisper turbo', 'On-device'),
    ('whisper-base-en', 'Whisper base.en', 'On-device'),
    ('qwen3-asr', 'Qwen3-ASR', 'On-device'),
    ('parakeet-tdt-0.6b-v2', 'Parakeet-TDT-0.6b-v2', 'On-device'),
    ('sensevoice-no-itn', 'SenseVoice (no ITN)', 'On-device'),
    ('sensevoice', 'SenseVoice', 'On-device'),
    ('paraformer-en', 'Paraformer-en', 'On-device'),
    ('nemo-fastconformer', 'NeMo FastConformer', 'On-device'),
    ('nemo-fastconformer-int8', 'NeMo FastConformer int8', 'On-device'),
    ('nemotron', 'Nemotron', 'On-device'),
    ('zipformer-zh-en', 'Zipformer', 'On-device'),
    ('medasr', 'MedASR', 'On-device'),
    ('medasr-int8', 'MedASR int8', 'On-device'),
    ('cloud-azure', 'Azure Speech', 'Cloud (general)'),
    ('cloud-deepgram', 'Deepgram Nova-2 Medical', 'Cloud (medical)'),
    ('cloud-assemblyai', 'AssemblyAI', 'Cloud (general)'),
    ('cloud-aws', 'AWS Transcribe Medical†', 'Cloud (medical)'),
    ('cloud-google', 'Google medical_conversation', 'Cloud (medical)'),
]

DATASETS = [
    ('figshare-osce', 'OSCE (n=272)'),
    ('primock57', 'PriMock57 (n=57)'),
    ('nazmulkazi', 'Psychiatric (n=71)'),
]


def load_ci() -> dict:
    """Return {(dataset, model): (ctr, lo, hi, n_files)}."""
    out = {}
    with CI_PATH.open() as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        for r in reader:
            out[(r['dataset'], r['model'])] = (
                float(r['ctr_pct']),
                float(r['ci_lo_pct']),
                float(r['ci_hi_pct']),
                int(r['n_files']),
            )
    return out


def fmt_cell(ci: tuple | None) -> str:
    if ci is None:
        return 'n/a'
    ctr, lo, hi, _ = ci
    return f'{ctr:.1f} [{lo:.1f}, {hi:.1f}]'


def main() -> int:
    cis = load_ci()
    print(f'[loaded] {len(cis)} CI entries', flush=True)

    fh = open(OUT_PATH, 'w', newline='')
    writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
    header = ['Model', 'Type'] + [label for _, label in DATASETS]
    writer.writerow(header)
    fh.flush()

    for model_key, display, type_ in ROWS:
        row = [display, type_]
        for ds_key, _ in DATASETS:
            ci = cis.get((ds_key, model_key))
            row.append(fmt_cell(ci))
        writer.writerow(row)
        fh.flush()
        print(' | '.join(str(c) for c in row), flush=True)

    fh.close()
    print(f'[done] wrote {OUT_PATH}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
