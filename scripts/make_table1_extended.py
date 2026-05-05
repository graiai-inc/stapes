#!/usr/bin/env python3
"""Generate Table 1 with both standard (whisper-normalized) and clinical
(extended-normalized) WER columns side by side.

Reads stapes/results/extended_normalization/summary.tsv and writes
paper/table1_wer.csv with the dual-WER layout.

Per stapes/CLAUDE.md ABSOLUTE RULE #1, every row is written and flushed.
"""
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STAPES = SCRIPT_DIR.parent
SUMMARY = STAPES / 'results' / 'extended_normalization' / 'summary.tsv'
OUT_CSV = STAPES / 'paper' / 'table1_wer.csv'


# (display_name, type, internal_id) — display order matches the prior table.
ROWS = [
    ('Whisper distil-v3.5',       'On-device',        'whisper-distil-v3.5'),
    ('Whisper turbo',             'On-device',        'whisper-turbo'),
    ('Whisper base.en',           'On-device',        'whisper-base-en'),
    ('Qwen3-ASR',                 'On-device',        'qwen3-asr'),
    ('Parakeet-TDT-0.6b-v2',      'On-device',        'parakeet-tdt-0.6b-v2'),
    ('SenseVoice (no ITN)',       'On-device',        'sensevoice-no-itn'),
    ('SenseVoice',                'On-device',        'sensevoice'),
    ('Paraformer-en',             'On-device',        'paraformer-en'),
    ('NeMo FastConformer',        'On-device',        'nemo-fastconformer'),
    ('NeMo FastConformer int8',   'On-device',        'nemo-fastconformer-int8'),
    ('Nemotron',                  'On-device',        'nemotron'),
    ('Zipformer',                 'On-device',        'zipformer-zh-en'),
    ('MedASR',                    'On-device',        'medasr'),
    ('MedASR int8',               'On-device',        'medasr-int8'),
    ('Azure Speech',              'Cloud (general)',  'azure'),
    ('Deepgram Nova-2 Medical',   'Cloud (medical)',  'deepgram'),
    ('AssemblyAI',                'Cloud (general)',  'assemblyai'),
    ('AWS Transcribe Medical',    'Cloud (medical)',  'aws'),       # subset cells get †
    ('Google medical_conversation', 'Cloud (medical)','google'),    # psychiatric gets ‡
]

# Cells that are subsets (AWS) or partial (Google psychiatric).
AWS_MARK = '†'
GOOGLE_PSY_MARK = '‡'


def load_summary():
    """Return dict keyed by (dataset, model) with whisper_pct and clinical_pct."""
    out = {}
    with open(SUMMARY, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            key = (row['dataset'], row['model'])
            out[key] = {
                'whisper_pct': float(row['wer_whisper_pct']),
                'clinical_pct': float(row['wer_clinical_pct']),
                'n_files': int(row['n_files']),
            }
    return out


def fmt_cell(value: float, mark: str = '') -> str:
    """Two-decimal WER with optional dagger marker."""
    return f'{value:.2f}{mark}'


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    summary = load_summary()

    fh = open(OUT_CSV, 'w', encoding='utf-8', newline='')
    writer = csv.writer(fh)

    header = [
        'Model', 'Type',
        'OSCE Std (n=272)*', 'OSCE MP (n=272)*',
        'PriMock57 Std (n=57)', 'PriMock57 MP (n=57)',
        'Psychiatric Std (n=71)', 'Psychiatric MP (n=71)',
    ]
    writer.writerow(header)
    fh.flush()

    for display_name, model_type, model_id in ROWS:
        # Determine subset markers.
        aws = model_id == 'aws'
        google = model_id == 'google'

        cells = []
        for dataset in ('figshare-osce', 'primock57', 'nazmulkazi'):
            row = summary.get((dataset, model_id))
            if row is None:
                cells.extend(['n/a', 'n/a'])
                continue

            mark = ''
            if aws:
                mark = AWS_MARK  # all AWS cells are subset
            if google and dataset == 'nazmulkazi':
                mark = GOOGLE_PSY_MARK

            cells.append(fmt_cell(row['whisper_pct'], mark))
            cells.append(fmt_cell(row['clinical_pct'], mark))

        writer.writerow([display_name, model_type] + cells)
        fh.flush()
        print(f'  {display_name:<28} {cells}', flush=True)

    fh.close()
    print(f'\nWrote {OUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
