#!/usr/bin/env python3
"""Build Supplementary Table S4 (per-model headline-drug error rates).

Converts results/_audit/paper_drug_rates.tsv (produced by
scripts/recompute_paper_drug_rates.py) into
paper/supplementary_table_S4_drug_rates.csv, with rows in Table 1 order plus
pooled per-mode rows. Cells show errors/occurrences (rate%).
"""

import csv
from pathlib import Path

ROOT = Path('/home/grey/dev/graiai/stapes')
SRC = ROOT / 'results' / '_audit' / 'paper_drug_rates.tsv'
OUT = ROOT / 'paper' / 'supplementary_table_S4_drug_rates.csv'

DISPLAY = [
    ('whisper-distil-v3.5', 'Whisper distil-v3.5', 'On-device'),
    ('whisper-turbo', 'Whisper turbo', 'On-device'),
    ('whisper-base-en', 'Whisper base.en', 'On-device'),
    ('qwen3-asr', 'Qwen3-ASR', 'On-device'),
    ('parakeet-tdt-0.6b-v2', 'Parakeet-TDT-0.6b-v2', 'On-device'),
    ('sensevoice', 'SenseVoice', 'On-device'),
    ('sensevoice-no-itn', 'SenseVoice (no ITN)', 'On-device'),
    ('paraformer-en', 'Paraformer-en', 'On-device'),
    ('nemo-fastconformer-int8', 'NeMo FastConformer', 'On-device'),
    ('nemotron', 'Nemotron', 'On-device'),
    ('zipformer-zh-en', 'Zipformer', 'On-device'),
    ('medasr-int8', 'MedASR', 'On-device'),
    ('cloud-azure', 'Azure Speech', 'Cloud'),
    ('cloud-google', 'Google medical_conversation', 'Cloud'),
    ('cloud-deepgram', 'Deepgram Nova-2 Medical', 'Cloud'),
    ('cloud-assemblyai', 'AssemblyAI', 'Cloud'),
    ('cloud-aws', 'AWS Transcribe Medical', 'Cloud'),
]
COLUMNS = [
    ('nazmulkazi', 'wellbutrin', 'Wellbutrin (Psych)'),
    ('primock57', 'lisinopril', 'Lisinopril (PriMock57)'),
    ('primock57', 'amoxicillin', 'Amoxicillin (PriMock57)'),
    ('figshare-osce', 'ramipril', 'Ramipril (OSCE)'),
    ('figshare-osce', 'rosuvastatin', 'Rosuvastatin (OSCE)'),
    ('figshare-osce', 'lisinopril', 'Lisinopril (OSCE)'),
]

rows = list(csv.DictReader(open(SRC), delimiter='\t'))
data = {(r['dataset'], r['drug'], r['model']): r for r in rows}

fh = open(OUT, 'w', newline='')
writer = csv.writer(fh)
writer.writerow(['Model', 'Type'] + [c[2] for c in COLUMNS])
fh.flush()


def cell(ds: str, drug: str, model: str) -> str:
    r = data.get((ds, drug, model))
    if r is None or r['rate'] == 'n/a':
        return 'n/a'
    return f"{r['errors']}/{r['occurrences']} ({float(r['rate']):.0%})"


for key, display, mtype in DISPLAY:
    writer.writerow([display, mtype] + [cell(ds, drug, key) for ds, drug, _ in COLUMNS])
    fh.flush()
    print(display, flush=True)

# pooled per mode
for mtype, pred in [('On-device', lambda m: not m.startswith('cloud-')),
                    ('Cloud', lambda m: m.startswith('cloud-'))]:
    out_row = [f'All {mtype.lower()} (pooled)', mtype]
    for ds, drug, _ in COLUMNS:
        e = sum(int(r['errors']) for r in rows
                if r['dataset'] == ds and r['drug'] == drug
                and r['model'] != 'POOLED-PAPER-17' and pred(r['model']) and r['rate'] != 'n/a')
        t = sum(int(r['occurrences']) for r in rows
                if r['dataset'] == ds and r['drug'] == drug
                and r['model'] != 'POOLED-PAPER-17' and pred(r['model']) and r['rate'] != 'n/a')
        out_row.append(f'{e}/{t} ({e/t:.0%})')
    writer.writerow(out_row)
    fh.flush()
    print(out_row[0], flush=True)

fh.close()
print(f'wrote {OUT}', flush=True)
