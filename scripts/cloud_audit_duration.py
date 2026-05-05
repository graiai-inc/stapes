#!/usr/bin/env python3
"""Sum audio durations per dataset for cost reconciliation. Incremental output."""

import wave
from pathlib import Path

BASE = Path('/home/grey/dev/graiai/ossicles/assets/audio')
OUT = Path('/home/grey/dev/graiai/stapes/results/_audit/durations.tsv')
OUT.parent.mkdir(parents=True, exist_ok=True)

fh = OUT.open('w')
fh.write('dataset\tfid\tseconds\n')
fh.flush()

totals = {}
for dataset in ('figshare-osce', 'primock57', 'nazmulkazi'):
    total = 0.0
    count = 0
    for w in sorted((BASE / dataset).glob('*.wav')):
        try:
            with wave.open(str(w), 'rb') as wf:
                sec = wf.getnframes() / wf.getframerate()
        except Exception as e:
            print(f'[err] {w}: {e}', flush=True)
            continue
        total += sec
        count += 1
        fh.write(f'{dataset}\t{w.stem}\t{sec:.2f}\n')
        fh.flush()
    totals[dataset] = (count, total)
    print(f'{dataset}: n={count} total_sec={total:.1f} total_min={total/60:.1f} total_hr={total/3600:.2f}',
          flush=True)
fh.close()

print('', flush=True)
grand_sec = sum(t[1] for t in totals.values())
grand_files = sum(t[0] for t in totals.values())
print(f'GRAND TOTAL: files={grand_files} hours={grand_sec/3600:.2f} minutes={grand_sec/60:.1f}',
      flush=True)

# AWS subset (88 files) duration
aws_ids = set()
SUBSET_FILE_IDS = [
    'CAR0001', 'CAR0002', 'CAR0003', 'CAR0004', 'CAR0005',
    'DER0001',
    'GAS0001', 'GAS0002', 'GAS0003', 'GAS0004', 'GAS0005', 'GAS0007',
    'GEN0001',
    'MSK0003', 'MSK0004', 'MSK0007', 'MSK0008', 'MSK0009', 'MSK0010',
    'MSK0015', 'MSK0016', 'MSK0017', 'MSK0019', 'MSK0029', 'MSK0037',
    'MSK0039', 'MSK0043', 'MSK0046',
    'RES0002', 'RES0007', 'RES0043', 'RES0053', 'RES0059', 'RES0062',
    'RES0074', 'RES0090', 'RES0110', 'RES0111', 'RES0118', 'RES0133',
    'RES0143', 'RES0147', 'RES0154', 'RES0159', 'RES0171', 'RES0183',
    'RES0184', 'RES0188', 'RES0199', 'RES0203',
]
aws_ids |= set(SUBSET_FILE_IDS)
aws_ids |= set(Path('/home/grey/dev/graiai/ossicles/scripts/aws_primock57_subset.txt').read_text().split())
aws_ids |= set(Path('/home/grey/dev/graiai/ossicles/scripts/aws_nazmulkazi_subset.txt').read_text().split())

aws_total = 0.0
aws_n = 0
for dataset in ('figshare-osce', 'primock57', 'nazmulkazi'):
    for w in sorted((BASE / dataset).glob('*.wav')):
        if w.stem in aws_ids:
            with wave.open(str(w), 'rb') as wf:
                aws_total += wf.getnframes() / wf.getframerate()
                aws_n += 1
print(f'AWS subset: files={aws_n} hours={aws_total/3600:.2f} minutes={aws_total/60:.1f}',
      flush=True)

# Google's 396 files (71 - 67 = 4 missing on nazmulkazi)
# compute Google-only duration
import json
google_minutes_per_dataset = {}
for dataset in ('figshare-osce', 'primock57', 'nazmulkazi'):
    p = Path(f'/home/grey/dev/graiai/ossicles/benchmark_results_cloud_{dataset}/google.json')
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    fids = {r['file_id'] for r in d['results']}
    tot = 0.0
    for w in sorted((BASE / dataset).glob('*.wav')):
        if w.stem in fids:
            with wave.open(str(w), 'rb') as wf:
                tot += wf.getnframes() / wf.getframerate()
    google_minutes_per_dataset[dataset] = (len(fids), tot)
    print(f'Google {dataset}: files={len(fids)} minutes={tot/60:.1f}', flush=True)
gtotal = sum(v[1] for v in google_minutes_per_dataset.values())
gtotn = sum(v[0] for v in google_minutes_per_dataset.values())
print(f'Google TOTAL: files={gtotn} hours={gtotal/3600:.2f} minutes={gtotal/60:.1f}', flush=True)
