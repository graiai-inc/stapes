#!/usr/bin/env python3
"""Recompute the headline per-drug error rates over exactly the paper's models.

The pooled rates in medical_error_analysis.json were computed over 19
evaluation runs per dataset, which include non-paper builds (full-precision
duplicates, parakeet-v3, zipformer-libriheavy) and lack AWS on OSCE. This
script recomputes the per-drug error rates with the same methodology as
scripts/analyze_medical_errors.py (identical normalize() and jiwer
substitute/delete error indices), restricted to the paper's 12 on-device int8
configurations + 5 cloud services, using the per-file results in
results/on_device/ and results/cloud/.

All five headline drugs are single words, so their reference spans are the
word positions of the drug token itself (matching the QuickUMLS exact match).

Writes results/_audit/paper_drug_rates.tsv incrementally.
"""

import json
import re
from pathlib import Path

import jiwer

ROOT = Path('/home/grey/dev/graiai/stapes')
OUT = ROOT / 'results' / '_audit' / 'paper_drug_rates.tsv'
OUT.parent.mkdir(parents=True, exist_ok=True)

ONDEV = {
    'whisper-distil-v3.5': 'whisper-distil-v3.5.json',
    'whisper-turbo': 'whisper-turbo.json',
    'whisper-base-en': 'whisper-base-en.json',
    'qwen3-asr': 'qwen3-asr.json',
    'parakeet-tdt-0.6b-v2': 'parakeet-tdt-0.6b-v2.json',
    'nemo-fastconformer-int8': 'nemo-fastconformer-int8.json',
    'nemotron': 'nemotron.json',
    'zipformer-zh-en': 'zipformer-zh-en.json',
    'sensevoice': 'sensevoice.json',
    'sensevoice-no-itn': 'sensevoice-no-itn.json',
    'medasr-int8': 'medasr-int8.json',
    'paraformer-en': 'paraformer-en.json',
}
CLOUD = ['azure', 'google', 'deepgram', 'assemblyai', 'aws']
DRUGS = {
    'nazmulkazi': ['wellbutrin'],
    'primock57': ['lisinopril', 'amoxicillin'],
    'figshare-osce': ['ramipril', 'rosuvastatin', 'lisinopril'],
}

fh = open(OUT, 'w')


def emit(line: str) -> None:
    fh.write(line + '\n')
    fh.flush()
    print(line, flush=True)


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s'-]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_error_indices(ref: str, hyp: str) -> set:
    out = jiwer.process_words(ref, hyp)
    errors = set()
    for chunk in out.alignments[0]:
        if chunk.type in ('substitute', 'delete'):
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                errors.add(i)
    return errors


emit('dataset\tdrug\tmodel\terrors\toccurrences\trate')

for ds, drugs in DRUGS.items():
    # load every paper config's per-file results
    runs = {}
    for name, fname in ONDEV.items():
        p = ROOT / 'results' / 'on_device' / ds / fname
        runs[name] = {r['file_id']: r for r in json.load(open(p))['results']}
    for svc in CLOUD:
        p = ROOT / 'results' / 'cloud' / ds / f'{svc}.json'
        runs[f'cloud-{svc}'] = {r['file_id']: r for r in json.load(open(p))['results']}

    # reference drug word positions, from any run that has the file
    ref_positions = {}
    all_fids = set()
    for files in runs.values():
        all_fids.update(files)
    for fid in sorted(all_fids):
        ref = None
        for files in runs.values():
            if fid in files and files[fid].get('reference'):
                ref = normalize(files[fid]['reference'])
                break
        if ref is None:
            continue
        words = ref.split()
        for drug in drugs:
            pos = [i for i, w in enumerate(words) if w == drug]
            if pos:
                ref_positions.setdefault(fid, {})[drug] = (ref, pos)

    for drug in drugs:
        pooled_err = pooled_tot = 0
        for model, files in sorted(runs.items()):
            err = tot = 0
            for fid, drugmap in ref_positions.items():
                if drug not in drugmap or fid not in files:
                    continue
                ref, positions = drugmap[drug]
                hyp = normalize(files[fid].get('hypothesis', ''))
                error_idx = get_error_indices(ref, hyp)
                for p in positions:
                    tot += 1
                    if p in error_idx:
                        err += 1
            pooled_err += err
            pooled_tot += tot
            emit(f'{ds}\t{drug}\t{model}\t{err}\t{tot}\t{err/tot:.4f}' if tot else
                 f'{ds}\t{drug}\t{model}\t0\t0\tn/a')
        emit(f'{ds}\t{drug}\tPOOLED-PAPER-17\t{pooled_err}\t{pooled_tot}\t{pooled_err/pooled_tot:.4f}')

fh.close()
