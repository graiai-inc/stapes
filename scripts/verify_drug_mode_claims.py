#!/usr/bin/env python3
"""Decompose the medication-error findings by model and deployment mode.

The per-drug error rates in results/medical_errors/<ds>/medical_error_analysis.json
are pooled across all models. This script verifies the manuscript's medication
claims at the scope they are stated:

1. Per model x dataset: is DRUG the highest-error UMLS category (among
   categories with >= 50 reference spans)?
2. Pooled per deployment mode: category error-rate ranking.
3. Per service: surface transcription counts of the five highest-error drugs
   named in the paper, versus their reference occurrence counts.

Writes results/_audit/drug_mode_claims.tsv incrementally.
"""

import json
import re
from pathlib import Path

ROOT = Path('/home/grey/dev/graiai/stapes')
OUT = ROOT / 'results' / '_audit' / 'drug_mode_claims.tsv'
OUT.parent.mkdir(parents=True, exist_ok=True)

PAPER_ONDEV = ['whisper-distil-v3.5', 'whisper-turbo', 'whisper-base-en', 'qwen3-asr',
               'parakeet-tdt-0.6b-v2', 'nemo-fastconformer-int8', 'nemotron',
               'zipformer-zh-en', 'sensevoice', 'sensevoice-no-itn', 'medasr-int8',
               'paraformer-en']
PAPER_CLOUD = ['cloud-azure', 'cloud-google', 'cloud-deepgram', 'cloud-assemblyai',
               'cloud-aws']
DATASETS = ['nazmulkazi', 'primock57', 'figshare-osce']
HEADLINE_DRUGS = {
    'nazmulkazi': ['wellbutrin'],
    'primock57': ['lisinopril', 'amoxicillin'],
    'figshare-osce': ['ramipril', 'rosuvastatin', 'lisinopril'],
}
MIN_CATEGORY_SPANS = 50

fh = open(OUT, 'w')


def emit(line: str) -> None:
    fh.write(line + '\n')
    fh.flush()
    print(line, flush=True)


emit('check\tdataset\tsubject\tresult')

for ds in DATASETS:
    d = json.load(open(ROOT / 'results' / 'medical_errors' / ds / 'medical_error_analysis.json'))
    n_models = d['models']
    stats = d['model_medical_stats']

    # reference span count per category (identical denominator for every model)
    denom: dict = {}
    for term, v in d['medical_term_errors'].items():
        denom[v['category']] = denom.get(v['category'], 0) + v['total_occurrences'] / n_models
    big = {c for c, n in denom.items() if n >= MIN_CATEGORY_SPANS}
    emit(f'category-denoms\t{ds}\tall\t' + json.dumps({c: round(n) for c, n in sorted(denom.items())}))

    # 1. per paper model: top category
    n_drug_top = n_total = 0
    for m in PAPER_ONDEV + PAPER_CLOUD:
        if m not in stats:
            emit(f'model-top-category\t{ds}\t{m}\tABSENT from analysis json')
            continue
        rates = {c: stats[m]['by_category'].get(c, 0) / denom[c] for c in big}
        top = max(rates, key=rates.get)
        n_total += 1
        n_drug_top += (top == 'DRUG')
        emit(f'model-top-category\t{ds}\t{m}\t{top} ({rates[top]:.3f}; DRUG {rates["DRUG"]:.3f})')
    emit(f'drug-top-summary\t{ds}\tpaper models\tDRUG highest for {n_drug_top}/{n_total}')

    # 2. pooled per mode
    for label, group in [('on-device', PAPER_ONDEV), ('cloud', PAPER_CLOUD)]:
        pool: dict = {}
        n_in = 0
        for m in group:
            if m not in stats:
                continue
            n_in += 1
            for c, e in stats[m]['by_category'].items():
                pool[c] = pool.get(c, 0) + e
        rates = {c: pool.get(c, 0) / (denom[c] * n_in) for c in big}
        rank = sorted(rates.items(), key=lambda kv: -kv[1])
        emit(f'pooled-mode-ranking\t{ds}\t{label}\t' + ', '.join(f'{c} {r:.3f}' for c, r in rank))

    # 3. per-service surface counts of the headline drugs
    for drug in HEADLINE_DRUGS[ds]:
        ref_n = round(d['medical_term_errors'][drug]['total_occurrences'] / n_models)
        for f in sorted((ROOT / 'results' / 'cloud' / ds).glob('*.json')):
            res = json.load(open(f))['results']
            n = sum(len(re.findall(drug, r['hypothesis'], re.I)) for r in res)
            emit(f'drug-surface-count\t{ds}\t{drug} cloud-{f.stem}\t{n} (reference {ref_n})')
        for f in sorted((ROOT / 'results' / 'on_device' / ds).glob('*.json')):
            res = json.load(open(f))['results']
            n = sum(len(re.findall(drug, r['hypothesis'], re.I)) for r in res)
            emit(f'drug-surface-count\t{ds}\t{drug} {f.stem}\t{n} (reference {ref_n})')

fh.close()
