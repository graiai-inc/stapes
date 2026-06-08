#!/usr/bin/env python3
"""Span-length distribution for the UMLS clinical-term spans used in CTR.

Addresses the reviewer request to quantify how many medical concept spans are
single- vs multi-word, since span-level scoring (any word wrong -> span error)
can overestimate error for longer phrases. Mirrors find_medical_spans /
normalize from compute_per_file_ctr.py. Writes results incrementally.

Run with the ossicles venv python (has quickumls):
    /home/grey/dev/graiai/ossicles/venv/bin/python scripts/span_length_stats.py
"""

import json
import re
from collections import Counter
from pathlib import Path

from quickumls import QuickUMLS

OSSICLES = Path('/home/grey/dev/graiai/ossicles')
QUICKUMLS_PATH = OSSICLES / 'quickumls_data'
DATASETS = ['figshare-osce', 'primock57', 'nazmulkazi']
REF_MODEL = 'whisper-distil-v3.5'  # any model; references are identical across models

MEDICAL_SEMANTIC_TYPES = {
    'T116', 'T121', 'T200', 'T195',
    'T047', 'T048', 'T046', 'T191',
    'T184', 'T033', 'T034',
    'T023', 'T029', 'T030',
    'T058', 'T059', 'T060', 'T061',
}
UMLS_STOPWORDS = {
    'bit', 'get', 'got', 'little', 'lot', 'said', 'still', 'today',
    'changes', 'well', 'cant', 'nothing', 'maybe', 'take', 'yes',
    'much', 'probably', 'worse', 'sharp', 'only', 'more', 'less',
    'stop', 'ask', 'bad', 'dad', 'may', 'nice', 'new', 'used',
    'find', 'care', 'place', 'always', 'wanted', 'play',
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def char_span_to_word_indices(text, start, end):
    word_starts = []
    in_word = False
    for i, c in enumerate(text):
        if c != ' ' and not in_word:
            word_starts.append(i)
            in_word = True
        elif c == ' ':
            in_word = False
    indices = []
    for wi, ws in enumerate(word_starts):
        we = word_starts[wi + 1] - 1 if wi + 1 < len(word_starts) else len(text)
        if ws < end and we > start:
            indices.append(wi)
    return indices


def find_spans(text, matcher):
    spans = []
    for group in matcher.match(text):
        best = group[0]
        ngram = best['ngram'].lower().strip()
        if ngram in UMLS_STOPWORDS:
            continue
        if ' ' not in ngram and len(ngram) < 6 and best['similarity'] < 1.0:
            continue
        if not any(s in MEDICAL_SEMANTIC_TYPES for s in best.get('semtypes', [])):
            continue
        wi = char_span_to_word_indices(text, best['start'], best['end'])
        if wi:
            spans.append(len(wi))
    return spans


def main():
    out = Path('/home/grey/dev/graiai/stapes/results/span_length_stats.tsv')
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open('w')
    fh.write('dataset\tn_spans\tn_single\tn_multi\tpct_single\tmean_len\tmax_len\n')
    fh.flush()

    matcher = QuickUMLS(quickumls_fp=str(QUICKUMLS_PATH), threshold=0.8,
                        similarity_name='jaccard', window=5)

    overall = Counter()
    for dataset in DATASETS:
        p = OSSICLES / f'benchmark_results_{dataset}' / f'{REF_MODEL}.json'
        data = json.loads(p.read_text())
        lengths = Counter()
        seen = set()
        for r in data['results']:
            fid = r['file_id']
            if fid in seen:
                continue
            seen.add(fid)
            ref = normalize(r.get('reference', '') or '')
            if not ref:
                continue
            for ln in find_spans(ref, matcher):
                lengths[ln] += 1
                overall[ln] += 1
        n = sum(lengths.values())
        n_single = lengths.get(1, 0)
        n_multi = n - n_single
        mean_len = sum(k * v for k, v in lengths.items()) / n if n else 0
        max_len = max(lengths) if lengths else 0
        pct_single = 100 * n_single / n if n else 0
        fh.write(f'{dataset}\t{n}\t{n_single}\t{n_multi}\t{pct_single:.1f}\t{mean_len:.2f}\t{max_len}\n')
        fh.flush()
        print(f'{dataset}: n={n} single={n_single} ({pct_single:.1f}%) multi={n_multi} '
              f'mean_len={mean_len:.2f} max={max_len}', flush=True)

    n = sum(overall.values())
    n_single = overall.get(1, 0)
    mean_len = sum(k * v for k, v in overall.items()) / n if n else 0
    fh.write(f'ALL\t{n}\t{n_single}\t{n - n_single}\t{100*n_single/n:.1f}\t{mean_len:.2f}\t{max(overall)}\n')
    fh.flush()
    print(f'ALL: n={n} single={n_single} ({100*n_single/n:.1f}%) mean_len={mean_len:.2f}', flush=True)
    fh.close()


if __name__ == '__main__':
    main()
