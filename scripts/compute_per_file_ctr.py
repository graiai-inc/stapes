#!/usr/bin/env python3
"""Compute per-file clinical term recall (CTR) counts for every model × dataset.

Writes results/per_file_ctr.tsv incrementally: one row per (dataset, model,
file_id) tuple with counts (medical_errors, medical_total). These counts are
the raw inputs for bootstrap confidence intervals over files — see
scripts/bootstrap_ctr.py.

Logic mirrors ossicles/scripts/analyze_medical_errors.py, but keeps the
per-file counts instead of aggregating immediately. QuickUMLS + jiwer are
used the same way as in the existing analysis; numbers in the aggregate
should match the existing medical_error_analysis.json files.

Run with the ossicles venv python (it has quickumls installed):
    /home/grey/dev/graiai/ossicles/venv/bin/python scripts/compute_per_file_ctr.py
"""

import collections
import json
import logging
import re
import sys
from pathlib import Path

import jiwer
from quickumls import QuickUMLS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
STAPES_DIR = SCRIPT_DIR.parent
OSSICLES_DIR = STAPES_DIR.parent / 'ossicles'
QUICKUMLS_PATH = OSSICLES_DIR / 'quickumls_data'

DATASETS = ['figshare-osce', 'primock57', 'nazmulkazi']

MEDICAL_SEMANTIC_TYPES = {
    'T116': 'DRUG', 'T121': 'DRUG', 'T200': 'DRUG', 'T195': 'DRUG',
    'T047': 'CONDITION', 'T048': 'CONDITION', 'T046': 'CONDITION', 'T191': 'CONDITION',
    'T184': 'SYMPTOM', 'T033': 'FINDING', 'T034': 'LAB_RESULT',
    'T023': 'ANATOMY', 'T029': 'ANATOMY', 'T030': 'ANATOMY',
    'T058': 'PROCEDURE', 'T059': 'PROCEDURE', 'T060': 'PROCEDURE', 'T061': 'PROCEDURE',
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


def get_error_indices(ref: str, hyp: str) -> set[int]:
    out = jiwer.process_words(ref, hyp)
    errors = set()
    for chunk in out.alignments[0]:
        if chunk.type in ('substitute', 'delete'):
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                errors.add(i)
    return errors


def char_span_to_word_indices(text: str, start: int, end: int) -> list[int]:
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


def find_medical_spans(text: str, matcher) -> list[dict]:
    matches = matcher.match(text)
    spans = []
    for match_group in matches:
        best = match_group[0]
        ngram = best['ngram'].lower().strip()
        if ngram in UMLS_STOPWORDS:
            continue
        if ' ' not in ngram and len(ngram) < 6 and best['similarity'] < 1.0:
            continue
        category = None
        for stype in best.get('semtypes', []):
            if stype in MEDICAL_SEMANTIC_TYPES:
                category = MEDICAL_SEMANTIC_TYPES[stype]
                break
        if category is None:
            continue
        word_indices = char_span_to_word_indices(text, best['start'], best['end'])
        if not word_indices:
            continue
        spans.append({'ngram': best['ngram'], 'category': category, 'word_indices': word_indices})
    return spans


def load_model_results(dataset: str) -> dict:
    """Return {model_name: {file_id: {hypothesis, reference, ...}}}."""
    out = {}
    on_device_dir = OSSICLES_DIR / f'benchmark_results_{dataset}'
    if on_device_dir.exists():
        for f in sorted(on_device_dir.glob('*.json')):
            if '.old' in f.name or 'normalized' in f.name or 'fusion' in f.stem or 'medical' in f.stem:
                continue
            try:
                data = json.loads(f.read_text())
                if isinstance(data, dict) and 'aggregates' in data:
                    out[f.stem] = {r['file_id']: r for r in data.get('results', [])}
            except (json.JSONDecodeError, KeyError):
                continue

    cloud_dir = OSSICLES_DIR / f'benchmark_results_cloud_{dataset}'
    if cloud_dir.exists():
        for f in sorted(cloud_dir.glob('*.json')):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, dict) and 'results' in data:
                    out[f'cloud-{f.stem}'] = {r['file_id']: r for r in data.get('results', [])}
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def main() -> int:
    out_path = STAPES_DIR / 'results' / 'per_file_ctr.tsv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, 'w')
    fh.write('dataset\tmodel\tfile_id\tmedical_errors\tmedical_total\n')
    fh.flush()

    log.info('initializing QuickUMLS...')
    matcher = QuickUMLS(
        quickumls_fp=str(QUICKUMLS_PATH),
        threshold=0.8,
        similarity_name='jaccard',
        window=5,
    )

    for dataset in DATASETS:
        log.info(f'=== dataset: {dataset} ===')
        model_results = load_model_results(dataset)
        if not model_results:
            log.warning(f'no model results for {dataset}')
            continue
        log.info(f'{dataset}: {len(model_results)} models loaded')

        # Compute medical spans per reference file once (shared across models).
        first_model = next(iter(model_results.values()))
        file_spans = {}
        file_refs = {}
        log.info(f'  computing medical spans for {len(first_model)} reference files...')
        for i, (fid, result) in enumerate(first_model.items()):
            ref = result.get('reference', '')
            if not ref:
                continue
            ref_norm = normalize(ref)
            file_refs[fid] = ref_norm
            file_spans[fid] = find_medical_spans(ref_norm, matcher)
            if (i + 1) % 50 == 0 or i + 1 == len(first_model):
                total = sum(len(s) for s in file_spans.values())
                log.info(f'    [{i + 1}/{len(first_model)}] {total} medical spans so far')

        total_spans = sum(len(s) for s in file_spans.values())
        log.info(f'  {total_spans} medical spans across {len(file_spans)} files')

        # For each model, align per file and record per-file counts.
        for model_idx, (model_name, files) in enumerate(sorted(model_results.items())):
            n_errors = 0
            n_total = 0
            for fid in file_spans:
                if fid not in files:
                    continue
                ref_norm = file_refs[fid]
                hyp_norm = normalize(files[fid].get('hypothesis', ''))
                error_indices = get_error_indices(ref_norm, hyp_norm)

                med_errors = 0
                med_total = 0
                for span in file_spans[fid]:
                    med_total += 1
                    if any(wi in error_indices for wi in span['word_indices']):
                        med_errors += 1

                fh.write(f'{dataset}\t{model_name}\t{fid}\t{med_errors}\t{med_total}\n')
                fh.flush()
                n_errors += med_errors
                n_total += med_total

            mtr = 100 * (1.0 - n_errors / n_total) if n_total else float('nan')
            print(
                f'[{dataset}] [{model_idx + 1}/{len(model_results)}] {model_name}: '
                f'MTR={mtr:.2f}% ({n_errors}/{n_total})',
                flush=True,
            )

    fh.close()
    print(f'[done] wrote {out_path}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
