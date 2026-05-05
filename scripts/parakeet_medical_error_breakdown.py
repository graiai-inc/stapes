#!/usr/bin/env python3
"""Break down Parakeet's medical-term errors by N-gram length AND category.

For each (reference, parakeet_hypothesis) pair on PriMock57 / figshare-OSCE /
nazmulkazi:

  1. Run QuickUMLS on reference to find every medical term span. Each span
     carries: term text, category (DRUG/CONDITION/...), word-length (1-gram,
     2-gram, ..., 5-gram).
  2. jiwer-align reference to Parakeet's hypothesis at the word level → set
     of error word indices in the reference.
  3. For every medical span, mark it as ERROR if ANY of its constituent
     reference words is in the error set.
  4. Aggregate by length, category, and (length × category).
  5. Capture each error case with: file, term, category, length, parakeet
     substitution, neighboring-word context (3 words each side).

Outputs (incremental writes, per-file flush):
  results/parakeet_medical_breakdown/<dataset>_per_error.tsv
    file_id  term  category  length  parakeet_sub  context
  results/parakeet_medical_breakdown/<dataset>_by_length.tsv
    length  n_total  n_errors  error_rate
  results/parakeet_medical_breakdown/<dataset>_by_category_length.tsv
    category  length  n_total  n_errors  error_rate
  results/parakeet_medical_breakdown/<dataset>_top_missed.tsv
    term  category  length  n_occurrences  n_errors  error_rate
  results/parakeet_medical_breakdown/summary.tsv

Run with the ossicles venv (has quickumls + jiwer + pandas):
    /home/grey/dev/graiai/ossicles/venv/bin/python \\
        scripts/parakeet_medical_error_breakdown.py
"""
import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

import jiwer
from quickumls import QuickUMLS

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
STAPES_DIR = SCRIPT_DIR.parent
GRAIAI_DIR = STAPES_DIR.parent
OSSICLES_DIR = GRAIAI_DIR / 'ossicles'
QUICKUMLS_PATH = OSSICLES_DIR / 'quickumls_data'

OUT_DIR = STAPES_DIR / 'results' / 'parakeet_medical_breakdown'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_TSV = OUT_DIR / 'summary.tsv'

DATASETS = ['primock57', 'figshare-osce', 'nazmulkazi']

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
    'hmm', 'mmm', 'mm', 'uh', 'um', 'oh', 'ah',  # backchannel noise
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s'\-]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


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
        if not match_group:
            continue
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
        n_tokens = len(ngram.split())
        spans.append({
            'ngram': best['ngram'],
            'category': category,
            'length': n_tokens,
            'word_indices': word_indices,
        })
    return spans


def get_error_indices_and_subs(ref: str, hyp: str) -> tuple[set, dict]:
    """Returns (error_word_indices_in_ref, ref_idx → substitution_text)."""
    out = jiwer.process_words(ref, hyp)
    errors = set()
    subs = {}
    for chunk in out.alignments[0]:
        if chunk.type in ('substitute', 'delete'):
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                errors.add(i)
            if chunk.type == 'substitute':
                # Map ref words to corresponding hyp words
                hyp_words = hyp.split()
                hyp_chunk = ' '.join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    subs[i] = hyp_chunk
            elif chunk.type == 'delete':
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    subs[i] = '<DELETED>'
    return errors, subs


def process_dataset(dataset: str, matcher) -> dict | None:
    pdir = OSSICLES_DIR / f'benchmark_results_{dataset}'
    pp = pdir / 'parakeet-tdt-0.6b-v2.json'
    if not pp.exists():
        print(f'[{dataset}] no parakeet results', flush=True)
        return None

    p_data = json.load(open(pp))
    p_results = p_data.get('results', [])
    print(f'[{dataset}] {len(p_results)} files to analyze', flush=True)

    fh_err = open(OUT_DIR / f'{dataset}_per_error.tsv', 'w')
    fh_err.write('file_id\tterm\tcategory\tlength\tparakeet_sub\tref_context\n')
    fh_err.flush()

    by_length = collections.Counter()      # length → n_total
    by_length_err = collections.Counter()  # length → n_errors
    by_cat_len = collections.Counter()     # (cat, len) → n_total
    by_cat_len_err = collections.Counter() # (cat, len) → n_errors
    term_total = collections.Counter()     # (term, cat, len) → n
    term_err = collections.Counter()

    n_files = 0
    for r in p_results:
        ref_raw = r.get('reference', '')
        hyp_raw = r.get('hypothesis', '')
        fid = r.get('file_id', '')
        if not ref_raw:
            continue

        ref_norm = normalize(ref_raw)
        hyp_norm = normalize(hyp_raw)

        spans = find_medical_spans(ref_norm, matcher)
        if not spans:
            continue

        errs, subs = get_error_indices_and_subs(ref_norm, hyp_norm)
        ref_words = ref_norm.split()

        for span in spans:
            length = span['length']
            cat = span['category']
            term = span['ngram']

            by_length[length] += 1
            by_cat_len[(cat, length)] += 1
            term_total[(term, cat, length)] += 1

            if any(wi in errs for wi in span['word_indices']):
                by_length_err[length] += 1
                by_cat_len_err[(cat, length)] += 1
                term_err[(term, cat, length)] += 1

                # Capture parakeet's substitution
                first_idx = span['word_indices'][0]
                p_sub = subs.get(first_idx, '<UNKNOWN>')
                # Context: 3 words on each side
                ctx_start = max(0, first_idx - 3)
                ctx_end = min(len(ref_words), span['word_indices'][-1] + 4)
                ctx = ' '.join(ref_words[ctx_start:ctx_end])

                fh_err.write(
                    f'{fid}\t{term}\t{cat}\t{length}\t{p_sub}\t{ctx}\n'
                )
                fh_err.flush()

        n_files += 1

    fh_err.close()

    # ── write aggregates ──
    fh_len = open(OUT_DIR / f'{dataset}_by_length.tsv', 'w')
    fh_len.write('length\tn_total\tn_errors\terror_rate_pct\n')
    for length in sorted(by_length):
        tot = by_length[length]
        err = by_length_err[length]
        rate = 100 * err / tot if tot else 0
        fh_len.write(f'{length}\t{tot}\t{err}\t{rate:.2f}\n')
        fh_len.flush()
    fh_len.close()

    fh_cl = open(OUT_DIR / f'{dataset}_by_category_length.tsv', 'w')
    fh_cl.write('category\tlength\tn_total\tn_errors\terror_rate_pct\n')
    for (cat, length) in sorted(by_cat_len):
        tot = by_cat_len[(cat, length)]
        err = by_cat_len_err[(cat, length)]
        rate = 100 * err / tot if tot else 0
        fh_cl.write(f'{cat}\t{length}\t{tot}\t{err}\t{rate:.2f}\n')
        fh_cl.flush()
    fh_cl.close()

    fh_top = open(OUT_DIR / f'{dataset}_top_missed.tsv', 'w')
    fh_top.write('term\tcategory\tlength\tn_occurrences\tn_errors\terror_rate_pct\n')
    # Sort by error count descending
    for (term, cat, length), nerr in sorted(term_err.items(), key=lambda x: -x[1])[:200]:
        ntot = term_total[(term, cat, length)]
        rate = 100 * nerr / ntot if ntot else 0
        fh_top.write(f'{term}\t{cat}\t{length}\t{ntot}\t{nerr}\t{rate:.2f}\n')
        fh_top.flush()
    fh_top.close()

    total_spans = sum(by_length.values())
    total_errors = sum(by_length_err.values())
    overall_rate = 100 * total_errors / total_spans if total_spans else 0

    print(
        f'[{dataset}] DONE: {n_files} files, '
        f'{total_spans} medical spans, {total_errors} errors '
        f'({overall_rate:.2f}% error rate)',
        flush=True,
    )
    print(f'  By length:', flush=True)
    for length in sorted(by_length):
        tot = by_length[length]
        err = by_length_err[length]
        rate = 100 * err / tot if tot else 0
        print(f'    {length}-gram: {tot} total, {err} errors ({rate:.1f}%)', flush=True)
    print(f'  By category:', flush=True)
    cat_total = collections.Counter()
    cat_err = collections.Counter()
    for (cat, length), n in by_cat_len.items():
        cat_total[cat] += n
    for (cat, length), n in by_cat_len_err.items():
        cat_err[cat] += n
    for cat in sorted(cat_total):
        tot = cat_total[cat]; err = cat_err[cat]
        rate = 100 * err / tot if tot else 0
        print(f'    {cat}: {tot} total, {err} errors ({rate:.1f}%)', flush=True)

    return {
        'dataset': dataset,
        'n_files': n_files,
        'n_spans': total_spans,
        'n_errors': total_errors,
        'overall_error_rate': round(overall_rate, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', type=str, default=','.join(DATASETS))
    args = p.parse_args()

    print('Loading QuickUMLS...', flush=True)
    matcher = QuickUMLS(
        quickumls_fp=str(QUICKUMLS_PATH),
        threshold=0.8,
        similarity_name='jaccard',
        window=5,
    )
    print('QuickUMLS loaded.', flush=True)

    fh_summary = open(SUMMARY_TSV, 'w')
    fh_summary.write('dataset\tn_files\tn_spans\tn_errors\toverall_error_rate_pct\n')
    fh_summary.flush()

    for ds in args.datasets.split(','):
        r = process_dataset(ds, matcher)
        if r is None:
            continue
        fh_summary.write(
            f'{r["dataset"]}\t{r["n_files"]}\t{r["n_spans"]}\t{r["n_errors"]}\t{r["overall_error_rate"]}\n'
        )
        fh_summary.flush()
    fh_summary.close()
    print(f'\nSummary: {SUMMARY_TSV}', flush=True)


if __name__ == '__main__':
    main()
