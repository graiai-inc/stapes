#!/usr/bin/env python3
"""Drug-only Clinical Term Recall (CTR-on-drugs): parakeet baseline vs fused output.

Why this script
---------------
The targeted-fusion v3 substitutions (Parakeet+MedASR drug-name corrections) move
WER by < 0.01pp because drug tokens are a tiny fraction of the transcript word
count. The clinical impact is invisible at the WER level. The right metric is
**drug-name recall** — out of every drug name spoken in the reference, how often
does the hypothesis preserve it?

This script computes that metric for:
  - Parakeet baseline (the production conversation ASR)
  - Fused output (Parakeet with v3 targeted MedASR drug corrections layered on)

Span detection
--------------
QuickUMLS with the same matcher config as compute_per_file_ctr.py, but spans
are filtered to DRUG-only semtypes (T116, T121, T200, T195). The same
UMLS_STOPWORDS filter applies — short single-word matches require similarity=1.0
and aren't in the stopword list.

Fusion is run via measure_targeted_fusion_v3.fuse(), reusing the exact code path
that produced the published substitutions. No re-implementation drift.

Output
------
results/drug_ctr/per_file.tsv  — per (dataset, file_id) row with both baselines
results/drug_ctr/summary.tsv   — aggregate per dataset
results/drug_ctr/sub_outcomes.tsv — for each fusion sub, did it correct an error?

Run with the ossicles venv python:
    /home/grey/dev/graiai/ossicles/venv/bin/python scripts/compute_drug_ctr.py
"""
import csv
import json
import logging
import re
import sys
from pathlib import Path

import jiwer
from quickumls import QuickUMLS

# Reuse fuse() and helpers from the v3 fusion script directly — avoids drift.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_targeted_fusion_v3 import (  # noqa: E402
    COMMON_ENGLISH_NOISE,
    fuse,
    load_targeted_vocab,
    normalize,
)

sys.stdout.reconfigure(line_buffering=True)
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

VOCAB_TSV = STAPES_DIR / 'results' / 'medical_vocab' / 'medical_vocab_drug_curated.tsv'
OUT_DIR = STAPES_DIR / 'results' / 'drug_ctr'

DATASETS = ['primock57', 'figshare-osce', 'nazmulkazi', 'kokoro-va-sample']
DRUG_SEMTYPES = {'T116', 'T121', 'T200', 'T195'}

# Same stopword set as compute_per_file_ctr.py — words QuickUMLS over-matches as medical.
UMLS_STOPWORDS = {
    'bit', 'get', 'got', 'little', 'lot', 'said', 'still', 'today',
    'changes', 'well', 'cant', 'nothing', 'maybe', 'take', 'yes',
    'much', 'probably', 'worse', 'sharp', 'only', 'more', 'less',
    'stop', 'ask', 'bad', 'dad', 'may', 'nice', 'new', 'used',
    'find', 'care', 'place', 'always', 'wanted', 'play',
}

# Fusion config (DRUG-specific). Sweep showed (2, None) catches the most drug
# corrections (68 vs 56 with edit_ratio=0.5), but per-substitution inspection
# revealed the gain comes with ~2-3 non-drug-position content misfires across
# 400 files (e.g. PriMock57 'asthma' anchored to 'aspirin' edit-ratio 0.71;
# Kazi 'relief' to 'aleve' 0.83). Drug-CTR doesn't see these because the
# reference word isn't a drug, but they're real clinical content errors.
# (2, 0.5) blocks them at the cost of 12 drug catches — net 56 corrected, 0
# introduced clinical errors. Worth the trade for the drug case.
#
# Different entity classes (CONDITION, SYMPTOM, etc.) will likely want
# different thresholds — conditions/symptoms are common English words and
# need tighter gating. Future fusion runs over those categories should
# re-sweep with category-specific defaults.
FUSION_MIN_ANCHOR_WORDS = 2
FUSION_MAX_ANCHOR_EDIT_RATIO = 0.5


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


def find_drug_spans(text: str, matcher) -> list[dict]:
    """Return DRUG-only medical spans (semtype filter)."""
    matches = matcher.match(text)
    spans = []
    for match_group in matches:
        best = match_group[0]
        ngram = best['ngram'].lower().strip()
        if ngram in UMLS_STOPWORDS:
            continue
        # Same single-word minimum-length / similarity gate as compute_per_file_ctr.py
        if ' ' not in ngram and len(ngram) < 6 and best['similarity'] < 1.0:
            continue
        # DRUG-only semtype filter
        is_drug = any(s in DRUG_SEMTYPES for s in best.get('semtypes', []))
        if not is_drug:
            continue
        word_indices = char_span_to_word_indices(text, best['start'], best['end'])
        if not word_indices:
            continue
        spans.append({
            'ngram': best['ngram'],
            'word_indices': word_indices,
            'similarity': best['similarity'],
        })
    return spans


def get_error_indices(ref: str, hyp: str) -> set[int]:
    out = jiwer.process_words(ref, hyp)
    errors = set()
    for chunk in out.alignments[0]:
        if chunk.type in ('substitute', 'delete'):
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                errors.add(i)
    return errors


def count_drug_errors(ref_norm: str, hyp_norm: str, drug_spans: list[dict]) -> tuple[int, int]:
    if not drug_spans:
        return 0, 0
    error_indices = get_error_indices(ref_norm, hyp_norm)
    n_errors = 0
    for span in drug_spans:
        if any(wi in error_indices for wi in span['word_indices']):
            n_errors += 1
    return n_errors, len(drug_spans)


def load_dataset_pair(dataset: str) -> tuple[dict, dict] | None:
    pdir = OSSICLES_DIR / f'benchmark_results_{dataset}'
    pp = pdir / 'parakeet-tdt-0.6b-v2.json'
    mp = pdir / 'medasr.json'
    if not pp.exists() or not mp.exists():
        log.warning(f'  missing parakeet or medasr JSON for {dataset}')
        return None
    p_data = json.load(open(pp))
    m_data = json.load(open(mp))
    p_idx = {r['file_id']: r for r in p_data.get('results', []) if 'file_id' in r}
    m_idx = {r['file_id']: r for r in m_data.get('results', []) if 'file_id' in r}
    return p_idx, m_idx


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f'loading curated drug vocab from {VOCAB_TSV}')
    vocab, max_n = load_targeted_vocab(VOCAB_TSV, ['DRUG'])
    log.info(f'  {len(vocab)} terms, max phrase length {max_n}')

    log.info('initializing QuickUMLS...')
    matcher = QuickUMLS(
        quickumls_fp=str(QUICKUMLS_PATH),
        threshold=0.8,
        similarity_name='jaccard',
        window=5,
    )

    fh_per = open(OUT_DIR / 'per_file.tsv', 'w')
    fh_per.write(
        'dataset\tfile_id\tparakeet_drug_errors\tparakeet_drug_total\t'
        'fused_drug_errors\tfused_drug_total\tfusion_subs_acoustic\tfusion_subs_anchor\n'
    )
    fh_per.flush()

    fh_summary = open(OUT_DIR / 'summary.tsv', 'w')
    fh_summary.write(
        f'# vocab_size={len(vocab)}\n'
        f'# fusion_min_anchor_words={FUSION_MIN_ANCHOR_WORDS}\n'
        f'# fusion_max_anchor_edit_ratio={FUSION_MAX_ANCHOR_EDIT_RATIO}\n'
        f'# drug_semtypes={sorted(DRUG_SEMTYPES)}\n'
        'dataset\tn_files\tn_drug_spans\t'
        'parakeet_drug_recall\tfused_drug_recall\tdelta_pp\t'
        'parakeet_errors\tfused_errors\terrors_corrected\terrors_introduced\n'
    )
    fh_summary.flush()

    fh_outcomes = open(OUT_DIR / 'sub_outcomes.tsv', 'w')
    fh_outcomes.write(
        'dataset\tfile_id\tparakeet_drug_errors\tfused_drug_errors\t'
        'errors_delta\tn_subs_acoustic\tn_subs_anchor\n'
    )
    fh_outcomes.flush()

    grand_summary = []

    for dataset in DATASETS:
        log.info(f'=== {dataset} ===')
        pair = load_dataset_pair(dataset)
        if pair is None:
            continue
        p_idx, m_idx = pair
        common = sorted(set(p_idx) & set(m_idx))
        log.info(f'  {len(common)} files in common')

        n_files_with_drugs = 0
        ds_total_spans = 0
        ds_p_errors = 0
        ds_f_errors = 0
        ds_corrected = 0
        ds_introduced = 0

        for i, fid in enumerate(common):
            p_row = p_idx[fid]
            m_row = m_idx[fid]
            ref = p_row.get('reference', '')
            if not ref:
                continue
            ref_norm = normalize(ref)
            p_hyp_norm = normalize(p_row.get('hypothesis', ''))
            m_hyp_norm = normalize(m_row.get('hypothesis', ''))

            # Find DRUG spans in the reference
            drug_spans = find_drug_spans(ref_norm, matcher)
            if not drug_spans:
                # No drugs in reference — file contributes 0/0; record but skip stats
                fh_per.write(f'{dataset}\t{fid}\t0\t0\t0\t0\t0\t0\n')
                fh_per.flush()
                continue
            n_files_with_drugs += 1
            ds_total_spans += len(drug_spans)

            # Run fusion in-memory using the exact same code as v3 script
            fused_hyp, fstats = fuse(
                p_hyp_norm, m_hyp_norm, vocab, max_n,
                min_anchor_words=FUSION_MIN_ANCHOR_WORDS,
                max_anchor_edit_ratio=FUSION_MAX_ANCHOR_EDIT_RATIO,
            )

            p_err, p_tot = count_drug_errors(ref_norm, p_hyp_norm, drug_spans)
            f_err, f_tot = count_drug_errors(ref_norm, fused_hyp, drug_spans)
            assert p_tot == f_tot == len(drug_spans), 'span count mismatch'

            ds_p_errors += p_err
            ds_f_errors += f_err
            corrected = max(0, p_err - f_err)
            introduced = max(0, f_err - p_err)
            ds_corrected += corrected
            ds_introduced += introduced

            fh_per.write(
                f'{dataset}\t{fid}\t{p_err}\t{p_tot}\t{f_err}\t{f_tot}\t'
                f'{fstats["subs_acoustic"]}\t{fstats["subs_anchor"]}\n'
            )
            fh_per.flush()

            n_subs = fstats['subs_acoustic'] + fstats['subs_anchor']
            if n_subs > 0:
                fh_outcomes.write(
                    f'{dataset}\t{fid}\t{p_err}\t{f_err}\t{f_err - p_err}\t'
                    f'{fstats["subs_acoustic"]}\t{fstats["subs_anchor"]}\n'
                )
                fh_outcomes.flush()

            if (i + 1) % 25 == 0 or i + 1 == len(common):
                log.info(f'  [{i + 1}/{len(common)}] running totals: '
                         f'parakeet_errors={ds_p_errors} fused_errors={ds_f_errors} '
                         f'corrected={ds_corrected} introduced={ds_introduced}')

        if ds_total_spans == 0:
            log.warning(f'  {dataset}: no drug spans found, skipping')
            continue
        p_recall = 100 * (1.0 - ds_p_errors / ds_total_spans)
        f_recall = 100 * (1.0 - ds_f_errors / ds_total_spans)
        delta = f_recall - p_recall
        fh_summary.write(
            f'{dataset}\t{n_files_with_drugs}\t{ds_total_spans}\t'
            f'{p_recall:.4f}\t{f_recall:.4f}\t{delta:+.4f}\t'
            f'{ds_p_errors}\t{ds_f_errors}\t{ds_corrected}\t{ds_introduced}\n'
        )
        fh_summary.flush()
        grand_summary.append((dataset, ds_total_spans, p_recall, f_recall, delta,
                              ds_corrected, ds_introduced))
        log.info(
            f'  {dataset} DONE: drug-CTR parakeet={p_recall:.2f}% fused={f_recall:.2f}% '
            f'(Δ {delta:+.2f}pp), corrected={ds_corrected} introduced={ds_introduced}'
        )

    fh_per.close()
    fh_summary.close()
    fh_outcomes.close()

    print('\n=== DRUG-CTR SUMMARY ===', flush=True)
    print(f'{"dataset":<18s} {"spans":>6s} {"p_recall":>10s} {"f_recall":>10s} '
          f'{"Δpp":>8s} {"corr":>5s} {"intr":>5s}', flush=True)
    for ds, n, pr, fr, d, cor, intr in grand_summary:
        print(f'{ds:<18s} {n:>6d} {pr:>9.2f}% {fr:>9.2f}% {d:>+7.2f} {cor:>5d} {intr:>5d}',
              flush=True)
    print(f'\nWrote: {OUT_DIR}/per_file.tsv, summary.tsv, sub_outcomes.tsv', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
