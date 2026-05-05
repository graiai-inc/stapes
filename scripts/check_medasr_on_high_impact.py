#!/usr/bin/env python3
"""For Parakeet's NEGATION_FLIP / NUMERIC errors, check what MedASR transcribed
at the same reference position. Reports MedASR's recovery rate per pattern.

If MedASR does better on these patterns, fusion can be extended to cover them.
If it doesn't, fusion can't help — different mitigation needed.

Method
------
For each file:
  1. Align ref ↔ parakeet → find error chunks where the pattern fires
  2. Align ref ↔ medasr  → for each of those ref positions, get medasr's words
  3. Check if medasr's words match the reference at that position (exact match
     OR contains the right negation/number token)

Per dataset, report:
  - n parakeet-error positions for each pattern
  - n where medasr got it right
  - n where medasr also got it wrong
  - n where medasr's chunk doesn't align cleanly (skipped)

Run with the ossicles venv python:
    /home/grey/dev/graiai/ossicles/venv/bin/python \\
        scripts/check_medasr_on_high_impact.py
"""
import json
import logging
import re
import sys
from pathlib import Path

import jiwer

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from measure_targeted_fusion_v3 import normalize  # noqa: E402
from mine_actual_error_patterns import (  # noqa: E402
    LATERALITY_TOKENS,
    NEGATION_TOKENS,
    NUMERIC_DIGIT_RE,
    SPELLED_NUMBER_TOKENS,
    detect_laterality,
    detect_negation_flip,
    detect_numeric_mismatch,
    tokens_of,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

STAPES_DIR = SCRIPT_DIR.parent
GRAIAI_DIR = STAPES_DIR.parent
OSSICLES_DIR = GRAIAI_DIR / 'ossicles'
OUT_DIR = STAPES_DIR / 'results' / 'medasr_on_high_impact'

DATASETS = ['primock57', 'figshare-osce', 'nazmulkazi', 'kokoro-va-sample']


def load_pair(dataset: str):
    pdir = OSSICLES_DIR / f'benchmark_results_{dataset}'
    pp = pdir / 'parakeet-tdt-0.6b-v2.json'
    mp = pdir / 'medasr.json'
    if not (pp.exists() and mp.exists()):
        return None
    p_idx = {r['file_id']: r for r in json.load(open(pp)).get('results', [])
             if 'file_id' in r}
    m_idx = {r['file_id']: r for r in json.load(open(mp)).get('results', [])
             if 'file_id' in r}
    return p_idx, m_idx


def get_medasr_words_at_ref_position(
    ref_words: list[str], medasr_words: list[str],
    ref_start: int, ref_end: int,
    fused_chunks,
) -> list[str]:
    """Return medasr's words that align to ref[ref_start:ref_end]."""
    # For each ref index in [ref_start, ref_end), find which medasr chunk it falls in
    # then collect medasr's hyp range across all those chunks.
    h_starts = []
    h_ends = []
    for fc in fused_chunks:
        for ri in range(fc.ref_start_idx, fc.ref_end_idx):
            if ref_start <= ri < ref_end:
                h_starts.append(fc.hyp_start_idx)
                h_ends.append(fc.hyp_end_idx)
                break
    if not h_starts:
        return []
    return medasr_words[min(h_starts):max(h_ends)]


def has_negation(words: list[str]) -> bool:
    return any(t in NEGATION_TOKENS for t in words)


def numeric_set(words: list[str]) -> set[str]:
    out = set()
    for w in words:
        if NUMERIC_DIGIT_RE.search(w) or w in SPELLED_NUMBER_TOKENS:
            out.add(w)
    return out


def laterality_set(words: list[str]) -> set[str]:
    return set(t for t in words if t in LATERALITY_TOKENS)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fh = open(OUT_DIR / 'check_results.tsv', 'w')
    fh.write(
        'dataset\tfile_id\tpattern\tref_words\tparakeet_words\tmedasr_words\t'
        'medasr_correct\n'
    )
    fh.flush()

    fh_summary = open(OUT_DIR / 'summary.tsv', 'w')
    fh_summary.write(
        'pattern\tdataset\tn_parakeet_errors\tn_medasr_correct\tn_medasr_wrong\t'
        'n_medasr_unalignable\tmedasr_recovery_rate_pct\n'
    )
    fh_summary.flush()

    overall_counts = {}

    for dataset in DATASETS:
        log.info(f'=== {dataset} ===')
        pair = load_pair(dataset)
        if pair is None:
            log.warning(f'  skip {dataset}, missing JSONs')
            continue
        p_idx, m_idx = pair

        # Per-pattern: counts of (parakeet_errors, medasr_correct, medasr_wrong, unalignable)
        counts = {'NEGATION_FLIP': [0, 0, 0, 0],
                  'LATERALITY': [0, 0, 0, 0],
                  'NUMERIC': [0, 0, 0, 0]}

        common = sorted(set(p_idx) & set(m_idx))
        for fid in common:
            ref = p_idx[fid].get('reference', '')
            if not ref:
                continue
            ref_norm = normalize(ref)
            p_norm = normalize(p_idx[fid].get('hypothesis', ''))
            m_norm = normalize(m_idx[fid].get('hypothesis', ''))

            ref_words = ref_norm.split()
            p_words = p_norm.split()
            m_words = m_norm.split()

            # Align ref vs parakeet — find error chunks
            p_align = jiwer.process_words(ref_norm, p_norm)
            # Align ref vs medasr — for cross-referencing
            m_align = jiwer.process_words(ref_norm, m_norm)
            m_chunks = m_align.alignments[0]

            for chunk in p_align.alignments[0]:
                if chunk.type not in ('substitute', 'delete'):
                    continue
                rs, re_ = chunk.ref_start_idx, chunk.ref_end_idx
                hs, he = chunk.hyp_start_idx, chunk.hyp_end_idx
                ref_chunk = ref_words[rs:re_]
                p_chunk = p_words[hs:he] if chunk.type == 'substitute' else []

                # Also need primary_cat for NEGATION (from the per-error categorization).
                # For this script, just pass 'generic' — we want to count all parakeet
                # negation errors here, not exclude based on cat.
                patterns = []
                if detect_negation_flip(ref_chunk, p_chunk, primary_cat='generic'):
                    patterns.append('NEGATION_FLIP')
                if detect_laterality(ref_chunk, p_chunk):
                    patterns.append('LATERALITY')
                if detect_numeric_mismatch(ref_chunk, p_chunk):
                    patterns.append('NUMERIC')
                if not patterns:
                    continue

                m_chunk = get_medasr_words_at_ref_position(
                    ref_words, m_words, rs, re_, m_chunks,
                )

                for pat in patterns:
                    counts[pat][0] += 1

                    # Did MedASR get this right?
                    if pat == 'NEGATION_FLIP':
                        # Correct = medasr's alignment-region matches the reference's
                        # negation polarity (both have neg or both don't).
                        ref_neg = has_negation(ref_chunk)
                        m_neg = has_negation(m_chunk)
                        if not m_chunk:
                            counts[pat][3] += 1
                            correct = 'unalignable'
                        elif ref_neg == m_neg:
                            counts[pat][1] += 1
                            correct = '1'
                        else:
                            counts[pat][2] += 1
                            correct = '0'

                    elif pat == 'NUMERIC':
                        ref_n = numeric_set(ref_chunk)
                        m_n = numeric_set(m_chunk)
                        if not m_chunk:
                            counts[pat][3] += 1
                            correct = 'unalignable'
                        elif ref_n == m_n:
                            counts[pat][1] += 1
                            correct = '1'
                        else:
                            counts[pat][2] += 1
                            correct = '0'

                    elif pat == 'LATERALITY':
                        ref_l = laterality_set(ref_chunk)
                        m_l = laterality_set(m_chunk)
                        if not m_chunk:
                            counts[pat][3] += 1
                            correct = 'unalignable'
                        elif ref_l == m_l:
                            counts[pat][1] += 1
                            correct = '1'
                        else:
                            counts[pat][2] += 1
                            correct = '0'

                    fh.write(
                        f'{dataset}\t{fid}\t{pat}\t{" ".join(ref_chunk)}\t'
                        f'{" ".join(p_chunk)}\t{" ".join(m_chunk)}\t{correct}\n'
                    )
                    fh.flush()

        # Write summary rows for this dataset
        for pat in ['NEGATION_FLIP', 'LATERALITY', 'NUMERIC']:
            n_total, n_correct, n_wrong, n_unalign = counts[pat]
            rate = 100 * n_correct / (n_total - n_unalign) if (n_total - n_unalign) else 0
            fh_summary.write(
                f'{pat}\t{dataset}\t{n_total}\t{n_correct}\t{n_wrong}\t{n_unalign}\t{rate:.1f}\n'
            )
            fh_summary.flush()
            log.info(f'  {pat}: parakeet wrong on {n_total}, '
                     f'medasr correct on {n_correct} ({rate:.1f}% recovery), '
                     f'wrong on {n_wrong}, unalignable {n_unalign}')

            # Aggregate
            if pat not in overall_counts:
                overall_counts[pat] = [0, 0, 0, 0]
            for j in range(4):
                overall_counts[pat][j] += counts[pat][j]

    # Overall rows
    print(f'\n=== OVERALL: where MedASR could rescue Parakeet ===', flush=True)
    print(f'{"pattern":<16s} {"parakeet wrong":>14s}  {"medasr correct":>15s}  '
          f'{"recovery %":>10s}  {"medasr also wrong":>17s}', flush=True)
    for pat in ['NEGATION_FLIP', 'LATERALITY', 'NUMERIC']:
        n_total, n_correct, n_wrong, n_unalign = overall_counts[pat]
        denom = n_total - n_unalign
        rate = 100 * n_correct / denom if denom else 0
        print(
            f'{pat:<16s} {n_total:>14d}  {n_correct:>15d}  '
            f'{rate:>9.1f}%  {n_wrong:>17d}',
            flush=True,
        )
        fh_summary.write(
            f'{pat}\tALL\t{n_total}\t{n_correct}\t{n_wrong}\t{n_unalign}\t{rate:.1f}\n'
        )
        fh_summary.flush()

    fh.close()
    fh_summary.close()
    print(f'\nFull check log: {OUT_DIR}/check_results.tsv', flush=True)
    print(f'Summary: {OUT_DIR}/summary.tsv', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
