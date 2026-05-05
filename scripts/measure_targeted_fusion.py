#!/usr/bin/env python3
"""Measure UMLS-gated targeted Parakeet + MedASR fusion on conversation datasets.

The hypothesis: at positions where Parakeet and MedASR disagree, if MedASR's
word/phrase is in the medical vocab and Parakeet's is not, substitute MedASR's
output. Outside those positions, keep Parakeet (overall stronger). This avoids
the "MedASR contaminates non-medical content" failure mode that killed equal-
weight ROVER.

Inputs (read-only):
    - medical vocab TSV (snapshot)
    - existing Parakeet outputs in benchmark_results_<dataset>/parakeet-tdt-0.6b-v2.json
    - existing MedASR outputs in benchmark_results_<dataset>/medasr.json

Outputs (incremental, per-file flush):
    - results/targeted_fusion/<dataset>_per_file.tsv
        file_id, parakeet_wer, medasr_wer, fused_wer, n_substitutions, n_insertions
    - results/targeted_fusion/<dataset>_substitutions.tsv
        file_id, position, parakeet_word, medasr_word, substitution_type
    - results/targeted_fusion/summary.tsv (rewritten on each dataset completion)
        dataset, n_files, parakeet_wer, medasr_wer, fused_wer, fused_delta_pp

Usage:
    /home/grey/dev/graiai/ossicles/venv/bin/python scripts/measure_targeted_fusion.py
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import jiwer

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
STAPES_DIR = SCRIPT_DIR.parent
GRAIAI_DIR = STAPES_DIR.parent
OSSICLES_DIR = GRAIAI_DIR / 'ossicles'

VOCAB_TSV = STAPES_DIR / 'results' / 'medical_vocab' / 'medical_vocab_snapshot_v1.tsv'
OUT_DIR = STAPES_DIR / 'results' / 'targeted_fusion'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_TSV = OUT_DIR / 'summary.tsv'

DATASETS = ['primock57', 'figshare-osce', 'nazmulkazi']


def normalize(text: str) -> str:
    """Match the script's existing normalize() — used for WER computation."""
    text = text.lower()
    text = re.sub(r"[^\w\s'\-]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_vocab(vocab_tsv: Path) -> tuple[set, int]:
    """Load medical vocab into a set; also report max phrase length in tokens."""
    vocab = set()
    max_n = 1
    with open(vocab_tsv) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            term = row.get('term', '').strip().lower()
            if not term:
                continue
            vocab.add(term)
            n = term.count(' ') + 1
            if n > max_n:
                max_n = n
    return vocab, max_n


def is_medical_at(words: list[str], start: int, vocab: set, max_n: int) -> int:
    """Return length (in tokens) of longest vocab match starting at words[start],
    or 0 if no match. Tries longest-first so multi-word phrases beat single-word.
    """
    for n in range(min(max_n, len(words) - start), 0, -1):
        phrase = ' '.join(words[start:start + n])
        if phrase in vocab:
            return n
    return 0


def medical_span_overlap(words: list[str], start: int, end: int, vocab: set, max_n: int) -> bool:
    """True if any n-gram starting between [start, end) is in vocab."""
    for i in range(start, end):
        if is_medical_at(words, i, vocab, max_n):
            return True
    return False


def fuse(parakeet_text: str, medasr_text: str, vocab: set, max_n: int):
    """Apply UMLS-gated targeted substitution. Returns (fused_text, stats)."""
    p_words = parakeet_text.split()
    m_words = medasr_text.split()

    if not p_words and not m_words:
        return '', {'subs': 0, 'inserts': 0, 'sub_log': []}
    if not p_words:
        # Empty Parakeet — keep MedASR's medical content if any
        if any(is_medical_at(m_words, i, vocab, max_n) for i in range(len(m_words))):
            return ' '.join(m_words), {'subs': 0, 'inserts': len(m_words), 'sub_log': []}
        return '', {'subs': 0, 'inserts': 0, 'sub_log': []}

    # jiwer alignment: ref=parakeet, hyp=medasr
    out = jiwer.process_words(parakeet_text, medasr_text)
    fused = []
    sub_log = []
    n_subs = 0
    n_inserts = 0

    for chunk in out.alignments[0]:
        ctype = chunk.type
        # ref=parakeet indices, hyp=medasr indices
        p_start, p_end = chunk.ref_start_idx, chunk.ref_end_idx
        m_start, m_end = chunk.hyp_start_idx, chunk.hyp_end_idx

        if ctype == 'equal':
            fused.extend(p_words[p_start:p_end])

        elif ctype == 'substitute':
            # Different words at same aligned position
            p_span_medical = medical_span_overlap(p_words, p_start, p_end, vocab, max_n)
            m_span_medical = medical_span_overlap(m_words, m_start, m_end, vocab, max_n)
            if m_span_medical and not p_span_medical:
                # Case 1: MedASR has medical, Parakeet doesn't — substitute
                fused.extend(m_words[m_start:m_end])
                n_subs += 1
                sub_log.append({
                    'type': 'sub',
                    'parakeet': ' '.join(p_words[p_start:p_end]),
                    'medasr': ' '.join(m_words[m_start:m_end]),
                    'position': p_start,
                })
            else:
                fused.extend(p_words[p_start:p_end])

        elif ctype == 'delete':
            # Word in Parakeet (ref) but not in MedASR (hyp). Keep Parakeet.
            fused.extend(p_words[p_start:p_end])

        elif ctype == 'insert':
            # Word in MedASR (hyp) but not in Parakeet (ref). If medical, insert.
            if medical_span_overlap(m_words, m_start, m_end, vocab, max_n):
                fused.extend(m_words[m_start:m_end])
                n_inserts += 1
                sub_log.append({
                    'type': 'ins',
                    'parakeet': '',
                    'medasr': ' '.join(m_words[m_start:m_end]),
                    'position': p_start,  # Parakeet position where insertion goes
                })
            # else drop: Parakeet's "absence" of non-medical is correct

    return ' '.join(fused), {'subs': n_subs, 'inserts': n_inserts, 'sub_log': sub_log}


def wer(ref: str, hyp: str) -> tuple[float, int, int]:
    """Returns (wer, n_errors, n_words)."""
    ref_n = normalize(ref)
    hyp_n = normalize(hyp)
    if not ref_n:
        return 0.0, 0, 0
    n_words = len(ref_n.split())
    w = jiwer.wer(ref_n, hyp_n)
    n_errors = round(w * n_words)
    return w, n_errors, n_words


def process_dataset(dataset: str, vocab: set, max_n: int) -> dict | None:
    pdir = OSSICLES_DIR / f'benchmark_results_{dataset}'
    pp = pdir / 'parakeet-tdt-0.6b-v2.json'
    mp = pdir / 'medasr.json'
    if not pp.exists() or not mp.exists():
        print(f'[{dataset}] missing inputs (parakeet={pp.exists()} medasr={mp.exists()})', flush=True)
        return None

    p_data = json.load(open(pp))
    m_data = json.load(open(mp))

    p_results = p_data.get('results', [])
    m_results = m_data.get('results', [])
    p_idx = {r['file_id']: r for r in p_results}
    m_idx = {r['file_id']: r for r in m_results}
    common = sorted(set(p_idx.keys()) & set(m_idx.keys()))
    if not common:
        # Try matching by reference prefix
        p_by_ref = {r.get('reference', '')[:120]: r for r in p_results}
        m_by_ref = {r.get('reference', '')[:120]: r for r in m_results}
        common_keys = set(p_by_ref) & set(m_by_ref)
        common = sorted(common_keys)
        print(f'[{dataset}] file_id mismatch — falling back to reference matching: {len(common)} matched', flush=True)
        p_idx = {k: p_by_ref[k] for k in common}
        m_idx = {k: m_by_ref[k] for k in common}

    print(f'[{dataset}] {len(common)} files to process', flush=True)

    per_file_path = OUT_DIR / f'{dataset}_per_file.tsv'
    sub_log_path = OUT_DIR / f'{dataset}_substitutions.tsv'

    fh_per = open(per_file_path, 'w')
    fh_per.write('file_id\tparakeet_wer\tmedasr_wer\tfused_wer\tn_subs\tn_inserts\tn_words\n')
    fh_per.flush()

    fh_sub = open(sub_log_path, 'w')
    fh_sub.write('file_id\tposition\ttype\tparakeet_word\tmedasr_word\n')
    fh_sub.flush()

    tot_p_err = tot_m_err = tot_f_err = 0
    tot_words_p = tot_words_m = tot_words_f = 0
    tot_subs = tot_inserts = 0
    fused_better = fused_worse = fused_tied = 0

    for fid in common:
        p_row = p_idx[fid]
        m_row = m_idx[fid]
        ref = p_row.get('reference', '')
        if not ref:
            continue
        p_hyp_raw = p_row.get('hypothesis', '')
        m_hyp_raw = m_row.get('hypothesis', '')

        p_hyp = normalize(p_hyp_raw)
        m_hyp = normalize(m_hyp_raw)

        fused_hyp, stats = fuse(p_hyp, m_hyp, vocab, max_n)

        p_wer, p_e, p_w = wer(ref, p_hyp_raw)
        m_wer, m_e, m_w = wer(ref, m_hyp_raw)
        f_wer, f_e, f_w = wer(ref, fused_hyp)

        fh_per.write(
            f'{fid}\t{p_wer * 100:.2f}\t{m_wer * 100:.2f}\t{f_wer * 100:.2f}\t'
            f'{stats["subs"]}\t{stats["inserts"]}\t{p_w}\n'
        )
        fh_per.flush()

        for s in stats['sub_log']:
            fh_sub.write(
                f'{fid}\t{s["position"]}\t{s["type"]}\t'
                f'{s["parakeet"]}\t{s["medasr"]}\n'
            )
            fh_sub.flush()

        tot_p_err += p_e; tot_words_p += p_w
        tot_m_err += m_e; tot_words_m += m_w
        tot_f_err += f_e; tot_words_f += f_w
        tot_subs += stats['subs']; tot_inserts += stats['inserts']

        if f_wer < p_wer - 1e-9: fused_better += 1
        elif f_wer > p_wer + 1e-9: fused_worse += 1
        else: fused_tied += 1

        if (fused_better + fused_worse + fused_tied) % 20 == 0 or stats['subs'] + stats['inserts'] > 0:
            print(
                f'  {fid}: parakeet {p_wer * 100:.1f}% medasr {m_wer * 100:.1f}% '
                f'fused {f_wer * 100:.1f}% (subs={stats["subs"]} ins={stats["inserts"]})',
                flush=True,
            )

    fh_per.close(); fh_sub.close()

    p_agg = 100 * tot_p_err / tot_words_p if tot_words_p else 0
    m_agg = 100 * tot_m_err / tot_words_m if tot_words_m else 0
    f_agg = 100 * tot_f_err / tot_words_f if tot_words_f else 0
    delta = f_agg - p_agg

    print(
        f'[{dataset}] DONE: '
        f'parakeet {p_agg:.2f}% | medasr {m_agg:.2f}% | fused {f_agg:.2f}% '
        f'(Δ vs parakeet: {delta:+.2f}pp). '
        f'better={fused_better} worse={fused_worse} tied={fused_tied}, '
        f'total subs={tot_subs} inserts={tot_inserts}',
        flush=True,
    )

    return {
        'dataset': dataset,
        'n_files': len(common),
        'parakeet_wer': round(p_agg, 4),
        'medasr_wer': round(m_agg, 4),
        'fused_wer': round(f_agg, 4),
        'fused_delta_pp': round(delta, 4),
        'better': fused_better,
        'worse': fused_worse,
        'tied': fused_tied,
        'n_substitutions': tot_subs,
        'n_insertions': tot_inserts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab', type=Path, default=VOCAB_TSV)
    p.add_argument('--datasets', type=str, default=','.join(DATASETS))
    args = p.parse_args()

    print(f'Loading vocab from {args.vocab}', flush=True)
    vocab, max_n = load_vocab(args.vocab)
    print(f'Loaded {len(vocab)} terms, max phrase length {max_n} tokens', flush=True)

    fh_summary = open(SUMMARY_TSV, 'w')
    fh_summary.write('dataset\tn_files\tparakeet_wer\tmedasr_wer\tfused_wer\tfused_delta_pp\tbetter\tworse\ttied\tn_subs\tn_inserts\n')
    fh_summary.flush()

    for ds in args.datasets.split(','):
        result = process_dataset(ds, vocab, max_n)
        if result is None:
            continue
        fh_summary.write(
            f'{result["dataset"]}\t{result["n_files"]}\t'
            f'{result["parakeet_wer"]}\t{result["medasr_wer"]}\t{result["fused_wer"]}\t'
            f'{result["fused_delta_pp"]}\t{result["better"]}\t{result["worse"]}\t{result["tied"]}\t'
            f'{result["n_substitutions"]}\t{result["n_insertions"]}\n'
        )
        fh_summary.flush()

    fh_summary.close()
    print(f'\nSummary: {SUMMARY_TSV}', flush=True)


if __name__ == '__main__':
    main()
