#!/usr/bin/env python3
"""V2 targeted Parakeet+MedASR fusion: drug-focused vocab + acoustic similarity.

Changes from v1:
  - TARGETED vocab subset (DRUG-only by default; configurable to add PROCEDURE,
    selected CONDITION/LAB_RESULT) instead of all medical categories.
  - Acoustic-similarity gate: substitute only when Parakeet's joined span and
    MedASR's joined vocab phrase are likely the same word (low char edit dist,
    space-stripped). Catches "ram a pril" → "ramipril" (Parakeet split a single
    drug into multiple words). Rejects "okay all right" → "ramipril" (totally
    different — no acoustic anchor).
  - Multi-word vocab support: MedASR's chunk is scanned with a sliding window
    for the LONGEST matching vocab phrase, not just word-by-word.

Inputs (read-only):
  - results/medical_vocab/medical_vocab_snapshot_v1.tsv (filtered by category)
  - benchmark_results_<dataset>/parakeet-tdt-0.6b-v2.json
  - benchmark_results_<dataset>/medasr.json

Outputs (incremental):
  - results/targeted_fusion_v2/<dataset>_per_file.tsv
  - results/targeted_fusion_v2/<dataset>_substitutions.tsv
  - results/targeted_fusion_v2/summary.tsv
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
OUT_DIR = STAPES_DIR / 'results' / 'targeted_fusion_v2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ['primock57', 'figshare-osce', 'nazmulkazi']
DEFAULT_CATEGORIES = ['DRUG']  # the clear win category


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s'\-]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_targeted_vocab(vocab_tsv: Path, categories: list[str]) -> tuple[set, int]:
    vocab = set()
    max_n = 1
    cat_set = set(categories)
    with open(vocab_tsv) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            term = row.get('term', '').strip().lower()
            cat = row.get('category', '').strip()
            if not term or cat not in cat_set:
                continue
            vocab.add(term)
            n = term.count(' ') + 1
            if n > max_n:
                max_n = n
    return vocab, max_n


def find_longest_vocab_phrase(words: list[str], start: int, end: int,
                               vocab: set, max_n: int) -> tuple[int, int, str] | None:
    """Within words[start:end], find the longest vocab phrase. Returns
    (match_start, match_end, phrase) or None.

    Tries longest-first within the window for each starting position.
    """
    best = None  # (length, start, end, phrase)
    for s in range(start, end):
        for n in range(min(max_n, end - s), 0, -1):
            phrase = ' '.join(words[s:s + n])
            if phrase in vocab:
                if best is None or n > best[0]:
                    best = (n, s, s + n, phrase)
                break  # found longest at this position, move to next start
    if best is None:
        return None
    return best[1], best[2], best[3]


def levenshtein(a: str, b: str) -> int:
    """Standard Levenshtein edit distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            ins = curr[j] + 1
            dele = prev[j + 1] + 1
            sub = prev[j] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def acoustic_similar(parakeet_span: str, medasr_phrase: str) -> tuple[bool, int, int]:
    """Are these likely the same word/phrase, just with different word boundaries?

    Strips spaces and computes char-level edit distance. Returns (is_similar,
    edit_dist, max_len). is_similar = ratio < 0.3 OR edit_dist <= 1.
    """
    p_join = parakeet_span.replace(' ', '').replace('-', '').lower()
    m_join = medasr_phrase.replace(' ', '').replace('-', '').lower()
    if not p_join or not m_join:
        return False, 0, 0
    dist = levenshtein(p_join, m_join)
    max_len = max(len(p_join), len(m_join))
    ratio = dist / max_len if max_len else 1.0
    is_similar = (dist <= 1) or (ratio < 0.3 and dist <= 4)
    return is_similar, dist, max_len


def fuse(parakeet_text: str, medasr_text: str, vocab: set, max_n: int):
    """Apply v2 fusion rule. Returns (fused_text, stats)."""
    p_words = parakeet_text.split()
    m_words = medasr_text.split()

    if not p_words and not m_words:
        return '', {'subs': 0, 'sub_log': []}
    if not p_words:
        return medasr_text, {'subs': 0, 'sub_log': []}
    if not m_words:
        return parakeet_text, {'subs': 0, 'sub_log': []}

    out = jiwer.process_words(parakeet_text, medasr_text)
    fused = []
    sub_log = []
    n_subs = 0

    for chunk in out.alignments[0]:
        ctype = chunk.type
        p_start, p_end = chunk.ref_start_idx, chunk.ref_end_idx
        m_start, m_end = chunk.hyp_start_idx, chunk.hyp_end_idx

        if ctype == 'equal':
            fused.extend(p_words[p_start:p_end])

        elif ctype == 'substitute':
            # Look for vocab match in MedASR's chunk
            vocab_match = find_longest_vocab_phrase(m_words, m_start, m_end, vocab, max_n)
            if vocab_match is not None:
                m_match_start, m_match_end, m_phrase = vocab_match
                p_span_text = ' '.join(p_words[p_start:p_end])

                # Check acoustic similarity between Parakeet's span and MedASR's vocab phrase
                similar, dist, max_len = acoustic_similar(p_span_text, m_phrase)
                if similar:
                    # Substitute Parakeet's span with MedASR's chunk (preserving any
                    # surrounding non-vocab words MedASR included)
                    fused.extend(m_words[m_start:m_end])
                    n_subs += 1
                    sub_log.append({
                        'type': 'sub',
                        'parakeet': p_span_text,
                        'medasr': ' '.join(m_words[m_start:m_end]),
                        'vocab_phrase': m_phrase,
                        'edit_dist': dist,
                        'max_len': max_len,
                    })
                else:
                    fused.extend(p_words[p_start:p_end])
            else:
                fused.extend(p_words[p_start:p_end])

        elif ctype == 'delete':
            # Word in Parakeet but not in MedASR — keep Parakeet
            fused.extend(p_words[p_start:p_end])

        elif ctype == 'insert':
            # Word in MedASR but not in Parakeet. Without an acoustic anchor on
            # Parakeet's side, inserting risks adding hallucination. Skip.
            # (Could revisit with a more sophisticated rule that uses surrounding
            # context to infer acoustic location.)
            pass

    return ' '.join(fused), {'subs': n_subs, 'sub_log': sub_log}


def wer(ref: str, hyp: str) -> tuple[float, int, int]:
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
        return None

    p_data = json.load(open(pp))
    m_data = json.load(open(mp))
    p_results = p_data.get('results', [])
    m_results = m_data.get('results', [])
    p_idx = {r['file_id']: r for r in p_results if 'file_id' in r}
    m_idx = {r['file_id']: r for r in m_results if 'file_id' in r}
    if not (set(p_idx.keys()) & set(m_idx.keys())):
        # fallback: ref-prefix match
        p_by_ref = {r.get('reference', '')[:120]: r for r in p_results}
        m_by_ref = {r.get('reference', '')[:120]: r for r in m_results}
        common = sorted(set(p_by_ref) & set(m_by_ref))
        p_idx = {k: p_by_ref[k] for k in common}
        m_idx = {k: m_by_ref[k] for k in common}
    common = sorted(set(p_idx.keys()) & set(m_idx.keys()))
    print(f'[{dataset}] {len(common)} files', flush=True)

    fh_per = open(OUT_DIR / f'{dataset}_per_file.tsv', 'w')
    fh_per.write('file_id\tparakeet_wer\tmedasr_wer\tfused_wer\tn_subs\n')
    fh_per.flush()

    fh_sub = open(OUT_DIR / f'{dataset}_substitutions.tsv', 'w')
    fh_sub.write('file_id\tparakeet_span\tmedasr_replacement\tvocab_phrase\tedit_dist\tmax_len\n')
    fh_sub.flush()

    tot_p_err = tot_m_err = tot_f_err = 0
    tot_w_p = tot_w_m = tot_w_f = 0
    tot_subs = 0
    better = worse = tied = 0

    for fid in common:
        p_row = p_idx[fid]; m_row = m_idx[fid]
        ref = p_row.get('reference', '')
        if not ref:
            continue
        p_hyp_raw = p_row.get('hypothesis', '')
        m_hyp_raw = m_row.get('hypothesis', '')

        fused_hyp, stats = fuse(normalize(p_hyp_raw), normalize(m_hyp_raw), vocab, max_n)

        p_wer, p_e, p_w = wer(ref, p_hyp_raw)
        m_wer, m_e, m_w = wer(ref, m_hyp_raw)
        f_wer, f_e, f_w = wer(ref, fused_hyp)

        fh_per.write(f'{fid}\t{p_wer*100:.2f}\t{m_wer*100:.2f}\t{f_wer*100:.2f}\t{stats["subs"]}\n')
        fh_per.flush()

        for s in stats['sub_log']:
            fh_sub.write(
                f'{fid}\t{s["parakeet"]}\t{s["medasr"]}\t{s["vocab_phrase"]}\t'
                f'{s["edit_dist"]}\t{s["max_len"]}\n'
            )
            fh_sub.flush()

        tot_p_err += p_e; tot_w_p += p_w
        tot_m_err += m_e; tot_w_m += m_w
        tot_f_err += f_e; tot_w_f += f_w
        tot_subs += stats['subs']

        if f_wer < p_wer - 1e-9: better += 1
        elif f_wer > p_wer + 1e-9: worse += 1
        else: tied += 1

    fh_per.close(); fh_sub.close()

    p_agg = 100 * tot_p_err / tot_w_p if tot_w_p else 0
    m_agg = 100 * tot_m_err / tot_w_m if tot_w_m else 0
    f_agg = 100 * tot_f_err / tot_w_f if tot_w_f else 0
    delta = f_agg - p_agg

    print(
        f'[{dataset}] DONE: parakeet {p_agg:.2f}% | medasr {m_agg:.2f}% | '
        f'fused {f_agg:.2f}% (Δ {delta:+.2f}pp). '
        f'better={better} worse={worse} tied={tied}, total subs={tot_subs}',
        flush=True,
    )

    return {
        'dataset': dataset,
        'n_files': len(common),
        'parakeet_wer': round(p_agg, 4),
        'medasr_wer': round(m_agg, 4),
        'fused_wer': round(f_agg, 4),
        'fused_delta_pp': round(delta, 4),
        'better': better, 'worse': worse, 'tied': tied,
        'n_subs': tot_subs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab', type=Path, default=VOCAB_TSV)
    p.add_argument('--categories', type=str, default=','.join(DEFAULT_CATEGORIES))
    p.add_argument('--datasets', type=str, default=','.join(DATASETS))
    args = p.parse_args()

    cats = args.categories.split(',')
    print(f'Loading vocab filtered to categories: {cats}', flush=True)
    vocab, max_n = load_targeted_vocab(args.vocab, cats)
    print(f'Loaded {len(vocab)} terms (categories={cats}), max phrase length {max_n}', flush=True)

    summary_path = OUT_DIR / 'summary.tsv'
    fh_summary = open(summary_path, 'w')
    fh_summary.write(
        f'# categories={",".join(cats)}\n'
        f'# vocab_size={len(vocab)}\n'
        f'dataset\tn_files\tparakeet_wer\tmedasr_wer\tfused_wer\tfused_delta_pp\tbetter\tworse\ttied\tn_subs\n'
    )
    fh_summary.flush()

    for ds in args.datasets.split(','):
        r = process_dataset(ds, vocab, max_n)
        if r is None:
            continue
        fh_summary.write(
            f'{r["dataset"]}\t{r["n_files"]}\t'
            f'{r["parakeet_wer"]}\t{r["medasr_wer"]}\t{r["fused_wer"]}\t{r["fused_delta_pp"]}\t'
            f'{r["better"]}\t{r["worse"]}\t{r["tied"]}\t{r["n_subs"]}\n'
        )
        fh_summary.flush()
    fh_summary.close()
    print(f'\nSummary: {summary_path}', flush=True)


if __name__ == '__main__':
    main()
