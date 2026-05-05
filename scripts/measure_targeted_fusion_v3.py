#!/usr/bin/env python3
"""V3 targeted fusion: contextual anchor in addition to acoustic similarity.

Two paths to fire a substitution. Either:

  A) ACOUSTIC: Parakeet's joined span is acoustically similar to MedASR's vocab
     phrase (joined, no spaces, edit distance / max_len < 0.3). Catches cases
     like "ram a pril" → "ramipril" — Parakeet heard the right phonemes but
     couldn't map to a known drug.

  B) CONTEXTUAL ANCHOR: even if acoustic similarity fails, substitute if both
     the chunk IMMEDIATELY BEFORE and IMMEDIATELY AFTER the substitute chunk
     are 'equal' (both models agree on those words exactly), AND each anchor
     has at least MIN_ANCHOR_WORDS exact matches. Catches "rhymer po f" →
     "ramipril" — the words inside the disagreement don't sound alike but
     surrounding context proves they're at the same audio position.

  C) NEITHER → don't substitute (rejects MedASR hallucinations and isolated noise).

Runs against existing Parakeet+MedASR outputs, no new ASR runs needed.

Output:
  results/targeted_fusion_v3/<dataset>_per_file.tsv
  results/targeted_fusion_v3/<dataset>_substitutions.tsv  (path: A or B noted)
  results/targeted_fusion_v3/summary.tsv

Usage:
    /home/grey/dev/graiai/ossicles/venv/bin/python \\
        scripts/measure_targeted_fusion_v3.py \\
        --categories DRUG,PROCEDURE,LAB_RESULT --min-anchor-words 2
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
OUT_DIR = STAPES_DIR / 'results' / 'targeted_fusion_v3'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ['primock57', 'figshare-osce', 'nazmulkazi']
DEFAULT_CATEGORIES = ['DRUG', 'PROCEDURE', 'LAB_RESULT']
DEFAULT_MIN_ANCHOR = 2          # DRUG-tuned; condition/symptom fusion will likely
                                # want different values — re-sweep per category.
DEFAULT_MAX_ANCHOR_EDIT_RATIO = 0.5  # Same caveat. 0.5 blocks asthma→aspirin
                                     # family of misfires for drug fusion.


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s'\-]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


COMMON_ENGLISH_NOISE = {
    'others', 'others\'', 'other', 'comfort', 'comforts', 'renewal', 'renewals',
    'active', 'commit', 'rise', 'rises', 'date', 'dates', 'pat', 'snow', 'fat',
    'cap', 'caps', 'tube', 'tubes', 'face', 'faces', 'hand', 'hands', 'back',
    'lower', 'upper', 'else', 'couple', 'definitely', 'symptoms', 'medications',
    'cough', 'coughs', 'cold', 'colds', 'hot', 'happy', 'sad', 'fun',
    'history', 'name', 'normal', 'medical', 'clinic', 'home', 'work',
    'family', 'group', 'friend', 'mother', 'father', 'parent', 'child',
    'children', 'sister', 'brother', 'wife', 'husband',
}


def load_targeted_vocab(
    vocab_tsv: Path, categories: list[str], min_chars: int = 5,
) -> tuple[set, int]:
    """Load vocab subset filtered to categories + minimum character length +
    excluding common-English noise that pollutes UMLS matches."""
    vocab = set()
    max_n = 1
    cat_set = set(categories)
    n_skipped_short = 0
    n_skipped_noise = 0
    with open(vocab_tsv) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            term = row.get('term', '').strip().lower()
            cat = row.get('category', '').strip()
            if not term or cat not in cat_set:
                continue
            # Filter: minimum character length (excludes "ther", "ast", "fat" etc)
            if len(term.replace(' ', '').replace('-', '')) < min_chars:
                n_skipped_short += 1
                continue
            # Filter: common English words that match UMLS coincidentally
            if term in COMMON_ENGLISH_NOISE:
                n_skipped_noise += 1
                continue
            vocab.add(term)
            n = term.count(' ') + 1
            if n > max_n:
                max_n = n
    print(f'  skipped {n_skipped_short} short (<{min_chars} chars) + {n_skipped_noise} noise', flush=True)
    return vocab, max_n


def find_longest_vocab_phrase(words, start, end, vocab, max_n):
    best = None
    for s in range(start, end):
        for n in range(min(max_n, end - s), 0, -1):
            phrase = ' '.join(words[s:s + n])
            if phrase in vocab:
                if best is None or n > best[0]:
                    best = (n, s, s + n, phrase)
                break
    if best is None:
        return None
    return best[1], best[2], best[3]


def levenshtein(a: str, b: str) -> int:
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
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def acoustic_similar(parakeet_span: str, medasr_phrase: str):
    p_join = parakeet_span.replace(' ', '').replace('-', '').lower()
    m_join = medasr_phrase.replace(' ', '').replace('-', '').lower()
    if not p_join or not m_join:
        return False, 0, 0
    dist = levenshtein(p_join, m_join)
    max_len = max(len(p_join), len(m_join))
    ratio = dist / max_len if max_len else 1.0
    is_similar = (dist <= 1) or (ratio < 0.3 and dist <= 4)
    return is_similar, dist, max_len


def chunk_is_equal_with_min_words(chunk, min_words: int) -> bool:
    if chunk is None:
        return False
    if chunk.type != 'equal':
        return False
    n_words = chunk.ref_end_idx - chunk.ref_start_idx
    return n_words >= min_words


def fuse(parakeet_text, medasr_text, vocab, max_n, min_anchor_words,
         max_anchor_edit_ratio=None):
    p_words = parakeet_text.split()
    m_words = medasr_text.split()

    if not p_words and not m_words:
        return '', {'subs_acoustic': 0, 'subs_anchor': 0, 'sub_log': []}
    if not p_words:
        return medasr_text, {'subs_acoustic': 0, 'subs_anchor': 0, 'sub_log': []}
    if not m_words:
        return parakeet_text, {'subs_acoustic': 0, 'subs_anchor': 0, 'sub_log': []}

    out = jiwer.process_words(parakeet_text, medasr_text)
    chunks = out.alignments[0]

    fused = []
    sub_log = []
    n_acoustic = 0
    n_anchor = 0

    for i, chunk in enumerate(chunks):
        ctype = chunk.type
        p_start, p_end = chunk.ref_start_idx, chunk.ref_end_idx
        m_start, m_end = chunk.hyp_start_idx, chunk.hyp_end_idx

        if ctype == 'equal':
            fused.extend(p_words[p_start:p_end])
            continue

        if ctype == 'substitute':
            vocab_match = find_longest_vocab_phrase(m_words, m_start, m_end, vocab, max_n)
            if vocab_match is None:
                fused.extend(p_words[p_start:p_end])
                continue
            _, _, m_phrase = vocab_match
            p_span_text = ' '.join(p_words[p_start:p_end])

            # Path A: acoustic similarity
            similar, dist, max_len = acoustic_similar(p_span_text, m_phrase)

            # Path B: contextual anchor
            prev_chunk = chunks[i - 1] if i > 0 else None
            next_chunk = chunks[i + 1] if i + 1 < len(chunks) else None
            prev_anchor = chunk_is_equal_with_min_words(prev_chunk, min_anchor_words)
            next_anchor = chunk_is_equal_with_min_words(next_chunk, min_anchor_words)
            has_anchor = prev_anchor and next_anchor

            if similar:
                fused.extend(m_words[m_start:m_end])
                n_acoustic += 1
                sub_log.append({
                    'path': 'A:acoustic',
                    'parakeet': p_span_text,
                    'medasr': ' '.join(m_words[m_start:m_end]),
                    'vocab_phrase': m_phrase,
                    'edit_dist': dist,
                    'max_len': max_len,
                })
            elif has_anchor and (
                max_anchor_edit_ratio is None
                or (max_len > 0 and dist / max_len <= max_anchor_edit_ratio)
            ):
                # Anchor path: optionally require partial acoustic similarity
                # to block clinically-dangerous misfires like 'asthma → aspirin'
                # or patient name 'madison → medicine'.
                fused.extend(m_words[m_start:m_end])
                n_anchor += 1
                sub_log.append({
                    'path': 'B:anchor',
                    'parakeet': p_span_text,
                    'medasr': ' '.join(m_words[m_start:m_end]),
                    'vocab_phrase': m_phrase,
                    'edit_dist': dist,
                    'max_len': max_len,
                })
            else:
                fused.extend(p_words[p_start:p_end])

        elif ctype == 'delete':
            fused.extend(p_words[p_start:p_end])

        elif ctype == 'insert':
            # Insertion path with anchor: if both surrounding chunks are equal AND
            # MedASR's inserted span has a vocab match, insert.
            vocab_match = find_longest_vocab_phrase(m_words, m_start, m_end, vocab, max_n)
            if vocab_match is None:
                continue
            _, _, m_phrase = vocab_match
            prev_chunk = chunks[i - 1] if i > 0 else None
            next_chunk = chunks[i + 1] if i + 1 < len(chunks) else None
            prev_anchor = chunk_is_equal_with_min_words(prev_chunk, min_anchor_words)
            next_anchor = chunk_is_equal_with_min_words(next_chunk, min_anchor_words)
            if prev_anchor and next_anchor:
                fused.extend(m_words[m_start:m_end])
                n_anchor += 1
                sub_log.append({
                    'path': 'B:anchor-insert',
                    'parakeet': '',
                    'medasr': ' '.join(m_words[m_start:m_end]),
                    'vocab_phrase': m_phrase,
                    'edit_dist': 0,
                    'max_len': 0,
                })

    return ' '.join(fused), {
        'subs_acoustic': n_acoustic,
        'subs_anchor': n_anchor,
        'sub_log': sub_log,
    }


def wer(ref, hyp):
    ref_n = normalize(ref)
    hyp_n = normalize(hyp)
    if not ref_n:
        return 0.0, 0, 0
    n_words = len(ref_n.split())
    w = jiwer.wer(ref_n, hyp_n)
    return w, round(w * n_words), n_words


def process_dataset(dataset, vocab, max_n, min_anchor, max_anchor_edit_ratio=None):
    pdir = OSSICLES_DIR / f'benchmark_results_{dataset}'
    pp = pdir / 'parakeet-tdt-0.6b-v2.json'
    mp = pdir / 'medasr.json'
    if not pp.exists() or not mp.exists():
        return None
    p_data = json.load(open(pp))
    m_data = json.load(open(mp))
    p_idx = {r['file_id']: r for r in p_data.get('results', []) if 'file_id' in r}
    m_idx = {r['file_id']: r for r in m_data.get('results', []) if 'file_id' in r}
    common = sorted(set(p_idx) & set(m_idx))
    print(f'[{dataset}] {len(common)} files', flush=True)

    fh_per = open(OUT_DIR / f'{dataset}_per_file.tsv', 'w')
    fh_per.write('file_id\tparakeet_wer\tmedasr_wer\tfused_wer\tsubs_acoustic\tsubs_anchor\n')
    fh_per.flush()

    fh_sub = open(OUT_DIR / f'{dataset}_substitutions.tsv', 'w')
    fh_sub.write('file_id\tpath\tparakeet_span\tmedasr_replacement\tvocab_phrase\tedit_dist\tmax_len\n')
    fh_sub.flush()

    tot_p_err = tot_m_err = tot_f_err = 0
    tot_w_p = tot_w_m = tot_w_f = 0
    tot_acoustic = tot_anchor = 0
    better = worse = tied = 0

    for fid in common:
        p_row = p_idx[fid]; m_row = m_idx[fid]
        ref = p_row.get('reference', '')
        if not ref:
            continue
        p_hyp_raw = p_row.get('hypothesis', '')
        m_hyp_raw = m_row.get('hypothesis', '')

        fused_hyp, stats = fuse(
            normalize(p_hyp_raw), normalize(m_hyp_raw), vocab, max_n, min_anchor,
            max_anchor_edit_ratio=max_anchor_edit_ratio,
        )

        p_wer, p_e, p_w = wer(ref, p_hyp_raw)
        m_wer, m_e, m_w = wer(ref, m_hyp_raw)
        f_wer, f_e, f_w = wer(ref, fused_hyp)

        fh_per.write(
            f'{fid}\t{p_wer*100:.2f}\t{m_wer*100:.2f}\t{f_wer*100:.2f}\t'
            f'{stats["subs_acoustic"]}\t{stats["subs_anchor"]}\n'
        )
        fh_per.flush()

        for s in stats['sub_log']:
            fh_sub.write(
                f'{fid}\t{s["path"]}\t{s["parakeet"]}\t{s["medasr"]}\t'
                f'{s["vocab_phrase"]}\t{s["edit_dist"]}\t{s["max_len"]}\n'
            )
            fh_sub.flush()

        tot_p_err += p_e; tot_w_p += p_w
        tot_m_err += m_e; tot_w_m += m_w
        tot_f_err += f_e; tot_w_f += f_w
        tot_acoustic += stats['subs_acoustic']
        tot_anchor += stats['subs_anchor']

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
        f'fused {f_agg:.2f}% (Δ {delta:+.2f}pp). better={better} worse={worse} tied={tied}, '
        f'acoustic_subs={tot_acoustic} anchor_subs={tot_anchor}',
        flush=True,
    )
    return {
        'dataset': dataset, 'n_files': len(common),
        'parakeet_wer': round(p_agg, 4), 'medasr_wer': round(m_agg, 4),
        'fused_wer': round(f_agg, 4), 'fused_delta_pp': round(delta, 4),
        'better': better, 'worse': worse, 'tied': tied,
        'subs_acoustic': tot_acoustic, 'subs_anchor': tot_anchor,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab', type=Path, default=VOCAB_TSV)
    p.add_argument('--categories', type=str, default=','.join(DEFAULT_CATEGORIES))
    p.add_argument('--datasets', type=str, default=','.join(DATASETS))
    p.add_argument('--min-anchor-words', type=int, default=DEFAULT_MIN_ANCHOR)
    p.add_argument(
        '--max-anchor-edit-ratio', type=float, default=DEFAULT_MAX_ANCHOR_EDIT_RATIO,
        help='If set (e.g. 0.5), the anchor path also requires '
             'edit_dist/max_len <= this ratio. Blocks clinically dangerous '
             'misfires like asthma→aspirin where anchors agree but words sound '
             'nothing alike. Pass an explicit None (or large value) to disable.',
    )
    args = p.parse_args()

    cats = args.categories.split(',')
    print(f'Loading vocab filtered to: {cats}', flush=True)
    vocab, max_n = load_targeted_vocab(args.vocab, cats)
    print(f'Loaded {len(vocab)} terms, max phrase length {max_n}, min_anchor={args.min_anchor_words}', flush=True)

    summary_path = OUT_DIR / 'summary.tsv'
    fh_summary = open(summary_path, 'w')
    fh_summary.write(
        f'# categories={",".join(cats)}\n'
        f'# vocab_size={len(vocab)}\n'
        f'# min_anchor_words={args.min_anchor_words}\n'
        f'# max_anchor_edit_ratio={args.max_anchor_edit_ratio}\n'
        f'dataset\tn_files\tparakeet_wer\tmedasr_wer\tfused_wer\tdelta_pp\tbetter\tworse\ttied\tsubs_acoustic\tsubs_anchor\n'
    )
    fh_summary.flush()

    for ds in args.datasets.split(','):
        r = process_dataset(
            ds, vocab, max_n, args.min_anchor_words,
            max_anchor_edit_ratio=args.max_anchor_edit_ratio,
        )
        if r is None:
            continue
        fh_summary.write(
            f'{r["dataset"]}\t{r["n_files"]}\t'
            f'{r["parakeet_wer"]}\t{r["medasr_wer"]}\t{r["fused_wer"]}\t{r["fused_delta_pp"]}\t'
            f'{r["better"]}\t{r["worse"]}\t{r["tied"]}\t'
            f'{r["subs_acoustic"]}\t{r["subs_anchor"]}\n'
        )
        fh_summary.flush()
    fh_summary.close()
    print(f'\nSummary: {summary_path}', flush=True)


if __name__ == '__main__':
    main()
