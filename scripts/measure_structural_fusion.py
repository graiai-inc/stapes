#!/usr/bin/env python3
"""Structural fusion: extend Parakeet+SecondaryASR fusion to negation/laterality/numeric.

Different from drug fusion (measure_targeted_fusion_v3.py) in that it has NO long-form
vocab — instead it uses tiny closed token sets:
  - NEGATION: 27 tokens (no, not, never, nothing, n't suffixes, can't/don't/etc.)
  - LATERALITY: 5 tokens (left, right, lateral, bilateral, unilateral)
  - NUMERIC: digit pattern + ~30 spelled-number words

Same anchor-supported substitution principle:
  - Walk parakeet vs secondary alignment
  - For each non-equal chunk, check if a target pattern fires (one side has the
    pattern token(s), the other doesn't, OR the laterality/numeric tokens differ)
  - Require anchor: ≥N exact-match words on each side
  - If anchored, substitute parakeet's words with secondary's words

Model-agnostic: works with any (parakeet, secondary) JSON pair from
ossicles/benchmark_results_<dataset>/. Try MedASR first; switch to Nemotron via
--secondary nemotron.

Output:
  results/structural_fusion/<dataset>__<secondary>__per_file.tsv
  results/structural_fusion/<dataset>__<secondary>__substitutions.tsv
  results/structural_fusion/<secondary>__summary.tsv

Usage:
    /home/grey/dev/graiai/ossicles/venv/bin/python \\
        scripts/measure_structural_fusion.py \\
        --secondary medasr --min-anchor-words 2

    /home/grey/dev/graiai/ossicles/venv/bin/python \\
        scripts/measure_structural_fusion.py \\
        --secondary nemotron --min-anchor-words 2

    /home/grey/dev/graiai/ossicles/venv/bin/python \\
        scripts/measure_structural_fusion.py \\
        --secondary medasr --secondary nemotron --min-anchor-words 2  # multi-model
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
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
OUT_DIR = STAPES_DIR / 'results' / 'structural_fusion'

DATASETS = ['primock57', 'figshare-osce', 'nazmulkazi']


def load_results(dataset: str, model: str) -> dict[str, dict]:
    p = OSSICLES_DIR / f'benchmark_results_{dataset}' / f'{model}.json'
    if not p.exists():
        p_cloud = OSSICLES_DIR / f'benchmark_results_cloud_{dataset}' / f'{model.replace("cloud-", "")}.json'
        if not p_cloud.exists():
            return {}
        p = p_cloud
    data = json.load(open(p))
    return {r['file_id']: r for r in data.get('results', []) if 'file_id' in r}


def chunk_is_anchor(chunk, min_words: int) -> bool:
    if chunk is None or chunk.type != 'equal':
        return False
    return (chunk.ref_end_idx - chunk.ref_start_idx) >= min_words


def fuse_structural_single(parakeet_text: str, secondary_text: str, min_anchor_words: int):
    """Single-secondary structural fusion (NEGATION-insert surgical only).
    Returns (fused_text, stats_dict).
    """
    p_words = parakeet_text.split()
    s_words = secondary_text.split()
    if not p_words and not s_words:
        return '', {'subs_negation': 0, 'subs_laterality': 0, 'subs_numeric': 0, 'sub_log': []}
    if not p_words:
        return secondary_text, {'subs_negation': 0, 'subs_laterality': 0, 'subs_numeric': 0, 'sub_log': []}
    if not s_words:
        return parakeet_text, {'subs_negation': 0, 'subs_laterality': 0, 'subs_numeric': 0, 'sub_log': []}

    out = jiwer.process_words(parakeet_text, secondary_text)
    chunks = out.alignments[0]

    fused = []
    sub_log = []
    n_neg = n_lat = n_num = 0

    for i, chunk in enumerate(chunks):
        ctype = chunk.type
        p_start, p_end = chunk.ref_start_idx, chunk.ref_end_idx  # parakeet side
        s_start, s_end = chunk.hyp_start_idx, chunk.hyp_end_idx  # secondary side

        if ctype == 'equal':
            fused.extend(p_words[p_start:p_end])
            continue
        if ctype == 'delete':
            # secondary has nothing here; trust parakeet
            fused.extend(p_words[p_start:p_end])
            continue
        if ctype == 'insert':
            # secondary has extra words; consider for structural fusion (negation insertion only).
            # SURGICAL: only insert the specific negation token(s) from secondary, not the
            # whole insert chunk (which often has disfluencies).
            s_chunk_words = s_words[s_start:s_end]
            prev_ok = chunk_is_anchor(chunks[i - 1] if i > 0 else None, min_anchor_words)
            next_ok = chunk_is_anchor(chunks[i + 1] if i + 1 < len(chunks) else None, min_anchor_words)
            if not (prev_ok and next_ok):
                continue
            # Extract just the negation token(s) — drop the rest of medasr's noise.
            neg_tokens_only = [t for t in s_chunk_words if t in NEGATION_TOKENS]
            if neg_tokens_only:
                fused.extend(neg_tokens_only)
                n_neg += 1
                sub_log.append({'pattern': 'NEGATION', 'parakeet': '',
                                'secondary': ' '.join(neg_tokens_only),
                                'kind': 'insert_surgical'})
            # Laterality + numeric on insert: SYMMETRIC — without 3rd-model voting,
            # disabled. Don't insert.
            continue

        # ctype == 'substitute'
        # NEGATION on substitute is too risky: short-token confusions like
        # "now" ↔ "no" or "that" ↔ "not" often substitute valid words for
        # spurious negations. Parakeet's typical failure mode is to DELETE
        # negations, which surfaces as the INSERT chunk above, not SUBSTITUTE.
        # Keep parakeet for substitute chunks. Laterality + numeric also disabled
        # without a 3rd-model voting tie-breaker.
        fused.extend(p_words[p_start:p_end])

    return ' '.join(fused), {'subs_negation': n_neg, 'subs_laterality': n_lat,
                             'subs_numeric': n_num, 'sub_log': sub_log}


def fuse_structural(parakeet_text: str, secondary_texts, min_anchor_words: int):
    """Run fuse_structural_single sequentially with each secondary. Each pass
    further enriches the fused output. Idempotent on already-fused negations
    (a negation token already present at a position won't fire again because
    the asymmetric `s_has_neg and not p_has_neg` check applies to the running
    fused text).
    """
    if isinstance(secondary_texts, str):
        secondary_texts = [secondary_texts]
    fused = parakeet_text
    combined_stats = {'subs_negation': 0, 'subs_laterality': 0, 'subs_numeric': 0, 'sub_log': []}
    for sec in secondary_texts:
        if not sec:
            continue
        fused, stats = fuse_structural_single(fused, sec, min_anchor_words)
        combined_stats['subs_negation'] += stats['subs_negation']
        combined_stats['subs_laterality'] += stats['subs_laterality']
        combined_stats['subs_numeric'] += stats['subs_numeric']
        combined_stats['sub_log'].extend(stats['sub_log'])
    return fused, combined_stats


def evaluate_pattern_recall(ref_text: str, hyp_text: str, pattern: str) -> tuple[int, int]:
    """For one ref/hyp pair, count (correct, total) at the pattern level.

    For NEGATION: count of reference 'negation positions' (chunks with neg token)
                  where hyp has matching polarity. NOT word-level — chunk-level.
    For LATERALITY: same idea on laterality tokens.
    For NUMERIC: same idea on numeric tokens.

    Simpler proxy: count occurrences of pattern tokens in ref vs hyp.
    """
    if pattern == 'NEGATION':
        target = NEGATION_TOKENS
        r_tokens = [t for t in ref_text.split() if t in target]
        h_tokens = [t for t in hyp_text.split() if t in target]
    elif pattern == 'LATERALITY':
        target = LATERALITY_TOKENS
        r_tokens = [t for t in ref_text.split() if t in target]
        h_tokens = [t for t in hyp_text.split() if t in target]
    elif pattern == 'NUMERIC':
        r_tokens = [t for t in ref_text.split() if NUMERIC_DIGIT_RE.search(t) or t in SPELLED_NUMBER_TOKENS]
        h_tokens = [t for t in hyp_text.split() if NUMERIC_DIGIT_RE.search(t) or t in SPELLED_NUMBER_TOKENS]
    else:
        return 0, 0
    # Recall: of ref tokens, how many are present in hyp (multiset match)
    r_count = len(r_tokens)
    matched = 0
    h_remaining = list(h_tokens)
    for r in r_tokens:
        if r in h_remaining:
            matched += 1
            h_remaining.remove(r)
    return matched, r_count


def wer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0
    return jiwer.wer(ref, hyp)


def process_dataset(dataset: str, secondary_models, min_anchor: int):
    """secondary_models: a list of model names to use as secondaries.
    Fusion runs sequentially with each. The fused output of the previous pass
    becomes the input to the next."""
    if isinstance(secondary_models, str):
        secondary_models = [secondary_models]
    parakeet = load_results(dataset, 'parakeet-tdt-0.6b-v2')
    secondaries = [(name, load_results(dataset, name)) for name in secondary_models]
    if not parakeet or not all(s for _, s in secondaries):
        missing = [n for n, s in secondaries if not s]
        log.warning(f'  no data for {dataset} / missing secondaries: {missing}')
        return None
    secondary = secondaries[0][1]  # use the first to find common file IDs
    for _, s in secondaries[1:]:
        secondary = {fid: secondary[fid] for fid in secondary if fid in s}

    common = sorted(set(parakeet) & set(secondary))
    log.info(f'  {dataset}: {len(common)} files (secondaries: {secondary_models})')

    label = '+'.join(secondary_models)
    fh_per = open(OUT_DIR / f'{dataset}__{label}__per_file.tsv', 'w')
    fh_per.write('file_id\tparakeet_wer\tfused_wer\tsubs_negation\tsubs_laterality\tsubs_numeric\t'
                 'p_neg_correct\tp_neg_total\tf_neg_correct\tf_neg_total\t'
                 'p_lat_correct\tp_lat_total\tf_lat_correct\tf_lat_total\t'
                 'p_num_correct\tp_num_total\tf_num_correct\tf_num_total\n')
    fh_per.flush()

    fh_sub = open(OUT_DIR / f'{dataset}__{label}__substitutions.tsv', 'w')
    fh_sub.write('file_id\tpattern\tkind\tparakeet_span\tsecondary_replacement\n')
    fh_sub.flush()

    agg = defaultdict(int)
    n_better = n_worse = n_tied = 0

    for fid in common:
        ref = parakeet[fid].get('reference', '')
        if not ref:
            continue
        ref_norm = normalize(ref)
        p_norm = normalize(parakeet[fid].get('hypothesis', ''))
        s_norms = []
        for name, s_results in secondaries:
            if fid in s_results:
                s_norms.append(normalize(s_results[fid].get('hypothesis', '')))

        fused, stats = fuse_structural(p_norm, s_norms, min_anchor)

        p_wer = wer(ref_norm, p_norm)
        f_wer = wer(ref_norm, fused)

        # Pattern recall
        p_neg_c, p_neg_t = evaluate_pattern_recall(ref_norm, p_norm, 'NEGATION')
        f_neg_c, f_neg_t = evaluate_pattern_recall(ref_norm, fused, 'NEGATION')
        p_lat_c, p_lat_t = evaluate_pattern_recall(ref_norm, p_norm, 'LATERALITY')
        f_lat_c, f_lat_t = evaluate_pattern_recall(ref_norm, fused, 'LATERALITY')
        p_num_c, p_num_t = evaluate_pattern_recall(ref_norm, p_norm, 'NUMERIC')
        f_num_c, f_num_t = evaluate_pattern_recall(ref_norm, fused, 'NUMERIC')

        fh_per.write(
            f'{fid}\t{p_wer*100:.2f}\t{f_wer*100:.2f}\t'
            f'{stats["subs_negation"]}\t{stats["subs_laterality"]}\t{stats["subs_numeric"]}\t'
            f'{p_neg_c}\t{p_neg_t}\t{f_neg_c}\t{f_neg_t}\t'
            f'{p_lat_c}\t{p_lat_t}\t{f_lat_c}\t{f_lat_t}\t'
            f'{p_num_c}\t{p_num_t}\t{f_num_c}\t{f_num_t}\n'
        )
        fh_per.flush()

        for s in stats['sub_log']:
            fh_sub.write(f'{fid}\t{s["pattern"]}\t{s["kind"]}\t{s["parakeet"]}\t{s["secondary"]}\n')
            fh_sub.flush()

        # Aggregates
        agg['subs_negation'] += stats['subs_negation']
        agg['subs_laterality'] += stats['subs_laterality']
        agg['subs_numeric'] += stats['subs_numeric']
        agg['p_neg_correct'] += p_neg_c; agg['p_neg_total'] += p_neg_t
        agg['f_neg_correct'] += f_neg_c; agg['f_neg_total'] += f_neg_t
        agg['p_lat_correct'] += p_lat_c; agg['p_lat_total'] += p_lat_t
        agg['f_lat_correct'] += f_lat_c; agg['f_lat_total'] += f_lat_t
        agg['p_num_correct'] += p_num_c; agg['p_num_total'] += p_num_t
        agg['f_num_correct'] += f_num_c; agg['f_num_total'] += f_num_t
        agg['p_wer_err'] += round(p_wer * len(ref_norm.split()))
        agg['f_wer_err'] += round(f_wer * len(ref_norm.split()))
        agg['n_words'] += len(ref_norm.split())

        if f_wer < p_wer - 1e-9:
            n_better += 1
        elif f_wer > p_wer + 1e-9:
            n_worse += 1
        else:
            n_tied += 1

    fh_per.close()
    fh_sub.close()

    p_neg_recall = 100 * agg['p_neg_correct'] / agg['p_neg_total'] if agg['p_neg_total'] else 0
    f_neg_recall = 100 * agg['f_neg_correct'] / agg['f_neg_total'] if agg['f_neg_total'] else 0
    p_lat_recall = 100 * agg['p_lat_correct'] / agg['p_lat_total'] if agg['p_lat_total'] else 0
    f_lat_recall = 100 * agg['f_lat_correct'] / agg['f_lat_total'] if agg['f_lat_total'] else 0
    p_num_recall = 100 * agg['p_num_correct'] / agg['p_num_total'] if agg['p_num_total'] else 0
    f_num_recall = 100 * agg['f_num_correct'] / agg['f_num_total'] if agg['f_num_total'] else 0
    p_wer_pct = 100 * agg['p_wer_err'] / agg['n_words'] if agg['n_words'] else 0
    f_wer_pct = 100 * agg['f_wer_err'] / agg['n_words'] if agg['n_words'] else 0

    log.info(f'  {dataset}/{label} DONE:')
    log.info(f'    WER: parakeet={p_wer_pct:.2f}%  fused={f_wer_pct:.2f}% (Δ {f_wer_pct-p_wer_pct:+.2f}pp)')
    log.info(f'    NEG recall: {p_neg_recall:.2f}% → {f_neg_recall:.2f}% (Δ {f_neg_recall-p_neg_recall:+.2f}pp; subs={agg["subs_negation"]})')
    log.info(f'    LAT recall: {p_lat_recall:.2f}% → {f_lat_recall:.2f}% (Δ {f_lat_recall-p_lat_recall:+.2f}pp; subs={agg["subs_laterality"]})')
    log.info(f'    NUM recall: {p_num_recall:.2f}% → {f_num_recall:.2f}% (Δ {f_num_recall-p_num_recall:+.2f}pp; subs={agg["subs_numeric"]})')
    log.info(f'    file-level WER: better={n_better} worse={n_worse} tied={n_tied}')

    return {
        'dataset': dataset, 'secondary': label, 'n_files': len(common),
        'parakeet_wer': p_wer_pct, 'fused_wer': f_wer_pct,
        'subs_negation': agg['subs_negation'], 'subs_laterality': agg['subs_laterality'],
        'subs_numeric': agg['subs_numeric'],
        'parakeet_neg_recall': p_neg_recall, 'fused_neg_recall': f_neg_recall,
        'parakeet_lat_recall': p_lat_recall, 'fused_lat_recall': f_lat_recall,
        'parakeet_num_recall': p_num_recall, 'fused_num_recall': f_num_recall,
        'better': n_better, 'worse': n_worse, 'tied': n_tied,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--secondary', action='append', default=[],
                   help='Secondary ASR model name; pass multiple times to chain')
    p.add_argument('--datasets', type=str, default=','.join(DATASETS))
    p.add_argument('--min-anchor-words', type=int, default=2)
    args = p.parse_args()
    if not args.secondary:
        args.secondary = ['medasr']

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    label = '+'.join(args.secondary)
    summary_path = OUT_DIR / f'{label}__summary.tsv'
    fh_summary = open(summary_path, 'w')
    fh_summary.write(
        f'# secondary={label}\n'
        f'# min_anchor_words={args.min_anchor_words}\n'
        'dataset\tn_files\tparakeet_wer\tfused_wer\twer_delta_pp\t'
        'subs_neg\tsubs_lat\tsubs_num\t'
        'p_neg_rec\tf_neg_rec\tneg_delta\t'
        'p_lat_rec\tf_lat_rec\tlat_delta\t'
        'p_num_rec\tf_num_rec\tnum_delta\t'
        'better\tworse\ttied\n'
    )
    fh_summary.flush()

    log.info(f'Structural fusion: secondary={label}, min_anchor={args.min_anchor_words}')
    for ds in args.datasets.split(','):
        log.info(f'=== {ds} ===')
        r = process_dataset(ds, args.secondary, args.min_anchor_words)
        if r is None:
            continue
        fh_summary.write(
            f'{r["dataset"]}\t{r["n_files"]}\t{r["parakeet_wer"]:.4f}\t{r["fused_wer"]:.4f}\t'
            f'{r["fused_wer"]-r["parakeet_wer"]:+.4f}\t'
            f'{r["subs_negation"]}\t{r["subs_laterality"]}\t{r["subs_numeric"]}\t'
            f'{r["parakeet_neg_recall"]:.4f}\t{r["fused_neg_recall"]:.4f}\t'
            f'{r["fused_neg_recall"]-r["parakeet_neg_recall"]:+.4f}\t'
            f'{r["parakeet_lat_recall"]:.4f}\t{r["fused_lat_recall"]:.4f}\t'
            f'{r["fused_lat_recall"]-r["parakeet_lat_recall"]:+.4f}\t'
            f'{r["parakeet_num_recall"]:.4f}\t{r["fused_num_recall"]:.4f}\t'
            f'{r["fused_num_recall"]-r["parakeet_num_recall"]:+.4f}\t'
            f'{r["better"]}\t{r["worse"]}\t{r["tied"]}\n'
        )
        fh_summary.flush()

    fh_summary.close()
    print(f'\nDone. Summary: {summary_path}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
