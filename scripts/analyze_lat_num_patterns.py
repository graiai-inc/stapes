#!/usr/bin/env python3
"""Look for patterns in WHICH model recovers Parakeet's LATERALITY / NUMERIC errors.

For each parakeet error position with LATERALITY or NUMERIC pattern:
  - Record the specific reference token(s) (e.g., "left", "5 mg", "fifteen")
  - Record which models in our benchmark transcripts got it right
  - Group by:
    1. ref token form (which specific word)
    2. left vs right (for LATERALITY)
    3. digit-form vs spelled-form (for NUMERIC)
    4. single-token vs multi-token disagreements
    5. position in file (early/middle/late)

If clear patterns emerge ("model X always gets digit forms right; model Y gets
spelled forms right"), we can build pattern-aware fusion. If recoveries look
randomly distributed, we need 3-model voting and there's no shortcut.
"""
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jiwer

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from check_all_models_on_high_impact import (  # noqa: E402
    get_words_at_ref_position,
    list_model_jsons,
    load_results,
    model_matches,
)
from measure_targeted_fusion_v3 import normalize  # noqa: E402
from mine_actual_error_patterns import (  # noqa: E402
    LATERALITY_TOKENS,
    NUMERIC_DIGIT_RE,
    SPELLED_NUMBER_TOKENS,
    detect_laterality,
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
OUT_DIR = STAPES_DIR / 'results' / 'lat_num_patterns'

DATASETS = ['primock57', 'figshare-osce', 'nazmulkazi']

# Models to track in detail (top performers for each pattern from prior cross-model check)
TRACK_MODELS = [
    'medasr', 'medasr-int8', 'nemotron', 'paraformer-en', 'zipformer-zh-en',
    'sensevoice', 'sensevoice-no-itn', 'whisper-distil-v3.5', 'whisper-base-en',
    'cloud-google', 'cloud-azure', 'cloud-deepgram', 'cloud-assemblyai',
]


def feature_lat(ref_chunk):
    """Feature dict for a LATERALITY error."""
    tokens = [t for t in ref_chunk if t in LATERALITY_TOKENS]
    return {
        'tokens': ' '.join(tokens) if tokens else '<none>',
        'has_left': 'left' in tokens,
        'has_right': 'right' in tokens,
        'has_lateral': any(t in {'lateral', 'bilateral', 'unilateral'} for t in tokens),
        'n_lat_tokens': len(tokens),
        'chunk_len': len(ref_chunk),
    }


def feature_num(ref_chunk):
    """Feature dict for a NUMERIC error."""
    digit_tokens = [t for t in ref_chunk if NUMERIC_DIGIT_RE.search(t)]
    spelled_tokens = [t for t in ref_chunk if t in SPELLED_NUMBER_TOKENS]
    return {
        'tokens': ' '.join(digit_tokens + spelled_tokens) if (digit_tokens or spelled_tokens) else '<none>',
        'has_digit': bool(digit_tokens),
        'has_spelled': bool(spelled_tokens),
        'mixed': bool(digit_tokens and spelled_tokens),
        'n_digit_tokens': len(digit_tokens),
        'n_spelled_tokens': len(spelled_tokens),
        'chunk_len': len(ref_chunk),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-position log: each row is one parakeet error with the wins/losses across models
    fh_lat = open(OUT_DIR / 'laterality_per_position.tsv', 'w')
    lat_header = ['dataset', 'file_id', 'ref_words', 'parakeet_words',
                  'tokens', 'has_left', 'has_right', 'has_lateral',
                  'n_lat_tokens', 'chunk_len']
    lat_header += [f'win_{m}' for m in TRACK_MODELS]
    fh_lat.write('\t'.join(lat_header) + '\n')
    fh_lat.flush()

    fh_num = open(OUT_DIR / 'numeric_per_position.tsv', 'w')
    num_header = ['dataset', 'file_id', 'ref_words', 'parakeet_words',
                  'tokens', 'has_digit', 'has_spelled', 'mixed',
                  'n_digit_tokens', 'n_spelled_tokens', 'chunk_len']
    num_header += [f'win_{m}' for m in TRACK_MODELS]
    fh_num.write('\t'.join(num_header) + '\n')
    fh_num.flush()

    for dataset in DATASETS:
        log.info(f'=== {dataset} ===')
        models = list_model_jsons(dataset)
        if 'parakeet-tdt-0.6b-v2' not in models:
            continue
        all_results = {name: load_results(path) for name, path in models.items()}
        parakeet_results = all_results['parakeet-tdt-0.6b-v2']

        for fid in sorted(parakeet_results.keys()):
            ref = parakeet_results[fid].get('reference', '')
            if not ref:
                continue
            ref_norm = normalize(ref)
            p_norm = normalize(parakeet_results[fid].get('hypothesis', ''))
            ref_words = ref_norm.split()
            p_words = p_norm.split()
            if not ref_words:
                continue
            p_align = jiwer.process_words(ref_norm, p_norm)

            for chunk in p_align.alignments[0]:
                if chunk.type not in ('substitute', 'delete'):
                    continue
                rs, re_ = chunk.ref_start_idx, chunk.ref_end_idx
                hs, he = chunk.hyp_start_idx, chunk.hyp_end_idx
                ref_chunk = ref_words[rs:re_]
                p_chunk = p_words[hs:he] if chunk.type == 'substitute' else []

                fire_lat = detect_laterality(ref_chunk, p_chunk)
                fire_num = detect_numeric_mismatch(ref_chunk, p_chunk)
                if not (fire_lat or fire_num):
                    continue

                # Get per-model win flags
                model_wins = {}
                for m in TRACK_MODELS:
                    if m not in all_results or fid not in all_results[m]:
                        model_wins[m] = ''  # not tracked
                        continue
                    m_norm = normalize(all_results[m][fid].get('hypothesis', ''))
                    m_words = m_norm.split()
                    if not m_words:
                        model_wins[m] = ''
                        continue
                    m_align = jiwer.process_words(ref_norm, m_norm)
                    m_chunk = get_words_at_ref_position(ref_words, m_words, rs, re_, m_align.alignments[0])
                    if fire_lat:
                        correct = model_matches('LATERALITY', ref_chunk, m_chunk)
                    else:
                        correct = model_matches('NUMERIC', ref_chunk, m_chunk)
                    if correct is None:
                        model_wins[m] = '?'
                    else:
                        model_wins[m] = '1' if correct else '0'

                if fire_lat:
                    feat = feature_lat(ref_chunk)
                    row = [dataset, fid, ' '.join(ref_chunk), ' '.join(p_chunk),
                           feat['tokens'], int(feat['has_left']), int(feat['has_right']),
                           int(feat['has_lateral']), feat['n_lat_tokens'], feat['chunk_len']]
                    row += [model_wins[m] for m in TRACK_MODELS]
                    fh_lat.write('\t'.join(str(x) for x in row) + '\n')
                    fh_lat.flush()
                if fire_num:
                    feat = feature_num(ref_chunk)
                    row = [dataset, fid, ' '.join(ref_chunk), ' '.join(p_chunk),
                           feat['tokens'], int(feat['has_digit']), int(feat['has_spelled']),
                           int(feat['mixed']), feat['n_digit_tokens'], feat['n_spelled_tokens'],
                           feat['chunk_len']]
                    row += [model_wins[m] for m in TRACK_MODELS]
                    fh_num.write('\t'.join(str(x) for x in row) + '\n')
                    fh_num.flush()

    fh_lat.close()
    fh_num.close()

    # Summary analyses
    # 1. LATERALITY: are some models systematically better at "left" vs "right"?
    log.info('Analyzing LATERALITY patterns by token...')
    lat_by_token = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # token → model → [n_correct, n_alignable]
    with open(OUT_DIR / 'laterality_per_position.tsv') as f:
        header = f.readline().strip().split('\t')
        win_cols = [(i, c.replace('win_', '')) for i, c in enumerate(header) if c.startswith('win_')]
        for line in f:
            parts = line.strip().split('\t')
            tokens = parts[header.index('tokens')]
            for col_i, model in win_cols:
                v = parts[col_i] if col_i < len(parts) else ''
                if v in ('1', '0'):
                    lat_by_token[tokens][model][1] += 1
                    if v == '1':
                        lat_by_token[tokens][model][0] += 1

    fh_summary = open(OUT_DIR / 'laterality_token_x_model.tsv', 'w')
    fh_summary.write('token\tmodel\tn_correct\tn_alignable\trecovery_pct\n')
    fh_summary.flush()
    for token in sorted(lat_by_token, key=lambda t: -sum(c[0] for c in lat_by_token[t].values())):
        for model, (n_corr, n_alg) in sorted(lat_by_token[token].items(), key=lambda x: -x[1][0])[:15]:
            pct = 100 * n_corr / n_alg if n_alg else 0
            fh_summary.write(f'{token}\t{model}\t{n_corr}\t{n_alg}\t{pct:.1f}\n')
            fh_summary.flush()
    fh_summary.close()

    # 2. NUMERIC: digit vs spelled — is one form easier for specific models?
    log.info('Analyzing NUMERIC patterns by form...')
    num_form_x_model = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # form → model → [correct, alignable]
    with open(OUT_DIR / 'numeric_per_position.tsv') as f:
        header = f.readline().strip().split('\t')
        win_cols = [(i, c.replace('win_', '')) for i, c in enumerate(header) if c.startswith('win_')]
        has_digit_idx = header.index('has_digit')
        has_spelled_idx = header.index('has_spelled')
        mixed_idx = header.index('mixed')
        for line in f:
            parts = line.strip().split('\t')
            mixed = parts[mixed_idx] == '1'
            has_digit = parts[has_digit_idx] == '1'
            has_spelled = parts[has_spelled_idx] == '1'
            if mixed:
                form = 'mixed'
            elif has_digit:
                form = 'digit_only'
            elif has_spelled:
                form = 'spelled_only'
            else:
                form = 'unknown'
            for col_i, model in win_cols:
                v = parts[col_i] if col_i < len(parts) else ''
                if v in ('1', '0'):
                    num_form_x_model[form][model][1] += 1
                    if v == '1':
                        num_form_x_model[form][model][0] += 1

    fh_form = open(OUT_DIR / 'numeric_form_x_model.tsv', 'w')
    fh_form.write('form\tmodel\tn_correct\tn_alignable\trecovery_pct\n')
    fh_form.flush()
    for form in ['digit_only', 'spelled_only', 'mixed']:
        for model, (n_corr, n_alg) in sorted(num_form_x_model[form].items(), key=lambda x: -x[1][0])[:15]:
            pct = 100 * n_corr / n_alg if n_alg else 0
            fh_form.write(f'{form}\t{model}\t{n_corr}\t{n_alg}\t{pct:.1f}\n')
            fh_form.flush()
    fh_form.close()

    # PRINT INSIGHTS
    print(f'\n=== LATERALITY: top 10 ref-tokens × top model ===', flush=True)
    for token in sorted(lat_by_token, key=lambda t: -sum(c[0] for c in lat_by_token[t].values()))[:10]:
        total = sum(c[0] for c in lat_by_token[token].values())
        top = sorted(lat_by_token[token].items(), key=lambda x: -x[1][0])[:3]
        top_str = ' / '.join(f'{m}={c[0]}/{c[1]}' for m, c in top)
        print(f'  "{token}" ({total} wins across all models): top → {top_str}', flush=True)

    print(f'\n=== NUMERIC: form × model ===', flush=True)
    for form in ['digit_only', 'spelled_only', 'mixed']:
        if form not in num_form_x_model:
            continue
        ranked = sorted(num_form_x_model[form].items(), key=lambda x: -x[1][0])
        total = sum(c[0] for c in num_form_x_model[form].values())
        print(f'\n  {form} (total wins: {total}):', flush=True)
        for m, (corr, alg) in ranked[:8]:
            pct = 100 * corr / alg if alg else 0
            print(f'    {m:<35s} {corr:>4d}/{alg:<4d} ({pct:5.1f}%)', flush=True)

    print(f'\nFull data: {OUT_DIR}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
