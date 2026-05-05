#!/usr/bin/env python3
"""For Parakeet's NEGATION_FLIP / LATERALITY / NUMERIC errors, check which OTHER
models in our benchmark transcripts have the right answer at the same reference
position.

Extends check_medasr_on_high_impact.py to compare every model in
ossicles/benchmark_results_<dataset>/. Cloud models from
ossicles/benchmark_results_cloud_<dataset>/ are also included.

This is the data the user asked for: are there models that handle short tokens
("no") or numbers better than MedASR?

Output:
  - results/cross_model_high_impact/by_model.tsv
    rows: pattern, dataset, model, n_parakeet_errors, n_correct, recovery_pct
  - results/cross_model_high_impact/best_per_pattern.tsv
    for each (pattern, dataset), the model with the highest recovery rate
  - results/cross_model_high_impact/coverage_union.tsv
    cumulative coverage if we union top-K models per pattern (does adding a 2nd
    or 3rd model meaningfully improve over MedASR alone?)
"""
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
OUT_DIR = STAPES_DIR / 'results' / 'cross_model_high_impact'

DATASETS = ['primock57', 'figshare-osce', 'nazmulkazi']  # Skipping kokoro — only 2 models

# Skip these JSON files in the benchmark dirs — they're not raw model outputs
SKIP_FILES = {
    'medical_error_analysis.json',
    'fusion_results',
    'medical_errors_per_model',
}


def list_model_jsons(dataset: str) -> dict[str, Path]:
    """Return {model_name: json_path} for all primary model results in a dataset."""
    out = {}
    for d in [
        OSSICLES_DIR / f'benchmark_results_{dataset}',
        OSSICLES_DIR / f'benchmark_results_cloud_{dataset}',
    ]:
        if not d.exists():
            continue
        for f in sorted(d.glob('*.json')):
            name = f.name
            if name in SKIP_FILES:
                continue
            if '.old' in name or 'normalized_' in name or 'medical_error' in name:
                continue
            stem = f.stem
            if d.name.startswith('benchmark_results_cloud_'):
                stem = f'cloud-{stem}'
            out[stem] = f
    return out


def load_results(json_path: Path) -> dict[str, dict]:
    """Return {file_id: row}."""
    try:
        data = json.load(open(json_path))
    except (json.JSONDecodeError, KeyError):
        return {}
    if not isinstance(data, dict):
        return {}
    rows = data.get('results', [])
    return {r['file_id']: r for r in rows if 'file_id' in r}


def get_words_at_ref_position(
    ref_words: list[str], hyp_words: list[str],
    ref_start: int, ref_end: int,
    align_chunks,
) -> list[str]:
    h_starts = []
    h_ends = []
    for fc in align_chunks:
        for ri in range(fc.ref_start_idx, fc.ref_end_idx):
            if ref_start <= ri < ref_end:
                h_starts.append(fc.hyp_start_idx)
                h_ends.append(fc.hyp_end_idx)
                break
    if not h_starts:
        return []
    return hyp_words[min(h_starts):max(h_ends)]


def has_negation(words):
    return any(t in NEGATION_TOKENS for t in words)


def numeric_set(words):
    out = set()
    for w in words:
        if NUMERIC_DIGIT_RE.search(w) or w in SPELLED_NUMBER_TOKENS:
            out.add(w)
    return out


def laterality_set(words):
    return set(t for t in words if t in LATERALITY_TOKENS)


def model_matches(pattern: str, ref_chunk: list[str], model_chunk: list[str]) -> bool | None:
    """Return True/False/None (None = unalignable)."""
    if not model_chunk:
        return None
    if pattern == 'NEGATION_FLIP':
        return has_negation(ref_chunk) == has_negation(model_chunk)
    if pattern == 'NUMERIC':
        return numeric_set(ref_chunk) == numeric_set(model_chunk)
    if pattern == 'LATERALITY':
        return laterality_set(ref_chunk) == laterality_set(model_chunk)
    return None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fh_by_model = open(OUT_DIR / 'by_model.tsv', 'w')
    fh_by_model.write(
        'pattern\tdataset\tmodel\tn_parakeet_errors\tn_correct\tn_wrong\t'
        'n_unalignable\trecovery_pct\n'
    )
    fh_by_model.flush()

    # Per (pattern, error_id) → which models got it right
    # Used for coverage-union analysis: how many parakeet errors are recovered
    # by AT LEAST ONE model in a candidate pool.
    error_recovery = defaultdict(set)  # (pattern, dataset, error_id) → {model names that got it right}
    error_counts = defaultdict(int)    # (pattern, dataset) → total parakeet errors

    for dataset in DATASETS:
        log.info(f'=== {dataset} ===')
        models = list_model_jsons(dataset)
        log.info(f'  found {len(models)} models: {sorted(models.keys())}')
        if 'parakeet-tdt-0.6b-v2' not in models:
            log.warning(f'  no parakeet baseline for {dataset}; skipping')
            continue

        # Load all model results once
        all_results = {name: load_results(path) for name, path in models.items()}
        parakeet_results = all_results['parakeet-tdt-0.6b-v2']

        # Per (pattern, model): counts
        counts = defaultdict(lambda: [0, 0, 0, 0])  # total, correct, wrong, unalignable

        for fid in sorted(parakeet_results.keys()):
            ref = parakeet_results[fid].get('reference', '')
            if not ref:
                continue
            ref_norm = normalize(ref)
            p_norm = normalize(parakeet_results[fid].get('hypothesis', ''))
            ref_words = ref_norm.split()
            p_words = p_norm.split()
            p_align = jiwer.process_words(ref_norm, p_norm)

            # Find Parakeet's high-impact error positions
            error_positions = []  # list of (rs, re_, pattern, ref_chunk, p_chunk)
            for chunk in p_align.alignments[0]:
                if chunk.type not in ('substitute', 'delete'):
                    continue
                rs, re_ = chunk.ref_start_idx, chunk.ref_end_idx
                hs, he = chunk.hyp_start_idx, chunk.hyp_end_idx
                ref_chunk = ref_words[rs:re_]
                p_chunk = p_words[hs:he] if chunk.type == 'substitute' else []

                patterns = []
                if detect_negation_flip(ref_chunk, p_chunk, primary_cat='generic'):
                    patterns.append('NEGATION_FLIP')
                if detect_laterality(ref_chunk, p_chunk):
                    patterns.append('LATERALITY')
                if detect_numeric_mismatch(ref_chunk, p_chunk):
                    patterns.append('NUMERIC')
                for pat in patterns:
                    error_positions.append((rs, re_, pat, ref_chunk, p_chunk))

            # For each candidate model, check at every error position
            for model_name, model_results in all_results.items():
                if model_name == 'parakeet-tdt-0.6b-v2':
                    continue
                if fid not in model_results:
                    continue
                m_norm = normalize(model_results[fid].get('hypothesis', ''))
                m_words = m_norm.split()
                if not m_words:
                    continue
                m_align = jiwer.process_words(ref_norm, m_norm)
                m_chunks = m_align.alignments[0]

                for rs, re_, pat, ref_chunk, p_chunk in error_positions:
                    m_chunk = get_words_at_ref_position(ref_words, m_words, rs, re_, m_chunks)
                    correct = model_matches(pat, ref_chunk, m_chunk)
                    counts[(pat, model_name)][0] += 1
                    if correct is None:
                        counts[(pat, model_name)][3] += 1
                    elif correct:
                        counts[(pat, model_name)][1] += 1
                        error_recovery[(pat, dataset, fid, rs, re_)].add(model_name)
                    else:
                        counts[(pat, model_name)][2] += 1

            # Track total error count per (pattern, dataset)
            for rs, re_, pat, _, _ in error_positions:
                error_counts[(pat, dataset)] += 1

        # Write rows
        for (pat, model), (n_total, n_corr, n_wrong, n_unalign) in sorted(counts.items()):
            denom = n_total - n_unalign
            rate = 100 * n_corr / denom if denom else 0
            fh_by_model.write(
                f'{pat}\t{dataset}\t{model}\t{n_total}\t{n_corr}\t{n_wrong}\t'
                f'{n_unalign}\t{rate:.1f}\n'
            )
            fh_by_model.flush()
        log.info(f'  {dataset}: scored {len(counts)} (pattern,model) pairs')

    fh_by_model.close()

    # Best per pattern
    log.info('Computing best-model and coverage-union analysis...')
    by_p_d_m = {}  # (pattern, dataset) → list of (model, recovery_pct, n_correct, n_total_alignable)
    with open(OUT_DIR / 'by_model.tsv') as f:
        next(f)
        for line in f:
            pat, ds, model, n_total, n_corr, n_wrong, n_unalign, rate = line.strip().split('\t')
            n_total = int(n_total); n_corr = int(n_corr); n_unalign = int(n_unalign)
            denom = n_total - n_unalign
            by_p_d_m.setdefault((pat, ds), []).append((model, float(rate), n_corr, denom))

    fh_best = open(OUT_DIR / 'best_per_pattern.tsv', 'w')
    fh_best.write('pattern\tdataset\trank\tmodel\trecovery_pct\tn_correct\tn_alignable\n')
    fh_best.flush()
    for (pat, ds), rows in sorted(by_p_d_m.items()):
        rows.sort(key=lambda r: -r[2])  # sort by absolute n_correct (most useful)
        for rank, (m, rate, n_corr, denom) in enumerate(rows[:10], 1):
            fh_best.write(f'{pat}\t{ds}\t{rank}\t{m}\t{rate:.1f}\t{n_corr}\t{denom}\n')
            fh_best.flush()
    fh_best.close()

    # Coverage union: for each (pattern, dataset), how many errors does at least
    # one of {medasr, top-2nd, top-3rd, ...} models recover?
    fh_union = open(OUT_DIR / 'coverage_union.tsv', 'w')
    fh_union.write('pattern\tdataset\tmodels_added\trecovery_count\trecovery_pct_of_total\n')
    fh_union.flush()
    for (pat, ds), rows in sorted(by_p_d_m.items()):
        rows.sort(key=lambda r: -r[2])
        # Build cumulative model set, count cumulative coverage
        used_models = []
        cum_recovered_errors = set()
        total_errors = error_counts.get((pat, ds), 0)
        for m, rate, n_corr, denom in rows[:10]:
            used_models.append(m)
            for key, recovering in error_recovery.items():
                if key[0] == pat and key[1] == ds and m in recovering:
                    cum_recovered_errors.add(key)
            pct = 100 * len(cum_recovered_errors) / total_errors if total_errors else 0
            fh_union.write(
                f'{pat}\t{ds}\t{",".join(used_models)}\t'
                f'{len(cum_recovered_errors)}\t{pct:.1f}\n'
            )
            fh_union.flush()
    fh_union.close()

    print(f'\n=== TOP MODELS BY ABSOLUTE RECOVERY (across 3 real datasets) ===', flush=True)
    for pat in ['NEGATION_FLIP', 'LATERALITY', 'NUMERIC']:
        # Aggregate across datasets
        agg = defaultdict(lambda: [0, 0])
        for (pat_, ds), rows in by_p_d_m.items():
            if pat_ != pat:
                continue
            for m, rate, n_corr, denom in rows:
                agg[m][0] += n_corr
                agg[m][1] += denom
        ranked = sorted(agg.items(), key=lambda x: -x[1][0])
        print(f'\n{pat}:', flush=True)
        print(f'  {"model":<35s} {"recovered":>10s}  {"of":>5s}  {"%":>6s}', flush=True)
        for m, (corr, denom) in ranked[:10]:
            pct = 100 * corr / denom if denom else 0
            print(f'  {m:<35s} {corr:>10d}  {denom:>5d}  {pct:>5.1f}%', flush=True)

    print(f'\nFull data: {OUT_DIR}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
