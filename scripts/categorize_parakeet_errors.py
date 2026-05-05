#!/usr/bin/env python3
"""Per-error categorization of Parakeet's errors across all 4 datasets.

For every Parakeet error position (substitute or delete in the jiwer alignment),
look up the reference word(s) against our 7 category vocabs (DRUG, CONDITION,
SYMPTOM, FINDING, PROCEDURE, ANATOMY, LAB_RESULT) and a small filler-word set.
Output a per-error TSV plus per-category counts and representative examples.

Also looks at the v3-fused hypothesis at the same position and flags whether
fusion changed Parakeet's output there. This lets us split errors into:
  - fusion fixed it (drug wins)
  - fusion didn't fix it (still wrong; future fusion work)
  - fusion made it different but not the reference (rare, the asthma→aspirin family)

Manual review consumes the per-error TSV. Bulk patterns are visible in the
per-category summary.

Run with the ossicles venv python (it has jiwer):
    /home/grey/dev/graiai/ossicles/venv/bin/python \\
        scripts/categorize_parakeet_errors.py
"""
import csv
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jiwer

# Reuse fuse() to recompute the fused hypothesis per file
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from measure_targeted_fusion_v3 import (  # noqa: E402
    DEFAULT_MAX_ANCHOR_EDIT_RATIO,
    DEFAULT_MIN_ANCHOR,
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

STAPES_DIR = SCRIPT_DIR.parent
GRAIAI_DIR = STAPES_DIR.parent
OSSICLES_DIR = GRAIAI_DIR / 'ossicles'

VOCAB_DIR = STAPES_DIR / 'results' / 'medical_vocab' / 'by_category'
DRUG_VOCAB_TSV = STAPES_DIR / 'results' / 'medical_vocab' / 'medical_vocab_drug_curated.tsv'
OUT_DIR = STAPES_DIR / 'results' / 'parakeet_error_categorization'

DATASETS = ['primock57', 'figshare-osce', 'nazmulkazi', 'kokoro-va-sample']

# Common conversational filler / discourse markers that should be classified as
# 'filler' rather than 'generic'. These almost always have low meaning impact
# when missed by ASR.
FILLERS = {
    'um', 'uh', 'uhm', 'umm', 'er', 'erm', 'ah', 'ahh', 'mm', 'mhm', 'mmhmm',
    'hmm', 'huh', 'oh', 'eh', 'yeah', 'yea', 'yup', 'nope', 'ya', 'okay',
    'ok', 'alright', 'right', 'so', 'well', 'like', 'just', 'kinda', 'sorta',
    'basically', 'actually', 'literally', 'really', 'pretty',
}

# Category priority for cases where a word appears in multiple vocabs. DRUG
# wins since drug names are the most specific clinically; symptoms and findings
# overlap heavily so we order by clinical decision-making impact.
CATEGORY_PRIORITY = [
    'DRUG', 'LAB_RESULT', 'PROCEDURE', 'CONDITION', 'SYMPTOM', 'FINDING', 'ANATOMY',
]


def load_category_vocabs() -> dict[str, dict[str, set[str]]]:
    """Returns {category: {'single': set of single-word terms, 'phrases': set of multi-word terms}}."""
    out = {}
    # Use the curated DRUG vocab (not the raw 19K) — that's what the fusion uses
    drug_single, drug_phrases = set(), set()
    with open(DRUG_VOCAB_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            t = row['term'].strip().lower()
            if not t:
                continue
            if ' ' in t:
                drug_phrases.add(t)
            else:
                drug_single.add(t)
    out['DRUG'] = {'single': drug_single, 'phrases': drug_phrases}

    # Other categories: use the raw split files (not yet curated; future work)
    for cat in ['CONDITION', 'FINDING', 'PROCEDURE', 'ANATOMY', 'SYMPTOM', 'LAB_RESULT']:
        path = VOCAB_DIR / f'{cat}.tsv'
        single, phrases = set(), set()
        if path.exists():
            with open(path) as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    t = row['term'].strip().lower()
                    if not t:
                        continue
                    # Min length filter: 5 chars on single-word entries to drop short noise
                    if ' ' not in t:
                        if len(t.replace('-', '')) < 5:
                            continue
                        single.add(t)
                    else:
                        phrases.add(t)
        out[cat] = {'single': single, 'phrases': phrases}

    return out


def categorize_words(ref_words: list[str], cat_vocabs: dict) -> list[str]:
    """Return list of categories that match these ref_words.

    Checks the joined phrase against multi-word vocabs, plus each word against
    single-word vocabs. Filler list is checked separately.
    """
    cats = set()
    if not ref_words:
        return []
    joined = ' '.join(ref_words).lower()
    # Filler check (only applies to single-word errors)
    if len(ref_words) == 1 and ref_words[0].lower().strip("'") in FILLERS:
        cats.add('filler')
    # Multi-word phrase check
    for cat, vocab in cat_vocabs.items():
        if joined in vocab['phrases']:
            cats.add(cat)
    # Single-word check
    for w in ref_words:
        wl = w.lower().strip("'-")
        for cat, vocab in cat_vocabs.items():
            if wl in vocab['single']:
                cats.add(cat)
    return sorted(cats)


def primary_category(cats: list[str]) -> str:
    """Pick a single primary category from the list, by priority order."""
    if 'filler' in cats:
        return 'filler'
    for c in CATEGORY_PRIORITY:
        if c in cats:
            return c
    return 'generic' if not cats else cats[0]


def likely_meaning_change_heuristic(ref_words: list[str], hyp_words: list[str]) -> tuple[bool, str]:
    """Coarse heuristic: returns (likely_changes_meaning, reason).

    This is NOT authoritative; manual review still required. It just flags the
    obviously low-impact cases (morphology, fillers) so the user can skip them.
    """
    if not ref_words:
        return True, 'pure_insertion'
    if not hyp_words:
        return False, 'pure_deletion_filler' if (len(ref_words) == 1
            and ref_words[0].lower() in FILLERS) else True
    # Same after stripping morphology suffixes (very rough)
    def strip_morph(w):
        w = w.lower().strip("'-")
        for suf in ['ies', 'ed', 'ing', 'es', 's']:
            if w.endswith(suf) and len(w) > len(suf) + 2:
                return w[:-len(suf)]
        return w
    r_stem = ' '.join(strip_morph(w) for w in ref_words)
    h_stem = ' '.join(strip_morph(w) for w in hyp_words)
    if r_stem == h_stem:
        return False, 'morphology_only'
    # Both filler
    if (len(ref_words) == 1 and ref_words[0].lower() in FILLERS
            and len(hyp_words) == 1 and hyp_words[0].lower() in FILLERS):
        return False, 'filler_to_filler'
    # Default: assume meaning change
    return True, 'unclassified'


def load_dataset_pair(dataset: str):
    pdir = OSSICLES_DIR / f'benchmark_results_{dataset}'
    pp = pdir / 'parakeet-tdt-0.6b-v2.json'
    mp = pdir / 'medasr.json'
    if not pp.exists() or not mp.exists():
        return None, None
    p_idx = {r['file_id']: r for r in json.load(open(pp)).get('results', [])
             if 'file_id' in r}
    m_idx = {r['file_id']: r for r in json.load(open(mp)).get('results', [])
             if 'file_id' in r}
    return p_idx, m_idx


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f'loading category vocabs from {VOCAB_DIR}')
    cat_vocabs = load_category_vocabs()
    for cat, v in cat_vocabs.items():
        log.info(f'  {cat}: {len(v["single"])} single + {len(v["phrases"])} multi-word')

    log.info(f'loading curated DRUG vocab for fusion replay from {DRUG_VOCAB_TSV}')
    fusion_vocab, fusion_max_n = load_targeted_vocab(DRUG_VOCAB_TSV, ['DRUG'])
    log.info(f'  {len(fusion_vocab)} terms, max phrase length {fusion_max_n}')
    log.info(f'  fusion config: min_anchor={DEFAULT_MIN_ANCHOR}, '
             f'edit_ratio={DEFAULT_MAX_ANCHOR_EDIT_RATIO}')

    fh_per_err = open(OUT_DIR / 'per_error.tsv', 'w')
    fh_per_err.write(
        'dataset\tfile_id\terror_type\tref_words\thyp_words\tfused_words\t'
        'primary_category\tall_categories\tfusion_changed_position\t'
        'fusion_fixed_position\tlikely_meaning_change\tmeaning_change_reason\n'
    )
    fh_per_err.flush()

    cat_counter = defaultdict(lambda: {'total': 0, 'fusion_fixed': 0, 'fusion_changed_not_fixed': 0,
                                       'meaning_change': 0, 'examples': []})
    dataset_counter = defaultdict(int)

    for dataset in DATASETS:
        log.info(f'=== {dataset} ===')
        p_idx, m_idx = load_dataset_pair(dataset)
        if p_idx is None:
            log.warning(f'  no benchmark results for {dataset}, skipping')
            continue

        common = sorted(set(p_idx) & set(m_idx))
        log.info(f'  {len(common)} files')

        for i, fid in enumerate(common):
            p_row = p_idx[fid]
            m_row = m_idx[fid]
            ref = p_row.get('reference', '')
            if not ref:
                continue
            ref_norm = normalize(ref)
            p_norm = normalize(p_row.get('hypothesis', ''))
            m_norm = normalize(m_row.get('hypothesis', ''))

            # Recompute the fused hypothesis (so we can see exactly what fusion did)
            fused_norm, _ = fuse(
                p_norm, m_norm, fusion_vocab, fusion_max_n,
                min_anchor_words=DEFAULT_MIN_ANCHOR,
                max_anchor_edit_ratio=DEFAULT_MAX_ANCHOR_EDIT_RATIO,
            )

            # Align ref vs parakeet
            ref_words = ref_norm.split()
            p_words = p_norm.split()
            f_words = fused_norm.split()

            out = jiwer.process_words(ref_norm, p_norm)
            chunks = out.alignments[0]

            # Also align ref vs fused so we can check fusion's behavior at error positions
            fused_out = jiwer.process_words(ref_norm, fused_norm)
            fused_chunks = fused_out.alignments[0]
            # Build a mapping ref_word_index -> fused_word_index range (where fused has the equivalent words)
            ref_to_fused_range = {}
            for fc in fused_chunks:
                for ri in range(fc.ref_start_idx, fc.ref_end_idx):
                    ref_to_fused_range[ri] = (fc.hyp_start_idx, fc.hyp_end_idx, fc.type)

            for chunk in chunks:
                if chunk.type not in ('substitute', 'delete'):
                    continue
                ref_chunk_words = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
                hyp_chunk_words = (p_words[chunk.hyp_start_idx:chunk.hyp_end_idx]
                                   if chunk.type == 'substitute' else [])

                # What did fusion put at this ref position?
                fused_chunk_words = []
                if chunk.ref_start_idx in ref_to_fused_range:
                    f_start, f_end, _ = ref_to_fused_range[chunk.ref_start_idx]
                    if chunk.ref_end_idx - 1 in ref_to_fused_range:
                        f_end_real = ref_to_fused_range[chunk.ref_end_idx - 1][1]
                    else:
                        f_end_real = f_end
                    fused_chunk_words = f_words[f_start:f_end_real]

                fusion_changed = (fused_chunk_words != hyp_chunk_words)
                fusion_fixed = (fused_chunk_words == ref_chunk_words)

                cats = categorize_words(ref_chunk_words, cat_vocabs)
                pcat = primary_category(cats)
                meaning_change, reason = likely_meaning_change_heuristic(
                    ref_chunk_words, hyp_chunk_words,
                )

                fh_per_err.write(
                    f'{dataset}\t{fid}\t{chunk.type}\t'
                    f'{" ".join(ref_chunk_words)}\t'
                    f'{" ".join(hyp_chunk_words)}\t'
                    f'{" ".join(fused_chunk_words)}\t'
                    f'{pcat}\t{",".join(cats) if cats else ""}\t'
                    f'{int(fusion_changed)}\t{int(fusion_fixed)}\t'
                    f'{int(meaning_change)}\t{reason}\n'
                )
                fh_per_err.flush()

                cat_counter[pcat]['total'] += 1
                if fusion_fixed:
                    cat_counter[pcat]['fusion_fixed'] += 1
                elif fusion_changed:
                    cat_counter[pcat]['fusion_changed_not_fixed'] += 1
                if meaning_change:
                    cat_counter[pcat]['meaning_change'] += 1
                # Save up to 20 representative examples per category
                if len(cat_counter[pcat]['examples']) < 20:
                    cat_counter[pcat]['examples'].append({
                        'dataset': dataset, 'file_id': fid,
                        'ref': ' '.join(ref_chunk_words),
                        'hyp': ' '.join(hyp_chunk_words),
                        'fused': ' '.join(fused_chunk_words),
                    })
                dataset_counter[dataset] += 1

            if (i + 1) % 50 == 0 or i + 1 == len(common):
                log.info(f'  [{i+1}/{len(common)}] processed')

    fh_per_err.close()

    # Write summary by category
    fh_summary = open(OUT_DIR / 'summary.tsv', 'w')
    fh_summary.write(
        'category\ttotal_errors\tfusion_fixed\tfusion_changed_not_fixed\t'
        'fusion_didnt_touch\tlikely_meaning_change\tpct_total\n'
    )
    fh_summary.flush()
    grand_total = sum(c['total'] for c in cat_counter.values())
    for cat in sorted(cat_counter.keys(), key=lambda c: -cat_counter[c]['total']):
        c = cat_counter[cat]
        no_touch = c['total'] - c['fusion_fixed'] - c['fusion_changed_not_fixed']
        pct = 100 * c['total'] / grand_total if grand_total else 0
        fh_summary.write(
            f'{cat}\t{c["total"]}\t{c["fusion_fixed"]}\t{c["fusion_changed_not_fixed"]}\t'
            f'{no_touch}\t{c["meaning_change"]}\t{pct:.2f}\n'
        )
        fh_summary.flush()
    fh_summary.close()

    # Write examples per category
    fh_ex = open(OUT_DIR / 'examples_by_category.txt', 'w')
    for cat in sorted(cat_counter.keys(), key=lambda c: -cat_counter[c]['total']):
        c = cat_counter[cat]
        no_touch = c['total'] - c['fusion_fixed'] - c['fusion_changed_not_fixed']
        fh_ex.write(f'\n========== {cat} ({c["total"]} errors, '
                    f'{c["fusion_fixed"]} fusion-fixed, {no_touch} fusion-didn\'t-touch) ==========\n')
        for ex in c['examples']:
            fh_ex.write(f"  {ex['dataset']:18s} {ex['file_id']:30s} "
                        f"  ref={ex['ref']!r}  parakeet={ex['hyp']!r}  "
                        f"fused={ex['fused']!r}\n")
        fh_ex.flush()
    fh_ex.close()

    print(f'\n=== SUMMARY ===', flush=True)
    print(f'Total errors across {len(DATASETS)} datasets: {grand_total}', flush=True)
    print(f'\n  per dataset:', flush=True)
    for ds, n in sorted(dataset_counter.items(), key=lambda x: -x[1]):
        print(f'    {ds}: {n}', flush=True)
    print(f'\n  per category:', flush=True)
    for cat in sorted(cat_counter.keys(), key=lambda c: -cat_counter[c]['total']):
        c = cat_counter[cat]
        no_touch = c['total'] - c['fusion_fixed'] - c['fusion_changed_not_fixed']
        pct = 100 * c['total'] / grand_total if grand_total else 0
        print(f'    {cat:12s}: {c["total"]:6d} ({pct:5.1f}%) | '
              f'fusion fixed: {c["fusion_fixed"]:4d}, '
              f'fusion changed-not-fixed: {c["fusion_changed_not_fixed"]:3d}, '
              f'fusion didn\'t touch: {no_touch:5d}', flush=True)
    print(f'\nFull per-error log: {OUT_DIR}/per_error.tsv', flush=True)
    print(f'Examples by category: {OUT_DIR}/examples_by_category.txt', flush=True)
    print(f'Summary: {OUT_DIR}/summary.tsv', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
