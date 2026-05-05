#!/usr/bin/env python3
"""Parameter sweep over min_anchor_words × max_anchor_edit_ratio for v3 fusion.

Justifies (or refutes) the chosen defaults: min_anchor_words=3, max_anchor_edit_ratio=0.5.

For each parameter combination, run the fusion across all 3 datasets and measure
drug-CTR delta + errors corrected/introduced. This re-uses fuse() and the
QuickUMLS drug-span finder from the existing scripts; no ASR re-runs needed.

Output:
    results/sweep_fusion_params/sweep_summary.tsv (incremental)
    results/sweep_fusion_params/sweep_per_file.tsv (incremental)

Run with the ossicles venv python:
    /home/grey/dev/graiai/ossicles/venv/bin/python scripts/sweep_fusion_params.py
"""
import json
import logging
import sys
from pathlib import Path

from quickumls import QuickUMLS

# Reuse existing implementations
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from compute_drug_ctr import (  # noqa: E402
    DRUG_SEMTYPES,
    count_drug_errors,
    find_drug_spans,
    load_dataset_pair,
)
from measure_targeted_fusion_v3 import (  # noqa: E402
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
OSSICLES_DIR = STAPES_DIR.parent / 'ossicles'
QUICKUMLS_PATH = OSSICLES_DIR / 'quickumls_data'
VOCAB_TSV = STAPES_DIR / 'results' / 'medical_vocab' / 'medical_vocab_drug_curated.tsv'
OUT_DIR = STAPES_DIR / 'results' / 'sweep_fusion_params'

DATASETS = ['primock57', 'figshare-osce', 'nazmulkazi']

# Parameter grid
MIN_ANCHOR_GRID = [1, 2, 3, 4]
EDIT_RATIO_GRID = [None, 0.7, 0.5, 0.3]


def precompute_drug_spans(matcher, datasets):
    """Compute drug spans per (dataset, file_id) once; reuse across all parameter combos."""
    spans_by_ds = {}
    refs_by_ds = {}
    p_norms_by_ds = {}
    m_norms_by_ds = {}
    for ds in datasets:
        log.info(f'  precomputing drug spans for {ds}')
        pair = load_dataset_pair(ds)
        if pair is None:
            continue
        p_idx, m_idx = pair
        common = sorted(set(p_idx) & set(m_idx))
        spans, refs, pnorms, mnorms = {}, {}, {}, {}
        for i, fid in enumerate(common):
            ref = p_idx[fid].get('reference', '')
            if not ref:
                continue
            ref_norm = normalize(ref)
            ds_spans = find_drug_spans(ref_norm, matcher)
            spans[fid] = ds_spans
            refs[fid] = ref_norm
            pnorms[fid] = normalize(p_idx[fid].get('hypothesis', ''))
            mnorms[fid] = normalize(m_idx[fid].get('hypothesis', ''))
            if (i + 1) % 100 == 0:
                log.info(f'    [{i+1}/{len(common)}]')
        spans_by_ds[ds] = spans
        refs_by_ds[ds] = refs
        p_norms_by_ds[ds] = pnorms
        m_norms_by_ds[ds] = mnorms
        n_with_drugs = sum(1 for s in spans.values() if s)
        n_drug_spans = sum(len(s) for s in spans.values())
        log.info(f'    {ds}: {n_with_drugs}/{len(spans)} files have drugs, {n_drug_spans} total drug spans')
    return spans_by_ds, refs_by_ds, p_norms_by_ds, m_norms_by_ds


def run_combo(min_anchor, edit_ratio, vocab, max_n,
              spans_by_ds, refs_by_ds, p_norms_by_ds, m_norms_by_ds,
              fh_per, fh_summary):
    """Run fusion + drug-CTR for one parameter combo across all datasets."""
    label = f'min_anchor={min_anchor}, edit_ratio={edit_ratio}'

    for ds in DATASETS:
        spans_map = spans_by_ds.get(ds, {})
        if not spans_map:
            continue

        n_files = 0
        n_drug_spans_total = 0
        ds_p_errors = 0
        ds_f_errors = 0
        ds_corrected = 0
        ds_introduced = 0
        ds_subs_acoustic = 0
        ds_subs_anchor = 0

        for fid, drug_spans in spans_map.items():
            if not drug_spans:
                continue
            ref_norm = refs_by_ds[ds][fid]
            p_norm = p_norms_by_ds[ds][fid]
            m_norm = m_norms_by_ds[ds][fid]

            fused, fstats = fuse(
                p_norm, m_norm, vocab, max_n,
                min_anchor_words=min_anchor,
                max_anchor_edit_ratio=edit_ratio,
            )
            p_err, _ = count_drug_errors(ref_norm, p_norm, drug_spans)
            f_err, _ = count_drug_errors(ref_norm, fused, drug_spans)

            n_files += 1
            n_drug_spans_total += len(drug_spans)
            ds_p_errors += p_err
            ds_f_errors += f_err
            ds_corrected += max(0, p_err - f_err)
            ds_introduced += max(0, f_err - p_err)
            ds_subs_acoustic += fstats['subs_acoustic']
            ds_subs_anchor += fstats['subs_anchor']

            fh_per.write(
                f'{min_anchor}\t{edit_ratio}\t{ds}\t{fid}\t{p_err}\t{f_err}\t'
                f'{fstats["subs_acoustic"]}\t{fstats["subs_anchor"]}\n'
            )
            fh_per.flush()

        p_recall = 100 * (1.0 - ds_p_errors / n_drug_spans_total) if n_drug_spans_total else 0.0
        f_recall = 100 * (1.0 - ds_f_errors / n_drug_spans_total) if n_drug_spans_total else 0.0
        delta = f_recall - p_recall
        net = ds_corrected - ds_introduced

        fh_summary.write(
            f'{min_anchor}\t{edit_ratio}\t{ds}\t{n_files}\t{n_drug_spans_total}\t'
            f'{p_recall:.4f}\t{f_recall:.4f}\t{delta:+.4f}\t'
            f'{ds_corrected}\t{ds_introduced}\t{net}\t'
            f'{ds_subs_acoustic}\t{ds_subs_anchor}\n'
        )
        fh_summary.flush()

        print(f'  [{label}] {ds}: Δ={delta:+.2f}pp corrected={ds_corrected} '
              f'introduced={ds_introduced} net={net:+d} '
              f'subs(A/B)={ds_subs_acoustic}/{ds_subs_anchor}', flush=True)


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

    log.info('precomputing drug spans (one-time)...')
    spans_by_ds, refs_by_ds, p_norms_by_ds, m_norms_by_ds = precompute_drug_spans(matcher, DATASETS)

    fh_summary = open(OUT_DIR / 'sweep_summary.tsv', 'w')
    fh_summary.write(
        'min_anchor\tedit_ratio\tdataset\tn_files\tn_drug_spans\t'
        'parakeet_recall\tfused_recall\tdelta_pp\t'
        'errors_corrected\terrors_introduced\tnet\t'
        'subs_acoustic\tsubs_anchor\n'
    )
    fh_summary.flush()

    fh_per = open(OUT_DIR / 'sweep_per_file.tsv', 'w')
    fh_per.write(
        'min_anchor\tedit_ratio\tdataset\tfile_id\tparakeet_err\tfused_err\t'
        'subs_acoustic\tsubs_anchor\n'
    )
    fh_per.flush()

    n_combos = len(MIN_ANCHOR_GRID) * len(EDIT_RATIO_GRID)
    i = 0
    for min_anchor in MIN_ANCHOR_GRID:
        for edit_ratio in EDIT_RATIO_GRID:
            i += 1
            print(f'\n=== combo {i}/{n_combos}: min_anchor={min_anchor}, '
                  f'edit_ratio={edit_ratio} ===', flush=True)
            run_combo(
                min_anchor, edit_ratio, vocab, max_n,
                spans_by_ds, refs_by_ds, p_norms_by_ds, m_norms_by_ds,
                fh_per, fh_summary,
            )

    fh_summary.close()
    fh_per.close()
    print(f'\nDone. Summary: {OUT_DIR / "sweep_summary.tsv"}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
