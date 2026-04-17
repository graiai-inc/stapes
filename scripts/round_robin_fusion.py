#!/usr/bin/env python3
"""Exhaustive pair + triple ROVER fusion search across all three datasets.

Curated 9-model on-device pool (drops int8 duplicates and models too weak
to survive fusion per malleus findings). Runs every pair and every triple
on PriMock57, Nazmulkazi, and figshare-osce. figshare-osce uses the
apostrophe-injected reference from figshare_fix_and_eval.py.

Three-model ROVER is implemented as progressive pairwise fusion: fuse
(A, B) with equal weights, then fuse that intermediate with C with equal
weights. This is standard progressive ROVER and is the approach used in
most deployed multi-hypothesis fusion systems.

Per-combination results are written incrementally to two TSVs (one for
pairs, one for triples) with fh.flush() on every row. A final block
prints the best pair and best triple per dataset plus the delta.
"""

import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rover_fusion as rf  # noqa: E402
from figshare_fix_and_eval import inject_apostrophes, rover_fuse_fixed  # noqa: E402

WNORM = EnglishTextNormalizer()
OSSICLES = Path('/home/grey/dev/graiai/ossicles')
RESULTS = Path('/home/grey/dev/graiai/stapes/results')
RESULTS.mkdir(parents=True, exist_ok=True)

# 9-model curated pool. Dropped: int8 variants, nemotron (38% baseline),
# zipformer-zh-en (34%), medasr (56%) — all too weak to help fusion per
# malleus. Kept strong on-device candidates across multiple labs and
# architectures.
MODELS = [
    'parakeet-tdt-0.6b-v2',
    'whisper-distil-v3.5',
    'whisper-turbo',
    'whisper-base-en',
    'sensevoice',
    'sensevoice-no-itn',
    'qwen3-asr',
    'paraformer-en',
    'nemo-fastconformer',
]

DATASETS = [
    # (display_name, subdir, inject_apostrophes_flag)
    ('primock57', 'benchmark_results_primock57', False),
    ('nazmulkazi', 'benchmark_results_nazmulkazi', False),
    ('figshare-osce', 'benchmark_results_figshare-osce', True),
]


def load_any(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    if isinstance(d, dict) and 'results' in d:
        return d['results']
    if isinstance(d, list):
        return d
    return []


def load_model(subdir: str, model: str) -> dict | None:
    p = OSSICLES / subdir / f'{model}.json'
    if not p.exists():
        return None
    rows = load_any(p)
    out = {}
    for r in rows:
        fid = r.get('file_id')
        if fid:
            out[fid] = r
    return out or None


def three_fuse(h1: str, h2: str, h3: str) -> str:
    """Progressive pairwise ROVER with equal weights at each step."""
    mid = rover_fuse_fixed(h1, h2)
    return rover_fuse_fixed(mid, h3)


def score_combo(args: tuple) -> dict | None:
    ds_name, subdir, inject, combo = args
    loaded: dict[str, dict] = {}
    for m in combo:
        r = load_model(subdir, m)
        if r is None:
            return {'dataset': ds_name, 'combo': '+'.join(combo),
                    'n_models': len(combo), 'error': f'missing:{m}'}
        loaded[m] = r

    common = sorted(set.intersection(*[set(r) for r in loaded.values()]))
    if not common:
        return {'dataset': ds_name, 'combo': '+'.join(combo),
                'n_models': len(combo), 'error': 'no_common_files'}

    total_err = 0
    total_words = 0
    files_scored = 0
    for fid in common:
        ref = loaded[combo[0]][fid].get('reference', '')
        hyps = [loaded[m][fid].get('hypothesis', '') for m in combo]
        if not ref or not all(hyps):
            continue
        if inject:
            ref = inject_apostrophes(ref)
        ref_w = WNORM(ref)
        if not ref_w:
            continue
        n = len(ref_w.split())
        try:
            if len(combo) == 2:
                fused = rover_fuse_fixed(hyps[0], hyps[1])
            else:
                fused = three_fuse(hyps[0], hyps[1], hyps[2])
            fused_w = WNORM(fused)
            wer = jiwer.wer(ref_w, fused_w)
        except Exception as exc:
            return {'dataset': ds_name, 'combo': '+'.join(combo),
                    'n_models': len(combo), 'error': f'fuse_err:{exc}'}
        total_err += round(wer * n)
        total_words += n
        files_scored += 1

    if total_words == 0:
        return {'dataset': ds_name, 'combo': '+'.join(combo),
                'n_models': len(combo), 'error': 'zero_words'}

    return {
        'dataset': ds_name,
        'combo': '+'.join(combo),
        'n_models': len(combo),
        'files': files_scored,
        'wer': total_err / total_words * 100,
        'words': total_words,
    }


def build_tasks() -> list[tuple]:
    tasks = []
    for ds_name, subdir, inject in DATASETS:
        # Filter pool to models available in this dataset
        available = [m for m in MODELS if (OSSICLES / subdir / f'{m}.json').exists()]
        print(f'[pool] {ds_name}: {len(available)}/{len(MODELS)} models available', flush=True)
        for combo in itertools.combinations(available, 2):
            tasks.append((ds_name, subdir, inject, combo))
        for combo in itertools.combinations(available, 3):
            tasks.append((ds_name, subdir, inject, combo))
    return tasks


def main() -> None:
    tasks = build_tasks()
    n_pairs = sum(1 for t in tasks if len(t[3]) == 2)
    n_trips = sum(1 for t in tasks if len(t[3]) == 3)
    print(f'[plan] {len(tasks)} total combinations ({n_pairs} pairs, {n_trips} triples)',
          flush=True)

    pair_f = (RESULTS / 'rr_pair_results.tsv').open('w')
    pair_f.write('dataset\tcombo\tfiles\twer\twords\n')
    pair_f.flush()
    trip_f = (RESULTS / 'rr_triple_results.tsv').open('w')
    trip_f.write('dataset\tcombo\tfiles\twer\twords\n')
    trip_f.flush()
    err_f = (RESULTS / 'rr_errors.tsv').open('w')
    err_f.write('dataset\tcombo\tn_models\terror\n')
    err_f.flush()

    pair_best: dict[str, dict] = {}
    trip_best: dict[str, dict] = {}

    done = 0
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(score_combo, t) for t in tasks]
        for fut in as_completed(futs):
            done += 1
            try:
                res = fut.result()
            except Exception as exc:
                print(f'[ERR] future raised: {exc}', flush=True)
                continue
            if res is None:
                continue
            if 'error' in res:
                err_f.write(f'{res["dataset"]}\t{res["combo"]}\t{res["n_models"]}\t{res["error"]}\n')
                err_f.flush()
                continue
            line = (f'{res["dataset"]}\t{res["combo"]}\t{res["files"]}'
                    f'\t{res["wer"]:.4f}\t{res["words"]}\n')
            if res['n_models'] == 2:
                pair_f.write(line)
                pair_f.flush()
                prev = pair_best.get(res['dataset'])
                if prev is None or res['wer'] < prev['wer']:
                    pair_best[res['dataset']] = res
            else:
                trip_f.write(line)
                trip_f.flush()
                prev = trip_best.get(res['dataset'])
                if prev is None or res['wer'] < prev['wer']:
                    trip_best[res['dataset']] = res

            if done % 10 == 0 or done == len(tasks):
                tag = 'pair' if res['n_models'] == 2 else 'triple'
                print(
                    f'[{done}/{len(tasks)}] {res["dataset"]:<14} {tag:<6} '
                    f'{res["combo"][:70]}: {res["wer"]:.2f}%',
                    flush=True,
                )

    pair_f.close()
    trip_f.close()
    err_f.close()

    print('', flush=True)
    print('=== BEST PAIR vs BEST TRIPLE per dataset ===', flush=True)
    print(f'{"dataset":<16}{"best_pair_wer":>16}{"best_triple_wer":>18}{"triple_delta_pp":>18}',
          flush=True)
    summary = (RESULTS / 'rr_summary.tsv').open('w')
    summary.write('dataset\tbest_pair\tbest_pair_wer\tbest_triple\tbest_triple_wer\ttriple_delta_pp\n')
    summary.flush()
    for ds_name, _, _ in DATASETS:
        p = pair_best.get(ds_name)
        t = trip_best.get(ds_name)
        if p is None or t is None:
            continue
        delta = t['wer'] - p['wer']
        print(
            f'{ds_name:<16}{p["wer"]:>15.2f}%{t["wer"]:>17.2f}%{delta:>+17.2f}',
            flush=True,
        )
        print(f'  best pair   : {p["combo"]}', flush=True)
        print(f'  best triple : {t["combo"]}', flush=True)
        summary.write(
            f'{ds_name}\t{p["combo"]}\t{p["wer"]:.4f}'
            f'\t{t["combo"]}\t{t["wer"]:.4f}\t{delta:+.4f}\n'
        )
        summary.flush()
    summary.close()

    print('', flush=True)
    print(f'[done] pairs:   {RESULTS}/rr_pair_results.tsv', flush=True)
    print(f'[done] triples: {RESULTS}/rr_triple_results.tsv', flush=True)
    print(f'[done] summary: {RESULTS}/rr_summary.tsv', flush=True)
    print(f'[done] errors:  {RESULTS}/rr_errors.tsv', flush=True)


if __name__ == '__main__':
    main()
