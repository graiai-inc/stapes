#!/usr/bin/env python3
"""Recompute cloud API WER on figshare-osce with reference apostrophes injected.

Uses the same 44-token injection dict as figshare_fix_and_eval.py so solo,
fused, and cloud numbers are directly comparable. Writes per-file TSV
incrementally.
"""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figshare_fix_and_eval import inject_apostrophes  # noqa: E402

WNORM = EnglishTextNormalizer()
OSSICLES = Path('/home/grey/dev/graiai/ossicles')
RESULTS = Path('/home/grey/dev/graiai/stapes/results')
RESULTS.mkdir(parents=True, exist_ok=True)

# (subdir, provider) — all cloud results on figshare-osce
SOURCES = [
    ('benchmark_results_cloud_figshare-osce', 'assemblyai'),
    ('benchmark_results_cloud_figshare-osce', 'azure'),
    ('benchmark_results_cloud_figshare-osce', 'deepgram'),
    ('benchmark_results_cloud_figshare-osce', 'google'),
    ('benchmark_results_cloud_figshare-osce-subset', 'assemblyai'),
    ('benchmark_results_cloud_figshare-osce-subset', 'aws'),
    ('benchmark_results_cloud_figshare-osce-subset', 'azure'),
    ('benchmark_results_cloud_figshare-osce-subset', 'google'),
]


def score(args: tuple) -> dict | None:
    subdir, provider, fid, ref_raw, hyp = args
    ref_fixed = inject_apostrophes(ref_raw)
    ref_w = WNORM(ref_fixed)
    ref_buggy_w = WNORM(ref_raw)
    if not ref_w:
        return None
    n = len(ref_w.split())
    n_buggy = len(ref_buggy_w.split())
    hyp_w = WNORM(hyp)
    fixed_err = round(jiwer.wer(ref_w, hyp_w) * n)
    buggy_err = round(jiwer.wer(ref_buggy_w, hyp_w) * n_buggy) if n_buggy else 0
    return {
        'subdir': subdir,
        'provider': provider,
        'fid': fid,
        'words_fixed': n,
        'words_buggy': n_buggy,
        'fixed_err': fixed_err,
        'buggy_err': buggy_err,
    }


def build_tasks() -> list[tuple]:
    tasks: list[tuple] = []
    for subdir, provider in SOURCES:
        p = OSSICLES / subdir / f'{provider}.json'
        if not p.exists():
            print(f'[skip] {subdir}/{provider}.json missing', flush=True)
            continue
        d = json.loads(p.read_text())
        for r in d.get('results', []):
            fid = r.get('file_id')
            ref = r.get('reference', '')
            hyp = r.get('hypothesis', '')
            if fid and ref and hyp:
                tasks.append((subdir, provider, fid, ref, hyp))
    return tasks


def main() -> None:
    tasks = build_tasks()
    print(f'[plan] {len(tasks)} (subdir, provider, file) tasks', flush=True)

    pf = (RESULTS / 'figshare_cloud_fix_per_file.tsv').open('w')
    pf.write('subdir\tprovider\tfid\twords_fixed\twords_buggy\tfixed_err\tbuggy_err\n')
    pf.flush()

    agg: dict[tuple, dict] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(score, t) for t in tasks]
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            if res is None:
                continue
            key = (res['subdir'], res['provider'])
            a = agg.setdefault(key, {'wf': 0, 'wb': 0, 'ef': 0, 'eb': 0, 'n': 0})
            a['wf'] += res['words_fixed']
            a['wb'] += res['words_buggy']
            a['ef'] += res['fixed_err']
            a['eb'] += res['buggy_err']
            a['n'] += 1
            pf.write(
                f'{res["subdir"]}\t{res["provider"]}\t{res["fid"]}\t'
                f'{res["words_fixed"]}\t{res["words_buggy"]}\t'
                f'{res["fixed_err"]}\t{res["buggy_err"]}\n'
            )
            pf.flush()
            if done % 50 == 0 or done == len(tasks):
                print(f'[progress] {done}/{len(tasks)}', flush=True)
    pf.close()

    sf = (RESULTS / 'figshare_cloud_fix_summary.tsv').open('w')
    sf.write('subdir\tprovider\tfiles\tfixed_wer\tbuggy_wer\tdelta_pp\n')
    sf.flush()

    print('', flush=True)
    print('=== CLOUD API WER (figshare-osce, apostrophe-injected ref) ===', flush=True)
    print(f'{"subdir":<48}{"provider":<14}{"files":>7}{"fixed":>10}{"buggy":>10}{"delta":>10}', flush=True)
    for (subdir, provider), a in sorted(agg.items()):
        if a['wf'] == 0:
            continue
        fixed = a['ef'] / a['wf'] * 100
        buggy = a['eb'] / a['wb'] * 100 if a['wb'] else 0
        delta = fixed - buggy
        print(
            f'{subdir:<48}{provider:<14}{a["n"]:>7}'
            f'{fixed:>9.2f}%{buggy:>9.2f}%{delta:>+9.2f}',
            flush=True,
        )
        sf.write(f'{subdir}\t{provider}\t{a["n"]}\t{fixed:.2f}\t{buggy:.2f}\t{delta:+.2f}\n')
        sf.flush()
    sf.close()
    print('', flush=True)
    print(f'[done] per-file: {RESULTS}/figshare_cloud_fix_per_file.tsv', flush=True)
    print(f'[done] summary:  {RESULTS}/figshare_cloud_fix_summary.tsv', flush=True)


if __name__ == '__main__':
    main()
