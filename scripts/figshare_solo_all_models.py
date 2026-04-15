#!/usr/bin/env python3
"""Corrected figshare-osce solo WER for ALL models in Table 1.

Covers the 8 Table 1 models not already in figshare_fix_and_eval.py:
whisper-base-en, paraformer-en, nemo-fastconformer, nemo-fastconformer-int8,
nemotron, zipformer (libriheavy and zh-en variants — paper's is ambiguous,
so we run both), medasr, medasr-int8. Also re-runs AWS cloud from the
subset directory.

Handles both JSON formats: {results: [...], aggregates: {...}} dict and
bare list-of-dicts (zipformer-libriheavy-large).
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

# (subdir, filename_stem, display_name)
TARGETS = [
    ('benchmark_results_figshare-osce', 'whisper-base-en', 'Whisper base.en'),
    ('benchmark_results_figshare-osce', 'paraformer-en', 'Paraformer-en'),
    ('benchmark_results_figshare-osce', 'nemo-fastconformer', 'NeMo FastConformer'),
    ('benchmark_results_figshare-osce', 'nemo-fastconformer-int8', 'NeMo FastConformer int8'),
    ('benchmark_results_figshare-osce', 'nemotron', 'Nemotron'),
    ('benchmark_results_figshare-osce', 'zipformer-libriheavy-large', 'Zipformer (libriheavy)'),
    ('benchmark_results_figshare-osce', 'zipformer-zh-en', 'Zipformer (zh-en)'),
    ('benchmark_results_figshare-osce', 'medasr', 'MedASR'),
    ('benchmark_results_figshare-osce', 'medasr-int8', 'MedASR int8'),
]


def load_any(path: Path) -> list[dict]:
    """Return a list of per-file dicts regardless of JSON shape."""
    d = json.loads(path.read_text())
    if isinstance(d, dict) and 'results' in d:
        return d['results']
    if isinstance(d, list):
        return d
    return []


def score(args: tuple) -> dict | None:
    display, fid, ref_raw, hyp = args
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
        'display': display,
        'fid': fid,
        'wf': n,
        'wb': n_buggy,
        'ef': fixed_err,
        'eb': buggy_err,
    }


def build_tasks() -> list[tuple]:
    tasks: list[tuple] = []
    for subdir, stem, display in TARGETS:
        p = OSSICLES / subdir / f'{stem}.json'
        if not p.exists():
            print(f'[skip] {p} missing', flush=True)
            continue
        rows = load_any(p)
        for r in rows:
            fid = r.get('file_id')
            ref = r.get('reference', '')
            hyp = r.get('hypothesis', '')
            if fid and ref and hyp:
                tasks.append((display, fid, ref, hyp))
        print(f'[load] {display}: {len(rows)} rows', flush=True)
    return tasks


def main() -> None:
    tasks = build_tasks()
    print(f'[plan] {len(tasks)} tasks', flush=True)

    pf = (RESULTS / 'figshare_solo_all_per_file.tsv').open('w')
    pf.write('display\tfid\twords_fixed\twords_buggy\tfixed_err\tbuggy_err\n')
    pf.flush()

    agg: dict[str, dict] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(score, t) for t in tasks]
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            if res is None:
                continue
            a = agg.setdefault(res['display'], {'wf': 0, 'wb': 0, 'ef': 0, 'eb': 0, 'n': 0})
            a['wf'] += res['wf']
            a['wb'] += res['wb']
            a['ef'] += res['ef']
            a['eb'] += res['eb']
            a['n'] += 1
            pf.write(
                f'{res["display"]}\t{res["fid"]}\t{res["wf"]}\t{res["wb"]}'
                f'\t{res["ef"]}\t{res["eb"]}\n'
            )
            pf.flush()
            if done % 50 == 0 or done == len(tasks):
                print(f'[progress] {done}/{len(tasks)}', flush=True)
    pf.close()

    sf = (RESULTS / 'figshare_solo_all_summary.tsv').open('w')
    sf.write('display\tfiles\tfixed_wer\tbuggy_wer\tdelta_pp\n')
    sf.flush()

    print('', flush=True)
    print('=== SOLO WER (figshare-osce, corrected ref) ===', flush=True)
    print(f'{"model":<32}{"files":>7}{"fixed":>10}{"buggy":>10}{"delta":>10}', flush=True)
    for display in [t[2] for t in TARGETS]:
        a = agg.get(display)
        if not a or a['wf'] == 0:
            continue
        fixed = a['ef'] / a['wf'] * 100
        buggy = a['eb'] / a['wb'] * 100 if a['wb'] else 0
        delta = fixed - buggy
        print(f'{display:<32}{a["n"]:>7}{fixed:>9.2f}%{buggy:>9.2f}%{delta:>+9.2f}', flush=True)
        sf.write(f'{display}\t{a["n"]}\t{fixed:.2f}\t{buggy:.2f}\t{delta:+.2f}\n')
        sf.flush()
    sf.close()
    print('', flush=True)
    print(f'[done] per-file: {RESULTS}/figshare_solo_all_per_file.tsv', flush=True)
    print(f'[done] summary:  {RESULTS}/figshare_solo_all_summary.tsv', flush=True)


if __name__ == '__main__':
    main()
