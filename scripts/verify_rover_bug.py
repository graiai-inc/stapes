#!/usr/bin/env python3
"""Verify the apostrophe-stripping bug's impact on stapes paper numbers.

Re-runs rover_fuse with a fix (preserve apostrophes in tokenization),
compares against the buggy version, and reports the delta on the model
pairs reported in the paper.

The bug: rover_fusion.py:44 uses re.sub(r'[^\\w\\s]', '', text), which
strips apostrophes. "I'm" -> "im". The reference is WNORM'd (expands
"I'm" -> "i am"). Every contraction in the ref becomes 2 false errors
in the fused WER. Solo WERs are unaffected because they use raw hyp
text with apostrophes intact.

Parallelism: we dispatch (dataset, pair, file_id) tasks to a process
pool. Each task computes per-file errors for BOTH the buggy and fixed
fuser and writes one per-file TSV row as soon as it returns. A second
aggregate TSV is written per-pair after all its files are in. No
buffering — every row is flushed immediately so partial progress is
recoverable if we crash.
"""

import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rover_fusion as rf  # existing buggy module

WNORM = EnglishTextNormalizer()
OSSICLES = Path('/home/grey/dev/graiai/ossicles')

RESULTS_DIR = Path('/home/grey/dev/graiai/stapes/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PER_FILE_TSV = RESULTS_DIR / 'rover_bug_per_file.tsv'
SUMMARY_TSV = RESULTS_DIR / 'rover_bug_verification.tsv'


# ── Fixed fuser: preserves apostrophes so WNORM can expand contractions ──

def tokenize_fixed(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s']", '', text)
    return [w for w in text.split() if w]


def rover_fuse_fixed(text1: str, text2: str, w1: float = 0.5, w2: float = 0.5) -> str:
    words1 = tokenize_fixed(text1)
    words2 = tokenize_fixed(text2)
    if not words1:
        return text2
    if not words2:
        return text1
    alignment = rf._align_sequences(words1, words2)
    out = []
    for word1, word2, _, _ in alignment:
        cand: dict[str, float] = {}
        if word1 is not None:
            cand[word1] = cand.get(word1, 0) + 0.8 * w1
        if word2 is not None:
            cand[word2] = cand.get(word2, 0) + 0.8 * w2
        if cand:
            out.append(max(cand, key=cand.get))
    return ' '.join(out)


# ── Data loading ──

def load(dataset: str, model: str) -> dict | None:
    p = OSSICLES / f'benchmark_results_{dataset}' / f'{model}.json'
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if not (isinstance(d, dict) and 'results' in d):
        return None
    return {r['file_id']: r for r in d['results'] if 'file_id' in r}


# Top pairs from paper (Figure 2 / text) and a few that lost
PAIRS = [
    ('parakeet-tdt-0.6b-v2', 'sensevoice'),
    ('parakeet-tdt-0.6b-v2', 'sensevoice-no-itn'),
    ('parakeet-tdt-0.6b-v2', 'whisper-distil-v3.5'),
    ('whisper-distil-v3.5', 'sensevoice'),
    ('whisper-distil-v3.5', 'sensevoice-no-itn'),
    ('whisper-distil-v3.5', 'qwen3-asr'),
    ('whisper-distil-v3.5', 'whisper-turbo'),
    ('parakeet-tdt-0.6b-v2', 'qwen3-asr'),
    ('sensevoice', 'sensevoice-no-itn'),
]

DATASETS = ['figshare-osce', 'primock57', 'nazmulkazi']

# Solo WER from paper table 1
SOLO = {
    'figshare-osce': {
        'parakeet-tdt-0.6b-v2': 22.93,
        'whisper-distil-v3.5': 17.59,
        'sensevoice': 18.75,
        'sensevoice-no-itn': 18.99,
        'qwen3-asr': 18.74,
        'whisper-turbo': 17.93,
    },
    'primock57': {
        'parakeet-tdt-0.6b-v2': 13.85,
        'whisper-distil-v3.5': 14.19,
        'sensevoice': 21.01,
        'sensevoice-no-itn': 19.95,
        'qwen3-asr': 14.74,
        'whisper-turbo': 15.53,
    },
    'nazmulkazi': {
        'parakeet-tdt-0.6b-v2': 7.30,
        'whisper-distil-v3.5': 7.35,
        'sensevoice': 7.68,
        'sensevoice-no-itn': 7.63,
        'qwen3-asr': 7.94,
        'whisper-turbo': 7.55,
    },
}


# ── Worker: one (dataset, m1, m2, fid) task ──

def score_one(args: tuple) -> dict | None:
    dataset, m1, m2, fid, hyp1, hyp2, ref = args
    ref_w = WNORM(ref)
    if not ref_w:
        return None
    n = len(ref_w.split())

    buggy_fused = rf.rover_fuse(hyp1, hyp2)
    fixed_fused = rover_fuse_fixed(hyp1, hyp2)
    buggy_err = round(jiwer.wer(ref_w, WNORM(buggy_fused)) * n)
    fixed_err = round(jiwer.wer(ref_w, WNORM(fixed_fused)) * n)
    return {
        'dataset': dataset,
        'm1': m1,
        'm2': m2,
        'fid': fid,
        'words': n,
        'buggy_err': buggy_err,
        'fixed_err': fixed_err,
    }


def build_tasks() -> list[tuple]:
    tasks: list[tuple] = []
    for ds in DATASETS:
        for m1, m2 in PAIRS:
            r1 = load(ds, m1)
            r2 = load(ds, m2)
            if r1 is None or r2 is None:
                print(f'[skip] {ds} {m1}+{m2}: missing results', flush=True)
                continue
            common = sorted(set(r1) & set(r2))
            for fid in common:
                hyp1 = r1[fid].get('hypothesis', '')
                hyp2 = r2[fid].get('hypothesis', '')
                ref = r1[fid].get('reference', '')
                tasks.append((ds, m1, m2, fid, hyp1, hyp2, ref))
    return tasks


def main() -> None:
    tasks = build_tasks()
    print(f'[plan] {len(tasks)} (dataset, pair, file) tasks', flush=True)

    pf = open(PER_FILE_TSV, 'w')
    pf.write('dataset\tm1\tm2\tfid\twords\tbuggy_err\tfixed_err\n')
    pf.flush()

    agg: dict[tuple, dict] = {}  # (ds, m1, m2) -> running sums

    done = 0
    with ProcessPoolExecutor(max_workers=12) as ex:
        futures = [ex.submit(score_one, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if res is None:
                continue
            pf.write(
                f'{res["dataset"]}\t{res["m1"]}\t{res["m2"]}\t{res["fid"]}\t'
                f'{res["words"]}\t{res["buggy_err"]}\t{res["fixed_err"]}\n'
            )
            pf.flush()

            key = (res['dataset'], res['m1'], res['m2'])
            a = agg.setdefault(key, {'words': 0, 'buggy': 0, 'fixed': 0})
            a['words'] += res['words']
            a['buggy'] += res['buggy_err']
            a['fixed'] += res['fixed_err']

            if done % 25 == 0 or done == len(tasks):
                print(f'[progress] {done}/{len(tasks)} files scored', flush=True)

    pf.close()

    # Write per-pair summary
    sf = open(SUMMARY_TSV, 'w')
    sf.write('dataset\tpair\tbuggy_wer\tfixed_wer\tdelta_pp\tbest_solo\tfixed_vs_solo_pp\n')
    sf.flush()

    header = (f'{"dataset":<16}{"pair":<50}{"buggy":>9}{"fixed":>9}'
              f'{"delta":>9}{"best_solo":>13}{"fixed_vs_solo":>17}')
    print('', flush=True)
    print(header, flush=True)
    print('-' * len(header), flush=True)

    last_ds = None
    for ds in DATASETS:
        best_solo = min(SOLO[ds].values())
        for m1, m2 in PAIRS:
            a = agg.get((ds, m1, m2))
            if not a or a['words'] == 0:
                continue
            buggy = a['buggy'] / a['words'] * 100
            fixed = a['fixed'] / a['words'] * 100
            delta = fixed - buggy
            vs_solo = fixed - best_solo
            tag = f'{m1} + {m2}'[:48]
            line = (f'{ds:<16}{tag:<50}{buggy:>8.2f}%{fixed:>8.2f}%'
                    f'{delta:>+8.2f}{best_solo:>12.2f}%{vs_solo:>+15.2f}pp')
            print(line, flush=True)
            sf.write(
                f'{ds}\t{m1}+{m2}\t{buggy:.2f}\t{fixed:.2f}'
                f'\t{delta:+.2f}\t{best_solo:.2f}\t{vs_solo:+.2f}\n'
            )
            sf.flush()
            last_ds = ds
        if last_ds == ds:
            print('', flush=True)
    sf.close()

    print(f'[done] per-file: {PER_FILE_TSV}', flush=True)
    print(f'[done] summary:  {SUMMARY_TSV}', flush=True)


if __name__ == '__main__':
    main()
