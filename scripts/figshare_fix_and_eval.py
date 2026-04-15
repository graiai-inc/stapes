#!/usr/bin/env python3
"""Fix figshare-osce reference corruption and recompute solo + fused WER.

The figshare-osce reference text has apostrophes stripped from contractions
("IM" for "I'm", "DONT" for "don't", etc.). This breaks WER computation when
the hyp is WNORM'd because WhisperNormalizer expands "i'm" -> "i am" while
the corrupted ref stays "im". Every contraction becomes an error.

Fix: inject apostrophes into ref tokens via a dictionary of 39 unambiguous
contractions observed in the data. Then run WNORM on both sides uniformly.
Ambiguous tokens (its, well, were, ill, id, lets, wed) are left alone —
they contribute bounded residual error.

This script recomputes:
  - Solo WER for each model on figshare-osce
  - Fused WER for each model pair on figshare-osce (fixed tokenizer)

with the reference-injection patch applied. Per-file incremental writes.
"""

import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rover_fusion as rf

WNORM = EnglishTextNormalizer()
OSSICLES = Path('/home/grey/dev/graiai/ossicles')
FIGSHARE = OSSICLES / 'benchmark_results_figshare-osce'
RESULTS = Path('/home/grey/dev/graiai/stapes/results')
RESULTS.mkdir(parents=True, exist_ok=True)

# 39 unambiguous contractions from figshare_contractions_recon.tsv
# + 5 ambiguous tokens validated via figshare_ambig_hyp_stats.tsv to be >50%
# contraction in the hyps: its (96.8%), ill (89.7%), id (98.0%),
# lets (99.4%), wed (100%). Skipped: well (27.3%) and were (19.1%) —
# the bare word dominates.
CONTRACTIONS = {
    'im': "i'm", 'ive': "i've", 'dont': "don't", 'thats': "that's",
    'havent': "haven't", 'youre': "you're", 'hes': "he's", 'youve': "you've",
    'didnt': "didn't", 'shes': "she's", 'theres': "there's", 'cant': "can't",
    'hasnt': "hasn't", 'doesnt': "doesn't", 'whats': "what's", 'whos': "who's",
    'weve': "we've", 'wasnt': "wasn't", 'theyre': "they're", 'theyve': "they've",
    'couldnt': "couldn't", 'youll': "you'll", 'wouldnt': "wouldn't", 'hows': "how's",
    'isnt': "isn't", 'werent': "weren't", 'theyll': "they'll", 'arent': "aren't",
    'wont': "won't", 'youd': "you'd", 'shouldnt': "shouldn't", 'hadnt': "hadn't",
    'whens': "when's", 'wheres': "where's", 'couldve': "could've", 'mustve': "must've",
    'shouldve': "should've", 'wouldve': "would've", 'theyd': "they'd",
    # Hyp-validated ambiguous tokens (>50% contraction in figshare hypotheses)
    'its': "it's", 'ill': "i'll", 'id': "i'd", 'lets': "let's", 'wed': "we'd",
}

TOKEN_RE = re.compile(r"\b([A-Za-z]+)\b")


def inject_apostrophes(text: str) -> str:
    """Replace apostrophe-less contractions with their apostrophe forms."""
    def _repl(m: re.Match) -> str:
        tok = m.group(1)
        low = tok.lower()
        return CONTRACTIONS.get(low, tok)
    return TOKEN_RE.sub(_repl, text)


def tokenize_fixed(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s']", '', text)
    return [w for w in text.split() if w]


def rover_fuse_fixed(text1: str, text2: str) -> str:
    w1 = tokenize_fixed(text1)
    w2 = tokenize_fixed(text2)
    if not w1:
        return text2
    if not w2:
        return text1
    alignment = rf._align_sequences(w1, w2)
    out = []
    for a, b, _, _ in alignment:
        cand: dict[str, float] = {}
        if a is not None:
            cand[a] = cand.get(a, 0) + 0.4
        if b is not None:
            cand[b] = cand.get(b, 0) + 0.4
        if cand:
            out.append(max(cand, key=cand.get))
    return ' '.join(out)


def load(model: str) -> dict | None:
    p = FIGSHARE / f'{model}.json'
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {r['file_id']: r for r in d['results'] if 'file_id' in r}


MODELS = [
    'parakeet-tdt-0.6b-v2',
    'whisper-distil-v3.5',
    'whisper-turbo',
    'sensevoice',
    'sensevoice-no-itn',
    'qwen3-asr',
]

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


def score_solo(args: tuple) -> dict | None:
    model, fid, ref_raw, hyp = args
    ref_fixed = inject_apostrophes(ref_raw)
    ref_w = WNORM(ref_fixed)
    ref_buggy_w = WNORM(ref_raw)  # original corrupted reference
    if not ref_w:
        return None
    n = len(ref_w.split())
    fixed_err = round(jiwer.wer(ref_w, WNORM(hyp)) * n)
    n_buggy = len(ref_buggy_w.split())
    buggy_err = round(jiwer.wer(ref_buggy_w, WNORM(hyp)) * n_buggy) if n_buggy else 0
    return {
        'kind': 'solo',
        'model': model,
        'fid': fid,
        'words_fixed': n,
        'words_buggy': n_buggy,
        'fixed_err': fixed_err,
        'buggy_err': buggy_err,
    }


def score_pair(args: tuple) -> dict | None:
    m1, m2, fid, ref_raw, hyp1, hyp2 = args
    ref_fixed = inject_apostrophes(ref_raw)
    ref_w = WNORM(ref_fixed)
    ref_buggy_w = WNORM(ref_raw)
    if not ref_w:
        return None
    n = len(ref_w.split())
    n_buggy = len(ref_buggy_w.split())

    buggy_fused = rf.rover_fuse(hyp1, hyp2)
    fixed_fused = rover_fuse_fixed(hyp1, hyp2)
    # Buggy: compare buggy-fused to buggy-ref (matches paper computation)
    buggy_err = round(jiwer.wer(ref_buggy_w, WNORM(buggy_fused)) * n_buggy) if n_buggy else 0
    # Fixed: compare fixed-fused to ref-injected (proper computation)
    fixed_err = round(jiwer.wer(ref_w, WNORM(fixed_fused)) * n)
    return {
        'kind': 'pair',
        'm1': m1,
        'm2': m2,
        'fid': fid,
        'words_fixed': n,
        'words_buggy': n_buggy,
        'fixed_err': fixed_err,
        'buggy_err': buggy_err,
    }


def build_tasks() -> tuple[list, list]:
    solo_tasks = []
    pair_tasks = []
    loaded: dict[str, dict] = {}
    for m in MODELS:
        r = load(m)
        if r is None:
            print(f'[skip] missing {m}', flush=True)
            continue
        loaded[m] = r

    # Use first available model's refs as canonical ref source
    ref_source = next(iter(loaded.values()))

    for m, results in loaded.items():
        for fid, row in results.items():
            ref = ref_source.get(fid, {}).get('reference', '')
            hyp = row.get('hypothesis', '')
            if ref and hyp:
                solo_tasks.append((m, fid, ref, hyp))

    for m1, m2 in PAIRS:
        if m1 not in loaded or m2 not in loaded:
            continue
        r1 = loaded[m1]
        r2 = loaded[m2]
        common = sorted(set(r1) & set(r2))
        for fid in common:
            ref = r1[fid].get('reference', '')
            hyp1 = r1[fid].get('hypothesis', '')
            hyp2 = r2[fid].get('hypothesis', '')
            if ref and hyp1 and hyp2:
                pair_tasks.append((m1, m2, fid, ref, hyp1, hyp2))
    return solo_tasks, pair_tasks


def main() -> None:
    solo_tasks, pair_tasks = build_tasks()
    print(f'[plan] {len(solo_tasks)} solo + {len(pair_tasks)} pair tasks', flush=True)

    per_file = (RESULTS / 'figshare_fix_per_file.tsv').open('w')
    per_file.write('kind\tmodel_or_pair\tfid\twords_fixed\twords_buggy\tfixed_err\tbuggy_err\n')
    per_file.flush()

    solo_agg: dict[str, dict] = {}
    pair_agg: dict[tuple, dict] = {}

    done = 0
    total = len(solo_tasks) + len(pair_tasks)
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = []
        for t in solo_tasks:
            futs.append(ex.submit(score_solo, t))
        for t in pair_tasks:
            futs.append(ex.submit(score_pair, t))

        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            if res is None:
                continue
            if res['kind'] == 'solo':
                key = res['model']
                a = solo_agg.setdefault(key, {'wf': 0, 'wb': 0, 'ef': 0, 'eb': 0})
                a['wf'] += res['words_fixed']
                a['wb'] += res['words_buggy']
                a['ef'] += res['fixed_err']
                a['eb'] += res['buggy_err']
                per_file.write(
                    f'solo\t{key}\t{res["fid"]}\t{res["words_fixed"]}\t{res["words_buggy"]}'
                    f'\t{res["fixed_err"]}\t{res["buggy_err"]}\n'
                )
            else:
                key = (res['m1'], res['m2'])
                a = pair_agg.setdefault(key, {'wf': 0, 'wb': 0, 'ef': 0, 'eb': 0})
                a['wf'] += res['words_fixed']
                a['wb'] += res['words_buggy']
                a['ef'] += res['fixed_err']
                a['eb'] += res['buggy_err']
                per_file.write(
                    f'pair\t{key[0]}+{key[1]}\t{res["fid"]}\t{res["words_fixed"]}'
                    f'\t{res["words_buggy"]}\t{res["fixed_err"]}\t{res["buggy_err"]}\n'
                )
            per_file.flush()

            if done % 50 == 0 or done == total:
                print(f'[progress] {done}/{total}', flush=True)
    per_file.close()

    summary = (RESULTS / 'figshare_fix_summary.tsv').open('w')
    summary.write('kind\tkey\tfixed_wer\tbuggy_wer\tdelta_pp\n')
    summary.flush()

    print('', flush=True)
    print('=== SOLO WER (figshare-osce) ===', flush=True)
    print(f'{"model":<30}{"fixed":>10}{"buggy":>10}{"delta":>10}', flush=True)
    for m in MODELS:
        a = solo_agg.get(m)
        if not a or a['wf'] == 0:
            continue
        fixed = a['ef'] / a['wf'] * 100
        buggy = a['eb'] / a['wb'] * 100 if a['wb'] else 0
        delta = fixed - buggy
        print(f'{m:<30}{fixed:>9.2f}%{buggy:>9.2f}%{delta:>+9.2f}', flush=True)
        summary.write(f'solo\t{m}\t{fixed:.2f}\t{buggy:.2f}\t{delta:+.2f}\n')
        summary.flush()

    print('', flush=True)
    print('=== FUSED WER (figshare-osce) ===', flush=True)
    print(f'{"pair":<50}{"fixed":>10}{"buggy":>10}{"delta":>10}', flush=True)
    for m1, m2 in PAIRS:
        a = pair_agg.get((m1, m2))
        if not a or a['wf'] == 0:
            continue
        fixed = a['ef'] / a['wf'] * 100
        buggy = a['eb'] / a['wb'] * 100 if a['wb'] else 0
        delta = fixed - buggy
        tag = f'{m1}+{m2}'[:48]
        print(f'{tag:<50}{fixed:>9.2f}%{buggy:>9.2f}%{delta:>+9.2f}', flush=True)
        summary.write(f'pair\t{m1}+{m2}\t{fixed:.2f}\t{buggy:.2f}\t{delta:+.2f}\n')
        summary.flush()

    summary.close()
    print('', flush=True)
    print(f'[done] per-file: {RESULTS}/figshare_fix_per_file.tsv', flush=True)
    print(f'[done] summary:  {RESULTS}/figshare_fix_summary.tsv', flush=True)


if __name__ == '__main__':
    main()
