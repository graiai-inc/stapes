#!/usr/bin/env python3
"""Recon: find apostrophe-less contraction candidates in figshare-osce refs.

Scans all reference texts in benchmark_results_figshare-osce/*.json, splits
on whitespace, lowercases, and counts tokens that match a candidate list of
contractions (both unambiguous and ambiguous). Output is a per-token TSV
written incrementally.

The output informs the apostrophe-injection dictionary used to fix figshare
WER computation.
"""

import json
from collections import Counter
from pathlib import Path

OSSICLES = Path('/home/grey/dev/graiai/ossicles')
FIGSHARE = OSSICLES / 'benchmark_results_figshare-osce'

# Tokens to check — lowercase, no apostrophes
# Unambiguous: no other English word collides
UNAMBIGUOUS = {
    'im': "i'm",
    'ive': "i've",
    'youre': "you're",
    'youve': "you've",
    'youll': "you'll",
    'youd': "you'd",
    'theyre': "they're",
    'theyve': "they've",
    'theyll': "they'll",
    'theyd': "they'd",
    'weve': "we've",
    'whos': "who's",
    'whats': "what's",
    'thats': "that's",
    'wheres': "where's",
    'whens': "when's",
    'hows': "how's",
    'theres': "there's",
    'heres': "here's",
    'hes': "he's",
    'shes': "she's",
    'dont': "don't",
    'doesnt': "doesn't",
    'didnt': "didn't",
    'cant': "can't",
    'couldnt': "couldn't",
    'wouldnt': "wouldn't",
    'shouldnt': "shouldn't",
    'isnt': "isn't",
    'wasnt': "wasn't",
    'arent': "aren't",
    'werent': "weren't",
    'hasnt': "hasn't",
    'havent': "haven't",
    'hadnt': "hadn't",
    'wont': "won't",
    'aint': "ain't",
    'wouldve': "would've",
    'couldve': "could've",
    'shouldve': "should've",
    'mightve': "might've",
    'mustve': "must've",
    'neednt': "needn't",
    'mustnt': "mustn't",
    'shant': "shan't",
}
# Ambiguous: valid English word collides with contraction
AMBIGUOUS = {
    'its': "it's / its",
    'were': "we're / were",
    'well': "we'll / well",
    'ill': "i'll / ill",
    'wed': "we'd / wed",
    'lets': "let's / lets",
    'hed': "he'd / hed",
    'shed': "she'd / shed",
    'id': "i'd / id",
}

OUT = Path('/home/grey/dev/graiai/stapes/results/figshare_contractions_recon.tsv')
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    counts_un: Counter = Counter()
    counts_am: Counter = Counter()
    files_seen = 0
    tokens_seen = 0

    fh = OUT.open('w')
    fh.write('category\ttoken\texpansion\tcount\n')
    fh.flush()

    # Use one model's json as the source of refs (all models share refs)
    src = FIGSHARE / 'parakeet-tdt-0.6b-v2.json'
    d = json.loads(src.read_text())
    for r in d['results']:
        ref = r.get('reference', '')
        if not ref:
            continue
        files_seen += 1
        for tok in ref.lower().split():
            # Strip non-alpha edges
            core = ''.join(c for c in tok if c.isalpha())
            if not core:
                continue
            tokens_seen += 1
            if core in UNAMBIGUOUS:
                counts_un[core] += 1
            elif core in AMBIGUOUS:
                counts_am[core] += 1
        print(f'[scan] files={files_seen} tokens={tokens_seen}', flush=True)

    print('', flush=True)
    print(f'=== UNAMBIGUOUS contractions ({len(counts_un)} unique) ===', flush=True)
    for tok, n in counts_un.most_common():
        line = f'{tok:<12}-> {UNAMBIGUOUS[tok]:<12}  ({n})'
        print(line, flush=True)
        fh.write(f'unambiguous\t{tok}\t{UNAMBIGUOUS[tok]}\t{n}\n')
        fh.flush()

    print('', flush=True)
    print(f'=== AMBIGUOUS (valid word collision) ({len(counts_am)} unique) ===', flush=True)
    for tok, n in counts_am.most_common():
        line = f'{tok:<12}-> {AMBIGUOUS[tok]:<20}  ({n})'
        print(line, flush=True)
        fh.write(f'ambiguous\t{tok}\t{AMBIGUOUS[tok]}\t{n}\n')
        fh.flush()

    fh.close()
    print('', flush=True)
    print(f'[done] {files_seen} refs, {tokens_seen} tokens', flush=True)
    print(f'[done] wrote {OUT}', flush=True)


if __name__ == '__main__':
    main()
