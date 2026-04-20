#!/usr/bin/env python3
"""Check how many disfluency-like tokens survive whisper-normalizer in Google transcripts.

For each of N sample files per dataset, report:
  - filler tokens the normalizer did NOT strip
  - a word-level diff of extra-in-hypothesis tokens grouped by category

Writes per-file results incrementally to results/google_disfluency_audit.tsv.

Run with lens venv: /home/grey/dev/graiai/lens/venv/bin/python scripts/google_disfluency_audit.py
"""

import collections
import json
import re
import sys
from pathlib import Path

from whisper_normalizer.english import EnglishTextNormalizer

STAPES = Path('/home/grey/dev/graiai/stapes')
OSSICLES = STAPES.parent / 'ossicles'
OUT_PATH = STAPES / 'results' / 'google_disfluency_audit.tsv'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

DATASETS = ['figshare-osce', 'primock57', 'nazmulkazi']
SAMPLE_PER_DATASET = 20

WNORM = EnglishTextNormalizer()

FILLER_CANDIDATES = [
    # single-token fillers the whisper normalizer DOES strip
    'uh', 'um', 'hmm', 'mm', 'mhm', 'mmm',
    # candidates the whisper normalizer does NOT strip
    'er', 'eh', 'ah', 'oh', 'okay', 'ok', 'yeah', 'yep', 'nope', 'nah', 'right',
    'like', 'so', 'well', 'you know', 'i mean', 'sort of', 'kind of', 'basically',
    'alright', 'gotcha', 'sure', 'true',
]


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z' ]", ' ', text)
    return text.split()


def main() -> int:
    fh = open(OUT_PATH, 'w')
    fh.write('dataset\tfile_id\tref_len_norm\thyp_len_norm\thyp_minus_ref\tfiller_hits_pre_norm\tfiller_hits_post_norm\ttop_extra_tokens\n')
    fh.flush()

    for ds in DATASETS:
        path = OSSICLES / f'benchmark_results_cloud_{ds}' / 'google.json'
        if not path.exists():
            print(f'[skip] {path} missing', flush=True)
            continue
        data = json.loads(path.read_text())
        results = data.get('results', [])
        print(f'[load] {ds}: {len(results)} files', flush=True)

        for i, r in enumerate(results[:SAMPLE_PER_DATASET]):
            ref = r.get('reference', '')
            hyp = r.get('hypothesis', '')
            if not ref or not hyp:
                continue

            ref_tokens_raw = tokenize(ref)
            hyp_tokens_raw = tokenize(hyp)

            ref_norm = WNORM(ref)
            hyp_norm = WNORM(hyp)
            ref_tokens_norm = tokenize(ref_norm)
            hyp_tokens_norm = tokenize(hyp_norm)

            # Count fillers surviving each side after normalization
            hyp_counter_raw = collections.Counter(hyp_tokens_raw)
            hyp_counter_norm = collections.Counter(hyp_tokens_norm)
            filler_pre = sum(hyp_counter_raw.get(f, 0) for f in FILLER_CANDIDATES)
            filler_post = sum(hyp_counter_norm.get(f, 0) for f in FILLER_CANDIDATES)

            # Find which tokens are extra in hyp vs ref
            ref_counter_norm = collections.Counter(ref_tokens_norm)
            extra = hyp_counter_norm - ref_counter_norm
            top_extra = ', '.join(f'{w}({n})' for w, n in extra.most_common(10))

            fh.write(f'{ds}\t{r.get("file_id", "?")}\t{len(ref_tokens_norm)}\t{len(hyp_tokens_norm)}\t{len(hyp_tokens_norm) - len(ref_tokens_norm)}\t{filler_pre}\t{filler_post}\t{top_extra}\n')
            fh.flush()
            print(
                f'[{ds}] {r.get("file_id", "?"):<16} ref={len(ref_tokens_norm):<5} hyp={len(hyp_tokens_norm):<5} '
                f'delta={len(hyp_tokens_norm) - len(ref_tokens_norm):+5}  fillers_pre={filler_pre:<3} fillers_post={filler_post:<3}  '
                f'top: {top_extra[:80]}',
                flush=True,
            )

    fh.close()
    print(f'[done] {OUT_PATH}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
