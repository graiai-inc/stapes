#!/usr/bin/env python3
"""Read-only audit of cloud ASR results.

Writes per-row TSVs incrementally. No paper edits. No API calls."""

import json
import re
from pathlib import Path

OSSICLES = Path('/home/grey/dev/graiai/ossicles')
OUT = Path('/home/grey/dev/graiai/stapes/results/_audit')
OUT.mkdir(parents=True, exist_ok=True)

SUBDIRS = [
    ('benchmark_results_cloud_figshare-osce', ['assemblyai', 'azure', 'deepgram', 'google']),
    ('benchmark_results_cloud_primock57', ['assemblyai', 'aws', 'azure', 'deepgram', 'google']),
    ('benchmark_results_cloud_nazmulkazi', ['assemblyai', 'aws', 'azure', 'deepgram', 'google']),
]

DISFLUENCIES = {'um', 'uh', 'uhh', 'umm', 'hmm', 'mm', 'er', 'mhm', 'mm-hmm', 'uh-huh',
                'yeah', 'ah', 'oh'}


def count_words(s: str) -> int:
    return len(s.split())


def disfluency_count(s: str) -> int:
    tokens = re.findall(r"[A-Za-z\-']+", s.lower())
    return sum(1 for t in tokens if t in DISFLUENCIES)


def main():
    summary = (OUT / 'summary.tsv').open('w')
    summary.write('subdir\tprovider\tfiles\tref_words_total\thyp_words_total\t'
                  'mean_ref_words\tmean_hyp_words\tavg_disfl_ref\tavg_disfl_hyp\n')
    summary.flush()

    per_file = (OUT / 'per_file_samples.tsv').open('w')
    per_file.write('subdir\tprovider\tfid\tref_words\thyp_words\tdisfl_ref\tdisfl_hyp\t'
                   'hyp_excerpt\n')
    per_file.flush()

    # For the disfluency claim specifically: sample first 3 files per provider/subdir
    for subdir, providers in SUBDIRS:
        for prov in providers:
            p = OSSICLES / subdir / f'{prov}.json'
            if not p.exists():
                print(f'[skip] {p} missing', flush=True)
                continue
            d = json.loads(p.read_text())
            results = d.get('results', [])
            n = len(results)
            if n == 0:
                continue
            ref_total = 0
            hyp_total = 0
            df_ref_total = 0
            df_hyp_total = 0
            sampled = 0
            for r in results:
                fid = r.get('file_id', '')
                ref = r.get('reference', '') or ''
                hyp = r.get('hypothesis', '') or ''
                rw = count_words(ref)
                hw = count_words(hyp)
                dr = disfluency_count(ref)
                dh = disfluency_count(hyp)
                ref_total += rw
                hyp_total += hw
                df_ref_total += dr
                df_hyp_total += dh
                if sampled < 3:
                    # write an excerpt of the first 40 words
                    ex = ' '.join(hyp.split()[:40]).replace('\n', ' ').replace('\t', ' ')
                    per_file.write(f'{subdir}\t{prov}\t{fid}\t{rw}\t{hw}\t{dr}\t{dh}\t{ex}\n')
                    per_file.flush()
                    sampled += 1
            summary.write(
                f'{subdir}\t{prov}\t{n}\t{ref_total}\t{hyp_total}\t'
                f'{ref_total / n:.1f}\t{hyp_total / n:.1f}\t'
                f'{df_ref_total / n:.2f}\t{df_hyp_total / n:.2f}\n'
            )
            summary.flush()
            print(f'[done] {subdir}/{prov}: n={n} ref={ref_total} hyp={hyp_total} '
                  f'df_ref={df_ref_total} df_hyp={df_hyp_total}', flush=True)

    summary.close()
    per_file.close()
    print(f'[written] {OUT}/summary.tsv', flush=True)
    print(f'[written] {OUT}/per_file_samples.tsv', flush=True)


if __name__ == '__main__':
    main()
