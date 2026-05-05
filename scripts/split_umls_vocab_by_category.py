#!/usr/bin/env python3
"""Split medical_vocab.tsv (100K all-categories) into per-category files.

Output:
    results/medical_vocab/by_category/<CATEGORY>.tsv

Each file has the same columns as the master vocab:
    term, category, first_tier, first_source

The drug-curated file (medical_vocab_drug_curated.tsv) is the curated subset
of the DRUG split. Other categories remain raw — same noise risks the DRUG
split had before curation. Apply the same build_curated_drug_vocab.py pattern
to extend curation to other categories when fusion expands beyond drugs.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
STAPES_DIR = SCRIPT_DIR.parent
INPUT_TSV = STAPES_DIR / 'results' / 'medical_vocab' / 'medical_vocab.tsv'
OUT_DIR = STAPES_DIR / 'results' / 'medical_vocab' / 'by_category'


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    handles = {}
    counts = Counter()

    with open(INPUT_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        for row in reader:
            cat = row.get('category', '').strip()
            if not cat:
                continue
            if cat not in handles:
                fh = open(OUT_DIR / f'{cat}.tsv', 'w')
                fh.write('\t'.join(fieldnames) + '\n')
                fh.flush()
                handles[cat] = fh
            handles[cat].write('\t'.join(row.get(k, '') for k in fieldnames) + '\n')
            handles[cat].flush()
            counts[cat] += 1

    for fh in handles.values():
        fh.close()

    print(f'\nSplit {sum(counts.values())} terms into {len(counts)} category files:', flush=True)
    for cat, n in counts.most_common():
        print(f'  {cat}: {n} terms → {OUT_DIR / f"{cat}.tsv"}', flush=True)
    print(f'\nDirectory: {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
