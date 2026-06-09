#!/usr/bin/env python3
"""Pre-submission QA checks for the stapes manuscript.

Reads the assembled build/stapes_manuscript.md (ground truth of what is
submitted) plus the source full_manuscript.md and CSV tables, and writes a
plain-text report incrementally to build/qa_report.txt.
"""

import csv
import re
from pathlib import Path

ROOT = Path('/home/grey/dev/graiai/stapes')
BUILD = ROOT / 'build'
PAPER = ROOT / 'paper'
OUT = BUILD / 'qa_report.txt'

fh = open(OUT, 'w')


def emit(line: str = '') -> None:
    fh.write(line + '\n')
    fh.flush()
    print(line, flush=True)


manuscript = (BUILD / 'stapes_manuscript.md').read_text()
full = (PAPER / 'full_manuscript.md').read_text()

emit('=== STAPES PRE-SUBMISSION QA ===')
emit()

# --- 1. Em-dashes ---
for label, text in [('build manuscript', manuscript),
                    ('full_manuscript.md', full),
                    ('references.md', (PAPER / 'references.md').read_text()),
                    ('cover_letter.md', (PAPER / 'cover_letter.md').read_text())]:
    n = text.count('—')
    emit(f'[em-dash] {label}: {n} em-dashes (—)')
emit()

# --- 2. Abstract word count ---
abs = re.search(r'# Abstract\n(.*?)\n# Background', manuscript, re.S)
if abs:
    abs_text = abs.group(1)
    # strip the bold field labels like **Objective:**
    abs_words = re.sub(r'\*\*[^*]+\*\*', '', abs_text)
    n_abs = len(abs_words.split())
    emit(f'[abstract] word count (excl. bold labels): {n_abs}')
emit()

# --- 3. Body word count (Background..Conclusion) ---
body = re.search(r'# Background and Significance\n(.*?)\n# Data Availability',
                 manuscript, re.S)
if body:
    btext = body.group(1)
    # remove markdown bold/italic markers and figure/table markup
    clean = re.sub(r'[#*`]', '', btext)
    clean = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', clean)
    n_body = len(clean.split())
    emit(f'[body] Background..Conclusion word count: {n_body}')
    emit('       (title page claims 3,973; JAMIA limit is 4,000)')
emit()

# --- 4. Citation order of first appearance ---
# Get the body+discussion region only (before References section)
ref_idx = manuscript.index('\n# References')
body_region = manuscript[:ref_idx]
cites = re.findall(r'\[(\d+(?:[,\-]\d+)*)\]', body_region)
order = []
for grp in cites:
    for part in grp.split(','):
        if '-' in part:
            a, b = part.split('-')
            order.extend(range(int(a), int(b) + 1))
        else:
            order.append(int(part))
first_seen = []
for c in order:
    if c not in first_seen:
        first_seen.append(c)
emit(f'[citations] order of first appearance: {first_seen}')
out_of_order = [first_seen[i] for i in range(1, len(first_seen))
                if first_seen[i] < first_seen[i - 1]]
emit(f'[citations] count distinct cited: {len(first_seen)}')
# which reference numbers exist in references.md
ref_nums = sorted(int(m) for m in re.findall(r'^(\d+)\.\s', (PAPER / 'references.md').read_text(), re.M))
emit(f'[citations] references defined: {ref_nums}')
cited_set = set(first_seen)
defined_set = set(ref_nums)
emit(f'[citations] cited but undefined: {sorted(cited_set - defined_set)}')
emit(f'[citations] defined but never cited: {sorted(defined_set - cited_set)}')
is_sorted = first_seen == sorted(first_seen)
emit(f'[citations] in strict order-of-appearance? {is_sorted}')
if not is_sorted:
    # show the first descent
    for i in range(1, len(first_seen)):
        if first_seen[i] < first_seen[i - 1]:
            emit(f'           first out-of-order: ...{first_seen[i-1]} then {first_seen[i]}')
            break
emit()

# --- 5. WER gap ranges from CSV (verify "0.5 to 5" / "0.7 to 3") ---
emit('[gaps] recomputing on-device-vs-best-cloud WER gaps from table1_wer.csv')
rows = list(csv.reader((PAPER / 'table1_wer.csv').open()))
hdr = rows[0]
data = rows[1:]
cols = {
    'OSCE Std': 2, 'OSCE MP': 3,
    'PriMock Std': 4, 'PriMock MP': 5,
    'Psych Std': 6, 'Psych MP': 7,
}


def fnum(s: str) -> float:
    return float(s.replace('‡', '').replace('*', '').strip())


def best(rows_subset, col):
    return min(fnum(r[col]) for r in rows_subset)


ondev = [r for r in data if r[1] == 'On-device']
cloud = [r for r in data if r[1].startswith('Cloud')]
std_gaps = []
mp_gaps = []
for name, col in cols.items():
    od = best(ondev, col)
    cl = best(cloud, col)
    gap = round(od - cl, 2)
    emit(f'       {name}: best on-device {od} - best cloud {cl} = {gap} pp')
    if 'Std' in name:
        std_gaps.append(gap)
    else:
        mp_gaps.append(gap)
emit(f'       STD gap range: {min(std_gaps)} to {max(std_gaps)}  (paper says 0.5 to 5.2)')
emit(f'       MP  gap range: {min(mp_gaps)} to {max(mp_gaps)}  (paper says 0.7 to 3.3)')
emit()

# --- 6. CTR gap range (Table 2) ---
emit('[ctr] recomputing on-device-vs-best-cloud CTR gaps from table2_ctr.csv')
rows2 = list(csv.reader((PAPER / 'table2_ctr.csv').open()))
data2 = rows2[1:]


def point(cell: str) -> float:
    return float(cell.split('[')[0].strip())


ond2 = [r for r in data2 if r[1] == 'On-device']
cld2 = [r for r in data2 if r[1].startswith('Cloud')]
for didx, dname in [(2, 'OSCE'), (3, 'PriMock57'), (4, 'Psychiatric')]:
    best_od = max(point(r[didx]) for r in ond2)
    best_cl = max(point(r[didx]) for r in cld2)
    emit(f'       {dname}: best on-device {best_od} vs best cloud {best_cl} = {round(best_cl-best_od,2)} pp')
emit('       (paper says CTR within 1 to 3 pp)')
emit()

emit('=== END QA ===')
fh.close()
