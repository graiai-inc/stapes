#!/usr/bin/env python3
"""Renumber references into JAMIA order-of-appearance.

References 1-17 are already in order; 18-27 are not. This applies a fixed
old->new mapping to all in-text [n] citation tokens in full_manuscript.md and
to references.md (both the annotation block and the Formatted Reference List,
which is re-sorted by the new number).
"""

import re
from pathlib import Path

PAPER = Path('/home/grey/dev/graiai/stapes/paper')

# old -> new (identity for 1-17, derived from order of first appearance)
MAP = {n: n for n in range(1, 18)}
MAP.update({23: 18, 24: 19, 22: 20, 27: 21, 19: 22, 18: 23, 25: 24, 26: 25, 20: 26, 21: 27})


def remap_token(match: re.Match) -> str:
    """Remap a [..] citation token, expanding ranges, sorting ascending."""
    inner = match.group(1)
    nums = []
    for part in inner.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            nums.extend(range(int(a), int(b) + 1))
        else:
            nums.append(int(part))
    new = sorted(MAP[n] for n in nums)
    return '[' + ','.join(str(n) for n in new) + ']'


# Only treat [..] as a citation if it is purely digits/commas/dashes.
CITE = re.compile(r'\[(\d+(?:\s*[,\-]\s*\d+)*)\]')


def process(text: str) -> str:
    return CITE.sub(remap_token, text)


# --- full_manuscript.md ---
fm_path = PAPER / 'full_manuscript.md'
fm = fm_path.read_text()
fm_new = process(fm)
fm_path.write_text(fm_new)
print(f'[full_manuscript] rewrote, {fm.count("[") - fm_new.count("[")} bracket delta', flush=True)

# --- references.md ---
refs_path = PAPER / 'references.md'
refs = refs_path.read_text()
marker = '## Formatted Reference List'
head, _, tail = refs.partition(marker)

# Remap citation tokens in the annotation head.
head_new = process(head)

# Parse the formatted list (subtitle line + numbered entries).
# tail starts with " (NEJM AI style)\n\n1. ...."
subtitle_match = re.match(r'(.*?)\n', tail)
subtitle = subtitle_match.group(1)
body = tail[len(subtitle) + 1:]

entries = {}
for m in re.finditer(r'(?m)^(\d+)\.\s(.*)$', body):
    old = int(m.group(1))
    entries[MAP[old]] = m.group(2).strip()

lines = [marker + subtitle, '']
for n in sorted(entries):
    lines.append(f'{n}. {entries[n]}')
    lines.append('')
refs_new = head_new + '\n'.join(lines).rstrip() + '\n'
refs_path.write_text(refs_new)
print(f'[references] rewrote {len(entries)} entries, renumbered to order-of-appearance', flush=True)
print('[done]', flush=True)
