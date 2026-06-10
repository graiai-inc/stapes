#!/usr/bin/env python3
"""Move bracketed reference citations after punctuation, per JAMIA style.

JAMIA General Instructions: "Reference numbers in the text should be inserted
immediately after punctuation (with no word spacing)—for example,[6] not [6]."
and multiple references "should be separated by a comma, for example,[1, 4, 39]".

Transforms paper/full_manuscript.md in place:
    'burnout [1,2].'  -> 'burnout.[1, 2]'
    'satisfaction [3,4],' -> 'satisfaction,[3, 4]'
    'dataset [10] contains' -> 'dataset[10] contains'   (no word spacing)

Only integer-content brackets are touched; bootstrap CI brackets like
[91.3, 92.6] contain decimals and never match. Logs every change
incrementally to results/_audit/citation_placement_changes.txt.
"""

import re
from pathlib import Path

ROOT = Path('/home/grey/dev/graiai/stapes')
SRC = ROOT / 'paper' / 'full_manuscript.md'
OUT = ROOT / 'results' / '_audit' / 'citation_placement_changes.txt'
OUT.parent.mkdir(parents=True, exist_ok=True)

CITE = r'\[(\d+(?:\s*[,\-]\s*\d+)*)\]'

fh = open(OUT, 'w')


def emit(line: str = '') -> None:
    fh.write(line + '\n')
    fh.flush()
    print(line, flush=True)


def space_refs(grp: str) -> str:
    """Normalize '1,2' -> '1, 2' (commas spaced, hyphens untouched)."""
    return re.sub(r'\s*,\s*', ', ', grp)


text = SRC.read_text()
emit('=== citation placement fixes ===')

n_moved = 0
n_tightened = 0


def move_after_punct(m: re.Match) -> str:
    global n_moved
    refs, punct = m.group(1), m.group(2)
    new = f'{punct}[{space_refs(refs)}]'
    emit(f'[move] "{m.group(0)}" -> "{new}"')
    n_moved += 1
    return new


def tighten_space(m: re.Match) -> str:
    global n_tightened
    new = f'[{space_refs(m.group(1))}]'
    emit(f'[tighten] "{m.group(0)}" -> "{new}"')
    n_tightened += 1
    return new


# 1. citation (optionally space-preceded) directly before punctuation:
#    'word [1,2].' -> 'word.[1, 2]'
text = re.sub(r'\s*' + CITE + r'([.,;:])', move_after_punct, text)

# 2. remaining mid-sentence citations with a preceding space: 'word [10] ' -> 'word[10] '
text = re.sub(r' ' + CITE, tighten_space, text)

SRC.write_text(text)
emit()
emit(f'[done] moved-after-punctuation: {n_moved}, space-tightened: {n_tightened}')
emit(f'[done] wrote {SRC}')
fh.close()
