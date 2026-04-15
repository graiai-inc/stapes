#!/usr/bin/env python3
"""Count contraction vs bare-word frequency for ambiguous tokens in figshare hyps.

For each ambiguous token (its/it's, well/we'll, were/we're, etc), count how
often the apostrophe form appears vs the bare form across all figshare-osce
model hypotheses. If the apostrophe form dominates (>50%), it's safe to
inject the apostrophe into the REF for WER computation.
"""

import json
import re
from collections import Counter
from pathlib import Path

FIGSHARE = Path('/home/grey/dev/graiai/ossicles/benchmark_results_figshare-osce')
OUT = Path('/home/grey/dev/graiai/stapes/results/figshare_ambig_hyp_stats.tsv')
OUT.parent.mkdir(parents=True, exist_ok=True)

MODELS = [
    'parakeet-tdt-0.6b-v2',
    'whisper-distil-v3.5',
    'whisper-turbo',
    'sensevoice',
    'sensevoice-no-itn',
    'qwen3-asr',
]

# (contraction_form, bare_form, contraction_expansion)
PAIRS = [
    ("it's",    'its',  "it is"),
    ("we'll",   'well', "we will"),
    ("we're",   'were', "we are"),
    ("i'll",    'ill',  "i will"),
    ("i'd",     'id',   "i would"),
    ("let's",   'lets', "let us"),
    ("we'd",    'wed',  "we would"),
]

# Match word boundaries, case-insensitive, apostrophe optional
def mk_pattern(form: str) -> re.Pattern:
    esc = re.escape(form)
    return re.compile(r"(?<![\w'])" + esc + r"(?![\w'])", re.IGNORECASE)


def main() -> None:
    fh = OUT.open('w')
    fh.write('token\tmodel\tcontraction_count\tbare_count\tpct_contraction\n')
    fh.flush()

    # Per-token totals across all models
    totals_contr: Counter = Counter()
    totals_bare: Counter = Counter()

    for model in MODELS:
        path = FIGSHARE / f'{model}.json'
        if not path.exists():
            print(f'[skip] {model}', flush=True)
            continue
        data = json.loads(path.read_text())
        full_hyp = ' '.join(r.get('hypothesis', '') for r in data['results'])

        for contr, bare, _ in PAIRS:
            c_pat = mk_pattern(contr)
            b_pat = mk_pattern(bare)
            c_n = len(c_pat.findall(full_hyp))
            b_n = len(b_pat.findall(full_hyp))
            total = c_n + b_n
            pct = (c_n / total * 100) if total else 0.0
            totals_contr[contr] += c_n
            totals_bare[contr] += b_n
            line = f'{contr:<8}  {model:<28}  contr={c_n:>5}  bare={b_n:>5}  pct_contr={pct:>5.1f}%'
            print(line, flush=True)
            fh.write(f'{contr}\t{model}\t{c_n}\t{b_n}\t{pct:.1f}\n')
            fh.flush()
        print('', flush=True)

    print('=== TOTALS (all models combined) ===', flush=True)
    for contr, bare, _ in PAIRS:
        c_n = totals_contr[contr]
        b_n = totals_bare[contr]
        total = c_n + b_n
        pct = (c_n / total * 100) if total else 0.0
        line = f'{contr:<8} vs {bare:<6}  contr={c_n:>6}  bare={b_n:>6}  pct_contr={pct:>5.1f}%'
        print(line, flush=True)
        fh.write(f'{contr}\tTOTAL\t{c_n}\t{b_n}\t{pct:.1f}\n')
        fh.flush()

    fh.close()


if __name__ == '__main__':
    main()
