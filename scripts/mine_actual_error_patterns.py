#!/usr/bin/env python3
"""Mine actual error patterns from per_error.tsv to drive fusion + ASR work.

Two analyses:

PART 1 — What conditions/symptoms/findings does Parakeet actually get wrong?
For each medical category, aggregate the reference words to produce a frequency-ranked
list. This is the empirical seed list for the next round of curated vocab — better
than guessing what conditions to include.

PART 2 — How often do high-impact 'generic' errors happen?
Scan the 47K "generic" errors for four high-meaning-change subclasses and quantify
their frequency:
  - NEGATION_FLIP — reference contains a negation token (no/not/nothing/never/neither/n't)
                    that the hypothesis doesn't, or vice versa. Flips clinical fact.
  - LATERALITY    — left↔right (or close variants) substitutions. Wrong-side errors.
  - PRONOUN       — he↔she, his↔her, him↔her, etc. Attribution / deference errors.
  - NUMERIC       — number-token substitutions (digits OR spelled-out). Dose / duration
                    / vitals errors.

Run with the ossicles venv python:
    /home/grey/dev/graiai/ossicles/venv/bin/python \\
        scripts/mine_actual_error_patterns.py
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
STAPES_DIR = SCRIPT_DIR.parent
INPUT_TSV = STAPES_DIR / 'results' / 'parakeet_error_categorization' / 'per_error.tsv'
OUT_DIR = STAPES_DIR / 'results' / 'error_patterns'

# Negation tokens — reference and hypothesis are checked for any membership in this set
NEGATION_TOKENS = {
    'no', 'not', 'none', 'never', 'neither', 'nor', 'nothing', 'nobody',
    'nowhere', 'cannot', "can't", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "shouldn't", "couldn't", "haven't", "hasn't",
    "hadn't", "isn't", "aren't", "wasn't", "weren't",
    'nope', 'nah',  # informal negations — should count as negation polarity matches
}

LATERALITY_TOKENS = {
    'left', 'right', 'lateral', 'bilateral', 'unilateral',
}

PRONOUN_PAIRS = [
    ({'he', 'him', 'his'}, {'she', 'her', 'hers'}),  # M/F gender swap
    ({'he', 'she'}, {'they'}),  # singular/plural
    ({'his', 'her'}, {'their'}),
    ({'i', "i'm", "i've", 'me', 'my'}, {'you', "you're", "you've", 'your'}),  # 1st/2nd person — affects attribution
]

NUMERIC_DIGIT_RE = re.compile(r'\d')
SPELLED_NUMBER_TOKENS = {
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
    'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty',
    'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred',
    'thousand', 'million', 'first', 'second', 'third', 'fourth', 'fifth',
    'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
    'half', 'third', 'quarter', 'double', 'triple',
}


def tokens_of(s: str) -> list[str]:
    return [t.lower().strip("'-") for t in s.split() if t.strip()]


def has_any(tokens: list[str], target: set[str]) -> bool:
    return any(t in target for t in tokens)


def detect_negation_flip(ref_words: list[str], hyp_words: list[str], primary_cat: str) -> bool:
    """Negation flip: one side has a negation token, the other doesn't.

    Filters out chunks whose primary error is medical (drug/condition/etc.) — if the
    chunk happens to contain a negation token along with a misrecognized drug name,
    the real story is the drug error, not the negation. Only count negation flips on
    chunks whose primary category is filler or generic (i.e., conversational content).
    """
    if primary_cat not in ('generic', 'filler', 'FINDING'):
        return False  # don't double-count medical-category errors as negation flips
    r_neg = has_any(ref_words, NEGATION_TOKENS)
    h_neg = has_any(hyp_words, NEGATION_TOKENS)
    return r_neg != h_neg


def detect_laterality(ref_words: list[str], hyp_words: list[str]) -> bool:
    """Laterality flip: one side has left/right, other has the opposite or different.

    Filters out the `alright → all right` normalization artifact (alright is one token,
    splits into 'all right' which contains the bare 'right' token but means nothing about
    laterality)."""
    # Drop "right" if it's adjacent to "all" in either side — that's "all right" / "alright".
    def strip_alright_right(words):
        out = []
        for i, w in enumerate(words):
            if w == 'right' and i > 0 and words[i - 1] == 'all':
                continue
            out.append(w)
        return out
    r_clean = strip_alright_right(ref_words)
    h_clean = strip_alright_right(hyp_words)
    # Also: if either side has 'alright' as a single token, ignore any 'right' on the other side
    if 'alright' in r_clean:
        h_clean = [w for w in h_clean if w != 'right']
    if 'alright' in h_clean:
        r_clean = [w for w in r_clean if w != 'right']
    r_lat = [t for t in r_clean if t in LATERALITY_TOKENS]
    h_lat = [t for t in h_clean if t in LATERALITY_TOKENS]
    if not r_lat and not h_lat:
        return False
    return r_lat != h_lat


def detect_pronoun_swap(ref_words: list[str], hyp_words: list[str]) -> bool:
    """Pronoun swap: ref word in one pronoun group, hyp word in the OTHER group of the same pair."""
    for r in ref_words:
        for h in hyp_words:
            for group_a, group_b in PRONOUN_PAIRS:
                if (r in group_a and h in group_b) or (r in group_b and h in group_a):
                    return True
    return False


def detect_numeric_mismatch(ref_words: list[str], hyp_words: list[str]) -> bool:
    """Numeric mismatch: at least one side has a numeric token, AND ref-numerics != hyp-numerics."""
    def num_set(words):
        out = set()
        for w in words:
            if NUMERIC_DIGIT_RE.search(w) or w in SPELLED_NUMBER_TOKENS:
                out.add(w)
        return out
    r = num_set(ref_words)
    h = num_set(hyp_words)
    if not r and not h:
        return False
    return r != h


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Aggregations
    # Load category vocabs to filter ref words to actually-in-category ones
    sys.path.insert(0, str(SCRIPT_DIR))
    from categorize_parakeet_errors import load_category_vocabs
    cat_vocabs = load_category_vocabs()

    cat_word_freq = defaultdict(Counter)  # category -> Counter(ref_word -> count)
    cat_word_examples = defaultdict(lambda: defaultdict(list))  # category -> ref_word -> [(file, hyp), ...]
    pattern_counts = defaultdict(int)
    pattern_examples = defaultdict(list)  # 50 examples per pattern
    dataset_pattern_counts = defaultdict(lambda: defaultdict(int))

    n_total = 0
    print(f'Reading {INPUT_TSV}', flush=True)
    with open(INPUT_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            n_total += 1
            ref = row['ref_words']
            hyp = row['hyp_words']
            cat = row['primary_category']
            ds = row['dataset']
            fid = row['file_id']

            ref_toks = tokens_of(ref)
            hyp_toks = tokens_of(hyp)

            # PART 1: aggregate medical ref words per category — but ONLY count words
            # that actually match the category's vocab (not the function words that
            # came along for the ride in the chunk).
            if cat in ('CONDITION', 'SYMPTOM', 'FINDING', 'PROCEDURE', 'ANATOMY', 'LAB_RESULT'):
                cat_single = cat_vocabs.get(cat, {}).get('single', set())
                cat_phrases = cat_vocabs.get(cat, {}).get('phrases', set())
                joined = ' '.join(ref_toks)
                # Phrase match counts as one entry under the phrase
                if joined in cat_phrases:
                    cat_word_freq[cat][joined] += 1
                    if len(cat_word_examples[cat][joined]) < 5:
                        cat_word_examples[cat][joined].append({
                            'dataset': ds, 'file_id': fid,
                            'ref': ref, 'hyp': hyp,
                        })
                else:
                    for w in ref_toks:
                        if w in cat_single:
                            cat_word_freq[cat][w] += 1
                            if len(cat_word_examples[cat][w]) < 5:
                                cat_word_examples[cat][w].append({
                                    'dataset': ds, 'file_id': fid,
                                    'ref': ref, 'hyp': hyp,
                                })

            # PART 2: detect high-impact subclasses
            patterns_hit = []
            if detect_negation_flip(ref_toks, hyp_toks, cat):
                patterns_hit.append('NEGATION_FLIP')
            if detect_laterality(ref_toks, hyp_toks):
                patterns_hit.append('LATERALITY')
            if detect_pronoun_swap(ref_toks, hyp_toks):
                patterns_hit.append('PRONOUN')
            if detect_numeric_mismatch(ref_toks, hyp_toks):
                patterns_hit.append('NUMERIC')
            for p in patterns_hit:
                pattern_counts[p] += 1
                dataset_pattern_counts[ds][p] += 1
                if len(pattern_examples[p]) < 50:
                    pattern_examples[p].append({
                        'dataset': ds, 'file_id': fid, 'category': cat,
                        'ref': ref, 'hyp': hyp,
                    })

    # WRITE PART 1: medical reference word frequencies
    fh1 = open(OUT_DIR / 'medical_word_frequencies.tsv', 'w')
    fh1.write('category\trank\tref_word\tn_errors\n')
    fh1.flush()
    for cat in ['DRUG', 'CONDITION', 'SYMPTOM', 'FINDING', 'PROCEDURE', 'ANATOMY', 'LAB_RESULT']:
        for rank, (w, n) in enumerate(cat_word_freq.get(cat, Counter()).most_common(), 1):
            fh1.write(f'{cat}\t{rank}\t{w}\t{n}\n')
            fh1.flush()
    fh1.close()

    # WRITE PART 1 examples — top 30 ref-words per category with example errors
    fh1ex = open(OUT_DIR / 'medical_word_examples.txt', 'w')
    for cat in ['CONDITION', 'SYMPTOM', 'FINDING', 'PROCEDURE', 'ANATOMY']:
        fh1ex.write(f'\n========== {cat} (top 30 ref words by error frequency) ==========\n')
        for rank, (w, n) in enumerate(cat_word_freq.get(cat, Counter()).most_common(30), 1):
            fh1ex.write(f'\n  [{rank:2d}] "{w}" — {n} errors\n')
            for ex in cat_word_examples[cat][w][:3]:
                fh1ex.write(f"        {ex['dataset']:18s} {ex['file_id']:30s} "
                            f"  ref={ex['ref']!r}  parakeet={ex['hyp']!r}\n")
        fh1ex.flush()
    fh1ex.close()

    # WRITE PART 2: high-impact pattern counts and examples
    fh2 = open(OUT_DIR / 'high_impact_patterns.tsv', 'w')
    fh2.write('pattern\tdataset\tn_errors\n')
    fh2.flush()
    for p in ['NEGATION_FLIP', 'LATERALITY', 'PRONOUN', 'NUMERIC']:
        fh2.write(f'{p}\tALL\t{pattern_counts[p]}\n')
        for ds in ['primock57', 'figshare-osce', 'nazmulkazi', 'kokoro-va-sample']:
            fh2.write(f'{p}\t{ds}\t{dataset_pattern_counts[ds][p]}\n')
        fh2.flush()
    fh2.close()

    fh2ex = open(OUT_DIR / 'high_impact_examples.txt', 'w')
    for p in ['NEGATION_FLIP', 'LATERALITY', 'PRONOUN', 'NUMERIC']:
        fh2ex.write(f'\n========== {p} ({pattern_counts[p]} errors) ==========\n')
        for ex in pattern_examples[p]:
            fh2ex.write(f"  {ex['dataset']:18s} {ex['file_id']:30s} cat={ex['category']:10s}"
                        f"  ref={ex['ref']!r}  parakeet={ex['hyp']!r}\n")
        fh2ex.flush()
    fh2ex.close()

    # PRINT SUMMARY
    print(f'\n=== ERROR-PATTERN MINING SUMMARY ===', flush=True)
    print(f'Total errors scanned: {n_total}', flush=True)

    print(f'\n--- PART 1: Top medical ref words by error frequency ---', flush=True)
    for cat in ['CONDITION', 'SYMPTOM', 'FINDING', 'PROCEDURE', 'ANATOMY']:
        wf = cat_word_freq.get(cat, Counter())
        print(f'\n{cat}: {sum(wf.values())} word-tokens across {len(wf)} unique ref words', flush=True)
        print(f'  top 15: ', end='', flush=True)
        print(', '.join(f'{w}({n})' for w, n in wf.most_common(15)), flush=True)

    print(f'\n--- PART 2: High-impact "generic" subclasses ---', flush=True)
    print(f'{"pattern":<16s} {"total":>7s}  {"primock57":>10s}  {"OSCE":>6s}  {"Kazi":>6s}  {"Kokoro":>7s}', flush=True)
    for p in ['NEGATION_FLIP', 'LATERALITY', 'PRONOUN', 'NUMERIC']:
        print(
            f'{p:<16s} {pattern_counts[p]:>7d}  '
            f'{dataset_pattern_counts["primock57"][p]:>10d}  '
            f'{dataset_pattern_counts["figshare-osce"][p]:>6d}  '
            f'{dataset_pattern_counts["nazmulkazi"][p]:>6d}  '
            f'{dataset_pattern_counts["kokoro-va-sample"][p]:>7d}',
            flush=True,
        )
    print(f'\nWrote: {OUT_DIR}/medical_word_frequencies.tsv', flush=True)
    print(f'Wrote: {OUT_DIR}/medical_word_examples.txt', flush=True)
    print(f'Wrote: {OUT_DIR}/high_impact_patterns.tsv', flush=True)
    print(f'Wrote: {OUT_DIR}/high_impact_examples.txt', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
