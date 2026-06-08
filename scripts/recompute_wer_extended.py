#!/usr/bin/env python3
"""Recompute WER on existing per-file hypotheses with the clinical normalizer.

Reads each model's per-file hypothesis JSON, applies the extended (clinical)
normalizer to reference and hypothesis, and writes per-file rows incrementally
to a TSV. After all files for a (dataset, model) cell, writes a summary row.

Per stapes/CLAUDE.md ABSOLUTE RULE #1, every iteration writes to disk and
flushes. No accumulators with a final json.dumps.

Inputs (read-only):
    ossicles/benchmark_results_<dataset>/<model>.json
    ossicles/benchmark_results_cloud_<dataset>/<service>.json
    ossicles/benchmark_results_cloud_figshare-osce-subset/aws.json

Outputs:
    stapes/results/extended_normalization/per_file.tsv
        dataset  model  file_id  ref_words  errors_clinical  wer_clinical  errors_whisper  wer_whisper
    stapes/results/extended_normalization/summary.tsv
        dataset  model  n_files  total_words  errors_clinical  wer_clinical_pct  errors_whisper  wer_whisper_pct  delta_pp
"""
import json
import re
import sys
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

# Make our extended_normalizer importable.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from extended_normalizer import normalize_clinical  # noqa: E402

OSSICLES = Path('/home/grey/dev/graiai/ossicles')
STAPES = Path('/home/grey/dev/graiai/stapes')
OUT_DIR = STAPES / 'results' / 'extended_normalization'
OUT_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_BASELINE = EnglishTextNormalizer()


# ── Datasets and models ────────────────────────────────────────────────────
DATASETS = ['figshare-osce', 'primock57', 'nazmulkazi']

ON_DEVICE_MODELS = [
    'whisper-distil-v3.5',
    'whisper-turbo',
    'whisper-base-en',
    'qwen3-asr',
    'parakeet-tdt-0.6b-v2',
    'nemo-fastconformer',
    'nemo-fastconformer-int8',
    'nemotron',
    'zipformer-zh-en',
    'sensevoice',
    'sensevoice-no-itn',
    'medasr',
    'medasr-int8',
    'paraformer-en',
]

CLOUD_SERVICES = ['azure', 'google', 'deepgram', 'assemblyai', 'aws']


# ── figshare-OSCE apostrophe injection ─────────────────────────────────────
# The figshare reference text has apostrophes stripped from contractions.
# Whisper normalizer expands "i'm" → "i am" but the corrupted reference stays
# "im", inflating WER. Inject apostrophes before normalization. This is the
# same correction the paper currently uses (see figshare_fix_and_eval.py).
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
    'its': "it's", 'ill': "i'll", 'id': "i'd", 'lets': "let's", 'wed': "we'd",
}
TOKEN_RE = re.compile(r"\b([A-Za-z]+)\b")


def inject_apostrophes_for_figshare(text: str) -> str:
    """Inject apostrophes into figshare-OSCE reference contractions."""
    def _replace(match: 're.Match[str]') -> str:
        token = match.group(1)
        return CONTRACTIONS.get(token.lower(), token)
    return TOKEN_RE.sub(_replace, text)


# ── Hypothesis JSON locations ──────────────────────────────────────────────
def hypothesis_path(dataset: str, model: str, is_cloud: bool) -> Path | None:
    """Resolve the JSON path for a (dataset, model) cell. Returns None if missing."""
    if is_cloud:
        p = OSSICLES / f'benchmark_results_cloud_{dataset}' / f'{model}.json'
    else:
        p = OSSICLES / f'benchmark_results_{dataset}' / f'{model}.json'
    return p if p.exists() else None


# ── WER computation ────────────────────────────────────────────────────────
def compute_wer(ref: str, hyp: str) -> tuple[int, int, float]:
    """Return (errors, ref_words, wer). Treats empty ref as zero-everything."""
    if not ref:
        return 0, 0, 0.0
    n_words = len(ref.split())
    if n_words == 0:
        return 0, 0, 0.0
    if not hyp:
        return n_words, n_words, 1.0
    wer = jiwer.wer(ref, hyp)
    errors = round(wer * n_words)
    return errors, n_words, wer


# ── Main loop ──────────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)

    per_file_path = OUT_DIR / 'per_file.tsv'
    summary_path = OUT_DIR / 'summary.tsv'

    fh_per_file = open(per_file_path, 'w', encoding='utf-8')
    fh_per_file.write(
        'dataset\tmodel\tis_cloud\tfile_id'
        '\tref_words_clinical\terrors_clinical\twer_clinical'
        '\tref_words_whisper\terrors_whisper\twer_whisper\n'
    )
    fh_per_file.flush()

    fh_summary = open(summary_path, 'w', encoding='utf-8')
    fh_summary.write(
        'dataset\tmodel\tis_cloud\tn_files'
        '\twords_clinical\terrors_clinical\twer_clinical_pct'
        '\twords_whisper\terrors_whisper\twer_whisper_pct'
        '\tdelta_pp\n'
    )
    fh_summary.flush()

    cells = (
        [(ds, m, False) for ds in DATASETS for m in ON_DEVICE_MODELS]
        + [(ds, m, True) for ds in DATASETS for m in CLOUD_SERVICES]
    )

    for dataset, model, is_cloud in cells:
        hp = hypothesis_path(dataset, model, is_cloud)
        if hp is None:
            print(f'SKIP {dataset}/{model} (no JSON)', flush=True)
            continue

        try:
            data = json.loads(hp.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError) as e:
            print(f'ERROR reading {hp}: {e}', flush=True)
            continue

        results = data.get('results', [])
        if not results:
            print(f'SKIP {dataset}/{model} (empty results)', flush=True)
            continue

        cell_errors_clinical = 0
        cell_errors_whisper = 0
        cell_words_clinical = 0
        cell_words_whisper = 0
        n_files = 0

        for entry in results:
            file_id = entry.get('file_id', '')
            ref = entry.get('reference', '') or ''
            hyp = entry.get('hypothesis', '') or ''
            if not file_id or not ref:
                continue

            # figshare-OSCE: inject apostrophes into the reference, matching
            # the correction documented in Methods.
            if dataset == 'figshare-osce':
                ref_for_norm = inject_apostrophes_for_figshare(ref)
            else:
                ref_for_norm = ref

            ref_clinical = normalize_clinical(ref_for_norm)
            hyp_clinical = normalize_clinical(hyp)
            ref_whisper = WHISPER_BASELINE(ref_for_norm)
            hyp_whisper = WHISPER_BASELINE(hyp)

            err_c, words_c, wer_c = compute_wer(ref_clinical, hyp_clinical)
            err_w, words_w, wer_w = compute_wer(ref_whisper, hyp_whisper)

            fh_per_file.write(
                f'{dataset}\t{model}\t{int(is_cloud)}\t{file_id}'
                f'\t{words_c}\t{err_c}\t{wer_c:.6f}'
                f'\t{words_w}\t{err_w}\t{wer_w:.6f}\n'
            )
            fh_per_file.flush()

            cell_errors_clinical += err_c
            cell_errors_whisper += err_w
            cell_words_clinical += words_c
            cell_words_whisper += words_w
            n_files += 1

        wer_clinical_pct = (
            100 * cell_errors_clinical / cell_words_clinical
        ) if cell_words_clinical else 0.0
        wer_whisper_pct = (
            100 * cell_errors_whisper / cell_words_whisper
        ) if cell_words_whisper else 0.0
        delta_pp = wer_clinical_pct - wer_whisper_pct

        fh_summary.write(
            f'{dataset}\t{model}\t{int(is_cloud)}\t{n_files}'
            f'\t{cell_words_clinical}\t{cell_errors_clinical}\t{wer_clinical_pct:.4f}'
            f'\t{cell_words_whisper}\t{cell_errors_whisper}\t{wer_whisper_pct:.4f}'
            f'\t{delta_pp:.4f}\n'
        )
        fh_summary.flush()

        print(
            f'{dataset:>14} / {model:<25}  n={n_files:>3}  '
            f'whisper={wer_whisper_pct:>5.2f}%  '
            f'clinical={wer_clinical_pct:>5.2f}%  '
            f'delta={delta_pp:+.2f}pp',
            flush=True,
        )

    fh_per_file.close()
    fh_summary.close()
    print(f'\nDONE. Outputs in {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
