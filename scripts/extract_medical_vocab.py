#!/usr/bin/env python3
"""Extract a static medical-term dictionary for on-device fusion gating.

Walks tiered text sources in order of relevance to the deployment target,
runs QuickUMLS on each text, and accumulates a unique-term dictionary that
gets written incrementally — one row per new unique term, flushed
immediately. Stops at TIER_EXHAUSTED or SIZE_LIMIT_MB, whichever fires
first.

Output:
    results/medical_vocab/medical_vocab.tsv
        term \\t semantic_category \\t first_seen_tier \\t first_seen_source
    results/medical_vocab/extraction_progress.tsv
        tier \\t source \\t file_id \\t n_terms_in_file \\t n_new_unique \\t total_unique \\t bytes_so_far
    results/medical_vocab/coverage_stats.json
        per-tier counts and exhaustion curve

Parallelism: 8 worker processes (half of 16 cores). Each loads its own
QuickUMLS instance once via initializer. Results stream back to the main
process via imap_unordered and are written incrementally — NO accumulator-
then-dump pattern.

Run with the ossicles venv:
    /home/grey/dev/graiai/ossicles/venv/bin/python scripts/extract_medical_vocab.py

Optional flags:
    --size-limit-mb N        (default 50)
    --workers N              (default 8)
    --tiers t1,t2,...        (default: 1a,1b,1c,1d,1e,4a)
"""
import argparse
import csv
import json
import logging
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# ──────────────────────── paths ────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
STAPES_DIR = SCRIPT_DIR.parent
GRAIAI_DIR = STAPES_DIR.parent
OSSICLES_DIR = GRAIAI_DIR / 'ossicles'
COCHLEA_DIR = GRAIAI_DIR / 'cochlea'
QUICKUMLS_PATH = OSSICLES_DIR / 'quickumls_data'

OUT_DIR = STAPES_DIR / 'results' / 'medical_vocab'
OUT_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_TSV = OUT_DIR / 'medical_vocab.tsv'
PROGRESS_TSV = OUT_DIR / 'extraction_progress.tsv'
STATS_JSON = OUT_DIR / 'coverage_stats.json'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ──────────────────────── QuickUMLS config (matches stapes paper) ────────
MEDICAL_SEMANTIC_TYPES = {
    'T116': 'DRUG', 'T121': 'DRUG', 'T200': 'DRUG', 'T195': 'DRUG',
    'T047': 'CONDITION', 'T048': 'CONDITION', 'T046': 'CONDITION', 'T191': 'CONDITION',
    'T184': 'SYMPTOM', 'T033': 'FINDING', 'T034': 'LAB_RESULT',
    'T023': 'ANATOMY', 'T029': 'ANATOMY', 'T030': 'ANATOMY',
    'T058': 'PROCEDURE', 'T059': 'PROCEDURE', 'T060': 'PROCEDURE', 'T061': 'PROCEDURE',
}
UMLS_STOPWORDS = {
    'bit', 'get', 'got', 'little', 'lot', 'said', 'still', 'today',
    'changes', 'well', 'cant', 'nothing', 'maybe', 'take', 'yes',
    'much', 'probably', 'worse', 'sharp', 'only', 'more', 'less',
    'stop', 'ask', 'bad', 'dad', 'may', 'nice', 'new', 'used',
    'find', 'care', 'place', 'always', 'wanted', 'play',
}

# ──────────────────────── worker globals ────────────────────────
_matcher = None


def init_worker():
    """Load QuickUMLS once per worker process."""
    global _matcher
    from quickumls import QuickUMLS
    _matcher = QuickUMLS(
        quickumls_fp=str(QUICKUMLS_PATH),
        threshold=0.8,
        similarity_name='jaccard',
        window=5,
    )


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s'\-]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_terms(text: str) -> list[tuple[str, str]]:
    """Return list of (canonical_term, category) for medical spans in text."""
    text_norm = normalize(text)
    if not text_norm:
        return []
    matches = _matcher.match(text_norm)
    out = []
    seen_local = set()
    for match_group in matches:
        if not match_group:
            continue
        best = match_group[0]
        ngram = best['ngram'].lower().strip()
        if ngram in UMLS_STOPWORDS:
            continue
        # Filter: short single-word low-similarity matches are probably noise
        if ' ' not in ngram and len(ngram) < 6 and best['similarity'] < 1.0:
            continue
        category = None
        for stype in best.get('semtypes', []):
            if stype in MEDICAL_SEMANTIC_TYPES:
                category = MEDICAL_SEMANTIC_TYPES[stype]
                break
        if category is None:
            continue
        # Use canonical form (the term that matched UMLS) — best['term'] is the
        # UMLS preferred name, best['ngram'] is the literal text. We want the
        # canonical CUI surface form for app deployment.
        canonical = best.get('term', ngram).lower().strip()
        if canonical in seen_local:
            continue
        seen_local.add(canonical)
        out.append((canonical, category))
    return out


def worker_process_text(args):
    """Worker entry: takes (text, source_tag, file_id), returns extracted terms.

    Errors are caught and returned as empty list — we don't want a single bad
    file to crash the pool.
    """
    text, source_tag, file_id = args
    try:
        terms = extract_terms(text)
        return source_tag, file_id, terms, None
    except Exception as e:
        return source_tag, file_id, [], f'{type(e).__name__}: {str(e)[:200]}'


# ──────────────────────── source loaders ────────────────────────
def iter_txt_dir(dir_path: Path, tier: str, source_label: str):
    """Yield (text, source_tag, file_id) for every .txt in dir."""
    if not dir_path.exists():
        return
    for f in sorted(dir_path.glob('*.txt')):
        try:
            text = f.read_text(encoding='utf-8', errors='replace')
            if text.strip():
                yield text, f'{tier}:{source_label}', f.stem
        except Exception as e:
            log.warning(f'  failed to read {f}: {e}')


def iter_rtf_dir(dir_path: Path, tier: str, source_label: str):
    """Yield (text, source_tag, file_id) for every .rtf, converted to plain text."""
    from striprtf.striprtf import rtf_to_text
    if not dir_path.exists():
        return
    for f in sorted(dir_path.glob('*.rtf')):
        try:
            raw = f.read_text(encoding='utf-8', errors='replace')
            text = rtf_to_text(raw)
            if text.strip():
                yield text, f'{tier}:{source_label}', f.stem
        except Exception as e:
            log.warning(f'  failed to read {f}: {e}')


def iter_manifest_dataset(dataset: str, tier: str, source_label: str):
    """Yield (text, source_tag, file_id) from cochlea unified_manifest.csv rows."""
    import pandas as pd
    manifest = COCHLEA_DIR / 'unified_manifest.csv'
    if not manifest.exists():
        log.warning(f'manifest not found: {manifest}')
        return
    log.info(f'  reading manifest for dataset={dataset}...')
    # Stream-read with chunksize to avoid full-table memory
    for chunk in pd.read_csv(
        manifest,
        usecols=['dataset', 'text', 'id'],
        chunksize=20000,
        low_memory=False,
    ):
        sub = chunk[chunk['dataset'] == dataset]
        for _, row in sub.iterrows():
            text = row['text']
            if isinstance(text, str) and text.strip():
                yield text, f'{tier}:{source_label}', str(row['id'])


# ──────────────────────── tier definitions ────────────────────────
def iter_thalamus_notes(tier: str, source_label: str):
    """Yield (text, source_tag, file_id) from all thalamus CleanNotes/*.txt files."""
    notes_dir = GRAIAI_DIR / 'thalamus' / 'CleanNotes'
    if not notes_dir.exists():
        return
    for f in notes_dir.rglob('*.txt'):
        try:
            text = f.read_text(encoding='utf-8', errors='replace')
            if text.strip():
                # Use parent dir + stem as file_id for traceability
                fid = f'{f.parent.name}/{f.stem}'
                yield text, f'{tier}:{source_label}', fid
        except Exception as e:
            log.warning(f'  failed to read {f}: {e}')


TIER_SOURCES = {
    # ── Tier 1: real conversational + clinical (closest to deployment) ──
    '1a': lambda: iter_txt_dir(OSSICLES_DIR / 'assets' / 'audio' / 'primock57', '1a', 'primock57'),
    '1b': lambda: iter_txt_dir(OSSICLES_DIR / 'assets' / 'audio' / 'figshare-osce', '1b', 'figshare-osce'),
    '1c': lambda: iter_txt_dir(OSSICLES_DIR / 'assets' / 'audio' / 'nazmulkazi', '1c', 'nazmulkazi'),
    '1d': lambda: iter_manifest_dataset('simulated', '1d', 'simulated'),
    '1e_eka': lambda: iter_manifest_dataset('eka-medical-asr-en', '1e', 'eka'),
    '1e_kaggle': lambda: iter_manifest_dataset('kaggle-medical-speech', '1e', 'kaggle'),

    # ── Tier 2: medical lecture / educational (real audio, real transcripts) ──
    '2a': lambda: iter_manifest_dataset('youtube_cc_medical_segmented', '2a', 'yt_med_seg'),
    '2b': lambda: iter_manifest_dataset('youtube_cc_medical', '2b', 'yt_med'),

    # ── Tier 3: dictation-style synthetic (text was hand-written, audio is TTS) ──
    '3a': lambda: iter_manifest_dataset('synthetic_dictation', '3a', 'syn_dict'),
    '3b': lambda: iter_manifest_dataset('united_syn_med', '3b', 'united_syn_med'),
    '3c': lambda: iter_manifest_dataset('synthetic_piper', '3c', 'syn_piper'),
    '3d': lambda: iter_manifest_dataset('kokoro_tts_va', '3d', 'kokoro_tts'),

    # ── Tier 4: dictation references + generated notes ──
    '4a': lambda: iter_rtf_dir(
        OSSICLES_DIR / 'assets' / 'audio' / 'ezDI-Medical-Dictation-Dataset',
        '4a', 'ezdi',
    ),
    '4b': lambda: iter_thalamus_notes('4b', 'thalamus_notes'),
}
TIER_DEFAULT_ORDER = ['1a', '1b', '1c', '1d', '1e_eka', '1e_kaggle', '4a']
TIER_FULL_ORDER = ['1a', '1b', '1c', '1d', '1e_eka', '1e_kaggle', '2b', '2a', '3a', '3b', '3c', '3d', '4a', '4b']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size-limit-mb', type=int, default=50)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--tiers', type=str, default=','.join(TIER_DEFAULT_ORDER))
    args = p.parse_args()

    size_limit_bytes = args.size_limit_mb * 1024 * 1024
    tier_order = args.tiers.split(',')

    # ── open output files for incremental writes ──
    fh_vocab = open(VOCAB_TSV, 'w')
    fh_vocab.write('term\tcategory\tfirst_tier\tfirst_source\n')
    fh_vocab.flush()

    fh_progress = open(PROGRESS_TSV, 'w')
    fh_progress.write('tier\tsource\tfile_id\tn_terms_file\tn_new_unique\ttotal_unique\tbytes_so_far\n')
    fh_progress.flush()

    seen = {}  # canonical_term → (category, first_tier, first_source)
    bytes_so_far = 0
    files_processed = 0
    per_tier_stats = {}
    stop_reason = 'tier_exhausted'

    log.info(f'starting extraction: workers={args.workers}, tier_order={tier_order}, size_limit={args.size_limit_mb}MB')
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as ex:
        for tier in tier_order:
            if tier not in TIER_SOURCES:
                log.warning(f'unknown tier: {tier}')
                continue
            tier_t0 = time.time()
            tier_files = 0
            tier_new = 0
            log.info(f'=== TIER {tier} ===')

            # Submit work in batches via imap_unordered for streaming results
            iterator = TIER_SOURCES[tier]()
            results_iter = ex.map(
                worker_process_text,
                iterator,
                chunksize=4,
            )

            for source_tag, file_id, terms, err in results_iter:
                if err:
                    log.warning(f'  err {source_tag}/{file_id}: {err}')
                    continue
                tier_files += 1
                files_processed += 1

                new_in_file = 0
                for canonical, category in terms:
                    if canonical not in seen:
                        seen[canonical] = (category, tier, source_tag)
                        # Write IMMEDIATELY for this new unique term
                        fh_vocab.write(f'{canonical}\t{category}\t{tier}\t{source_tag}\n')
                        fh_vocab.flush()
                        new_in_file += 1
                        tier_new += 1
                        bytes_so_far += len(canonical.encode('utf-8')) + 20  # term + columns

                fh_progress.write(
                    f'{tier}\t{source_tag}\t{file_id}\t{len(terms)}\t{new_in_file}\t{len(seen)}\t{bytes_so_far}\n'
                )
                fh_progress.flush()

                # Throttled console line
                if tier_files % 50 == 0 or new_in_file > 0:
                    print(
                        f'  [{tier}] file {tier_files} ({source_tag}/{file_id}): '
                        f'{len(terms)} terms, {new_in_file} new ({len(seen)} total, {bytes_so_far/1024/1024:.2f} MB)',
                        flush=True,
                    )

                if bytes_so_far >= size_limit_bytes:
                    stop_reason = 'size_limit'
                    break

            tier_dt = time.time() - tier_t0
            per_tier_stats[tier] = {
                'files_processed': tier_files,
                'new_unique_terms': tier_new,
                'cumulative_unique': len(seen),
                'cumulative_bytes': bytes_so_far,
                'wall_seconds': round(tier_dt, 1),
            }
            log.info(
                f'tier {tier} done: {tier_files} files, +{tier_new} new terms '
                f'({len(seen)} total, {bytes_so_far/1024/1024:.2f} MB) in {tier_dt:.1f}s'
            )

            if bytes_so_far >= size_limit_bytes:
                log.warning(f'STOPPED: size limit {args.size_limit_mb} MB reached')
                break

    fh_vocab.close()
    fh_progress.close()

    # Write final stats — single small JSON, not a pattern violation since
    # the per-iteration data is already in vocab.tsv and progress.tsv
    stats = {
        'stop_reason': stop_reason,
        'total_files': files_processed,
        'total_unique_terms': len(seen),
        'total_bytes': bytes_so_far,
        'wall_seconds': round(time.time() - t_start, 1),
        'workers': args.workers,
        'size_limit_mb': args.size_limit_mb,
        'tier_order': tier_order,
        'per_tier': per_tier_stats,
    }
    STATS_JSON.write_text(json.dumps(stats, indent=2))

    print(
        f'\nDONE: {len(seen)} unique medical terms from {files_processed} files '
        f'({bytes_so_far/1024/1024:.2f} MB) in {stats["wall_seconds"]}s. '
        f'stop_reason={stop_reason}',
        flush=True,
    )
    print(f'  vocab:    {VOCAB_TSV}', flush=True)
    print(f'  progress: {PROGRESS_TSV}', flush=True)
    print(f'  stats:    {STATS_JSON}', flush=True)


if __name__ == '__main__':
    main()
