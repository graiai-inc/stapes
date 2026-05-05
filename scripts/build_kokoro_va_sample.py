#!/usr/bin/env python3
"""Sample Kokoro VA conversations and concatenate into session-level audio for ASR benchmarking.

Background
----------
cochlea/audio/kokoro_tts/<session>/<turn_idx>_<speaker>.flac is per-turn synthetic
audio (Kokoro TTS) of VA conversation transcripts. presegmented_manifest.csv lists
115,956 turn rows across ~405 sessions, with reference text per turn.

For ASR benchmarking we want session-level audio (matching how PriMock57/OSCE/Kazi
are structured). This script:

1. Reads kokoro_tts_va rows from presegmented_manifest.csv
2. Groups by session_id, sorted by utterance_index
3. Scores each session by the number of drug-name mentions (from a small list of
   common drugs we already know are in the curated vocab)
4. Picks the top N sessions by drug-mention count (drugs are the fusion target —
   we want maximal CTR signal)
5. For each session: concatenates turn .flac files in order via ffmpeg, writes
   session.wav (16kHz mono pcm16) + session.txt (concatenated reference text) to
   ossicles/assets/audio/kokoro-va-sample/

Output structure mirrors the existing ossicles dataset layout so evaluate_benchmark.py
can run on it after a one-line addition.

Usage:
    /home/grey/dev/graiai/cochlea/venv/bin/python scripts/build_kokoro_va_sample.py
"""
import csv
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
STAPES_DIR = SCRIPT_DIR.parent
GRAIAI_DIR = STAPES_DIR.parent
COCHLEA_DIR = GRAIAI_DIR / 'cochlea'
OSSICLES_DIR = GRAIAI_DIR / 'ossicles'

MANIFEST = COCHLEA_DIR / 'presegmented_manifest.csv'
KOKORO_AUDIO_ROOT = COCHLEA_DIR / 'audio' / 'kokoro_tts'
OUT_AUDIO_DIR = OSSICLES_DIR / 'assets' / 'audio' / 'kokoro-va-sample'
OUT_INDEX = STAPES_DIR / 'data' / 'kokoro_va_sample_index.tsv'

# Sample size: enough to get statistical signal, small enough that ASR runs in a
# reasonable wall-clock time.
N_SESSIONS = 30

# Drug names to score sessions by (from the full curated vocab — but we only need
# common ones to rank, not all 2700). These are the high-frequency drug-name
# mentions found in kokoro_tts_va via earlier scan.
DRUG_VOCAB_FOR_RANKING = {
    'insulin', 'lisinopril', 'aspirin', 'metformin', 'tylenol', 'metoprolol',
    'simvastatin', 'glipizide', 'ibuprofen', 'amlodipine', 'albuterol',
    'plavix', 'tramadol', 'viagra', 'statin', 'prednisone', 'losartan',
    'aleve', 'lipitor', 'morphine', 'advil', 'synthroid', 'warfarin',
    'motrin', 'acetaminophen', 'rosuvastatin', 'atorvastatin', 'ramipril',
    'crestor', 'wellbutrin', 'klonopin', 'seroquel', 'ambien', 'xanax',
    'zoloft', 'prozac', 'celexa', 'lexapro', 'cymbalta', 'effexor',
    'oxycodone', 'percocet', 'vicodin', 'oxycontin',
    'singulair', 'flovent', 'advair', 'symbicort', 'spiriva', 'ventolin',
    'januvia', 'glucophage', 'prilosec', 'nexium', 'pepcid', 'zantac',
    'fosamax', 'boniva', 'coumadin', 'eliquis', 'xarelto',
    'levothyroxine', 'thyroxine', 'norvasc',
    'allopurinol', 'colchicine',
    'fentanyl', 'demerol', 'dilaudid',
    'penicillin', 'amoxicillin', 'azithromycin', 'doxycycline',
    'ciprofloxacin', 'cipro', 'flagyl', 'metronidazole',
    'benadryl', 'diphenhydramine', 'cetirizine', 'zyrtec',
    'gabapentin', 'pregabalin', 'lyrica',
}

WORD_RE = re.compile(r"[a-z']+")


def load_manifest_rows():
    rows = []
    with open(MANIFEST) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('dataset', '') != 'kokoro_tts_va':
                continue
            rows.append({
                'audio_path': row.get('audio_path', ''),
                'session_id': row.get('session_id', ''),
                'speaker': row.get('speaker', ''),
                'utterance_index': float(row.get('utterance_index', '0') or 0),
                'text': row.get('text', '') or '',
            })
    return rows


def group_by_session(rows):
    by_session = defaultdict(list)
    for r in rows:
        by_session[r['session_id']].append(r)
    for sid in by_session:
        by_session[sid].sort(key=lambda r: r['utterance_index'])
    return by_session


def count_drug_mentions(turns):
    n = 0
    for t in turns:
        for w in WORD_RE.findall(t['text'].lower()):
            if w in DRUG_VOCAB_FOR_RANKING:
                n += 1
    return n


def resolve_audio_file(turn):
    """Manifest paths use .wav extension but actual files on disk are .flac.
    Try both."""
    # Manifest path is relative to cochlea/
    manifest_rel = turn['audio_path']
    candidate_wav = COCHLEA_DIR / manifest_rel
    if candidate_wav.exists():
        return candidate_wav
    candidate_flac = candidate_wav.with_suffix('.flac')
    if candidate_flac.exists():
        return candidate_flac
    return None


def concat_audio(audio_paths: list[Path], out_wav: Path) -> bool:
    """Concatenate a list of audio files into a single 16kHz mono pcm16 WAV.

    Uses ffmpeg's concat demuxer; writes a temp listfile then runs ffmpeg.
    """
    listfile = out_wav.with_suffix('.concat.txt')
    with open(listfile, 'w') as f:
        for p in audio_paths:
            f.write(f"file '{p}'\n")
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(listfile),
            '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
            str(out_wav),
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print(f'  ffmpeg failed: {result.stderr[-500:]}', flush=True)
            return False
        return True
    finally:
        listfile.unlink(missing_ok=True)


def main() -> int:
    print('Loading manifest...', flush=True)
    rows = load_manifest_rows()
    print(f'  {len(rows)} kokoro_tts_va turn rows', flush=True)

    by_session = group_by_session(rows)
    print(f'  {len(by_session)} sessions', flush=True)

    print('Scoring sessions by drug-mention count...', flush=True)
    scored = []
    for sid, turns in by_session.items():
        score = count_drug_mentions(turns)
        scored.append((sid, score, len(turns)))
    scored.sort(key=lambda x: -x[1])
    print('  top 10 sessions by drug mentions:', flush=True)
    for sid, sc, nt in scored[:10]:
        print(f'    {sid}: {sc} drug mentions, {nt} turns', flush=True)

    # Pick top N. Skip sessions with <5 drug mentions or <10 turns.
    selected = [(sid, sc, nt) for sid, sc, nt in scored if sc >= 5 and nt >= 10][:N_SESSIONS]
    print(f'\nSelected {len(selected)} sessions (top by drug count, ≥5 drugs, ≥10 turns)',
          flush=True)

    if OUT_AUDIO_DIR.exists():
        print(f'  removing existing {OUT_AUDIO_DIR}', flush=True)
        shutil.rmtree(OUT_AUDIO_DIR)
    OUT_AUDIO_DIR.mkdir(parents=True)
    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)

    fh_idx = open(OUT_INDEX, 'w')
    fh_idx.write('session_id\tn_turns\tn_drug_mentions\tduration_s\tref_words\twav_path\ttxt_path\n')
    fh_idx.flush()

    n_succeeded = 0
    n_failed = 0
    for i, (sid, sc, nt) in enumerate(selected):
        turns = by_session[sid]
        # Resolve every turn's audio file; bail this session if any are missing.
        audio_paths = []
        missing = []
        for t in turns:
            p = resolve_audio_file(t)
            if p is None:
                missing.append(t['audio_path'])
            else:
                audio_paths.append(p)
        if missing:
            print(f'[{i+1}/{len(selected)}] {sid}: SKIPPED — {len(missing)} missing turns '
                  f'(first: {missing[0]})', flush=True)
            n_failed += 1
            continue

        out_wav = OUT_AUDIO_DIR / f'{sid}.wav'
        out_txt = OUT_AUDIO_DIR / f'{sid}.txt'

        ok = concat_audio(audio_paths, out_wav)
        if not ok:
            n_failed += 1
            continue

        ref = ' '.join(t['text'].strip() for t in turns if t['text'].strip())
        with open(out_txt, 'w') as f:
            f.write(ref + '\n')

        # Probe duration with ffprobe
        try:
            r = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'csv=p=0', str(out_wav)],
                capture_output=True, text=True,
            )
            duration = float(r.stdout.strip()) if r.stdout.strip() else 0.0
        except Exception:
            duration = 0.0

        ref_words = len(ref.split())
        fh_idx.write(f'{sid}\t{nt}\t{sc}\t{duration:.2f}\t{ref_words}\t{out_wav}\t{out_txt}\n')
        fh_idx.flush()
        n_succeeded += 1
        print(f'[{i+1}/{len(selected)}] {sid}: {nt} turns, {sc} drug mentions, '
              f'{duration:.1f}s, {ref_words} ref words → {out_wav.name}', flush=True)

    fh_idx.close()
    print(f'\nDone: {n_succeeded} sessions written, {n_failed} failed', flush=True)
    print(f'Audio: {OUT_AUDIO_DIR}', flush=True)
    print(f'Index: {OUT_INDEX}', flush=True)
    return 0 if n_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
