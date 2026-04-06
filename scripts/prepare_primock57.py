#!/usr/bin/env python3
"""Prepare primock57 dataset for ASR benchmarking.

Downloads the primock57 repo (Git LFS), mixes doctor+patient audio into
single-channel WAV files, and extracts reference transcripts from TextGrid
files — one WAV + one TXT per consultation.

Output: ossicles/assets/audio/primock57/{consultation_id}.wav
        ossicles/assets/audio/primock57/{consultation_id}.txt

Usage:
    python scripts/prepare_primock57.py
    python scripts/prepare_primock57.py --repo-dir /path/to/existing/primock57
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

REPO_URL = 'https://github.com/babylonhealth/primock57.git'
TARGET_SR = 16000
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / 'assets' / 'audio' / 'primock57'


def parse_textgrid(filepath: Path) -> list[dict]:
    """Parse a Praat TextGrid file, return list of {text, from, to} dicts."""
    content = filepath.read_text(encoding='utf-8')
    utterances = []
    pattern = re.compile(
        r'xmin\s*=\s*([0-9.]+)\s*\n\s*xmax\s*=\s*([0-9.]+)\s*\n\s*text\s*=\s*"((?:[^"\\]|\\.)*)"',
        re.MULTILINE,
    )
    for match in pattern.finditer(content):
        xmin = float(match.group(1))
        xmax = float(match.group(2))
        text = match.group(3).strip()
        if text:
            # Strip annotation tags
            for tag in ['<UNSURE>', '</UNSURE>', '<UNIN/>', '<INAUDIBLE_SPEECH/>']:
                text = text.replace(tag, '')
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                utterances.append({'text': text, 'from': xmin, 'to': xmax})
    return utterances


def clone_repo(dest: Path) -> Path:
    """Clone primock57 repo with Git LFS."""
    if (dest / '.git').exists():
        logger.info(f'Repo already cloned at {dest}')
        return dest
    logger.info(f'Cloning {REPO_URL} to {dest} (this may take a while with LFS)...')
    subprocess.run(
        ['git', 'clone', '--depth', '1', REPO_URL, str(dest)],
        check=True,
    )
    # Pull LFS files
    subprocess.run(
        ['git', '-C', str(dest), 'lfs', 'pull'],
        check=True,
    )
    return dest


def find_consultations(repo_dir: Path) -> list[dict]:
    """Find all consultation audio/transcript pairs in the repo.

    Returns list of dicts with keys:
        consultation_id, doctor_audio, patient_audio,
        doctor_textgrid, patient_textgrid
    """
    audio_dir = repo_dir / 'audio'
    transcript_dir = repo_dir / 'transcripts'
    consultations = []

    if not audio_dir.exists():
        logger.error(f'Audio directory not found: {audio_dir}')
        return []

    # Find all doctor audio files
    for doctor_wav in sorted(audio_dir.glob('*_doctor.wav')):
        stem = doctor_wav.stem.replace('_doctor', '')
        patient_wav = audio_dir / f'{stem}_patient.wav'

        if not patient_wav.exists():
            logger.warning(f'Missing patient audio for {stem}, skipping')
            continue

        # Find matching TextGrid files
        doctor_tg = transcript_dir / f'{stem}_doctor.TextGrid'
        patient_tg = transcript_dir / f'{stem}_patient.TextGrid'

        if not doctor_tg.exists() or not patient_tg.exists():
            logger.warning(f'Missing TextGrid for {stem}, skipping')
            continue

        consultations.append({
            'consultation_id': stem,
            'doctor_audio': doctor_wav,
            'patient_audio': patient_wav,
            'doctor_textgrid': doctor_tg,
            'patient_textgrid': patient_tg,
        })

    return consultations


def mix_audio(doctor_path: Path, patient_path: Path) -> tuple[np.ndarray, int]:
    """Mix doctor and patient audio into single channel.

    Reads both files, resamples to TARGET_SR if needed, zero-pads the
    shorter one, and sums them (clipping to [-1, 1]).
    """
    doc_audio, doc_sr = sf.read(doctor_path, dtype='float32')
    pat_audio, pat_sr = sf.read(patient_path, dtype='float32')

    # Handle stereo → mono
    if doc_audio.ndim > 1:
        doc_audio = doc_audio.mean(axis=1)
    if pat_audio.ndim > 1:
        pat_audio = pat_audio.mean(axis=1)

    # Resample if needed
    if doc_sr != TARGET_SR:
        import librosa
        doc_audio = librosa.resample(doc_audio, orig_sr=doc_sr, target_sr=TARGET_SR)
    if pat_sr != TARGET_SR:
        import librosa
        pat_audio = librosa.resample(pat_audio, orig_sr=pat_sr, target_sr=TARGET_SR)

    # Zero-pad shorter to match longer
    max_len = max(len(doc_audio), len(pat_audio))
    if len(doc_audio) < max_len:
        doc_audio = np.pad(doc_audio, (0, max_len - len(doc_audio)))
    if len(pat_audio) < max_len:
        pat_audio = np.pad(pat_audio, (0, max_len - len(pat_audio)))

    # Mix and clip
    mixed = doc_audio + pat_audio
    mixed = np.clip(mixed, -1.0, 1.0)

    return mixed, TARGET_SR


def build_transcript(doctor_tg: Path, patient_tg: Path) -> str:
    """Build combined transcript from doctor and patient TextGrids.

    Merges utterances from both speakers in chronological order.
    Returns plain text (no speaker labels) — just the words spoken.
    """
    doc_utts = parse_textgrid(doctor_tg)
    pat_utts = parse_textgrid(patient_tg)

    # Merge and sort by start time
    all_utts = []
    for u in doc_utts:
        all_utts.append(u)
    for u in pat_utts:
        all_utts.append(u)

    all_utts.sort(key=lambda x: x['from'])

    # Join all text, uppercase to match figshare-osce format
    lines = [u['text'].upper() for u in all_utts]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Prepare primock57 for benchmarking')
    parser.add_argument('--repo-dir', type=Path, default=None,
                        help='Path to existing primock57 repo clone (skips download)')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get repo
    if args.repo_dir:
        repo_dir = args.repo_dir
    else:
        repo_dir = Path(tempfile.mkdtemp()) / 'primock57'
        clone_repo(repo_dir)

    # Find consultations
    consultations = find_consultations(repo_dir)
    logger.info(f'Found {len(consultations)} consultations')

    if not consultations:
        logger.error('No consultations found. Check repo structure.')
        sys.exit(1)

    # Process each consultation
    processed = 0
    skipped = 0
    for cons in consultations:
        cid = cons['consultation_id']
        wav_out = output_dir / f'{cid}.wav'
        txt_out = output_dir / f'{cid}.txt'

        # Skip if already done (resume support)
        if wav_out.exists() and txt_out.exists():
            logger.info(f'SKIP {cid} — already exists')
            skipped += 1
            continue

        try:
            # Mix audio
            mixed, sr = mix_audio(cons['doctor_audio'], cons['patient_audio'])

            # Build transcript
            transcript = build_transcript(
                cons['doctor_textgrid'], cons['patient_textgrid']
            )

            # Write outputs
            sf.write(str(wav_out), mixed, sr, subtype='PCM_16')
            txt_out.write_text(transcript, encoding='utf-8')

            duration = len(mixed) / sr
            logger.info(f'OK {cid} — {duration:.1f}s, {len(transcript)} chars')
            processed += 1

        except Exception as e:
            logger.error(f'FAIL {cid}: {e}')

    logger.info(f'Done. Processed: {processed}, Skipped: {skipped}, Total: {len(consultations)}')


if __name__ == '__main__':
    main()
