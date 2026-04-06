#!/usr/bin/env python3
"""Prepare nazmulkazi psychiatric dataset for ASR benchmarking.

Downloads audio from GitHub, converts to 16kHz mono WAV, pairs with
transcripts from thalamus/NazmulkaziTranscripts/.

The nazmulkazi dataset has enacted psychiatric conversations — students
reading from real clinical transcripts. Audio is in the 'recordings'
directory of the repo.

Output: ossicles/assets/audio/nazmulkazi/{transcript_id}.wav
        ossicles/assets/audio/nazmulkazi/{transcript_id}.txt

Usage:
    python scripts/prepare_nazmulkazi.py
    python scripts/prepare_nazmulkazi.py --repo-dir /path/to/existing/repo
"""

import argparse
import logging
import os
import pickle
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

REPO_URL = 'https://github.com/nazmulkazi/dataset_automated_medical_transcription.git'
TARGET_SR = 16000
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / 'assets' / 'audio' / 'nazmulkazi'
TRANSCRIPT_DIR = Path('/home/grey/dev/graiai/thalamus/NazmulkaziTranscripts')


def clone_repo(dest: Path) -> Path:
    """Clone nazmulkazi repo."""
    if (dest / '.git').exists():
        logger.info(f'Repo already cloned at {dest}')
        return dest
    logger.info(f'Cloning {REPO_URL} to {dest}...')
    subprocess.run(
        ['git', 'clone', '--depth', '1', REPO_URL, str(dest)],
        check=True,
    )
    return dest


def strip_speaker_labels(text: str) -> str:
    """Remove CLINICIAN/PATIENT speaker labels and IDs from transcript.

    Input format:
        CLINICIAN D0420-S1-T01
        I am the psychiatrist here in this department.

        PATIENT D0420-S1-T01
        I came to see you because my GP sent me to see you didn't he?

    Output: plain text, all utterances concatenated, uppercase.
    """
    lines = text.strip().split('\n')
    utterances = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip speaker label lines (CLINICIAN xxx or PATIENT xxx)
        if re.match(r'^(CLINICIAN|PATIENT)\s+\S+', line):
            continue
        utterances.append(line.upper())
    return '\n'.join(utterances)


def find_audio_files(repo_dir: Path) -> dict[str, Path]:
    """Find audio files in the repo and map them to transcript IDs.

    Returns dict mapping transcript_id -> audio_path.
    """
    recordings_dir = repo_dir / 'recordings'
    audio_map = {}

    if not recordings_dir.exists():
        logger.warning(f'No recordings/ directory at {recordings_dir}')
        # Try loading from pickle
        pickle_path = repo_dir / 'dataset.pickle'
        if pickle_path.exists():
            logger.info('Found dataset.pickle — will try to extract audio from it')
            return {'_pickle': pickle_path}
        return {}

    # Map audio files to transcript IDs
    for audio_file in sorted(recordings_dir.rglob('*')):
        if audio_file.suffix.lower() in ('.wav', '.mp3', '.flac', '.ogg', '.m4a'):
            # Try to match filename to transcript ID pattern (D0420-S1-T01)
            match = re.search(r'(D\d{4}-S\d+-T\d+)', audio_file.stem)
            if match:
                tid = match.group(1)
                audio_map[tid] = audio_file
            else:
                logger.debug(f'Cannot map audio file to transcript: {audio_file.name}')

    return audio_map


def convert_audio(input_path: Path, output_path: Path) -> float:
    """Convert audio to 16kHz mono WAV. Returns duration in seconds."""
    audio, sr = sf.read(str(input_path), dtype='float32')

    # Stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != TARGET_SR:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        except ImportError:
            logger.error('librosa required for resampling. pip install librosa')
            sys.exit(1)

    sf.write(str(output_path), audio, TARGET_SR, subtype='PCM_16')
    return len(audio) / TARGET_SR


def main():
    parser = argparse.ArgumentParser(description='Prepare nazmulkazi for benchmarking')
    parser.add_argument('--repo-dir', type=Path, default=None,
                        help='Path to existing repo clone (skips download)')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--transcript-dir', type=Path, default=TRANSCRIPT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get repo
    if args.repo_dir:
        repo_dir = args.repo_dir
    else:
        repo_dir = Path(tempfile.mkdtemp()) / 'nazmulkazi'
        clone_repo(repo_dir)

    # List what's in the repo
    logger.info(f'Repo contents:')
    for item in sorted(repo_dir.iterdir()):
        if item.name.startswith('.'):
            continue
        if item.is_dir():
            count = sum(1 for _ in item.rglob('*') if _.is_file())
            logger.info(f'  {item.name}/ ({count} files)')
        else:
            logger.info(f'  {item.name} ({item.stat().st_size / 1024:.0f} KB)')

    # Find audio files — check downloaded dir first, then recordings dir
    downloaded_dir = repo_dir / 'recordings_downloaded'
    if downloaded_dir.exists():
        logger.info(f'Using downloaded recordings from {downloaded_dir}')
        audio_map = {}
        for f in sorted(downloaded_dir.glob('*.wav')):
            match = re.search(r'(D\d{4}-S\d+-T\d+)', f.stem)
            if match:
                audio_map[match.group(1)] = f
    else:
        audio_map = find_audio_files(repo_dir)
    logger.info(f'Found {len(audio_map)} audio files')

    # Find transcripts
    transcript_files = sorted(args.transcript_dir.glob('*.txt'))
    logger.info(f'Found {len(transcript_files)} transcripts')

    if not audio_map:
        logger.error('No audio files found. Listing repo structure for debugging:')
        for item in sorted(repo_dir.rglob('*'))[:30]:
            logger.error(f'  {item.relative_to(repo_dir)}')
        sys.exit(1)

    # Process each transcript that has matching audio
    processed = 0
    skipped = 0
    no_audio = 0

    for txt_file in transcript_files:
        tid = txt_file.stem  # e.g., D0420-S1-T01
        wav_out = output_dir / f'{tid}.wav'
        ref_out = output_dir / f'{tid}.txt'

        # Skip if already done
        if wav_out.exists() and ref_out.exists():
            logger.info(f'SKIP {tid} — already exists')
            skipped += 1
            continue

        # Find matching audio
        if tid not in audio_map:
            logger.warning(f'NO AUDIO for {tid}')
            no_audio += 1
            continue

        try:
            # Convert audio
            duration = convert_audio(audio_map[tid], wav_out)

            # Build reference transcript (strip speaker labels, uppercase)
            raw_text = txt_file.read_text(encoding='utf-8')
            transcript = strip_speaker_labels(raw_text)
            ref_out.write_text(transcript, encoding='utf-8')

            logger.info(f'OK {tid} — {duration:.1f}s, {len(transcript)} chars')
            processed += 1

        except Exception as e:
            logger.error(f'FAIL {tid}: {e}')

    logger.info(f'Done. Processed: {processed}, Skipped: {skipped}, No audio: {no_audio}')


if __name__ == '__main__':
    main()
