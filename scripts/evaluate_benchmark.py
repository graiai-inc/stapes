#!/usr/bin/env python3
"""Clinical ASR Benchmark Evaluation Harness.

Runs sherpa-onnx ASR models on benchmark datasets (figshare-osce, primock57,
nazmulkazi) and computes WER with both raw and Whisper normalization.

Results are saved per-file immediately and support resume after interruption.

Usage:
    # Run one model on one dataset
    python scripts/evaluate_benchmark.py --model whisper-distil-v3.5 --dataset figshare-osce

    # Run one model on all datasets
    python scripts/evaluate_benchmark.py --model whisper-distil-v3.5 --dataset all

    # Run all models on all datasets
    python scripts/evaluate_benchmark.py --model all --dataset all

    # List available models and datasets
    python scripts/evaluate_benchmark.py --list
"""

import argparse
import json
import logging
import time
import wave as wave_mod
from pathlib import Path

import jiwer
import numpy as np
import sherpa_onnx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
OSSICLES_DIR = SCRIPT_DIR.parent
ASSETS_DIR = OSSICLES_DIR / 'assets'
MODELS_DIR = ASSETS_DIR / 'models'

# ── Datasets ──

DATASETS = {
    'figshare-osce': ASSETS_DIR / 'audio' / 'figshare-osce',
    'primock57': ASSETS_DIR / 'audio' / 'primock57',
    'nazmulkazi': ASSETS_DIR / 'audio' / 'nazmulkazi',
}

# ── Model Configs ──
# Each model config specifies how to create a sherpa_onnx.OfflineRecognizer.
# 'factory' is the method name on OfflineRecognizer (e.g., 'from_whisper').
# 'kwargs' are the arguments to pass.
# 'chunk_seconds' is the max audio chunk size for models with length limits.

def _model_path(relative: str) -> str:
    """Resolve a model path relative to assets/models/."""
    return str(MODELS_DIR / relative)


BATCH_MODELS = {
    'whisper-distil-v3.5': {
        'factory': 'from_whisper',
        'kwargs': {
            'encoder': _model_path('sherpa-onnx-whisper-distil-large-v3.5/distil-large-v3.5-encoder.int8.onnx'),
            'decoder': _model_path('sherpa-onnx-whisper-distil-large-v3.5/distil-large-v3.5-decoder.int8.onnx'),
            'tokens': _model_path('sherpa-onnx-whisper-distil-large-v3.5/distil-large-v3.5-tokens.txt'),
            'language': 'en',
            'task': 'transcribe',
        },
        'chunk_seconds': 29,
    },
    'whisper-turbo': {
        'factory': 'from_whisper',
        'kwargs': {
            'encoder': _model_path('sherpa-onnx-whisper-turbo/turbo-encoder.int8.onnx'),
            'decoder': _model_path('sherpa-onnx-whisper-turbo/turbo-decoder.int8.onnx'),
            'tokens': _model_path('sherpa-onnx-whisper-turbo/turbo-tokens.txt'),
            'language': 'en',
            'task': 'transcribe',
        },
        'chunk_seconds': 29,
    },
    'whisper-base-en': {
        'factory': 'from_whisper',
        'kwargs': {
            'encoder': _model_path('sherpa-onnx-whisper-base.en/base.en-encoder.int8.onnx'),
            'decoder': _model_path('sherpa-onnx-whisper-base.en/base.en-decoder.int8.onnx'),
            'tokens': _model_path('sherpa-onnx-whisper-base.en/base.en-tokens.txt'),
            'language': 'en',
            'task': 'transcribe',
        },
        'chunk_seconds': 29,
    },
    'sensevoice': {
        'factory': 'from_sense_voice',
        'kwargs': {
            'model': _model_path('sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx'),
            'tokens': _model_path('sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt'),
            'language': 'en',
            'use_itn': True,
        },
        'chunk_seconds': 60,
    },
    'sensevoice-no-itn': {
        'factory': 'from_sense_voice',
        'kwargs': {
            'model': _model_path('sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx'),
            'tokens': _model_path('sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt'),
            'language': 'en',
            'use_itn': False,
        },
        'chunk_seconds': 60,
    },
    'medasr': {
        'factory': 'from_medasr_ctc',
        'kwargs': {
            'model': _model_path('sherpa-onnx-medasr-ctc-en-2025-12-25/model.onnx'),
            'tokens': _model_path('sherpa-onnx-medasr-ctc-en-2025-12-25/tokens.txt'),
        },
        'chunk_seconds': 60,
    },
    'medasr-int8': {
        'factory': 'from_medasr_ctc',
        'kwargs': {
            'model': _model_path('sherpa-onnx-medasr-ctc-en-int8-2025-12-25/model.int8.onnx'),
            'tokens': _model_path('sherpa-onnx-medasr-ctc-en-int8-2025-12-25/tokens.txt'),
        },
        'chunk_seconds': 60,
    },
    'parakeet-tdt-0.6b-v2': {
        'factory': 'from_transducer',
        'kwargs': {
            'encoder': _model_path('sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx'),
            'decoder': _model_path('sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx'),
            'joiner': _model_path('sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx'),
            'tokens': _model_path('sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt'),
            'model_type': 'nemo_transducer',
            'feature_dim': 128,
        },
        'chunk_seconds': 120,
    },
    'parakeet-tdt-0.6b-v3': {
        'factory': 'from_transducer',
        'kwargs': {
            'encoder': _model_path('sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx'),
            'decoder': _model_path('sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx'),
            'joiner': _model_path('sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx'),
            'tokens': _model_path('sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt'),
            'model_type': 'nemo_transducer',
            'feature_dim': 128,
        },
        'chunk_seconds': 120,
    },
    'paraformer-en': {
        'factory': 'from_paraformer',
        'kwargs': {
            'paraformer': _model_path('sherpa-onnx-paraformer-en-2024-03-09/model.int8.onnx'),
            'tokens': _model_path('sherpa-onnx-paraformer-en-2024-03-09/tokens.txt'),
        },
        'chunk_seconds': 60,
    },
    'fire-red-asr': {
        'factory': 'from_fire_red_asr',
        'kwargs': {
            'encoder': _model_path('sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx'),
            'decoder': _model_path('sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/decoder.int8.onnx'),
            'tokens': _model_path('sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/tokens.txt'),
        },
        'chunk_seconds': 60,
    },
    'fire-red-asr2-ctc': {
        'factory': 'from_fire_red_asr',
        'kwargs': {
            'encoder': _model_path('sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/model.int8.onnx'),
            'decoder': '',
            'tokens': _model_path('sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/tokens.txt'),
        },
        'chunk_seconds': 60,
    },
    'zipformer-zh-en': {
        'factory': 'from_transducer',
        'kwargs': {
            'encoder': _model_path('sherpa-onnx-zipformer-zh-en-2023-11-22/encoder-epoch-34-avg-19.int8.onnx'),
            'decoder': _model_path('sherpa-onnx-zipformer-zh-en-2023-11-22/decoder-epoch-34-avg-19.onnx'),
            'joiner': _model_path('sherpa-onnx-zipformer-zh-en-2023-11-22/joiner-epoch-34-avg-19.int8.onnx'),
            'tokens': _model_path('sherpa-onnx-zipformer-zh-en-2023-11-22/tokens.txt'),
        },
        'chunk_seconds': 120,
    },
    'moonshine-base': {
        'factory': 'from_moonshine',
        'kwargs': {
            'preprocessor': _model_path('sherpa-onnx-moonshine-base-en-int8/preprocess.onnx'),
            'encoder': _model_path('sherpa-onnx-moonshine-base-en-int8/encode.int8.onnx'),
            'uncached_decoder': _model_path('sherpa-onnx-moonshine-base-en-int8/uncached_decode.int8.onnx'),
            'cached_decoder': _model_path('sherpa-onnx-moonshine-base-en-int8/cached_decode.int8.onnx'),
            'tokens': _model_path('sherpa-onnx-moonshine-base-en-int8/tokens.txt'),
        },
        'chunk_seconds': None,  # process full audio
    },
    'qwen3-asr': {
        'factory': 'from_qwen3_asr',
        'kwargs': {
            'conv_frontend': _model_path('sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx'),
            'encoder': _model_path('sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx'),
            'decoder': _model_path('sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx'),
            'tokenizer': _model_path('sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer'),
            'max_total_len': 4096,
            'max_new_tokens': 2048,
        },
        'chunk_seconds': 30,
    },
}

# Streaming models
STREAMING_MODELS = {
    'nemo-fastconformer': {
        'factory': 'streaming',
        'kwargs': {
            'encoder': _model_path('sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-1040ms/encoder.onnx'),
            'decoder': _model_path('sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-1040ms/decoder.onnx'),
            'joiner': _model_path('sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-1040ms/joiner.onnx'),
            'tokens': _model_path('sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-1040ms/tokens.txt'),
            'model_type': 'nemo_transducer',
        },
    },
    'nemo-fastconformer-int8': {
        'factory': 'streaming',
        'kwargs': {
            'encoder': _model_path('sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-1040ms-int8/encoder.int8.onnx'),
            'decoder': _model_path('sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-1040ms-int8/decoder.int8.onnx'),
            'joiner': _model_path('sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-1040ms-int8/joiner.int8.onnx'),
            'tokens': _model_path('sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-1040ms-int8/tokens.txt'),
            'model_type': 'nemo_transducer',
        },
    },
    'nemotron': {
        'factory': 'streaming',
        'kwargs': {
            'encoder': _model_path('sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/encoder.int8.onnx'),
            'decoder': _model_path('sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/decoder.int8.onnx'),
            'joiner': _model_path('sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/joiner.int8.onnx'),
            'tokens': _model_path('sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14/tokens.txt'),
            'model_type': 'nemo_transducer',
            'feature_dim': 128,
        },
    },
    'zipformer': {
        'factory': 'streaming',
        'kwargs': {
            'encoder': _model_path('sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx'),
            'decoder': _model_path('sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.onnx'),
            'joiner': _model_path('sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx'),
            'tokens': _model_path('sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt'),
        },
    },
}

ALL_MODELS = {**BATCH_MODELS, **STREAMING_MODELS}


# ── Normalization ──

def normalize_raw(text: str) -> str:
    """Raw normalization: lowercase + strip punctuation."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_whisper(text: str) -> str:
    """Whisper EnglishTextNormalizer — the de facto standard."""
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
        normalizer = EnglishTextNormalizer()
        return normalizer(text)
    except ImportError:
        log.warning('whisper-normalizer not installed. pip install whisper-normalizer. Falling back to raw.')
        return normalize_raw(text)


NORMALIZERS = {
    'raw': normalize_raw,
    'whisper': normalize_whisper,
}


# ── Audio ──

def read_wav(path: str) -> tuple[int, list[float]]:
    """Read WAV file → (sample_rate, float32_samples)."""
    with wave_mod.open(path, 'rb') as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)
    return sr, samples.tolist()


# ── Recognition ──

def create_recognizer(model_name: str, num_threads: int = 2) -> object:
    """Create a sherpa-onnx recognizer for the given model."""
    if model_name in BATCH_MODELS:
        config = BATCH_MODELS[model_name]
        factory = getattr(sherpa_onnx.OfflineRecognizer, config['factory'])
        return factory(**config['kwargs'], num_threads=num_threads)
    elif model_name in STREAMING_MODELS:
        config = STREAMING_MODELS[model_name]
        kwargs = config['kwargs'].copy()
        model_type = kwargs.pop('model_type', '')
        feature_dim = kwargs.pop('feature_dim', 80)
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            **kwargs,
            num_threads=num_threads,
            model_type=model_type,
            feature_dim=feature_dim,
        )
    else:
        raise ValueError(f'Unknown model: {model_name}')


def transcribe_batch(recognizer, audio_path: str, chunk_seconds: int | None) -> str:
    """Transcribe audio with an offline recognizer, chunking if needed."""
    sr, samples = read_wav(audio_path)

    if chunk_seconds is None or len(samples) <= chunk_seconds * sr:
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, samples)
        recognizer.decode_stream(stream)
        return recognizer.get_result(stream).strip()

    # Chunk processing
    max_samples = chunk_seconds * sr
    transcriptions = []
    offset = 0
    while offset < len(samples):
        end = min(offset + max_samples, len(samples))
        chunk = samples[offset:end]
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, chunk)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        if text:
            transcriptions.append(text)
        offset = end
    return ' '.join(transcriptions)


def transcribe_streaming(recognizer, audio_path: str) -> str:
    """Transcribe audio with an online (streaming) recognizer."""
    sr, samples = read_wav(audio_path)
    stream = recognizer.create_stream()

    # Process in 100ms chunks (matching ossicles Dart implementation)
    chunk_size = int(sr * 0.1)
    offset = 0
    while offset < len(samples):
        end = min(offset + chunk_size, len(samples))
        stream.accept_waveform(sr, samples[offset:end])
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        offset = end

    # Flush
    tail_paddings = [0.0] * int(sr * 0.5)
    stream.accept_waveform(sr, tail_paddings)
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    return recognizer.get_result(stream).strip()


# ── Evaluation ──

def discover_files(dataset_dir: Path) -> list[dict]:
    """Find WAV + TXT pairs in a dataset directory."""
    files = []
    for wav_path in sorted(dataset_dir.glob('*.wav')):
        txt_path = dataset_dir / f'{wav_path.stem}.txt'
        if not txt_path.exists():
            continue
        ref_text = txt_path.read_text(encoding='utf-8', errors='replace').strip()
        if not ref_text:
            continue
        files.append({
            'audio': str(wav_path),
            'reference': ref_text,
            'file_id': wav_path.stem,
        })
    return files


def evaluate_model_on_dataset(
    model_name: str,
    dataset_name: str,
    output_dir: Path,
    num_threads: int = 2,
) -> None:
    """Run a single model on a single dataset with per-file saving."""
    dataset_dir = DATASETS[dataset_name]
    files = discover_files(dataset_dir)
    if not files:
        log.error(f'No files found in {dataset_dir}')
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{model_name}.json'

    # Load existing results for resume (only our format — dict with 'results' + 'aggregates')
    existing_results = {}
    if output_file.exists():
        try:
            data = json.loads(output_file.read_text())
            if isinstance(data, dict) and 'aggregates' in data:
                # Our format — safe to resume
                for r in data.get('results', []):
                    if isinstance(r, dict) and 'file_id' in r and 'hypothesis' in r:
                        existing_results[r['file_id']] = r
            else:
                # Old ossicles format — back up and regenerate
                backup = output_file.with_suffix('.old.json')
                if not backup.exists():
                    output_file.rename(backup)
                    log.info(f'Backed up old-format results to {backup.name}')
                else:
                    output_file.unlink()
        except (json.JSONDecodeError, KeyError):
            pass

    if len(existing_results) >= len(files):
        log.info(f'SKIP {model_name} on {dataset_name} — already complete ({len(existing_results)} files)')
        return

    log.info(f'Evaluating {model_name} on {dataset_name}: {len(files)} files '
             f'({len(existing_results)} already done)')

    # Create recognizer
    is_streaming = model_name in STREAMING_MODELS
    recognizer = create_recognizer(model_name, num_threads)

    chunk_seconds = None
    if not is_streaming and model_name in BATCH_MODELS:
        chunk_seconds = BATCH_MODELS[model_name].get('chunk_seconds')

    results = list(existing_results.values())

    for i, entry in enumerate(files):
        file_id = entry['file_id']

        # Skip already processed
        if file_id in existing_results:
            continue

        t0 = time.time()
        try:
            if is_streaming:
                hyp_text = transcribe_streaming(recognizer, entry['audio'])
            else:
                hyp_text = transcribe_batch(recognizer, entry['audio'], chunk_seconds)
        except Exception as e:
            log.error(f'ERROR {file_id}: {e}')
            hyp_text = ''
        elapsed = time.time() - t0

        ref_text = entry['reference']

        # Compute WER with both normalizers
        wer_results = {}
        for norm_name, norm_fn in NORMALIZERS.items():
            ref_norm = norm_fn(ref_text)
            hyp_norm = norm_fn(hyp_text)
            if ref_norm:
                wer = jiwer.wer(ref_norm, hyp_norm)
                n_words = len(ref_norm.split())
                errors = round(wer * n_words)
            else:
                wer = 0.0
                n_words = 0
                errors = 0
            wer_results[norm_name] = {
                'wer': round(wer, 6),
                'wer_percent': round(wer * 100, 2),
                'errors': errors,
                'ref_words': n_words,
            }

        result = {
            'file_id': file_id,
            'hypothesis': hyp_text,
            'reference': ref_text,
            'inference_time_s': round(elapsed, 2),
            'wer': wer_results,
        }
        results.append(result)

        # Compute running aggregate
        for norm_name in NORMALIZERS:
            total_errors = sum(r['wer'][norm_name]['errors'] for r in results if norm_name in r.get('wer', {}))
            total_words = sum(r['wer'][norm_name]['ref_words'] for r in results if norm_name in r.get('wer', {}))
            agg_wer = total_errors / total_words if total_words > 0 else 0.0
            wer_results[f'{norm_name}_aggregate'] = round(agg_wer * 100, 2)

        done = len(results)
        log.info(f'[{dataset_name}] {done}/{len(files)} {file_id}: '
                 f'raw={wer_results["raw"]["wer_percent"]:.1f}% '
                 f'whisper={wer_results["whisper"]["wer_percent"]:.1f}% '
                 f'[{elapsed:.1f}s]')

        # Save immediately
        _save_results(output_file, model_name, dataset_name, results, len(files))

    log.info(f'DONE {model_name} on {dataset_name}: {len(results)} files')


def _save_results(output_file: Path, model_name: str, dataset_name: str,
                  results: list[dict], total_files: int) -> None:
    """Save results JSON — called after every file."""
    aggregates = {}
    for norm_name in NORMALIZERS:
        total_errors = sum(r['wer'][norm_name]['errors'] for r in results if norm_name in r.get('wer', {}))
        total_words = sum(r['wer'][norm_name]['ref_words'] for r in results if norm_name in r.get('wer', {}))
        agg_wer = total_errors / total_words if total_words > 0 else 0.0
        aggregates[norm_name] = {
            'wer': round(agg_wer, 6),
            'wer_percent': round(agg_wer * 100, 2),
            'total_errors': total_errors,
            'total_words': total_words,
        }

    data = {
        'model': model_name,
        'dataset': dataset_name,
        'files_completed': len(results),
        'files_total': total_files,
        'aggregates': aggregates,
        'results': results,
    }
    output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description='Clinical ASR Benchmark Evaluation')
    parser.add_argument('--model', type=str, default='all',
                        help='Model name or "all"')
    parser.add_argument('--dataset', type=str, default='all',
                        help='Dataset name or "all"')
    parser.add_argument('--threads', type=int, default=2,
                        help='Threads per model (default: 2)')
    parser.add_argument('--list', action='store_true',
                        help='List available models and datasets')
    args = parser.parse_args()

    if args.list:
        print('Datasets:')
        for name, path in DATASETS.items():
            n_files = len(list(path.glob('*.wav'))) if path.exists() else 0
            print(f'  {name}: {n_files} files ({path})')
        print('\nBatch models:')
        for name in sorted(BATCH_MODELS):
            print(f'  {name}')
        print('\nStreaming models:')
        for name in sorted(STREAMING_MODELS):
            print(f'  {name}')
        return

    # Models to skip (too slow, not competitive)
    SKIP_MODELS = {'fire-red-asr', 'fire-red-asr2-ctc', 'parakeet-tdt-0.6b-v3', 'moonshine-base'}

    # Resolve models
    if args.model == 'all':
        models = [m for m in ALL_MODELS.keys() if m not in SKIP_MODELS]
    else:
        models = [args.model]
        if args.model not in ALL_MODELS:
            log.error(f'Unknown model: {args.model}. Use --list to see available models.')
            return

    # Resolve datasets
    if args.dataset == 'all':
        datasets = list(DATASETS.keys())
    else:
        datasets = [args.dataset]
        if args.dataset not in DATASETS:
            log.error(f'Unknown dataset: {args.dataset}. Use --list to see available datasets.')
            return

    for model_name in models:
        # Check model files exist
        config = ALL_MODELS[model_name]
        kwargs = config['kwargs'] if 'kwargs' in config else config
        non_file_keys = {'model_type', 'language', 'task', 'feature_dim', 'use_itn',
                         'decoding_method', 'debug', 'provider'}
        missing = False
        for key, val in kwargs.items():
            if key in non_file_keys:
                continue
            if isinstance(val, str) and val and not Path(val).exists():
                log.warning(f'Missing file for {model_name}: {key}={val}')
                missing = True
        if missing:
            log.error(f'Skipping {model_name} — missing model files')
            continue

        for dataset_name in datasets:
            output_dir = OSSICLES_DIR / f'benchmark_results_{dataset_name}'
            evaluate_model_on_dataset(
                model_name, dataset_name, output_dir,
                num_threads=args.threads,
            )


if __name__ == '__main__':
    main()
