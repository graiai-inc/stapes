#!/usr/bin/env python3
"""Count parameters directly from each on-device model's ONNX initializers.

This is the authoritative, guess-free source for the Table 1 parameter column:
it sums the element count of every weight initializer across all .onnx files in
a model directory (encoder/decoder/joiner). int8 and fp32 share the same param
COUNT, so quantized dirs give the true count. Writes results incrementally.
"""

from pathlib import Path

import onnx

MODELS_DIR = Path('/home/grey/dev/graiai/ossicles/assets/models')
OUT = Path('/home/grey/dev/graiai/stapes/results/_audit/model_params.tsv')
OUT.parent.mkdir(parents=True, exist_ok=True)

# Table 1 label -> on-disk model directory (one entry per unique model file set).
MODELS = [
    ('Whisper distil-v3.5', 'sherpa-onnx-whisper-distil-large-v3.5'),
    ('Whisper turbo', 'sherpa-onnx-whisper-turbo'),
    ('Whisper base.en', 'sherpa-onnx-whisper-base.en'),
    ('Qwen3-ASR', 'sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25'),
    ('Parakeet-TDT-0.6b-v2', 'sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8'),
    ('SenseVoice', 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17'),
    ('Paraformer-en', 'sherpa-onnx-paraformer-en-2024-03-09'),
    ('NeMo FastConformer', 'sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-1040ms'),
    ('Nemotron', 'sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14'),
    ('Zipformer', 'sherpa-onnx-zipformer-zh-en-2023-11-22'),
    ('MedASR', 'sherpa-onnx-medasr-ctc-en-2025-12-25'),
]


def numel(dims) -> int:
    n = 1
    for d in dims:
        n *= d
    return n


def count_dir(d: Path) -> tuple[int, list[str]]:
    total = 0
    detail = []
    # Dedup int8/fp32 twins: if both 'X.onnx' and 'X.int8.onnx' exist (same
    # component), count the component once. Group by stem with '.int8' removed.
    by_stem: dict[str, Path] = {}
    for f in sorted(d.glob('*.onnx')):
        stem = f.name.replace('.int8.onnx', '').replace('.onnx', '')
        # Prefer fp32 if present; otherwise keep whatever (int8) we have.
        if stem not in by_stem or '.int8.' in by_stem[stem].name:
            by_stem[stem] = f
    for stem, f in sorted(by_stem.items()):
        model = onnx.load(str(f), load_external_data=False)
        sub = sum(numel(init.dims) for init in model.graph.initializer)
        detail.append(f'{f.name}={sub/1e6:.1f}M')
        total += sub
    return total, detail


fh = open(OUT, 'w')
fh.write('label\tparams_millions\tdir\tonnx_breakdown\n')
fh.flush()
for label, dirname in MODELS:
    d = MODELS_DIR / dirname
    if not d.is_dir():
        line = f'{label}\tNA\t{dirname}\t(dir missing)'
        fh.write(line + '\n'); fh.flush(); print(line, flush=True)
        continue
    total, detail = count_dir(d)
    line = f'{label}\t{total/1e6:.1f}\t{dirname}\t{"; ".join(detail)}'
    fh.write(line + '\n'); fh.flush()
    print(f'{label}: {total/1e6:.1f}M params  [{"; ".join(detail)}]', flush=True)
fh.close()
print('\nwrote', OUT, flush=True)
