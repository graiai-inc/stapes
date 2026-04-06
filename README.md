# stapes

Open clinical ASR benchmark comparing on-device and cloud speech recognition models across 400 medical conversations.

## Overview

First open, reproducible benchmark for medical automatic speech recognition (ASR) using freely available datasets. Compares 14+ on-device models (via sherpa-onnx) against 5 commercial cloud APIs, evaluating Word Error Rate (WER), medical term accuracy, and ROVER fusion performance.

## Datasets

| Dataset | Conversations | Hours | Domain | License |
|---------|--------------|-------|--------|---------|
| figshare-osce | 272 | ~51h | 6 specialties (OSCE) | CC-BY |
| primock57 | 57 | ~13h | UK primary care | CC-BY |
| nazmulkazi | 71 | ~10-15h | Psychiatric | CC-BY 4.0 |

## On-Device Models

All on-device models run via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (v1.12.34):

- whisper-distil-v3.5, whisper-turbo, whisper-base-en
- sensevoice, sensevoice-no-itn
- medasr, medasr-int8
- parakeet-tdt-0.6b-v2
- paraformer-en
- nemo-fastconformer, nemo-fastconformer-int8
- nemotron
- qwen3-asr
- zipformer-zh-en

## Cloud APIs

- Azure Speech (medical model)
- Google Cloud Speech (medical_conversation)
- Deepgram (Nova-2 Medical)
- AssemblyAI (Best)
- AWS Transcribe Medical (representative subset)

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/evaluate_benchmark.py` | Run on-device models via sherpa-onnx |
| `scripts/evaluate_cloud_asr.py` | Run cloud API evaluations |
| `scripts/rover_fusion.py` | ROVER fusion across all model pairs |
| `scripts/analyze_medical_errors.py` | Medical term error analysis |
| `scripts/prepare_primock57.py` | Prepare primock57 dataset |
| `scripts/prepare_nazmulkazi.py` | Prepare nazmulkazi dataset |

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or: source venv/bin/activate.fish
pip install -r requirements.txt
```

## Normalization

All WER comparisons use the [Whisper EnglishTextNormalizer](https://github.com/openai/whisper) (HuggingFace standard). Both raw and whisper-normalized WER are reported in all result files.

## License

MIT
