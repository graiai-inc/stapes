#!/usr/bin/env python3
"""Cloud ASR Benchmark Evaluation.

Runs cloud ASR services (Deepgram, Google, AWS, AssemblyAI, Azure) on a subset
of benchmark audio files and saves results in the same format as evaluate_benchmark.py.

Usage:
    # Run all services on the 50-file subset
    python scripts/evaluate_cloud_asr.py --service all

    # Run one service
    python scripts/evaluate_cloud_asr.py --service deepgram

    # Run on a custom file list
    python scripts/evaluate_cloud_asr.py --service google --subset custom_files.txt

    # List available services
    python scripts/evaluate_cloud_asr.py --list
"""

import argparse
import json
import logging
import os
import subprocess
import time
import wave as wave_mod
from datetime import datetime, timezone
from pathlib import Path

import jiwer
from dotenv import load_dotenv
from whisper_normalizer.english import EnglishTextNormalizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
OSSICLES_DIR = SCRIPT_DIR.parent
ASSETS_DIR = OSSICLES_DIR / 'assets'

DATASETS = {
    'figshare-osce': ASSETS_DIR / 'audio' / 'figshare-osce',
    'primock57': ASSETS_DIR / 'audio' / 'primock57',
    'nazmulkazi': ASSETS_DIR / 'audio' / 'nazmulkazi',
}

load_dotenv(OSSICLES_DIR / '.env')

# Google SDK auth — use service account JSON if set, otherwise ADC
if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
    creds_path = Path(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    if not creds_path.is_absolute():
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(OSSICLES_DIR / creds_path)

# ── 50-file representative subset ──
# All small specialties + random sample from MSK/RES (seed=42)
# ~9.8 hours, all 6 specialties covered

SUBSET_FILE_IDS = [
    'CAR0001', 'CAR0002', 'CAR0003', 'CAR0004', 'CAR0005',
    'DER0001',
    'GAS0001', 'GAS0002', 'GAS0003', 'GAS0004', 'GAS0005', 'GAS0007',
    'GEN0001',
    'MSK0003', 'MSK0004', 'MSK0007', 'MSK0008', 'MSK0009', 'MSK0010',
    'MSK0015', 'MSK0016', 'MSK0017', 'MSK0019', 'MSK0029', 'MSK0037',
    'MSK0039', 'MSK0043', 'MSK0046',
    'RES0002', 'RES0007', 'RES0043', 'RES0053', 'RES0059', 'RES0062',
    'RES0074', 'RES0090', 'RES0110', 'RES0111', 'RES0118', 'RES0133',
    'RES0143', 'RES0147', 'RES0154', 'RES0159', 'RES0171', 'RES0183',
    'RES0184', 'RES0188', 'RES0199', 'RES0203',
]


def load_files(dataset_name: str, subset: bool = False,
               file_list: str | None = None) -> list[dict]:
    """Load audio file paths and reference transcripts for a dataset.

    If file_list is provided, only load those file IDs.
    If subset=True and dataset is figshare-osce, only load the 50-file
    representative subset. Otherwise load all files.
    """
    audio_dir = DATASETS[dataset_name]

    if file_list:
        file_ids = Path(file_list).read_text().strip().split('\n')
        wav_files = [audio_dir / f'{fid.strip()}.wav' for fid in file_ids if fid.strip()]
    elif subset and dataset_name == 'figshare-osce':
        file_ids = SUBSET_FILE_IDS
        wav_files = [audio_dir / f'{fid}.wav' for fid in file_ids]
    else:
        wav_files = sorted(audio_dir.glob('*.wav'))

    files = []
    for wav in wav_files:
        fid = wav.stem
        ref = audio_dir / f'{fid}.txt'
        if not wav.exists():
            log.warning(f'Audio not found: {wav}')
            continue
        if not ref.exists():
            log.warning(f'Reference not found: {ref}')
            continue
        files.append({
            'file_id': fid,
            'audio': str(wav),
            'reference': ref.read_text().strip(),
        })
    return files


WHISPER_NORM = EnglishTextNormalizer()


def compute_wer(reference: str, hypothesis: str) -> dict:
    """Compute WER with both raw and Whisper normalization."""
    ref_raw = reference.lower().strip()
    hyp_raw = hypothesis.lower().strip()

    ref_whisper = WHISPER_NORM(reference)
    hyp_whisper = WHISPER_NORM(hypothesis)

    result = {}
    for name, ref_n, hyp_n in [('raw', ref_raw, hyp_raw),
                                 ('whisper', ref_whisper, hyp_whisper)]:
        if not ref_n:
            result[name] = {'wer': 0.0, 'wer_percent': 0.0, 'errors': 0, 'ref_words': 0}
            continue
        wer_score = jiwer.wer(ref_n, hyp_n)
        ref_words = len(ref_n.split())
        errors = round(wer_score * ref_words)
        result[name] = {
            'wer': round(wer_score, 6),
            'wer_percent': round(wer_score * 100, 2),
            'errors': errors,
            'ref_words': ref_words,
        }

    return result


# ── Service implementations ──

def transcribe_deepgram(wav_path: str) -> str:
    """Transcribe using Deepgram Nova-2 Medical (REST API)."""
    import requests

    url = 'https://api.deepgram.com/v1/listen?model=nova-2-medical&language=en-US&smart_format=true'
    headers = {
        'Authorization': f'Token {os.environ["DEEPGRAM_API_KEY"]}',
        'Content-Type': 'audio/wav',
    }

    with open(wav_path, 'rb') as f:
        resp = requests.post(url, headers=headers, data=f, timeout=600)

    resp.raise_for_status()
    return resp.json()['results']['channels'][0]['alternatives'][0]['transcript']


def transcribe_google(wav_path: str) -> str:
    """Transcribe using Google Cloud Speech-to-Text v1 medical_conversation."""
    from google.cloud import speech_v1, storage

    bucket_name = os.environ['GCS_BUCKET']

    # Upload to GCS (required for files > 1 min)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob_name = f'asr-benchmark/{Path(wav_path).name}'
    blob = bucket.blob(blob_name)
    if not blob.exists():
        blob.upload_from_filename(wav_path)

    gcs_uri = f'gs://{bucket_name}/{blob_name}'

    # Long-running recognize
    client = speech_v1.SpeechClient()
    audio = speech_v1.RecognitionAudio(uri=gcs_uri)
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US',
        model='medical_conversation',
        enable_automatic_punctuation=True,
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=600)

    transcript = ''
    for result in response.results:
        transcript += result.alternatives[0].transcript
    return transcript.strip()


def transcribe_aws(wav_path: str, job_name: str) -> str:
    """Transcribe using AWS Transcribe Medical."""
    import boto3
    import urllib.request

    region = os.environ['AWS_REGION']
    bucket = os.environ['AWS_S3_BUCKET']

    s3 = boto3.client('s3', region_name=region)
    transcribe = boto3.client('transcribe', region_name=region)

    # Upload to S3
    s3_key = f'asr-benchmark/{Path(wav_path).name}'
    s3.upload_file(wav_path, bucket, s3_key)
    media_uri = f's3://{bucket}/{s3_key}'

    # Start job
    transcribe.start_medical_transcription_job(
        MedicalTranscriptionJobName=job_name,
        Media={'MediaFileUri': media_uri},
        MediaFormat='wav',
        MediaSampleRateHertz=16000,
        LanguageCode='en-US',
        Specialty='PRIMARYCARE',
        Type='CONVERSATION',
        OutputBucketName=bucket,
        OutputKey=f'asr-benchmark-output/{job_name}.json',
    )

    # Poll
    while True:
        status = transcribe.get_medical_transcription_job(
            MedicalTranscriptionJobName=job_name
        )
        job_status = status['MedicalTranscriptionJob']['TranscriptionJobStatus']
        if job_status in ('COMPLETED', 'FAILED'):
            break
        time.sleep(5)

    if job_status == 'FAILED':
        reason = status['MedicalTranscriptionJob'].get('FailureReason', 'unknown')
        raise RuntimeError(f'AWS Transcribe failed: {reason}')

    # Read result directly from S3 (pre-signed URL can be forbidden)
    result_key = f'asr-benchmark-output/{job_name}.json'
    obj = s3.get_object(Bucket=bucket, Key=result_key)
    result = json.loads(obj['Body'].read())

    return result['results']['transcripts'][0]['transcript']


def transcribe_assemblyai(wav_path: str) -> str:
    """Transcribe using AssemblyAI (REST API, universal-3-pro)."""
    import requests

    base_url = 'https://api.assemblyai.com'
    headers = {'authorization': os.environ['ASSEMBLY_AI_KEY']}

    # Upload audio
    with open(wav_path, 'rb') as f:
        upload_resp = requests.post(f'{base_url}/v2/upload', headers=headers, data=f, timeout=300)
    upload_resp.raise_for_status()
    audio_url = upload_resp.json()['upload_url']

    # Start transcription
    data = {
        'audio_url': audio_url,
        'speech_models': ['universal-3-pro', 'universal-2'],
        'language_code': 'en_us',
        'speaker_labels': True,
        'speakers_expected': 2,
        'temperature': 0.0,
        'prompt': 'This is a doctor-patient clinical conversation.',
    }
    resp = requests.post(f'{base_url}/v2/transcript', json=data, headers=headers, timeout=30)
    resp.raise_for_status()
    transcript_id = resp.json()['id']

    # Poll
    polling_url = f'{base_url}/v2/transcript/{transcript_id}'
    while True:
        result = requests.get(polling_url, headers=headers, timeout=120).json()
        if result['status'] == 'completed':
            return result['text']
        elif result['status'] == 'error':
            raise RuntimeError(f'AssemblyAI failed: {result["error"]}')
        time.sleep(3)


def transcribe_azure(wav_path: str) -> str:
    """Transcribe using Azure Speech-to-Text batch API."""
    import requests
    from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
    from datetime import datetime, timedelta, timezone

    region = os.environ['AZURE_REGION']
    speech_key = os.environ['AZURE_KEY_1']
    storage_account = os.environ['AZURE_STORAGE_ACCOUNT_NAME']
    storage_key = os.environ['AZURE_STORAGE_KEY_1']
    container = os.environ['AZURE_STORAGE_CONTAINER']
    conn_str = os.environ['AZURE_STORAGE_CONNECTION_STRING_1']

    base_url = f'https://{region}.api.cognitive.microsoft.com/speechtotext/v3.2'
    headers = {
        'Ocp-Apim-Subscription-Key': speech_key,
        'Content-Type': 'application/json',
    }

    # Upload to Azure Blob Storage
    blob_name = f'asr-benchmark/{Path(wav_path).name}'
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    blob_client = blob_service.get_blob_client(container, blob_name)
    if not blob_client.exists():
        with open(wav_path, 'rb') as f:
            blob_client.upload_blob(f)

    # Generate SAS URL
    sas_token = generate_blob_sas(
        account_name=storage_account,
        container_name=container,
        blob_name=blob_name,
        account_key=storage_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(hours=2),
    )
    sas_url = f'https://{storage_account}.blob.core.windows.net/{container}/{blob_name}?{sas_token}'

    # Submit batch transcription
    body = {
        'contentUrls': [sas_url],
        'locale': 'en-US',
        'displayName': f'bench-{Path(wav_path).stem}-{int(time.time())}',
        'properties': {
            'punctuationMode': 'DictatedAndAutomatic',
            'profanityFilterMode': 'None',
        },
    }
    resp = requests.post(f'{base_url}/transcriptions', headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    transcription_url = resp.json()['self']

    # Poll until done
    while True:
        status_resp = requests.get(transcription_url, headers=headers, timeout=30)
        status_resp.raise_for_status()
        status = status_resp.json()['status']
        if status in ('Succeeded', 'Failed'):
            break
        time.sleep(5)

    if status == 'Failed':
        raise RuntimeError(f'Azure batch transcription failed: {status_resp.json()}')

    # Fetch results
    files_resp = requests.get(f'{transcription_url}/files', headers=headers, timeout=30)
    files_resp.raise_for_status()

    transcript = ''
    for f in files_resp.json()['values']:
        if f['kind'] != 'Transcription':
            continue
        content_url = f['links']['contentUrl']
        result = requests.get(content_url, timeout=30).json()
        for segment in result['combinedRecognizedPhrases']:
            transcript += segment['display'] + ' '

    return transcript.strip()


SERVICES = {
    'deepgram': transcribe_deepgram,
    'google': transcribe_google,
    'aws': transcribe_aws,
    'assemblyai': transcribe_assemblyai,
    'azure': transcribe_azure,
}

# Exact request configuration per service, recorded in each result's
# provenance block so cloud runs are self-documenting (these have no local
# model files to hash; the API/model/params are the reproducibility surface).
SERVICE_CONFIG = {
    'deepgram': {
        'api': 'Deepgram /v1/listen',
        'model': 'nova-2-medical',
        'params': {'language': 'en-US', 'smart_format': True},
    },
    'google': {
        'api': 'Google Cloud Speech-to-Text v1',
        'model': 'medical_conversation',
        'params': {'encoding': 'LINEAR16', 'sample_rate_hertz': 16000,
                   'language_code': 'en-US'},
    },
    'aws': {
        'api': 'AWS Transcribe Medical (start_medical_transcription_job)',
        'model': 'medical',
        'params': {'Specialty': 'PRIMARYCARE', 'Type': 'CONVERSATION'},
    },
    'assemblyai': {
        'api': 'AssemblyAI /v2/transcript',
        'model': 'universal-3-pro',
        'params': {'speech_models': ['universal-3-pro', 'universal-2'],
                   'language_code': 'en_us',
                   'prompt': 'This is a doctor-patient clinical conversation.'},
    },
    'azure': {
        'api': 'Azure Speech Batch Transcription v3.2',
        'model': 'default',
        'params': {'locale': 'en-US'},
    },
}


def _git_commit() -> str | None:
    """Short git commit of the repo, or None if unavailable."""
    try:
        out = subprocess.run(
            ['git', '-C', str(OSSICLES_DIR), 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _cloud_provenance(service_name: str) -> dict:
    """Record the exact API, model, and request config used for a cloud run,
    so each result JSON self-documents what produced it."""
    return {
        'service': service_name,
        'request_config': SERVICE_CONFIG.get(service_name),
        'git_commit': _git_commit(),
        'generated_utc': datetime.now(timezone.utc).isoformat(
            timespec='seconds'),
    }


def run_service(service_name: str, dataset_name: str, files: list[dict],
                subset: bool = False) -> None:
    """Run a single cloud ASR service on all files, saving results incrementally."""
    suffix = f'{dataset_name}-subset' if subset else dataset_name
    output_dir = OSSICLES_DIR / f'benchmark_results_cloud_{suffix}'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'{service_name}.json'

    transcribe_fn = SERVICES[service_name]
    provenance = _cloud_provenance(service_name)

    # Resume support
    existing_results = {}
    if output_file.exists():
        try:
            data = json.loads(output_file.read_text())
            if isinstance(data, dict) and 'results' in data:
                for r in data.get('results', []):
                    if isinstance(r, dict) and 'file_id' in r and 'hypothesis' in r:
                        existing_results[r['file_id']] = r
        except (json.JSONDecodeError, KeyError):
            pass

    if len(existing_results) >= len(files):
        log.info(f'SKIP {service_name} — already complete ({len(existing_results)} files)')
        return

    log.info(f'Evaluating {service_name}: {len(files)} files '
             f'({len(existing_results)} already done)')

    results = list(existing_results.values())

    for i, entry in enumerate(files):
        file_id = entry['file_id']

        if file_id in existing_results:
            continue

        t0 = time.time()
        try:
            if service_name == 'aws':
                job_name = f'bench-{file_id}-{int(time.time())}'
                hyp_text = transcribe_fn(entry['audio'], job_name)
            else:
                hyp_text = transcribe_fn(entry['audio'])
        except Exception as e:
            log.error(f'[{service_name}] {file_id}: FAILED — {e}')
            continue

        elapsed = time.time() - t0
        wer = compute_wer(entry['reference'], hyp_text)

        result = {
            'file_id': file_id,
            'hypothesis': hyp_text,
            'reference': entry['reference'],
            'inference_time_s': round(elapsed, 1),
            'wer': wer,
        }
        results.append(result)

        # Compute running aggregates
        aggregates = {}
        for norm_name in ('raw', 'whisper'):
            total_errors = sum(r['wer'][norm_name]['errors'] for r in results)
            total_words = sum(r['wer'][norm_name]['ref_words'] for r in results)
            agg_wer = total_errors / total_words if total_words > 0 else 0.0
            aggregates[norm_name] = {
                'wer': round(agg_wer, 6),
                'wer_percent': round(agg_wer * 100, 2),
                'total_errors': total_errors,
                'total_words': total_words,
            }

        output = {
            'service': service_name,
            'dataset': dataset_name,
            'provenance': provenance,
            'files_completed': len(results),
            'files_total': len(files),
            'aggregates': aggregates,
            'results': results,
        }

        output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False))

        log.info(f'[{service_name}] {i + 1}/{len(files)} {file_id}: '
                 f'raw={wer["raw"]["wer_percent"]:.1f}% '
                 f'whisper={wer["whisper"]["wer_percent"]:.1f}% [{elapsed:.1f}s]')

    log.info(f'DONE {service_name}: {len(results)} files, '
             f'raw={output["aggregates"]["raw"]["wer_percent"]:.2f}% '
             f'whisper={output["aggregates"]["whisper"]["wer_percent"]:.2f}%')


def main():
    parser = argparse.ArgumentParser(description='Cloud ASR benchmark evaluation')
    parser.add_argument('--service', default='all',
                        help='Service to run (deepgram, google, aws, assemblyai, azure, all)')
    parser.add_argument('--dataset', default='all',
                        help='Dataset to run (figshare-osce, primock57, nazmulkazi, all)')
    parser.add_argument('--subset', action='store_true',
                        help='Use 50-file representative subset (figshare-osce only)')
    parser.add_argument('--file-list', default=None,
                        help='Text file with file IDs to process (one per line)')
    parser.add_argument('--list', action='store_true', help='List available services and datasets')
    args = parser.parse_args()

    if args.list:
        print('Available services:')
        for name in SERVICES:
            print(f'  {name}')
        print('\nAvailable datasets:')
        for name in DATASETS:
            print(f'  {name}')
        return

    if args.service == 'all':
        services = list(SERVICES.keys())
    else:
        services = [args.service]
        if args.service not in SERVICES:
            log.error(f'Unknown service: {args.service}. Use --list.')
            return

    if args.dataset == 'all':
        datasets = list(DATASETS.keys())
    else:
        datasets = [args.dataset]
        if args.dataset not in DATASETS:
            log.error(f'Unknown dataset: {args.dataset}. Use --list.')
            return

    for dataset_name in datasets:
        files = load_files(dataset_name, subset=args.subset, file_list=args.file_list)
        log.info(f'Dataset {dataset_name}: {len(files)} files'
                 f'{" (subset)" if args.subset else ""}')

        for service_name in services:
            try:
                run_service(service_name, dataset_name, files, subset=args.subset)
            except Exception as e:
                log.error(f'{service_name} on {dataset_name} failed: {e}')
                continue


if __name__ == '__main__':
    main()
