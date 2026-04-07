#!/usr/bin/env python3
"""ROVER fusion for all model pairs across all datasets.

Reads saved benchmark results (hypothesis text per file), fuses all model
pairs using ROVER, and computes WER on the fused output.

This operates entirely on saved text — no audio reprocessing needed.

Usage:
    # Run all pairs on all datasets
    python scripts/rover_fusion.py

    # Run specific pair on specific dataset
    python scripts/rover_fusion.py --model1 whisper-distil-v3.5 --model2 sensevoice-no-itn --dataset figshare-osce
"""

import argparse
import itertools
import json
import logging
import re
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
OSSICLES_DIR = SCRIPT_DIR.parent

DATASETS = ['figshare-osce', 'primock57', 'nazmulkazi']


# ── ROVER Algorithm (ported from Dart) ──

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return [w for w in text.split() if w]


def _levenshtein(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def _word_similarity(w1: str, w2: str) -> float:
    if w1 == w2:
        return 2.0
    dist = _levenshtein(w1, w2)
    max_len = max(len(w1), len(w2))
    sim = 1.0 - (dist / max_len)
    return sim if sim > 0.7 else -1.0


def _align_sequences(seq1: list[str], seq2: list[str]) -> list[tuple]:
    """DP alignment returning list of (word1_or_None, word2_or_None, idx1, idx2)."""
    m, n = len(seq1), len(seq2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    bt = [[''] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] - 1
        bt[i][0] = 'D'
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] - 1
        bt[0][j] = 'I'

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = _word_similarity(seq1[i - 1], seq2[j - 1])
            scores = [
                dp[i - 1][j - 1] + match_score,
                dp[i - 1][j] - 1,
                dp[i][j - 1] - 1,
            ]
            best = max(scores)
            dp[i][j] = best
            if best == scores[0]:
                bt[i][j] = 'M'
            elif best == scores[1]:
                bt[i][j] = 'D'
            else:
                bt[i][j] = 'I'

    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i == 0:
            alignment.append((None, seq2[j - 1], -1, j - 1))
            j -= 1
        elif j == 0:
            alignment.append((seq1[i - 1], None, i - 1, -1))
            i -= 1
        elif bt[i][j] == 'M':
            alignment.append((seq1[i - 1], seq2[j - 1], i - 1, j - 1))
            i -= 1
            j -= 1
        elif bt[i][j] == 'D':
            alignment.append((seq1[i - 1], None, i - 1, -1))
            i -= 1
        else:
            alignment.append((None, seq2[j - 1], -1, j - 1))
            j -= 1

    return list(reversed(alignment))


def rover_fuse(text1: str, text2: str, w1: float = 0.5, w2: float = 0.5) -> str:
    """Fuse two ASR hypotheses using ROVER."""
    words1 = _tokenize(text1)
    words2 = _tokenize(text2)

    if not words1:
        return text2
    if not words2:
        return text1

    alignment = _align_sequences(words1, words2)

    # Build word transition network
    best_path = []
    for word1, word2, idx1, idx2 in alignment:
        candidates = {}
        if word1 is not None:
            candidates[word1] = candidates.get(word1, 0) + 0.8 * w1
        if word2 is not None:
            candidates[word2] = candidates.get(word2, 0) + 0.8 * w2
        if candidates:
            best = max(candidates, key=candidates.get)
            best_path.append(best)

    return ' '.join(best_path)


# ── Load results ──

def load_model_results(dataset: str, model: str) -> dict[str, dict] | None:
    """Load benchmark results for a model on a dataset.

    Returns dict mapping file_id → {hypothesis, reference, wer}.
    """
    results_dir = OSSICLES_DIR / f'benchmark_results_{dataset}'
    result_file = results_dir / f'{model}.json'

    if not result_file.exists():
        return None

    data = json.loads(result_file.read_text())
    results = {}
    # Only accept Python-format results (dict with 'results' + 'aggregates')
    if isinstance(data, dict) and 'aggregates' in data and 'results' in data:
        for r in data['results']:
            if isinstance(r, dict) and 'file_id' in r and 'hypothesis' in r:
                results[r['file_id']] = r
    return results if results else None


WHISPER_NORM = EnglishTextNormalizer()


def normalize_raw(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── Main ──

def run_fusion(model1: str, model2: str, dataset: str) -> dict | None:
    """Run ROVER fusion for one model pair on one dataset."""
    r1 = load_model_results(dataset, model1)
    r2 = load_model_results(dataset, model2)

    if r1 is None:
        log.debug(f'No results for {model1} on {dataset}')
        return None
    if r2 is None:
        log.debug(f'No results for {model2} on {dataset}')
        return None

    # Find common files
    common_ids = set(r1.keys()) & set(r2.keys())
    if not common_ids:
        log.debug(f'No common files for {model1}+{model2} on {dataset}')
        return None

    total_errors_fused = 0
    total_errors_m1 = 0
    total_errors_m2 = 0
    total_words = 0
    wins_fused = 0
    wins_m1 = 0
    wins_m2 = 0
    ties = 0
    per_file = []

    for fid in sorted(common_ids):
        hyp1 = r1[fid].get('hypothesis', '')
        hyp2 = r2[fid].get('hypothesis', '')
        ref = r1[fid].get('reference', '')

        # Fuse
        fused = rover_fuse(hyp1, hyp2)

        # Normalize for WER (Whisper normalizer — primary metric)
        ref_w = WHISPER_NORM(ref)
        hyp1_w = WHISPER_NORM(hyp1)
        hyp2_w = WHISPER_NORM(hyp2)
        fused_w = WHISPER_NORM(fused)

        if not ref_w:
            continue

        n_words = len(ref_w.split())
        wer_fused = jiwer.wer(ref_w, fused_w)
        wer_m1 = jiwer.wer(ref_w, hyp1_w)
        wer_m2 = jiwer.wer(ref_w, hyp2_w)

        err_fused = round(wer_fused * n_words)
        err_m1 = round(wer_m1 * n_words)
        err_m2 = round(wer_m2 * n_words)

        total_errors_fused += err_fused
        total_errors_m1 += err_m1
        total_errors_m2 += err_m2
        total_words += n_words

        best_single = min(err_m1, err_m2)
        if err_fused < best_single:
            wins_fused += 1
        elif err_fused > best_single:
            if err_m1 <= err_m2:
                wins_m1 += 1
            else:
                wins_m2 += 1
        else:
            ties += 1

        per_file.append({
            'file_id': fid,
            'wer_fused': round(wer_fused * 100, 2),
            'wer_m1': round(wer_m1 * 100, 2),
            'wer_m2': round(wer_m2 * 100, 2),
            'fusion_hypothesis': fused,
        })

    if total_words == 0:
        return None

    agg_fused = total_errors_fused / total_words
    agg_m1 = total_errors_m1 / total_words
    agg_m2 = total_errors_m2 / total_words

    return {
        'model1': model1,
        'model2': model2,
        'dataset': dataset,
        'files': len(per_file),
        'aggregate_wer_fused': round(agg_fused * 100, 2),
        'aggregate_wer_m1': round(agg_m1 * 100, 2),
        'aggregate_wer_m2': round(agg_m2 * 100, 2),
        'wins_fused': wins_fused,
        'wins_m1': wins_m1,
        'wins_m2': wins_m2,
        'ties': ties,
        'per_file': per_file,
    }


def main():
    parser = argparse.ArgumentParser(description='ROVER fusion for all model pairs')
    parser.add_argument('--model1', type=str, default=None)
    parser.add_argument('--model2', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='all')
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]

    for dataset in datasets:
        results_dir = OSSICLES_DIR / f'benchmark_results_{dataset}'
        if not results_dir.exists():
            log.warning(f'No results directory for {dataset}')
            continue

        # Find all models with results for this dataset
        # Only include Python-format results (dict with 'aggregates')
        available_models = []
        for f in sorted(results_dir.glob('*.json')):
            if '.old' in f.name or 'normalized' in f.name or 'fusion' in f.stem:
                continue
            try:
                data = json.loads(f.read_text())
                if isinstance(data, dict) and 'aggregates' in data:
                    available_models.append(f.stem)
            except (json.JSONDecodeError, KeyError):
                continue

        if args.model1 and args.model2:
            pairs = [(args.model1, args.model2)]
        else:
            pairs = list(itertools.combinations(available_models, 2))

        log.info(f'{dataset}: {len(available_models)} models, {len(pairs)} pairs')

        # Run pairs in parallel
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing

        n_workers = min(multiprocessing.cpu_count(), 12)
        log.info(f'Running with {n_workers} workers')

        all_results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(run_fusion, m1, m2, dataset): (m1, m2)
                for m1, m2 in pairs
            }
            for future in as_completed(futures):
                m1, m2 = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    log.error(f'  {m1} + {m2}: FAILED — {e}')
                    continue
                if result:
                    all_results.append(result)
                    log.info(f'  {m1} + {m2}: fused={result["aggregate_wer_fused"]:.2f}% '
                             f'(m1={result["aggregate_wer_m1"]:.2f}%, m2={result["aggregate_wer_m2"]:.2f}%) '
                             f'wins={result["wins_fused"]}/{result["files"]}')

        # Save all fusion results
        if all_results:
            output_dir = results_dir / 'fusion_results'
            output_dir.mkdir(exist_ok=True)

            # Save per-pair files
            for result in all_results:
                pair_file = output_dir / f'{result["model1"]}_+_{result["model2"]}.json'
                pair_file.write_text(json.dumps(result, indent=2))

            # Save summary sorted by fused WER
            summary = sorted(all_results, key=lambda x: x['aggregate_wer_fused'])
            summary_file = output_dir / 'summary.txt'
            lines = [f'ROVER Fusion Results — {dataset}',
                     f'{"="*60}',
                     f'{"Pair":<50s} {"Fused":>8s} {"M1":>8s} {"M2":>8s} {"Wins":>8s}',
                     f'{"-"*60}']
            for r in summary:
                pair = f'{r["model1"]} + {r["model2"]}'
                lines.append(f'{pair:<50s} {r["aggregate_wer_fused"]:>7.2f}% '
                             f'{r["aggregate_wer_m1"]:>7.2f}% {r["aggregate_wer_m2"]:>7.2f}% '
                             f'{r["wins_fused"]:>3d}/{r["files"]}')
            lines.append('')
            summary_file.write_text('\n'.join(lines))
            log.info(f'Saved {len(all_results)} fusion results to {output_dir}')


if __name__ == '__main__':
    main()
