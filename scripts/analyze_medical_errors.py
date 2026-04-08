#!/usr/bin/env python3
"""Medical term error analysis for ASR benchmark using UMLS (QuickUMLS).

1. Runs QuickUMLS on each reference transcript to find all medical concept
   spans (single and multi-word).
2. Uses jiwer to align hypothesis to reference and identify error positions.
3. For each medical span, checks if any of its words were wrong.
4. Reports medical term recall per model, per category, and per term.

Usage:
    python scripts/analyze_medical_errors.py --dataset figshare-osce
    python scripts/analyze_medical_errors.py --dataset all
"""

import argparse
import collections
import json
import logging
import re
from pathlib import Path

import jiwer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
OSSICLES_DIR = SCRIPT_DIR.parent

DATASETS = ['figshare-osce', 'primock57', 'nazmulkazi']
QUICKUMLS_PATH = OSSICLES_DIR / 'quickumls_data'

MEDICAL_SEMANTIC_TYPES = {
    'T116': 'DRUG',
    'T121': 'DRUG',
    'T200': 'DRUG',
    'T195': 'DRUG',
    'T047': 'CONDITION',
    'T048': 'CONDITION',
    'T046': 'CONDITION',
    'T191': 'CONDITION',
    'T184': 'SYMPTOM',
    'T033': 'FINDING',
    'T034': 'LAB_RESULT',
    'T023': 'ANATOMY',
    'T029': 'ANATOMY',
    'T030': 'ANATOMY',
    'T058': 'PROCEDURE',
    'T059': 'PROCEDURE',
    'T060': 'PROCEDURE',
    'T061': 'PROCEDURE',
}

# Common English words that collide with UMLS abbreviations/eponyms.
# "said"->Simian AIDS, "little"->Little's Disease, "today"->brand drug,
# "bit"->benzimidazole, "get"->Generalized Essential Telangiectasia,
# "lot"->Olfactory Tract, "well/cant/nothing/etc"->T033 Finding noise.
UMLS_STOPWORDS = {
    'bit', 'get', 'got', 'little', 'lot', 'said', 'still', 'today',
    'changes', 'well', 'cant', 'nothing', 'maybe', 'take', 'yes',
    'much', 'probably', 'worse', 'sharp', 'only', 'more', 'less',
    'stop', 'ask', 'bad', 'dad', 'may', 'nice', 'new', 'used',
    'find', 'care', 'place', 'always', 'wanted', 'play',
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_error_indices(ref: str, hyp: str) -> set[int]:
    """Use jiwer to find which reference word indices have errors."""
    out = jiwer.process_words(ref, hyp)
    errors = set()
    for chunk in out.alignments[0]:
        if chunk.type in ('substitute', 'delete'):
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                errors.add(i)
    return errors


def char_span_to_word_indices(text: str, start: int, end: int) -> list[int]:
    """Convert character span to word indices."""
    word_starts = []
    in_word = False
    for i, c in enumerate(text):
        if c != ' ' and not in_word:
            word_starts.append(i)
            in_word = True
        elif c == ' ':
            in_word = False
    indices = []
    for wi, ws in enumerate(word_starts):
        we = word_starts[wi + 1] - 1 if wi + 1 < len(word_starts) else len(text)
        if ws < end and we > start:
            indices.append(wi)
    return indices


def find_medical_spans(text: str, matcher) -> list[dict]:
    """Run QuickUMLS on text, return clinically relevant spans."""
    matches = matcher.match(text)
    spans = []
    for match_group in matches:
        best = match_group[0]
        ngram = best['ngram'].lower().strip()
        if ngram in UMLS_STOPWORDS:
            continue
        # Reject fuzzy matches on short single words (e.g., "talked"->"stalked")
        if ' ' not in ngram and len(ngram) < 6 and best['similarity'] < 1.0:
            continue
        category = None
        for stype in best.get('semtypes', []):
            if stype in MEDICAL_SEMANTIC_TYPES:
                category = MEDICAL_SEMANTIC_TYPES[stype]
                break
        if category is None:
            continue
        word_indices = char_span_to_word_indices(text, best['start'], best['end'])
        if not word_indices:
            continue
        spans.append({
            'ngram': best['ngram'],
            'term': best['term'],
            'cui': best['cui'],
            'category': category,
            'word_indices': word_indices,
        })
    return spans


def analyze_dataset(dataset: str, matcher) -> dict:
    """Analyze medical term errors across all models for one dataset."""
    results_dir = OSSICLES_DIR / f'benchmark_results_{dataset}'
    if not results_dir.exists():
        log.error(f'No results for {dataset}')
        return {}

    model_results = {}
    for f in sorted(results_dir.glob('*.json')):
        if '.old' in f.name or 'normalized' in f.name or 'fusion' in f.stem or 'medical' in f.stem:
            continue
        try:
            data = json.loads(f.read_text())
            if isinstance(data, dict) and 'aggregates' in data:
                model_results[f.stem] = {
                    r['file_id']: r for r in data.get('results', [])
                }
        except (json.JSONDecodeError, KeyError):
            continue

    cloud_dir = OSSICLES_DIR / f'benchmark_results_cloud_{dataset}'
    if cloud_dir.exists():
        for f in sorted(cloud_dir.glob('*.json')):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, dict) and 'results' in data:
                    model_results[f'cloud-{f.stem}'] = {
                        r['file_id']: r for r in data.get('results', [])
                    }
            except (json.JSONDecodeError, KeyError):
                continue

    log.info(f'{dataset}: {len(model_results)} models loaded')

    # Step 1: Find medical spans in references (once, shared across models)
    first_model = next(iter(model_results.values()))
    file_spans = {}
    file_refs = {}

    log.info('  Finding medical concepts in references...')
    for i, (fid, result) in enumerate(first_model.items()):
        ref = result.get('reference', '')
        if not ref:
            continue
        ref_norm = normalize(ref)
        file_refs[fid] = ref_norm
        file_spans[fid] = find_medical_spans(ref_norm, matcher)
        if (i + 1) % 50 == 0 or i + 1 == len(first_model):
            total = sum(len(s) for s in file_spans.values())
            log.info(f'    [{i + 1}/{len(first_model)}] {total} medical spans')

    total_spans = sum(len(s) for s in file_spans.values())
    log.info(f'  {total_spans} medical spans across {len(file_spans)} files')

    # Step 2: For each model, align and check medical spans
    output_dir = results_dir / 'medical_errors_per_model'
    output_dir.mkdir(exist_ok=True)

    all_term_errors = collections.Counter()
    all_term_totals = collections.Counter()
    model_medical_stats = {}

    for fid, spans in file_spans.items():
        for span in spans:
            all_term_totals[span['ngram'].lower()] += 1

    for model_name, files in model_results.items():
        med_errors = 0
        med_total = 0
        by_category = collections.Counter()
        term_errors = collections.Counter()

        for fid in file_spans:
            if fid not in files:
                continue
            ref_norm = file_refs[fid]
            hyp_norm = normalize(files[fid].get('hypothesis', ''))

            error_indices = get_error_indices(ref_norm, hyp_norm)

            for span in file_spans[fid]:
                med_total += 1
                if any(wi in error_indices for wi in span['word_indices']):
                    med_errors += 1
                    by_category[span['category']] += 1
                    term_errors[span['ngram'].lower()] += 1
                    all_term_errors[span['ngram'].lower()] += 1

        mtr = 1.0 - (med_errors / med_total) if med_total > 0 else 1.0
        model_medical_stats[model_name] = {
            'medical_term_recall': round(mtr * 100, 2),
            'medical_errors': med_errors,
            'medical_total': med_total,
            'by_category': dict(by_category),
        }

        model_file = output_dir / f'{model_name}.json'
        model_file.write_text(json.dumps({
            'model': model_name,
            'medical_term_recall': round(mtr * 100, 2),
            'medical_errors': med_errors,
            'medical_total': med_total,
            'by_category': dict(by_category),
            'top_term_errors': dict(term_errors.most_common(30)),
        }, indent=2))
        log.info(f'    {model_name}: MTR={mtr*100:.2f}% '
                 f'({med_errors}/{med_total})')

    term_summary = {}
    for term, errors in all_term_errors.most_common():
        total = all_term_totals.get(term, 0)
        category = 'UNKNOWN'
        for spans in file_spans.values():
            for span in spans:
                if span['ngram'].lower() == term:
                    category = span['category']
                    break
            if category != 'UNKNOWN':
                break
        term_summary[term] = {
            'errors': errors,
            'total_occurrences': total * len(model_results),
            'occurrences_per_file': total,
            'error_rate': round(errors / (total * len(model_results)), 4) if total > 0 else None,
            'category': category,
        }

    return {
        'dataset': dataset,
        'models': len(model_results),
        'umls_method': 'QuickUMLS (medspacy fork)',
        'umls_version': '2025AB',
        'umls_threshold': 0.8,
        'total_medical_spans': total_spans,
        'stopwords_excluded': sorted(UMLS_STOPWORDS),
        'medical_term_errors': term_summary,
        'model_medical_stats': model_medical_stats,
    }


def main():
    parser = argparse.ArgumentParser(description='Medical term error analysis')
    parser.add_argument('--dataset', type=str, default='all')
    parser.add_argument('--quickumls-path', type=str,
                        default=str(QUICKUMLS_PATH))
    args = parser.parse_args()

    from quickumls import QuickUMLS
    log.info(f'Loading QuickUMLS from {args.quickumls_path}...')
    matcher = QuickUMLS(args.quickumls_path, threshold=0.8)
    log.info('QuickUMLS loaded')

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]

    for dataset in datasets:
        results = analyze_dataset(dataset, matcher)
        if not results:
            continue

        output_dir = OSSICLES_DIR / f'benchmark_results_{dataset}'
        output_file = output_dir / 'medical_error_analysis.json'
        output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        log.info(f'Saved to {output_file}')

        print(f'\n{"="*60}')
        print(f'Medical Term Error Analysis - {dataset}')
        print(f'{"="*60}')
        print(f'Total medical spans: {results["total_medical_spans"]}')
        print(f'Terms with errors: {len(results["medical_term_errors"])}')

        print(f'\nTop 20 error terms:')
        for term, stats in list(results['medical_term_errors'].items())[:20]:
            rate = f'{stats["error_rate"]:.1%}' if stats['error_rate'] else 'n/a'
            print(f'  {term:<30s} {stats["category"]:<12s} '
                  f'errors={stats["errors"]:>5d}  rate={rate:>6s}')

        print(f'\nMedical Term Recall by model:')
        for model, stats in sorted(results['model_medical_stats'].items(),
                                    key=lambda x: x[1]['medical_term_recall'],
                                    reverse=True):
            print(f'  {model:<30s} MTR={stats["medical_term_recall"]:>6.2f}%')


if __name__ == '__main__':
    main()
