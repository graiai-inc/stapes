#!/usr/bin/env python3
"""Medical term error analysis for ASR benchmark.

Works backwards from ASR errors:
1. Loads all benchmark results (reference + hypothesis per file per model)
2. Identifies words that differ between reference and hypothesis
3. Validates error words against UMLS to determine which are medical terms
4. Reports: what fraction of ASR errors are clinically significant?

For UMLS lookup, uses QuickUMLS if available, otherwise falls back to
a curated medical term list.

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

# UMLS semantic types we care about
MEDICAL_SEMANTIC_TYPES = {
    'T121': 'DRUG',          # Pharmacologic Substance
    'T200': 'DRUG',          # Clinical Drug
    'T195': 'DRUG',          # Antibiotic
    'T047': 'CONDITION',     # Disease or Syndrome
    'T048': 'CONDITION',     # Mental or Behavioral Dysfunction
    'T184': 'SYMPTOM',       # Sign or Symptom
    'T033': 'FINDING',       # Finding
    'T034': 'LAB_RESULT',    # Laboratory or Test Result
    'T023': 'ANATOMY',       # Body Part, Organ, or Organ Component
    'T029': 'ANATOMY',       # Body Location or Region
    'T059': 'PROCEDURE',     # Laboratory Procedure
    'T060': 'PROCEDURE',     # Diagnostic Procedure
    'T061': 'PROCEDURE',     # Therapeutic or Preventive Procedure
}


def normalize_raw(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def word_level_diff(ref: str, hyp: str) -> dict:
    """Compute word-level alignment and identify error words.

    Returns dict with:
        substitutions: list of (ref_word, hyp_word)
        deletions: list of ref_words missing from hyp
        insertions: list of hyp_words not in ref
        correct: list of words that matched
    """
    ref_words = ref.split()
    hyp_words = hyp.split()

    # Use Levenshtein alignment (DP backtrack)
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    bt = [[''] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bt[i][j] = 'M'
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                best = min(sub, dele, ins)
                dp[i][j] = best
                if best == sub:
                    bt[i][j] = 'S'
                elif best == dele:
                    bt[i][j] = 'D'
                else:
                    bt[i][j] = 'I'

    # Backtrack
    substitutions = []
    deletions = []
    insertions = []
    correct = []

    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i][j] == 'M':
            correct.append(ref_words[i - 1])
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and bt[i][j] == 'S':
            substitutions.append((ref_words[i - 1], hyp_words[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or bt[i][j] == 'D'):
            deletions.append(ref_words[i - 1])
            i -= 1
        else:
            insertions.append(hyp_words[j - 1])
            j -= 1

    return {
        'substitutions': list(reversed(substitutions)),
        'deletions': list(reversed(deletions)),
        'insertions': list(reversed(insertions)),
        'correct': list(reversed(correct)),
    }


def try_umls_lookup(words: set[str]) -> dict[str, str]:
    """Try to look up words in UMLS. Returns word → category mapping.

    Falls back gracefully if QuickUMLS is not installed.
    """
    try:
        from quickumls import QuickUMLS
        # Try default install location
        matcher = QuickUMLS('/opt/quickumls', threshold=0.8)
        results = {}
        for word in words:
            matches = matcher.match(word)
            if matches:
                for match_group in matches:
                    for match in match_group:
                        for stype in match.get('semtypes', []):
                            if stype in MEDICAL_SEMANTIC_TYPES:
                                results[word] = MEDICAL_SEMANTIC_TYPES[stype]
                                break
                    if word in results:
                        break
        return results
    except (ImportError, Exception) as e:
        log.info(f'QuickUMLS not available ({e}), using scispaCy fallback')

    try:
        import spacy
        nlp = spacy.load('en_core_sci_md')
        # Add UMLS linker
        from scispacy.linking import EntityLinker
        nlp.add_pipe('scispacy_linker',
                      config={'resolve_abbreviations': True, 'linker_name': 'umls'})
        results = {}
        # Process in batches
        text = ' . '.join(sorted(words))
        doc = nlp(text)
        for ent in doc.ents:
            word_lower = ent.text.lower().strip()
            if word_lower in words:
                # Check linked UMLS entities for medical semantic types
                linker = nlp.get_pipe('scispacy_linker')
                for umls_ent in ent._.kb_ents:
                    cui = umls_ent[0]
                    entity = linker.kb.cui_to_entity.get(cui)
                    if entity:
                        for stype in entity.types:
                            if stype in MEDICAL_SEMANTIC_TYPES:
                                results[word_lower] = MEDICAL_SEMANTIC_TYPES[stype]
                                break
                    if word_lower in results:
                        break
        return results
    except (ImportError, Exception) as e:
        log.warning(f'scispaCy not available ({e}), using pattern-based fallback')

    # Pattern-based fallback (from cochlea/extract_medical_failures.py)
    return _pattern_based_classify(words)


def _pattern_based_classify(words: set[str]) -> dict[str, str]:
    """Classify medical terms using suffix/prefix patterns.

    This is the fallback when UMLS tools aren't available.
    """
    drug_suffixes = ['statin', 'pril', 'olol', 'azine', 'dipine', 'sartan',
                     'cillin', 'mycin', 'mide', 'done', 'pine', 'pam', 'lam',
                     'zole', 'oxin', 'vir', 'mab', 'nib']
    condition_suffixes = ['itis', 'osis', 'emia', 'pathy', 'uria', 'pnea',
                          'rrhea', 'algia', 'asis']
    procedure_suffixes = ['ectomy', 'oscopy', 'plasty', 'otomy', 'ization']
    anatomy_prefixes = ['cardio', 'pulmo', 'hepat', 'gastr', 'neuro', 'derma',
                        'oropharyn', 'lumbosacr']
    clinical_prefixes = ['hyper', 'hypo', 'tachy', 'brady']

    # Common abbreviations
    abbreviations = {'heent', 'jvd', 'wbc', 'rbc', 'mcg', 'pco2', 'ercp',
                     'picc', 'gerd', 'copd', 'chf', 'cad', 'dvt', 'uti',
                     'bun', 'hba1c', 'inr', 'bp', 'hr', 'ekg', 'ecg'}

    results = {}
    for word in words:
        w = word.lower()
        if w in abbreviations:
            results[word] = 'ABBREVIATION'
        elif any(w.endswith(s) for s in drug_suffixes):
            results[word] = 'DRUG'
        elif any(w.endswith(s) for s in condition_suffixes):
            results[word] = 'CONDITION'
        elif any(w.endswith(s) for s in procedure_suffixes):
            results[word] = 'PROCEDURE'
        elif any(w.startswith(p) for p in anatomy_prefixes):
            results[word] = 'ANATOMY'
        elif any(w.startswith(p) for p in clinical_prefixes):
            results[word] = 'CLINICAL_MODIFIER'

    return results


def _analyze_one_model(args: tuple) -> tuple:
    """Analyze errors for a single model. Returns (model_name, word_errors, ref_counts)."""
    model_name, files = args
    word_errors = collections.Counter()
    ref_counts = collections.Counter()

    for fid, result in files.items():
        ref = normalize_raw(result.get('reference', ''))
        hyp = normalize_raw(result.get('hypothesis', ''))
        if not ref:
            continue

        diff = word_level_diff(ref, hyp)

        for word in ref.split():
            ref_counts[word] += 1

        for ref_word, hyp_word in diff['substitutions']:
            word_errors[ref_word] += 1

        for word in diff['deletions']:
            word_errors[word] += 1

    return model_name, dict(word_errors), dict(ref_counts)


def analyze_dataset(dataset: str) -> dict:
    """Analyze medical term errors across all models for one dataset."""
    results_dir = OSSICLES_DIR / f'benchmark_results_{dataset}'
    if not results_dir.exists():
        log.error(f'No results for {dataset}')
        return {}

    # Load all model results (on-device + cloud)
    model_results = {}

    # On-device results
    for f in sorted(results_dir.glob('*.json')):
        if '.old' in f.name or 'normalized' in f.name or 'fusion' in f.stem:
            continue
        try:
            data = json.loads(f.read_text())
            if isinstance(data, dict) and 'aggregates' in data:
                model_results[f.stem] = {
                    r['file_id']: r for r in data.get('results', [])
                }
        except (json.JSONDecodeError, KeyError):
            continue

    # Cloud results
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

    # Collect all error words across all models
    all_error_words = collections.Counter()  # word → total error count
    all_ref_words = collections.Counter()    # word → total occurrence count
    model_errors = {}  # model → {word → error_count}

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    n_workers = min(multiprocessing.cpu_count(), 12)
    log.info(f'  Analyzing {len(model_results)} models with {n_workers} workers...')

    intermediate_dir = results_dir / 'medical_errors_per_model'
    intermediate_dir.mkdir(exist_ok=True)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_analyze_one_model, (model, files)): model
            for model, files in model_results.items()
        }
        for future in as_completed(futures):
            model_name, word_errors, ref_counts = future.result()
            model_errors[model_name] = word_errors
            for word, count in word_errors.items():
                all_error_words[word] += count
            for word, count in ref_counts.items():
                all_ref_words[word] += count
            # Write per-model results immediately
            model_file = intermediate_dir / f'{model_name}.json'
            model_file.write_text(json.dumps({
                'model': model_name,
                'unique_error_words': len(word_errors),
                'total_errors': sum(word_errors.values()),
                'top_errors': dict(sorted(word_errors.items(),
                                          key=lambda x: x[1], reverse=True)[:50]),
            }, indent=2))
            log.info(f'    [{len(model_errors)}/{len(model_results)}] {model_name}: '
                     f'{len(word_errors)} unique error words')

    log.info(f'  {len(all_error_words)} unique error words, '
             f'{sum(all_error_words.values())} total errors')

    # Filter to words with meaningful error rates (>= 3 occurrences, >= 20% error rate)
    significant_errors = {}
    for word, error_count in all_error_words.items():
        total_count = all_ref_words.get(word, 0)
        if total_count >= 3:
            error_rate = error_count / total_count
            if error_rate >= 0.2:
                significant_errors[word] = {
                    'errors': error_count,
                    'total': total_count,
                    'error_rate': round(error_rate, 4),
                }

    log.info(f'  {len(significant_errors)} words with >= 20% error rate and >= 3 occurrences')

    # Classify error words as medical or not
    error_word_set = set(significant_errors.keys())
    medical_classifications = try_umls_lookup(error_word_set)

    # Split into medical and non-medical
    medical_errors = {}
    non_medical_errors = {}
    for word, stats in significant_errors.items():
        if word in medical_classifications:
            stats['category'] = medical_classifications[word]
            medical_errors[word] = stats
        else:
            non_medical_errors[word] = stats

    log.info(f'  {len(medical_errors)} medical error words, '
             f'{len(non_medical_errors)} non-medical')

    # Per-model medical term error breakdown
    model_medical_stats = {}
    for model, word_errors in model_errors.items():
        medical_term_errors = 0
        total_medical_terms = 0
        by_category = collections.Counter()

        for word in medical_errors:
            total_in_refs = all_ref_words.get(word, 0)
            errors_this_model = word_errors.get(word, 0)
            total_medical_terms += total_in_refs
            medical_term_errors += errors_this_model
            if errors_this_model > 0:
                cat = medical_errors[word].get('category', 'UNKNOWN')
                by_category[cat] += errors_this_model

        mtr = 1.0 - (medical_term_errors / total_medical_terms) if total_medical_terms > 0 else 1.0
        model_medical_stats[model] = {
            'medical_term_recall': round(mtr * 100, 2),
            'medical_errors': medical_term_errors,
            'medical_total': total_medical_terms,
            'by_category': dict(by_category),
        }

    return {
        'dataset': dataset,
        'models': len(model_results),
        'medical_error_words': {
            k: v for k, v in sorted(medical_errors.items(),
                                     key=lambda x: x[1]['errors'], reverse=True)
        },
        'non_medical_error_words_count': len(non_medical_errors),
        'model_medical_stats': model_medical_stats,
        'top_non_medical_errors': {
            k: v for k, v in sorted(non_medical_errors.items(),
                                     key=lambda x: x[1]['errors'], reverse=True)[:50]
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Medical term error analysis')
    parser.add_argument('--dataset', type=str, default='all')
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]

    for dataset in datasets:
        results = analyze_dataset(dataset)
        if not results:
            continue

        # Save results
        output_dir = OSSICLES_DIR / f'benchmark_results_{dataset}'
        output_file = output_dir / 'medical_error_analysis.json'
        output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        log.info(f'Saved to {output_file}')

        # Print summary
        print(f'\n{"="*60}')
        print(f'Medical Term Error Analysis — {dataset}')
        print(f'{"="*60}')
        print(f'Medical error words: {len(results.get("medical_error_words", {}))}')
        print(f'\nTop 20 medical error words:')
        for word, stats in list(results.get('medical_error_words', {}).items())[:20]:
            print(f'  {word:<25s} {stats["category"]:<15s} '
                  f'errors={stats["errors"]:>5d} total={stats["total"]:>5d} '
                  f'rate={stats["error_rate"]:.1%}')

        print(f'\nMedical Term Recall by model:')
        for model, stats in sorted(results.get('model_medical_stats', {}).items(),
                                    key=lambda x: x[1]['medical_term_recall'],
                                    reverse=True):
            print(f'  {model:<30s} MTR={stats["medical_term_recall"]:>6.2f}%')


if __name__ == '__main__':
    main()
