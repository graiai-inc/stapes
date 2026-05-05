#!/usr/bin/env python3
"""Filter the medical vocab to remove UMLS-fuzzy-matching noise.

Rules (applied to medical_vocab_snapshot_v1.tsv → produces _clean.tsv):

1. Multi-word entries (`chest pain`, `general anesthesia`) — ALWAYS keep.
   Multi-word phrases are very unlikely to be false positives.
2. Single-word entries are kept only if ALL of:
   - length >= 5 characters
   - NOT in COMMON_ENGLISH_NOISE blocklist
   - EITHER not in standard English dictionary OR in DRUG_ALLOWLIST
3. Drop entries in EXPLICIT_NOISE blocklist regardless.

Output:
  results/medical_vocab/medical_vocab_clean.tsv
  results/medical_vocab/clean_stats.json
"""
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
STAPES_DIR = SCRIPT_DIR.parent
VOCAB_DIR = STAPES_DIR / 'results' / 'medical_vocab'
INPUT_TSV = VOCAB_DIR / 'medical_vocab_snapshot_v1.tsv'
OUTPUT_TSV = VOCAB_DIR / 'medical_vocab_clean.tsv'
STATS_PATH = VOCAB_DIR / 'clean_stats.json'

ENGLISH_DICT_PATH = Path('/usr/share/dict/words')

# Drugs that are also in standard English dict — keep them anyway.
# (Generic + top US brands. Add more as needed.)
DRUG_ALLOWLIST = {
    'tylenol', 'ibuprofen', 'acetaminophen', 'aspirin', 'advil', 'motrin',
    'benadryl', 'aleve', 'claritin', 'zyrtec', 'allegra', 'sudafed', 'mucinex',
    'pepcid', 'prilosec', 'nexium', 'prevacid', 'zantac', 'tums', 'rolaids',
    'imodium', 'pepto', 'metamucil', 'colace', 'miralax', 'senokot',
    'midol', 'pamprin', 'excedrin', 'bufferin', 'ecotrin',
    'naproxen', 'diclofenac', 'celecoxib', 'meloxicam',
    'codeine', 'morphine', 'fentanyl', 'oxycodone', 'hydrocodone', 'tramadol',
    'methadone', 'percocet', 'vicodin', 'norco', 'oxycontin', 'dilaudid',
    'ativan', 'xanax', 'valium', 'klonopin', 'lorazepam', 'diazepam',
    'alprazolam', 'clonazepam', 'temazepam',
    'ambien', 'lunesta', 'sonata', 'restoril', 'rozerem',
    'prozac', 'zoloft', 'paxil', 'lexapro', 'celexa', 'wellbutrin',
    'effexor', 'cymbalta', 'pristiq', 'remeron', 'trintellix',
    'fluoxetine', 'sertraline', 'paroxetine', 'citalopram', 'escitalopram',
    'venlafaxine', 'duloxetine', 'bupropion', 'mirtazapine',
    'zyprexa', 'risperdal', 'abilify', 'seroquel', 'geodon', 'latuda',
    'haldol', 'thorazine',
    'lithium', 'depakote', 'lamictal', 'tegretol', 'topamax',
    'metformin', 'glipizide', 'glyburide', 'januvia', 'jardiance', 'ozempic',
    'trulicity', 'humalog', 'lantus', 'novolog', 'levemir', 'tresiba',
    'lipitor', 'crestor', 'zocor', 'pravachol', 'lescol', 'livalo',
    'atorvastatin', 'rosuvastatin', 'simvastatin', 'pravastatin',
    'norvasc', 'lopressor', 'toprol', 'tenormin', 'inderal', 'corgard',
    'lasix', 'hydrochlorothiazide', 'spironolactone', 'aldactone',
    'altace', 'zestril', 'prinivil', 'vasotec', 'cozaar', 'diovan',
    'lisinopril', 'ramipril', 'enalapril', 'losartan', 'valsartan',
    'plavix', 'eliquis', 'xarelto', 'pradaxa', 'coumadin', 'warfarin',
    'amoxicillin', 'azithromycin', 'penicillin', 'ciprofloxacin', 'cipro',
    'levaquin', 'augmentin', 'keflex', 'bactrim', 'septra', 'flagyl',
    'doxycycline', 'erythromycin', 'tetracycline', 'clindamycin',
    'cephalexin', 'metronidazole',
    'albuterol', 'ventolin', 'proventil', 'singulair', 'flovent', 'advair',
    'symbicort', 'spiriva', 'breo',
    'salbutamol', 'fluticasone', 'budesonide', 'tiotropium', 'montelukast',
    'prednisone', 'prednisolone', 'medrol', 'dexamethasone', 'cortisone',
    'synthroid', 'levothyroxine', 'cytomel', 'armour',
    'metoprolol', 'atenolol', 'propranolol', 'carvedilol', 'bisoprolol',
    'amlodipine', 'diltiazem', 'verapamil', 'nifedipine',
    'omeprazole', 'pantoprazole', 'lansoprazole', 'esomeprazole',
    'gabapentin', 'pregabalin', 'lyrica', 'neurontin',
    'allopurinol', 'colchicine', 'tamiflu', 'oseltamivir',
    'gabapentin', 'levetiracetam', 'keppra',
    'methotrexate', 'humira', 'enbrel', 'remicade',
    'adderall', 'ritalin', 'vyvanse', 'concerta', 'strattera', 'focalin',
    'piriton', 'dioralyte', 'salbutamol', 'paracetamol', 'co-codamol',
    'microgynon', 'trimethoprim', 'fexofenadine', 'loratadine',
}

# Common-English noise that UMLS coincidentally tags with our semtypes.
# These are CONFIRMED via inspection of vocab — common English words.
EXPLICIT_NOISE = {
    'others', 'others\'', 'other', 'comfort', 'comforts', 'renewal', 'renewals',
    'active', 'commit', 'rise', 'rises', 'date', 'dates', 'pat', 'snow', 'fat',
    'cap', 'caps', 'tube', 'tubes', 'face', 'faces', 'hand', 'hands', 'back',
    'lower', 'upper', 'else', 'couple', 'definitely', 'symptoms', 'medications',
    'cough', 'coughs', 'cold', 'colds', 'hot', 'happy', 'sad', 'fun', 'food',
    'history', 'name', 'normal', 'medical', 'clinic', 'home', 'work',
    'family', 'group', 'friend', 'mother', 'father', 'parent', 'child',
    'children', 'sister', 'brother', 'wife', 'husband',
    'recap', 'admit', 'toilet', 'light', 'best', 'ate', 'general', 'control',
    'liquid', 'fluids', 'times', 'close', 'shaky', 'unwell', 'vomit', 'vomited',
    'cramp', 'stomach', 'tests', 'able', 'life', 'problem', 'problem:',
    'hmm', 'mmm', 'mm', 'uh', 'um', 'oh', 'ah', 'yeah', 'okay',
    'dops', 'dot', 'ther', 'spt', 'aleve',  # "aleve" is in DRUG_ALLOWLIST so this is overruled
    'biologic', 'biologics', 'medicine', 'medicines',  # too generic
    'maybe', 'probably', 'definitely', 'really', 'pretty',
    'stop', 'start', 'begin', 'end', 'change', 'changes', 'change',
    'taking', 'take', 'took', 'put', 'put on', 'getting',
    'better', 'worse', 'worst', 'great', 'fine', 'good', 'bad',
}

# Override: never block these even if they appear in EXPLICIT_NOISE
ALLOWLIST_OVERRIDE = DRUG_ALLOWLIST.copy()


def main():
    eng = set()
    if ENGLISH_DICT_PATH.exists():
        with open(ENGLISH_DICT_PATH) as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    eng.add(w)
                    # Also add stems without 's
                    if w.endswith("'s"):
                        eng.add(w[:-2])
        print(f'Loaded {len(eng)} English dict words', flush=True)
    else:
        print(f'WARNING: no English dict at {ENGLISH_DICT_PATH}', flush=True)

    fh_out = open(OUTPUT_TSV, 'w')
    fh_out.write('term\tcategory\tfirst_tier\tfirst_source\n')
    fh_out.flush()

    counts = {
        'total_input': 0,
        'kept_multi_word': 0,
        'kept_single_not_in_dict': 0,
        'kept_single_drug_allowlist': 0,
        'dropped_too_short': 0,
        'dropped_explicit_noise': 0,
        'dropped_in_english_dict': 0,
    }
    sample_kept = []
    sample_dropped = []

    with open(INPUT_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            term = row['term'].strip().lower()
            cat = row['category']
            if not term:
                continue
            counts['total_input'] += 1

            is_multi = ' ' in term

            # Always check explicit noise (unless allowlisted)
            if term in EXPLICIT_NOISE and term not in ALLOWLIST_OVERRIDE:
                counts['dropped_explicit_noise'] += 1
                if len(sample_dropped) < 20:
                    sample_dropped.append((term, cat, 'noise'))
                continue

            if is_multi:
                # Multi-word phrases are kept by default (unlikely false positives)
                counts['kept_multi_word'] += 1
                fh_out.write(f"{row['term']}\t{cat}\t{row['first_tier']}\t{row['first_source']}\n")
                fh_out.flush()
                continue

            # Single-word filters
            if len(term) < 5:
                counts['dropped_too_short'] += 1
                if len(sample_dropped) < 20:
                    sample_dropped.append((term, cat, 'too_short'))
                continue

            # English dict filter — but DRUG_ALLOWLIST overrides
            if term in eng and term not in ALLOWLIST_OVERRIDE:
                counts['dropped_in_english_dict'] += 1
                if len(sample_dropped) < 20:
                    sample_dropped.append((term, cat, 'in_eng_dict'))
                continue

            # Kept
            if term in ALLOWLIST_OVERRIDE:
                counts['kept_single_drug_allowlist'] += 1
            else:
                counts['kept_single_not_in_dict'] += 1
            fh_out.write(f"{row['term']}\t{cat}\t{row['first_tier']}\t{row['first_source']}\n")
            fh_out.flush()
            if len(sample_kept) < 20:
                sample_kept.append((term, cat))

    fh_out.close()

    total_kept = (counts['kept_multi_word']
                  + counts['kept_single_not_in_dict']
                  + counts['kept_single_drug_allowlist'])

    print(f'\nTotal input: {counts["total_input"]} terms', flush=True)
    print(f'Kept: {total_kept}', flush=True)
    print(f'  - multi-word: {counts["kept_multi_word"]}', flush=True)
    print(f'  - single-word, not in English dict: {counts["kept_single_not_in_dict"]}', flush=True)
    print(f'  - single-word in DRUG_ALLOWLIST: {counts["kept_single_drug_allowlist"]}', flush=True)
    print(f'Dropped: {counts["total_input"] - total_kept}', flush=True)
    print(f'  - too short (<5 chars): {counts["dropped_too_short"]}', flush=True)
    print(f'  - explicit noise blocklist: {counts["dropped_explicit_noise"]}', flush=True)
    print(f'  - in English dict (and not in DRUG_ALLOWLIST): {counts["dropped_in_english_dict"]}', flush=True)

    print(f'\nSample KEPT single-word entries:', flush=True)
    for s in sample_kept:
        print(f'  {s[0]}\t{s[1]}', flush=True)
    print(f'\nSample DROPPED entries:', flush=True)
    for s in sample_dropped:
        print(f'  {s[0]}\t{s[1]}\t({s[2]})', flush=True)

    STATS_PATH.write_text(json.dumps(counts, indent=2))
    print(f'\nClean vocab written to: {OUTPUT_TSV}', flush=True)
    print(f'Stats: {STATS_PATH}', flush=True)


if __name__ == '__main__':
    main()
