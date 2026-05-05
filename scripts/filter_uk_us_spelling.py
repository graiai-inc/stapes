#!/usr/bin/env python3
"""Filter UK/US spelling differences from Parakeet's medical-error log.

These aren't real ASR errors — they're normalization artifacts. The reference
uses one regional spelling, Parakeet outputs the other. Counts depend on:
  - explicit UK↔US pairs (medical and general)
  - suffix rules: -our/-or, -ise/-ize, -isation/-ization, -re/-er, -ae/-e, -oe/-e

Inputs:
  results/parakeet_medical_breakdown/<dataset>_per_error.tsv

Outputs:
  results/parakeet_medical_breakdown/<dataset>_uk_us_artifacts.tsv
    file_id, term, parakeet_sub, rule_matched
  results/parakeet_medical_breakdown/<dataset>_real_errors.tsv
    same columns as per_error but with UK/US artifacts removed
  results/parakeet_medical_breakdown/uk_us_summary.tsv
    dataset, total_errors, uk_us_artifacts, real_errors, pct_artifacts

Run with the ossicles venv (or any python — no special deps).
"""
import csv
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
STAPES_DIR = SCRIPT_DIR.parent
OUT_DIR = STAPES_DIR / 'results' / 'parakeet_medical_breakdown'

DATASETS = ['primock57', 'figshare-osce', 'nazmulkazi']

# ── Explicit UK→US pairs (medical first, then general) ──
UK_US_PAIRS = {
    # Medical
    'diarrhoea': 'diarrhea', 'oedema': 'edema', 'oesophagus': 'esophagus',
    'oestrogen': 'estrogen', 'oestrogens': 'estrogens',
    'haemorrhage': 'hemorrhage', 'haemorrhages': 'hemorrhages',
    'haematology': 'hematology', 'haematological': 'hematological',
    'haematoma': 'hematoma', 'haemoglobin': 'hemoglobin',
    'haemodynamic': 'hemodynamic', 'haemoptysis': 'hemoptysis',
    'haemophilia': 'hemophilia', 'haematuria': 'hematuria',
    'haemothorax': 'hemothorax', 'haemolysis': 'hemolysis',
    'anaemia': 'anemia', 'anaemic': 'anemic',
    'leukaemia': 'leukemia', 'leukaemic': 'leukemic',
    'caesarean': 'cesarean', 'caesarian': 'cesarian',
    'foetus': 'fetus', 'foetal': 'fetal',
    'paediatric': 'pediatric', 'paediatrics': 'pediatrics',
    'paediatrician': 'pediatrician',
    'gynaecology': 'gynecology', 'gynaecological': 'gynecological',
    'gynaecologist': 'gynecologist',
    'orthopaedic': 'orthopedic', 'orthopaedics': 'orthopedics',
    'aetiology': 'etiology',
    'coeliac': 'celiac', 'leucocyte': 'leukocyte',
    'leucocytes': 'leukocytes', 'leucocytosis': 'leukocytosis',
    'amoebic': 'amebic', 'amoeba': 'ameba',
    'oedematous': 'edematous',
    'tumour': 'tumor', 'tumours': 'tumors',
    'tonsillitis': 'tonsilitis',  # different
    'manoeuvre': 'maneuver', 'manoeuvres': 'maneuvers',
    'speciality': 'specialty', 'specialities': 'specialties',
    'theatre': 'theater', 'theatres': 'theaters',
    'centre': 'center', 'centres': 'centers',
    'fibre': 'fiber', 'fibres': 'fibers',
    'litre': 'liter', 'litres': 'liters',
    'metre': 'meter', 'metres': 'meters',
    'millilitre': 'milliliter', 'millilitres': 'milliliters',
    'colour': 'color', 'colours': 'colors',
    'coloured': 'colored', 'colouration': 'coloration',
    'behaviour': 'behavior', 'behavioural': 'behavioral',
    'humour': 'humor', 'odour': 'odor',
    'flavour': 'flavor', 'savour': 'savor',
    'tumourous': 'tumorous',
    'analyse': 'analyze', 'analysed': 'analyzed', 'analysing': 'analyzing',
    'analysis': 'analysis',  # same
    'recognise': 'recognize', 'recognised': 'recognized',
    'realise': 'realize', 'realised': 'realized',
    'organise': 'organize', 'organised': 'organized',
    'characterise': 'characterize', 'characterised': 'characterized',
    'standardise': 'standardize', 'standardised': 'standardized',
    'sterilise': 'sterilize', 'sterilised': 'sterilized',
    'minimise': 'minimize', 'maximise': 'maximize',
    'mobilise': 'mobilize', 'immobilise': 'immobilize',
    'familiarise': 'familiarize',
    'hospitalise': 'hospitalize', 'hospitalised': 'hospitalized',
    'specialise': 'specialize', 'specialised': 'specialized',
    'localise': 'localize', 'localised': 'localized',
    'utilise': 'utilize', 'utilised': 'utilized',
    'normalise': 'normalize', 'normalised': 'normalized',
    'prioritise': 'prioritize', 'prioritised': 'prioritized',
    'defence': 'defense', 'offence': 'offense',
    'licence': 'license',  # noun in UK; verb is license in both
    'practice': 'practice',  # same noun; verb diff but rare
    'practising': 'practicing', 'practised': 'practiced',
    'enquire': 'inquire', 'enquiry': 'inquiry',
    'storey': 'story', 'storeys': 'stories',
    'mould': 'mold', 'moulds': 'molds',
    'programme': 'program', 'programmes': 'programs',
    'kerb': 'curb', 'kerbs': 'curbs',
    'cheque': 'check', 'cheques': 'checks',
    'plough': 'plow', 'ploughed': 'plowed',
    'aluminium': 'aluminum',
    'sulphur': 'sulfur', 'sulphate': 'sulfate', 'sulphide': 'sulfide',
    'sulphonamide': 'sulfonamide',
    # Hyphenation common in medical
    'mini-stroke': 'mini stroke', 'light-headed': 'light headed',
    'short-term': 'short term', 'long-term': 'long term',
    'follow-up': 'follow up', 'follow-ups': 'follow ups',
    'check-up': 'check up', 'check-ups': 'check ups',
    'x-ray': 'xray', 'x-rays': 'xrays',
    'in-patient': 'inpatient', 'in-patients': 'inpatients',
    'out-patient': 'outpatient', 'out-patients': 'outpatients',
    'first-time': 'first time',
    'fit-and-well': 'fit and well',
    'home-care': 'home care', 'self-care': 'self care',
    'over-the-counter': 'over the counter',
}

# Reverse for US→UK (so we catch both directions)
US_UK_PAIRS = {us: uk for uk, us in UK_US_PAIRS.items() if us != uk}


def is_uk_us_artifact(term: str, sub: str) -> str | None:
    """Return rule name if term/sub is a UK/US spelling artifact, else None.

    Term is the reference word (already lower-cased & punct-stripped).
    Sub is Parakeet's substitution (may contain multiple words).
    """
    term = term.lower().strip()
    sub_norm = sub.lower().strip()
    sub_words = sub_norm.split()

    # 1. Direct dictionary lookup (UK in ref, US in sub)
    if term in UK_US_PAIRS:
        target = UK_US_PAIRS[term]
        if target == sub_norm or target in sub_words:
            return f'uk_to_us:{term}->{target}'
        # Also check if the sub is the no-space version (e.g., x-ray -> xray)
        if sub_norm.replace(' ', '') == target.replace(' ', ''):
            return f'uk_to_us:{term}->{target}'

    # 2. Reverse direction (US in ref, UK in sub)
    if term in US_UK_PAIRS:
        target = US_UK_PAIRS[term]
        if target == sub_norm or target in sub_words:
            return f'us_to_uk:{term}->{target}'

    # 3. Suffix rules — these catch pairs not in the explicit dict
    # -our/-or
    if term.endswith('our') and sub_norm.endswith('or'):
        if term[:-3] == sub_norm[:-2]:
            return 'suffix:-our/-or'
    if term.endswith('our') and len(sub_words) == 1 and sub_words[0].endswith('or'):
        if term[:-3] == sub_words[0][:-2]:
            return 'suffix:-our/-or'

    # -ise/-ize, -ised/-ized, -ising/-izing, -isation/-ization
    for uk_suf, us_suf in [
        ('ise', 'ize'), ('ised', 'ized'), ('ising', 'izing'),
        ('isation', 'ization'), ('isations', 'izations'),
    ]:
        if term.endswith(uk_suf):
            stem = term[:-len(uk_suf)]
            if sub_norm == stem + us_suf:
                return f'suffix:-{uk_suf}/-{us_suf}'

    # -re/-er
    if term.endswith('re') and len(term) > 4:
        stem = term[:-2]
        if sub_norm == stem + 'er':
            return 'suffix:-re/-er'

    # -ae- / -e- (like haemorrhage / hemorrhage already in dict, but generic)
    if 'ae' in term and term.replace('ae', 'e') == sub_norm:
        return 'suffix:-ae-/-e-'
    if 'oe' in term and term.replace('oe', 'e') == sub_norm:
        return 'suffix:-oe-/-e-'

    # -ogue/-og
    if term.endswith('ogue') and sub_norm == term[:-2]:
        return 'suffix:-ogue/-og'

    # -mme/-m (programme / program)
    if term.endswith('mme') and sub_norm == term[:-2]:
        return 'suffix:-mme/-m'

    # Hyphen vs space: "mini-stroke" → "mini stroke"
    if '-' in term and term.replace('-', ' ') == sub_norm:
        return 'hyphenation:hyphen->space'
    if '-' in term and term.replace('-', '') == sub_norm:
        return 'hyphenation:hyphen->joined'

    return None


def main():
    fh_summary = open(OUT_DIR / 'uk_us_summary.tsv', 'w')
    fh_summary.write('dataset\ttotal_errors\tuk_us_artifacts\treal_errors\tpct_artifacts\n')
    fh_summary.flush()

    rule_counter_global = {}

    for dataset in DATASETS:
        per_error_path = OUT_DIR / f'{dataset}_per_error.tsv'
        if not per_error_path.exists():
            print(f'[{dataset}] no per_error.tsv', flush=True)
            continue

        artifacts_path = OUT_DIR / f'{dataset}_uk_us_artifacts.tsv'
        real_errors_path = OUT_DIR / f'{dataset}_real_errors.tsv'

        fh_art = open(artifacts_path, 'w')
        fh_art.write('file_id\tterm\tcategory\tlength\tparakeet_sub\trule\n')
        fh_art.flush()

        fh_real = open(real_errors_path, 'w')
        fh_real.write('file_id\tterm\tcategory\tlength\tparakeet_sub\tref_context\n')
        fh_real.flush()

        n_total = n_artifact = 0
        rule_counter = {}
        with open(per_error_path) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                n_total += 1
                term = row['term']
                sub = row['parakeet_sub']
                rule = is_uk_us_artifact(term, sub)
                if rule:
                    n_artifact += 1
                    rule_counter[rule] = rule_counter.get(rule, 0) + 1
                    rule_counter_global[rule] = rule_counter_global.get(rule, 0) + 1
                    fh_art.write(
                        f'{row["file_id"]}\t{row["term"]}\t{row["category"]}\t'
                        f'{row["length"]}\t{row["parakeet_sub"]}\t{rule}\n'
                    )
                    fh_art.flush()
                else:
                    fh_real.write(
                        f'{row["file_id"]}\t{row["term"]}\t{row["category"]}\t'
                        f'{row["length"]}\t{row["parakeet_sub"]}\t{row["ref_context"]}\n'
                    )
                    fh_real.flush()
        fh_art.close(); fh_real.close()

        n_real = n_total - n_artifact
        pct = 100 * n_artifact / n_total if n_total else 0
        fh_summary.write(f'{dataset}\t{n_total}\t{n_artifact}\t{n_real}\t{pct:.2f}\n')
        fh_summary.flush()

        print(f'[{dataset}] {n_total} errors -> {n_artifact} UK/US artifacts ({pct:.1f}%) -> {n_real} real errors', flush=True)
        print(f'  Rules matched in this dataset:', flush=True)
        for rule, n in sorted(rule_counter.items(), key=lambda x: -x[1]):
            print(f'    {rule}: {n}', flush=True)

    fh_summary.close()

    print(f'\n=== Global rule counts across all datasets ===', flush=True)
    for rule, n in sorted(rule_counter_global.items(), key=lambda x: -x[1]):
        print(f'  {rule}: {n}', flush=True)


if __name__ == '__main__':
    main()
