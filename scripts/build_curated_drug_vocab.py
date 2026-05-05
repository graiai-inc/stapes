#!/usr/bin/env python3
"""Build a curated drug-only vocab by intersecting curated drug sources with UMLS DRUG vocab.

Why this exists
---------------
Our 100K-term UMLS vocab (medical_vocab.tsv) has ~19K DRUG-tagged terms, but UMLS
itself contains noise: 'food', 'fat', 'comfort', 'hmm', 'recap' are all tagged T121
(Pharmacologic Substance) at exact match because the Metathesaurus aggregates broad
sources (CHV, MTH). Using this vocab as-is causes targeted-fusion v3 to substitute
correct Parakeet output with wrong drug names when MedASR mentions one of these noise
tokens.

Approach
--------
Allowlist-style: a UMLS DRUG term is kept ONLY if at least one of its tokens appears
in a curated drug source we trust. Sources:

  1. ClinCalc Top 300 (2023): the most-prescribed US drugs, generics only.
  2. Orbweaver clinical PDFs (WVEMS Protocols, ProtocolBook, WHO Pocket Book): drug
     tokens are extracted from PDF text, then VALIDATED by appearing in UMLS DRUG
     single-word entries. This keeps brands ("tylenol", "advil", "lipitor") and
     uncommon drugs that PDFs reference but ClinCalc omits.

A small DENYLIST of UMLS-noise tokens (food, fat, comfort, comforts, hmm, recap)
is subtracted explicitly, so even if a PDF mentions 'food' it does not survive.

Final vocab is written in the same format as medical_vocab.tsv so it slots into
measure_targeted_fusion_v3.py without changes:
  columns: term, category, first_tier, first_source

Usage:
    /home/grey/dev/graiai/ossicles/venv/bin/python \\
        scripts/build_curated_drug_vocab.py
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
STAPES_DIR = SCRIPT_DIR.parent
DATA_DIR = STAPES_DIR / 'data'
RESULTS_DIR = STAPES_DIR / 'results' / 'medical_vocab'

UMLS_VOCAB_TSV = RESULTS_DIR / 'medical_vocab.tsv'
CLINCALC_TSV = DATA_DIR / 'clincalc_top300.tsv'
BRAND_SEEDS_TSV = DATA_DIR / 'drug_brand_seeds.tsv'
PDF_TEXT_DIR = DATA_DIR / 'orbweaver_pdf_text'
PDF_FILES = ['wvems_protocols.txt', 'protocolbook.txt', 'who_pocketbook.txt']

OUT_VOCAB_TSV = RESULTS_DIR / 'medical_vocab_drug_curated.tsv'
OUT_SEED_TSV = DATA_DIR / 'drug_seed_tokens.tsv'
OUT_STATS_JSON = DATA_DIR / 'drug_curation_stats.json'

# Tokens that UMLS tags as DRUG (T121 et al.) but are layperson/admin noise.
# Discovered through:
#   (1) prior session: food, fat, comfort, hmm, recap (direct QuickUMLS hits at threshold=1.0)
#   (2) sample inspection of the curated vocab — 200-sample showed ~25% noise rate from
#       PDF-derived tokens being over-tagged as DRUG.
DENYLIST = {
    # Direct UMLS-DRUG-tagged noise from prior session
    'food', 'foods', 'fat', 'fats', 'comfort', 'comforts', 'hmm', 'recap', 'recaps',
    # ----- Drug CLASS words (not specific drugs; would trigger spurious subs) -----
    'antibiotic', 'antibiotics', 'antimicrobial', 'antimicrobials',
    'antifungal', 'antifungals', 'antiviral', 'antivirals',
    'antimalarial', 'antimalarials', 'antiparasitic',
    'antipsychotic', 'antipsychotics', 'neuroleptic', 'neuroleptics',
    'antidepressant', 'antidepressants', 'tricyclic', 'tricyclics',
    'ssri', 'ssris', 'snri', 'snris', 'maoi', 'maois',
    'benzodiazepine', 'benzodiazepines', 'barbiturate', 'barbiturates',
    'sedative', 'sedatives', 'hypnotic', 'hypnotics',
    'stimulant', 'stimulants', 'amphetamine',  # note: 'amphetamine' is a class AND drug; rely on UMLS multi-word
    'opioid', 'opioids', 'opiate', 'opiates', 'narcotic', 'narcotics',
    'analgesic', 'analgesics',
    'anesthetic', 'anesthetics', 'anaesthetic', 'anaesthetics',
    'antiseptic', 'antiseptics', 'disinfectant', 'disinfectants',
    'antagonist', 'antagonists', 'agonist', 'agonists',
    'inhibitor', 'inhibitors',
    'antiarrhythmic', 'antiarrhythmics',
    'anticonvulsant', 'anticonvulsants',
    'antihistamine', 'antihistamines',
    'antiemetic', 'antiemetics',
    'antihypertensive', 'antihypertensives',
    'beta-blocker', 'betablocker', 'blockers', 'blocker',
    'diuretic', 'diuretics',
    'bronchodilator', 'bronchodilators',
    'corticosteroid', 'corticosteroids', 'steroid', 'steroids',
    'glucocorticoid', 'glucocorticoids',
    'mineralocorticoid', 'mineralocorticoids',
    'nsaid', 'nsaids',
    'thrombolytic', 'thrombolytics', 'fibrinolytic', 'fibrinolytics',
    'anticoagulant', 'anticoagulants',
    'antiplatelet', 'antiplatelets',
    'pressor', 'pressors', 'vasopressor', 'vasopressors',
    'vasodilator', 'vasodilators', 'vasoconstrictor', 'vasoconstrictors',
    'sulfa', 'sulfas',
    'sulfonylurea', 'sulfonylureas',
    'biguanide', 'biguanides',
    'statin', 'statins',
    'fibrate', 'fibrates',
    'nitrate', 'nitrates',
    'penicillin', 'penicillins',  # the class word; specific drugs like 'penicillin v' kept
    'cephalosporin', 'cephalosporins',
    'macrolide', 'macrolides',
    'fluoroquinolone', 'fluoroquinolones',
    'tetracycline', 'tetracyclines',  # 'tetracycline' is also a drug; drop to be safe
    'aminoglycoside', 'aminoglycosides',
    'lactam', 'lactams', 'beta-lactam', 'betalactam',
    'vitamin', 'vitamins', 'multivitamin', 'multivitamins',
    'mineral', 'minerals',
    'electrolyte', 'electrolytes',
    'supplement', 'supplements',
    'placebo', 'placebos',
    # ----- Foods -----
    'avocado', 'avocados', 'apple', 'apples', 'banana', 'bananas',
    'corn', 'potato', 'potatoes', 'lentils', 'lentil', 'spinach',
    'rice', 'wheat', 'bread', 'meat', 'meats', 'fish',
    'milk', 'cheese', 'butter', 'cream', 'creams', 'yogurt',
    'lemon', 'lemons', 'lime', 'limes', 'orange', 'oranges',
    'vinegar', 'honey',
    'carrot', 'carrots', 'onion', 'onions', 'garlic',
    # ----- Elements / generic chemicals -----
    'oxygen', 'nitrogen', 'hydrogen', 'helium',
    'copper', 'zinc', 'iron', 'gold', 'silver',
    'sodium', 'potassium', 'magnesium', 'phosphorus',  # bare element; specific salts (e.g. 'sodium chloride') kept
    'lead', 'mercury',
    'sucrose', 'glucose', 'fructose', 'lactose', 'sugar', 'sugars',
    'starch', 'starches',
    'kerosene', 'gasoline', 'alcohol',  # alcohols-as-class problem; specific 'isopropyl alcohol' kept
    'dioxide', 'monoxide', 'oxide',
    'hydrochloride', 'hydrochlorides',  # excipient/salt suffix that appears alone
    # ----- Body parts / anatomy -----
    'iris', 'lens', 'eye', 'eyes', 'ear', 'ears', 'nose', 'mouth',
    'throat', 'sinus', 'sinuses', 'lung', 'lungs', 'heart', 'liver',
    'kidney', 'kidneys', 'brain', 'spine', 'skin', 'bone', 'bones',
    'muscle', 'muscles', 'nerve', 'nerves', 'vein', 'veins',
    'artery', 'arteries', 'gland', 'glands', 'organ', 'organs',
    'posture', 'postures',
    # ----- Animals / insect references (UMLS includes venom/toxin tags) -----
    'insect', 'insects', 'bee', 'bees', 'wasp', 'wasps',
    'spider', 'spiders', 'scorpion', 'scorpions',
    'jellyfish', 'snake', 'snakes',
    'rotavirus',  # this is a VIRUS; 'rotavirus vaccine' is the drug, kept via multi-word
    'pneumococcal',  # adjective; 'pneumococcal vaccine' kept
    # ----- Common verbs / adjectives that UMLS occasionally over-tags -----
    'allay', 'soothe', 'soothes', 'prevent', 'prevents',
    'compete', 'competes', 'complete', 'completes',
    'reconcile', 'reconciles', 'damp', 'damps',
    'gentle', 'duration', 'durations',
    'trip', 'trips', 'probe', 'probes',
    'stopping', 'starting', 'continuing',
    'injectable', 'injectables', 'oral', 'topical', 'systemic',
    'natural', 'synthetic', 'recombinant',
    'normal', 'abnormal',
    'anorexic', 'anorexics',
    'viru',  # OCR/junk fragment
    # ----- Generic English noise observed in prior fusion runs -----
    'others', 'other', 'renewal', 'renewals', 'active', 'commit',
    'rise', 'rises', 'date', 'dates', 'pat', 'snow',
    'cap', 'caps', 'tube', 'tubes', 'face', 'faces', 'hand', 'hands',
    'back', 'lower', 'upper', 'else', 'couple', 'definitely',
    'symptoms', 'medications', 'medication',
    'cough', 'coughs', 'cold', 'colds', 'hot', 'happy', 'sad', 'fun',
    'history', 'name', 'medical', 'clinic', 'home', 'work',
    'family', 'group', 'friend', 'mother', 'father', 'parent', 'child',
    'children', 'sister', 'brother', 'wife', 'husband',
    'ill', 'hearts', 'results', 'patient', 'patients',
    'water', 'waters', 'air', 'gas', 'gases', 'oil', 'oils',
    'salt', 'salts',
    'visit', 'visits', 'today', 'morning', 'evening', 'night',
    # ----- Second-pass discoveries from 200-sample inspection -----
    # Drug-class plurals/hyphens missed in the first pass
    'anti-arrhythmics', 'antiarrhythmics', 'anticholinergic', 'anticholinergics',
    'anticoag', 'contraceptive', 'contraceptives',
    'crystalloid', 'crystalloids', 'colloid', 'colloids',
    # Generic chemicals/elements missed
    'carbon', 'chloride', 'chlorides', 'hydroxide', 'hydroxides',
    'nitrite', 'nitrites', 'phosphate', 'phosphates', 'sulfate', 'sulfates',
    'carbonate', 'carbonates',  # bare salt suffix; specific 'calcium carbonate' kept by multi-word
    'selenium', 'iodide', 'iodides',
    'petroleum', 'paraffin',
    # Procedure / class / object nouns
    'pill', 'pills', 'tablet', 'tablets', 'capsule', 'capsules',
    'vaccine', 'vaccines', 'serum', 'serums',
    'prbcs', 'rbcs', 'platelets', 'plasma',  # blood products; specific names kept multi-word
    'transfusion', 'transfusions', 'infusion', 'infusions',
    'maternity', 'pregnancy',
    'copd',  # diagnosis acronym (not a drug)
    # Lab / biomarker / chemistry words
    'ferritin', 'folate', 'cofactor', 'cofactors',
    'angiotensin',  # endogenous peptide; not a drug (drugs are 'angiotensin receptor blockers')
    # Common English / medical words still leaking
    'advantage', 'advantages', 'animals', 'animal',
    'basis', 'bore', 'cape', 'capes', 'chicken', 'chickens',
    'combinations', 'combination', 'control', 'controls',
    'cope', 'copes', 'correct', 'corrects',
    'crystal', 'crystals', 'glass', 'glasses',
    'hydrated', 'hydrates', 'hydration',
    'isolate', 'isolates', 'isolation',
    'linden', 'medusa', 'nips', 'various',
    # OCR/junk fragments observed
    'tinct',  # 'tincture' fragment
    # ----- Third-pass discoveries from 80-sample re-inspection -----
    'antifibrinolytic', 'antifibrinolytics',
    'antipyretic', 'antipyretics',
    'surfactant', 'surfactants',
    'legumes', 'legume', 'mango', 'mangoes', 'mangos',
    'pear', 'pears', 'peach', 'peaches', 'plum', 'plums', 'grape', 'grapes',
    'shot', 'shots', 'jab', 'jabs', 'injection', 'injections',
    'sleep', 'sleeps', 'sleeping', 'sleeper',
    'awake', 'awaken',
    'zita',  # likely OCR noise / non-standard fragment
    # ----- Discovered from fusion-v3 run substitution log -----
    # Class words that anchored spurious subs (medicine→medicines, biologic→biologics)
    'medicine', 'medicines', 'remedy', 'remedies',
    'biologic', 'biologics', 'biological', 'biologicals',
    'central',  # adjective (CNS); not a specific drug
    'compound', 'compounds',  # generic
    'agent', 'agents',  # generic
    'preparation', 'preparations',
    'product', 'products',
    'formula', 'formulas',
    'solution', 'solutions',  # IV "solution" is too generic
    'mixture', 'mixtures',
    'extract', 'extracts',
    'dose', 'doses', 'dosing', 'dosage', 'dosages',
}

# Match alphabetic + hyphen tokens (drugs are alphabetic; numbers like '500mg' are
# dose info, not drug names). Length ≥ 4 to drop short noise tokens.
TOKEN_RE = re.compile(r"[a-z][a-z\-]{3,}")


def load_clincalc_seeds(path: Path) -> set[str]:
    seeds = set()
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            generic = row['generic'].strip().lower()
            if generic:
                seeds.add(generic)
                # Also add each token of multi-word generics (e.g. 'insulin glargine'
                # contributes both 'insulin' and 'glargine' as token-level seeds).
                for tok in generic.split():
                    if len(tok) >= 4:
                        seeds.add(tok)
    return seeds


def load_brand_seeds(path: Path) -> set[str]:
    """Load hand-curated brand-name seeds. Returns brand names AND their generics
    (in case the generic isn't covered by ClinCalc, e.g. fentanyl)."""
    seeds = set()
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            brand = row.get('brand', '').strip().lower()
            generic = row.get('generic', '').strip().lower()
            if brand:
                seeds.add(brand)
            if generic:
                seeds.add(generic)
                for tok in generic.split():
                    if len(tok) >= 4:
                        seeds.add(tok)
    return seeds


def load_umls_drug_terms(path: Path) -> tuple[set[str], dict[str, dict]]:
    """Return (set of single-word DRUG tokens, full DRUG term metadata dict).

    full_meta maps term -> {category, first_tier, first_source}.
    """
    single = set()
    full = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            cat = row.get('category', '').strip()
            if cat != 'DRUG':
                continue
            term = row.get('term', '').strip().lower()
            if not term:
                continue
            full[term] = {
                'category': cat,
                'first_tier': row.get('first_tier', ''),
                'first_source': row.get('first_source', ''),
            }
            if ' ' not in term:
                single.add(term)
    return single, full


def extract_pdf_drug_tokens(pdf_dir: Path, files: list[str], umls_single: set[str]) -> Counter:
    """For each PDF: tokenize, lowercase, keep only tokens that appear in UMLS DRUG
    single-word vocab. Return Counter of (token -> count of PDFs it appeared in)."""
    presence = Counter()
    for fname in files:
        path = pdf_dir / fname
        if not path.exists():
            print(f'  MISSING: {path}', flush=True)
            continue
        text = path.read_text(errors='ignore').lower()
        tokens = set(TOKEN_RE.findall(text))
        drug_tokens = tokens & umls_single
        for t in drug_tokens:
            presence[t] += 1
        print(f'  {fname}: {len(tokens)} unique tokens, {len(drug_tokens)} are UMLS-DRUG',
              flush=True)
    return presence


def main():
    print('=== Curated drug vocab builder ===', flush=True)
    print(f'Loading UMLS DRUG vocab from {UMLS_VOCAB_TSV}', flush=True)
    umls_single, umls_full = load_umls_drug_terms(UMLS_VOCAB_TSV)
    print(f'  {len(umls_full)} total UMLS DRUG terms ({len(umls_single)} single-word)',
          flush=True)

    print(f'Loading ClinCalc seeds from {CLINCALC_TSV}', flush=True)
    clincalc = load_clincalc_seeds(CLINCALC_TSV)
    print(f'  {len(clincalc)} ClinCalc seed tokens (generics + their constituent tokens)',
          flush=True)

    print(f'Loading hand-curated brand seeds from {BRAND_SEEDS_TSV}', flush=True)
    brand_seeds = load_brand_seeds(BRAND_SEEDS_TSV)
    print(f'  {len(brand_seeds)} brand-seed tokens (brands + their generics)', flush=True)

    print(f'Tokenizing orbweaver PDFs from {PDF_TEXT_DIR}', flush=True)
    pdf_presence = extract_pdf_drug_tokens(PDF_TEXT_DIR, PDF_FILES, umls_single)
    print(f'  {len(pdf_presence)} unique UMLS-DRUG tokens found across PDFs',
          flush=True)

    # Build seed set: ClinCalc generics + brand seeds + PDF-validated drug tokens
    seeds = set(clincalc)
    seeds |= brand_seeds
    seeds |= set(pdf_presence.keys())
    n_before_denylist = len(seeds)
    seeds -= DENYLIST
    print(f'\nSeed set: {n_before_denylist} candidates → {len(seeds)} after denylist '
          f'(removed {n_before_denylist - len(seeds)})', flush=True)

    # Save seeds for inspection
    fh_seed = open(OUT_SEED_TSV, 'w')
    fh_seed.write('seed\tin_clincalc\tin_pdfs_count\n')
    fh_seed.flush()
    for s in sorted(seeds):
        in_cc = '1' if s in clincalc else '0'
        n_pdfs = pdf_presence.get(s, 0)
        fh_seed.write(f'{s}\t{in_cc}\t{n_pdfs}\n')
        fh_seed.flush()
    fh_seed.close()
    print(f'  seeds saved to {OUT_SEED_TSV}', flush=True)

    # Now build curated DRUG vocab: keep any UMLS DRUG term whose tokens include at
    # least one seed token. ALSO ensure no token of the term is on the DENYLIST
    # (prevents 'food' multi-word forms like 'food allergy' from sneaking in).
    fh_out = open(OUT_VOCAB_TSV, 'w')
    fh_out.write('term\tcategory\tfirst_tier\tfirst_source\n')
    fh_out.flush()
    n_kept = 0
    n_dropped_no_seed = 0
    n_dropped_denylist = 0
    n_progress = 0
    for term in sorted(umls_full):
        n_progress += 1
        if n_progress % 5000 == 0:
            print(f'  scanned {n_progress}/{len(umls_full)} UMLS DRUG terms, kept {n_kept}',
                  flush=True)
        toks = term.split()
        if any(t in DENYLIST for t in toks):
            n_dropped_denylist += 1
            continue
        if not any(t in seeds for t in toks):
            n_dropped_no_seed += 1
            continue
        meta = umls_full[term]
        fh_out.write(f'{term}\t{meta["category"]}\t{meta["first_tier"]}\t{meta["first_source"]}\n')
        fh_out.flush()
        n_kept += 1
    fh_out.close()

    # Stats
    import json
    stats = {
        'umls_drug_total': len(umls_full),
        'umls_drug_single_word': len(umls_single),
        'clincalc_seed_tokens': len(clincalc),
        'brand_seed_tokens': len(brand_seeds),
        'pdf_drug_tokens_unique': len(pdf_presence),
        'final_seed_set_after_denylist': len(seeds),
        'curated_vocab_size': n_kept,
        'dropped_no_seed_match': n_dropped_no_seed,
        'dropped_denylist_token': n_dropped_denylist,
    }
    with open(OUT_STATS_JSON, 'w') as f:
        json.dump(stats, f, indent=2)

    print('\n=== SUMMARY ===', flush=True)
    for k, v in stats.items():
        print(f'  {k}: {v}', flush=True)
    print(f'\nCurated vocab written to: {OUT_VOCAB_TSV}', flush=True)
    print(f'Stats: {OUT_STATS_JSON}', flush=True)

    # Sanity check: confirm denylist words are not in output
    print('\n=== SANITY CHECK ===', flush=True)
    written_terms = set()
    with open(OUT_VOCAB_TSV) as f:
        next(f)
        for line in f:
            written_terms.add(line.split('\t', 1)[0])
    leaked = DENYLIST & written_terms
    print(f'  denylist tokens leaked to output: {len(leaked)} ({sorted(leaked)})',
          flush=True)
    # Show drugs we know to test were caught
    test_drugs = ['atorvastatin', 'tylenol', 'lipitor', 'advil', 'ibuprofen',
                  'ramipril', 'klonopin', 'wellbutrin', 'seroquel', 'crestor',
                  'abilify', 'ambien', 'rosuvastatin', 'lisinopril']
    print('  test drug presence:', flush=True)
    for d in test_drugs:
        present = d in written_terms
        print(f'    {d}: {"YES" if present else "NO"}', flush=True)


if __name__ == '__main__':
    main()
