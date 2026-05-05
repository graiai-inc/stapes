#!/usr/bin/env python3
"""Clinical text normalizer for ASR WER computation.

Layered on top of Whisper EnglishTextNormalizer (the de facto baseline used
by the HuggingFace Open ASR Leaderboard and MLPerf). Adds meaning-preserving
normalizations that the Whisper normalizer leaves on the table:

  1. UK/US spelling variants (medical and general) → canonical US form
  2. Hyphenation (x-ray, xray, x ray) → single canonical form
  3. Open vs closed compound words (healthcare, health care) → canonical form
  4. Possessive eponyms (Crohn's, Crohns, Crohn) → canonical form
  5. Spaced acronyms (C T scan, c t scan) → joined form (ct scan)
  6. Honorifics (Dr., Doctor; Mrs., missus) → canonical form
  7. Dosing abbreviations (5 mg, 5 milligrams) → abbreviated form

These categories do not change clinical meaning. The normalizer is symmetric:
applied identically to reference and hypothesis. Idempotent.

Usage:
    from extended_normalizer import normalize_clinical
    ref = normalize_clinical("She has Crohn's, takes 5 milligrams of Lisinopril.")
    hyp = normalize_clinical("she has crohns takes 5 mg of lisinopril")
    # Both will normalize to the same string.

Self-test: `python extended_normalizer.py` runs assertions and prints PASS.
"""
import re
from functools import lru_cache

from whisper_normalizer.english import EnglishTextNormalizer

_WHISPER = EnglishTextNormalizer()


# ── Category 1: UK/US spelling ──────────────────────────────────────────────
# Direct pairs (UK form → US form). Reuses dictionary from
# stapes/scripts/filter_uk_us_spelling.py. Keep medical first.
UK_TO_US = {
    # Medical (digestive / hematology / oncology / etc.)
    'diarrhoea': 'diarrhea', 'oedema': 'edema', 'oesophagus': 'esophagus',
    'oestrogen': 'estrogen', 'oestrogens': 'estrogens',
    'haemorrhage': 'hemorrhage', 'haemorrhages': 'hemorrhages',
    'haematology': 'hematology', 'haematological': 'hematological',
    'haematoma': 'hematoma', 'haemoglobin': 'hemoglobin',
    'haemodynamic': 'hemodynamic', 'haemoptysis': 'hemoptysis',
    'haemophilia': 'hemophilia', 'haematuria': 'hematuria',
    'haemothorax': 'hemothorax', 'haemolysis': 'hemolysis',
    'haematogenous': 'hematogenous', 'haematocrit': 'hematocrit',
    'anaemia': 'anemia', 'anaemic': 'anemic',
    'leukaemia': 'leukemia', 'leukaemic': 'leukemic',
    'leucocyte': 'leukocyte', 'leucocytes': 'leukocytes',
    'leucocytosis': 'leukocytosis',
    'caesarean': 'cesarean', 'caesarian': 'cesarian',
    'foetus': 'fetus', 'foetal': 'fetal',
    'paediatric': 'pediatric', 'paediatrics': 'pediatrics',
    'paediatrician': 'pediatrician',
    'gynaecology': 'gynecology', 'gynaecological': 'gynecological',
    'gynaecologist': 'gynecologist',
    'orthopaedic': 'orthopedic', 'orthopaedics': 'orthopedics',
    'aetiology': 'etiology', 'aetiological': 'etiological',
    'coeliac': 'celiac',
    'amoebic': 'amebic', 'amoeba': 'ameba',
    'oedematous': 'edematous',
    'tumour': 'tumor', 'tumours': 'tumors', 'tumourous': 'tumorous',
    'manoeuvre': 'maneuver', 'manoeuvres': 'maneuvers',
    'speciality': 'specialty', 'specialities': 'specialties',
    'fibre': 'fiber', 'fibres': 'fibers',
    'litre': 'liter', 'litres': 'liters',
    'metre': 'meter', 'metres': 'meters',
    'millilitre': 'milliliter', 'millilitres': 'milliliters',
    'aluminium': 'aluminum',
    'sulphur': 'sulfur', 'sulphate': 'sulfate', 'sulphide': 'sulfide',
    'sulphonamide': 'sulfonamide',
    # General -our / -or
    'colour': 'color', 'colours': 'colors',
    'coloured': 'colored', 'colouration': 'coloration',
    'behaviour': 'behavior', 'behavioural': 'behavioral',
    'humour': 'humor', 'odour': 'odor',
    'flavour': 'flavor', 'savour': 'savor',
    # General -tre / -ter
    'centre': 'center', 'centres': 'centers',
    'theatre': 'theater', 'theatres': 'theaters',
    # General -ce / -se
    'defence': 'defense', 'offence': 'offense',
    # ise/ize verbs (Whisper normalizer sometimes leaves these)
    'analyse': 'analyze', 'analysed': 'analyzed', 'analysing': 'analyzing',
    'recognise': 'recognize', 'recognised': 'recognized', 'recognising': 'recognizing',
    'realise': 'realize', 'realised': 'realized', 'realising': 'realizing',
    'organise': 'organize', 'organised': 'organized', 'organising': 'organizing',
    'organisation': 'organization', 'organisations': 'organizations',
    'characterise': 'characterize', 'characterised': 'characterized',
    'standardise': 'standardize', 'standardised': 'standardized',
    'standardisation': 'standardization',
    'sterilise': 'sterilize', 'sterilised': 'sterilized',
    'sterilisation': 'sterilization',
    'minimise': 'minimize', 'minimised': 'minimized',
    'maximise': 'maximize', 'maximised': 'maximized',
    'mobilise': 'mobilize', 'immobilise': 'immobilize',
    'familiarise': 'familiarize',
    'hospitalise': 'hospitalize', 'hospitalised': 'hospitalized',
    'specialise': 'specialize', 'specialised': 'specialized',
    'localise': 'localize', 'localised': 'localized',
    'utilise': 'utilize', 'utilised': 'utilized',
    'normalise': 'normalize', 'normalised': 'normalized',
    'prioritise': 'prioritize', 'prioritised': 'prioritized',
    # Misc
    'practising': 'practicing', 'practised': 'practiced',
    'enquire': 'inquire', 'enquiry': 'inquiry',
    'storey': 'story', 'storeys': 'stories',
    'mould': 'mold', 'moulds': 'molds',
    'programme': 'program', 'programmes': 'programs',
    'kerb': 'curb', 'kerbs': 'curbs',
    'cheque': 'check', 'cheques': 'checks',
    'plough': 'plow', 'ploughed': 'plowed',
    'travelled': 'traveled', 'travelling': 'traveling', 'traveller': 'traveler',
    'cancelled': 'canceled', 'cancelling': 'canceling',
    'modelled': 'modeled', 'modelling': 'modeling',
    'labelled': 'labeled', 'labelling': 'labeling',
    'grey': 'gray', 'greys': 'grays',
}


# ── Category 2 + 3: Hyphenation and open-vs-closed compounds ────────────────
# Apply two transforms: collapse hyphens, then collapse known open compounds.
# CLOSED_COMPOUNDS are pairs of words that should be joined (e.g., "health care"
# → "healthcare"). HYPHENATED tokens become unhyphenated in normalization.
CLOSED_COMPOUNDS = {
    # Medical
    ('x', 'ray'): 'xray',
    ('chest', 'xray'): 'chestxray',
    ('mini', 'stroke'): 'ministroke',
    ('light', 'headed'): 'lightheaded',
    ('non', 'smoker'): 'nonsmoker',
    ('non', 'smokers'): 'nonsmokers',
    ('non', 'smoking'): 'nonsmoking',
    ('co', 'morbidity'): 'comorbidity',
    ('co', 'morbidities'): 'comorbidities',
    ('co', 'pay'): 'copay',
    ('post', 'operative'): 'postoperative',
    ('pre', 'operative'): 'preoperative',
    ('post', 'op'): 'postop',
    ('pre', 'op'): 'preop',
    ('in', 'patient'): 'inpatient',
    ('in', 'patients'): 'inpatients',
    ('out', 'patient'): 'outpatient',
    ('out', 'patients'): 'outpatients',
    ('check', 'up'): 'checkup',
    ('check', 'ups'): 'checkups',
    ('follow', 'up'): 'followup',
    ('follow', 'ups'): 'followups',
    ('on', 'call'): 'oncall',
    ('off', 'label'): 'offlabel',
    ('over', 'the', 'counter'): 'overthecounter',
    ('short', 'term'): 'shortterm',
    ('long', 'term'): 'longterm',
    ('well', 'being'): 'wellbeing',
    ('self', 'care'): 'selfcare',
    ('home', 'care'): 'homecare',
    ('health', 'care'): 'healthcare',
    ('day', 'care'): 'daycare',
    ('chicken', 'pox'): 'chickenpox',
    ('high', 'risk'): 'highrisk',
    ('low', 'risk'): 'lowrisk',
    ('long', 'standing'): 'longstanding',
    ('cross', 'sectional'): 'crosssectional',
    # General
    ('every', 'day'): 'everyday',  # context-dependent; safe to normalize for WER
    ('any', 'one'): 'anyone',
    ('every', 'one'): 'everyone',
    ('some', 'one'): 'someone',
    ('no', 'one'): 'noone',
}


# ── Category 4: Possessive eponyms ──────────────────────────────────────────
# Map all forms (with apostrophe-s, with bare -s, bare) to canonical bare form.
# This is harder than spelling because the bare form is already lowercase
# and looks like a regular word; we list named conditions explicitly.
EPONYMS = {
    'crohn', 'alzheimer', 'parkinson', 'huntington', 'hodgkin',
    'cushing', 'addison', 'wilson', 'tourette', 'graves',
    'down',  # Down's syndrome
    'asperger', 'menieres', 'meniere',
    'bell',  # Bell's palsy
    'reye',  # Reye's syndrome
    'raynaud', 'marfan', 'ehlers',  # Ehlers-Danlos handled separately if needed
    'guillain', 'hashimoto',
    'klinefelter', 'turner', 'kawasaki',
    'paget', 'whipple', 'crigler', 'gilbert',
}


# ── Category 5: Spaced acronyms ─────────────────────────────────────────────
# Patterns like "c t scan" → "ct scan", "m r i" → "mri".
# Regex matches 2-5 single letters separated by spaces. Then we look up
# in KNOWN_ACRONYMS to confirm before joining (avoids collapsing "I am a").
KNOWN_ACRONYMS = {
    # Imaging — common in clinical conversation
    'ct', 'mri', 'pet', 'mra', 'cta', 'ekg', 'ecg', 'eeg', 'emg',
    'cxr', 'kub',
    # Dosing frequency (3+ letters only — 2-letter forms like ac/pc/hs
    # collide with common letter sequences)
    'prn', 'bid', 'tid', 'qid', 'qhs',
    # Care settings (3+ letters only — er/or/iv handled at 3-letter context)
    'icu', 'ccu', 'sicu', 'micu', 'picu', 'nicu',
    'obgyn', 'pcp',
    # Labs and conditions
    'cbc', 'cmp', 'bmp', 'lft', 'tsh', 'ptt', 'inr',
    'cva', 'tia', 'chf', 'copd', 'cad', 'dvt',
    'gerd', 'ibs', 'ibd', 'uti', 'std', 'sti',
    'hiv', 'hep',
    # Acronyms typically written but sometimes spaced
    'fda', 'cdc', 'nih', 'cms',
    'ehr', 'emr', 'asr', 'nlp', 'usa',
}

# Match runs of 2+ single letters separated by single spaces.
# Build per-acronym substitution patterns: each acronym becomes a regex
# matching its letters separated by single spaces ("ct" → r"\bc t\b").
def _build_acronym_patterns() -> list[tuple[re.Pattern, str]]:
    patterns = []
    for acronym in sorted(KNOWN_ACRONYMS, key=len, reverse=True):
        if len(acronym) < 2:
            continue
        spaced = ' '.join(acronym)
        pattern = re.compile(r'\b' + re.escape(spaced) + r'\b')
        patterns.append((pattern, acronym))
    return patterns


_ACRONYM_PATTERNS: list[tuple[re.Pattern, str]] | None = None


# Pre-Whisper eponym strip: turn "crohn's" → "crohns" so Whisper doesn't
# expand the possessive 's into "is".
_EPONYM_POSSESSIVE_RE = re.compile(
    r"\b(" + '|'.join(re.escape(e) for e in EPONYMS) + r")'s\b",
    re.IGNORECASE,
)


# ── Category 6: Honorifics ──────────────────────────────────────────────────
# Whisper normalizer expands many but is inconsistent. Force uniform expansion.
HONORIFIC_MAP = {
    'dr': 'doctor', 'drs': 'doctors',
    'mr': 'mister',
    'mrs': 'missus',
    'ms': 'miss',
    'st': 'saint',  # mostly only matters in proper names
    'jr': 'junior', 'sr': 'senior',
    'prof': 'professor',
}


# ── Category 7: Dosing abbreviations ────────────────────────────────────────
# Canonical: abbreviated form (mg, ml, kg, mcg).
# Whisper normalizer leaves both as-is, so we collapse spelled-out → abbrev.
DOSING_MAP = {
    'milligrams': 'mg', 'milligram': 'mg',
    'micrograms': 'mcg', 'microgram': 'mcg',
    'kilograms': 'kg', 'kilogram': 'kg',
    'grams': 'g', 'gram': 'g',
    'milliliters': 'ml', 'milliliter': 'ml',
    'millilitres': 'ml', 'millilitre': 'ml',  # also UK spelling
    'liters': 'l', 'liter': 'l',
    'litres': 'l', 'litre': 'l',
    'centimeters': 'cm', 'centimeter': 'cm',
    'centimetres': 'cm', 'centimetre': 'cm',
    'millimeters': 'mm', 'millimeter': 'mm',
    'millimetres': 'mm', 'millimetre': 'mm',
}


# ── Category 8: Backchannel / filler suppression ───────────────────────────
# Non-content tokens that some ASR systems emit verbatim and others suppress.
# These do not affect clinical documentation: no scribe writes "okay yeah" into
# a SOAP note. Suppressing them symmetrically (both reference and hypothesis)
# isolates content-word accuracy, which is what matters for documentation.
# Conservative list: only tokens with no clinical content meaning.
# Excluded deliberately (clinical/contextual content possible): right, well,
# so, like, fine, sure, no, yes (yes/no can carry clinical decisions).
BACKCHANNELS = {
    # Non-lexical
    'um', 'uh', 'er', 'ah', 'hmm', 'hm', 'mm', 'mmm',
    'mhm', 'mhmm', 'mmhmm',
    'uhhuh', 'huh',
    'hum', 'huh',
    # Affirmation backchannels (continuers, not decisions)
    'yeah', 'yep', 'yup', 'yah',
    'okay', 'ok',
    'oh',
}


# ── Suffix rules for UK→US (catches words not in the explicit dict) ─────────
def _suffix_uk_to_us(token: str) -> str:
    """Apply suffix-based UK→US rules. Returns transformed token or original."""
    if len(token) <= 3:
        return token
    # -our → -or (e.g., "vapour" → "vapor"). Only if not in dict already.
    if token.endswith('our') and len(token) > 4:
        candidate = token[:-3] + 'or'
        # Avoid false positives like "four", "your", "hour", "tour", "sour", "pour"
        if token not in {'four', 'your', 'hour', 'tour', 'sour', 'pour', 'flour'}:
            return candidate
    # -ise → -ize
    if token.endswith('ise') and len(token) > 4:
        return token[:-3] + 'ize'
    # -ised → -ized
    if token.endswith('ised') and len(token) > 5:
        return token[:-4] + 'ized'
    # -ising → -izing
    if token.endswith('ising') and len(token) > 6:
        return token[:-5] + 'izing'
    # -isation → -ization
    if token.endswith('isation') and len(token) > 8:
        return token[:-7] + 'ization'
    # -re → -er (only for words ending in -tre, -bre, -cre to avoid false hits)
    if token.endswith('tre') and len(token) > 4:
        return token[:-3] + 'ter'
    if token.endswith('bre') and len(token) > 4:
        return token[:-3] + 'ber'
    return token


def _normalize_uk_us(text: str) -> str:
    """Apply UK→US dictionary lookup + suffix rules to each whitespace token."""
    out = []
    for token in text.split():
        if token in UK_TO_US:
            out.append(UK_TO_US[token])
        else:
            out.append(_suffix_uk_to_us(token))
    return ' '.join(out)


def _normalize_hyphens(text: str) -> str:
    """Replace hyphens with spaces (so 'x-ray' → 'x ray', then later collapsed)."""
    return text.replace('-', ' ')


def _normalize_compounds_once(text: str) -> str:
    """Single pass: collapse known open compounds to closed form."""
    tokens = text.split()
    out = []
    i = 0
    while i < len(tokens):
        matched = False
        for length in (3, 2):
            if i + length <= len(tokens):
                key = tuple(tokens[i:i + length])
                if key in CLOSED_COMPOUNDS:
                    out.append(CLOSED_COMPOUNDS[key])
                    i += length
                    matched = True
                    break
        if not matched:
            out.append(tokens[i])
            i += 1
    return ' '.join(out)


def _normalize_compounds(text: str) -> str:
    """Iterate compound collapse until convergence (handles 'chest x ray'
    → 'chest xray' → 'chestxray' via two passes through the dict)."""
    for _ in range(4):
        new = _normalize_compounds_once(text)
        if new == text:
            return text
        text = new
    return text


def _normalize_eponyms(text: str) -> str:
    """Map Crohn's, Crohns, Crohn → crohns (canonical bare-plural)."""
    out = []
    for token in text.split():
        # Strip apostrophe to handle "crohn's" remnants if any
        bare = token.replace("'", '')
        # If token is in EPONYMS (with 's' suffix optional), canonicalize to with-s form
        if bare in EPONYMS:
            out.append(bare + 's')
        elif bare.rstrip('s') in EPONYMS and bare.endswith('s'):
            out.append(bare)
        else:
            out.append(token)
    return ' '.join(out)


def _normalize_spaced_acronyms(text: str) -> str:
    """Join known spaced acronyms ('c t' → 'ct'). Iterates over each known
    acronym in length-descending order so longer matches win first."""
    global _ACRONYM_PATTERNS
    if _ACRONYM_PATTERNS is None:
        _ACRONYM_PATTERNS = _build_acronym_patterns()
    for pattern, replacement in _ACRONYM_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _strip_eponym_possessives(text: str) -> str:
    """Pre-Whisper: turn 'crohn's' → 'crohns' so Whisper does not expand the
    possessive 's into the verb 'is'. Case-insensitive lookup; output preserves
    the rest of the sentence's case for Whisper to handle."""
    return _EPONYM_POSSESSIVE_RE.sub(lambda m: m.group(1) + 's', text)


def _normalize_honorifics(text: str) -> str:
    """Expand abbreviated honorifics to spelled-out form."""
    out = []
    for token in text.split():
        if token in HONORIFIC_MAP:
            out.append(HONORIFIC_MAP[token])
        else:
            out.append(token)
    return ' '.join(out)


def _normalize_dosing(text: str) -> str:
    """Collapse spelled-out dosing units to abbreviations."""
    out = []
    for token in text.split():
        if token in DOSING_MAP:
            out.append(DOSING_MAP[token])
        else:
            out.append(token)
    return ' '.join(out)


def _suppress_backchannels(text: str) -> str:
    """Remove non-content backchannel tokens (um, uh, okay, yeah, etc.).
    Applied symmetrically to ref and hyp so verbatim-style transcripts
    don't pay an artificial WER penalty for content that wouldn't appear
    in clinical documentation."""
    return ' '.join(t for t in text.split() if t not in BACKCHANNELS)


@lru_cache(maxsize=8192)
def normalize_clinical(text: str) -> str:
    """Apply Whisper normalizer + all clinical extensions.

    The order matters:
      1. Whisper normalizer (case fold, punct, contractions, numbers-to-words)
      2. Hyphens → spaces (after Whisper, which preserves hyphens)
      3. UK → US spelling (token-level lookup + suffix rules)
      4. Honorifics, dosing, eponyms, spaced acronyms
      5. Compound collapse (after all token-level work)
      6. Whitespace collapse
    """
    if not text:
        return ''
    text = _strip_eponym_possessives(text)
    text = _WHISPER(text)
    text = _normalize_hyphens(text)
    text = _normalize_uk_us(text)
    text = _normalize_honorifics(text)
    text = _normalize_dosing(text)
    text = _normalize_eponyms(text)
    text = _normalize_spaced_acronyms(text)
    text = _normalize_compounds(text)
    text = _suppress_backchannels(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Self-test ───────────────────────────────────────────────────────────────
def _selftest() -> None:
    """Round-trip normalization tests. Run with `python extended_normalizer.py`."""
    cases = [
        # UK/US spelling
        ("She has diarrhoea", "she has diarrhea"),
        ("Behaviour was concerning", "behavior was concerning"),
        ("Tumour size", "tumor size"),
        ("Anaemia and leukaemia", "anemia and leukemia"),
        ("She was hospitalised", "she was hospitalized"),
        # Hyphenation
        ("chest x-ray", "chestxray"),
        ("she's light-headed", "lightheaded"),
        # Compound flexibility
        ("health care", "healthcare"),
        ("everyday symptoms", "everyday symptoms"),  # already closed
        # Eponyms
        ("Crohn's disease", "crohns disease"),
        ("Alzheimer's", "alzheimers"),
        # Spaced acronyms
        ("got a c t scan", "got a ct scan"),
        ("had an m r i", "had an mri"),
        # Honorifics
        ("Dr. Smith", "doctor smith"),
        ("Mrs. Jones", "missus jones"),
        # Dosing
        ("5 milligrams", "5 mg"),
        ("two hundred milligrams", "200 mg"),
        # Backchannel suppression
        ("Yeah, okay, the lung sounds are clear", "lung sounds are clear"),
        ("Um, so the patient has a cough", "so the patient has a cough"),
        ("Mm-hmm, right, that's the diagnosis", "right that is the diagnosis"),
        # Idempotence: applying twice is same as once
    ]
    passed = 0
    failed = 0
    for raw, expected_substr in cases:
        out = normalize_clinical(raw)
        if expected_substr in out:
            passed += 1
        else:
            failed += 1
            print(f'FAIL: {raw!r} → {out!r}, expected substring {expected_substr!r}')
    # Idempotence
    sample = "Mrs. Jones has Crohn's, takes 5 milligrams of Lisinopril, history of leukaemia."
    once = normalize_clinical(sample)
    twice = normalize_clinical(once)
    if once != twice:
        failed += 1
        print(f'FAIL idempotence: {once!r} != {twice!r}')
    else:
        passed += 1
    # Symmetric across UK/US
    uk = "She has diarrhoea, history of leukaemia and oedema, takes 5 milligrams of paracetamol."
    us = "She has diarrhea, history of leukemia and edema, takes 5 mg of paracetamol."
    if normalize_clinical(uk) != normalize_clinical(us):
        failed += 1
        print(f'FAIL symmetric:\n  uk={normalize_clinical(uk)!r}\n  us={normalize_clinical(us)!r}')
    else:
        passed += 1
    print(f'\n{passed} passed, {failed} failed', flush=True)
    if failed:
        raise SystemExit(1)
    print('PASS', flush=True)


if __name__ == '__main__':
    _selftest()
