#!/usr/bin/env python3
"""Audit whether the public stapes repo is self-contained: find any local
module import in stapes .py files that does not resolve to a module present in
the repo (i.e., a hidden dependency on private ossicles-only code).
"""

import re
from pathlib import Path

STAPES = Path('/home/grey/dev/graiai/stapes')
OUT = STAPES / 'results' / '_audit' / 'selfcontain_audit.txt'
OUT.parent.mkdir(parents=True, exist_ok=True)

# Third-party packages used in the project (treated as available via pip).
THIRD_PARTY = {
    'jiwer', 'numpy', 'np', 'sherpa_onnx', 'dotenv', 'whisper_normalizer',
    'requests', 'scipy', 'pandas', 'matplotlib', 'quickumls', 'boto3',
    'google', 'azure', 'assemblyai', 'tqdm', 'pypandoc', 'docx', 'onnx',
    'medspacy', 'spacy', 'rapidfuzz', 'soundfile', 'librosa', 'torch',
    'sklearn', 'jellyfish',
}
STDLIB = {
    'argparse', 'hashlib', 'json', 'logging', 'subprocess', 'time', 'wave',
    'datetime', 'pathlib', 're', 'os', 'sys', 'csv', 'math', 'random',
    'collections', 'itertools', 'functools', 'dataclasses', 'glob', 'shutil',
    'warnings', 'concurrent', 'typing', 'io', 'string', 'statistics',
    'tempfile', 'textwrap', 'unicodedata', 'difflib', 'enum', 'abc',
    'urllib', 'pickle', 'multiprocessing',
}

py_files = sorted(STAPES.glob('scripts/*.py')) + sorted(STAPES.glob('fusion_depth/**/*.py'))
local_modules = {p.stem for p in py_files}

imp_re = re.compile(r'^\s*(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)')

fh = open(OUT, 'w')
fh.write('file\timported_module\tstatus\n')
fh.flush()

unresolved = []
for p in py_files:
    for line in p.read_text().splitlines():
        m = imp_re.match(line)
        if not m:
            continue
        mod = m.group(1)
        if mod in STDLIB or mod in THIRD_PARTY:
            continue
        if mod in local_modules:
            status = 'local-ok'
        else:
            status = 'UNRESOLVED'
            unresolved.append((p.name, mod))
        # only log non-stdlib/third-party (the interesting ones)
        line_out = f'{p.relative_to(STAPES)}\t{mod}\t{status}'
        fh.write(line_out + '\n')
        fh.flush()

fh.write('\n=== SUMMARY ===\n')
fh.write(f'{len(py_files)} python files audited\n')
if unresolved:
    fh.write(f'UNRESOLVED local imports ({len(unresolved)}):\n')
    for fname, mod in unresolved:
        fh.write(f'  {fname} -> {mod}\n')
else:
    fh.write('All local imports resolve within stapes. Self-contained.\n')
fh.flush()
fh.close()

print(f'{len(py_files)} files audited; '
      f'{len(unresolved)} unresolved local imports', flush=True)
for fname, mod in unresolved:
    print(f'  UNRESOLVED: {fname} -> {mod}', flush=True)
