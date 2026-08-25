#!/usr/bin/env python3
"""Verify every URL and DOI cited in the stapes manuscript still resolves.

Written for the BMC MIDM technical check (2026-08-25), which reported two
Zenodo DOI links as broken when both in fact resolve; Zenodo bot-protects its
landing pages and intermittently returns 403 to automated link checkers.

Extracts URLs from the source manuscript, the reference list, and the
assembled build (if present), probes each by GET and by HEAD with both a
default and a browser user-agent, and for every DOI also queries DataCite,
which is not bot-protected and is therefore the citable proof of registration.

Writes build/link_check.tsv, one row per probe, flushed every iteration.
"""

import json
import re
import subprocess
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path('/home/grey/dev/graiai/stapes')
BUILD = ROOT / 'build'
PAPER = ROOT / 'paper'
OUT = BUILD / 'link_check.tsv'

SOURCES = [
    PAPER / 'full_manuscript.md',
    PAPER / 'references.md',
    BUILD / 'stapes_manuscript.md',
]
URL_RE = re.compile(r'https?://[^\s\)\]<>"]+')
TRAILING = '.,;:'
BROWSER_UA = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36')
PROBES: List[Tuple[str, List[str]]] = [
    ('GET/default-UA', ['-sL']),
    ('HEAD/default-UA', ['-sIL']),
    ('GET/browser-UA', ['-sL', '-A', BROWSER_UA]),
]
TIMEOUT_S = '45'


def collect_urls() -> Dict[str, List[str]]:
    """Map each cited URL to the source files it appears in."""
    found: Dict[str, List[str]] = {}
    for path in SOURCES:
        if not path.exists():
            continue
        for raw in URL_RE.findall(path.read_text()):
            url = raw.rstrip(TRAILING)
            found.setdefault(url, []).append(path.name)
    return found


def curl(url: str, flags: List[str]) -> Tuple[str, str]:
    """Return (http_code, final_url) for one curl probe."""
    cmd = ['curl', *flags, '-o', '/dev/null', '-m', TIMEOUT_S,
           '-w', '%{http_code}\t%{url_effective}', url]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return (f'curl-error-{res.returncode}', '')
    code, _, final = res.stdout.partition('\t')
    return (code, final)


def datacite_state(doi: str) -> str:
    """Return 'state|registered_date' from DataCite, or an error marker."""
    api = f'https://api.datacite.org/dois/{urllib.parse.quote(doi)}'
    res = subprocess.run(['curl', '-s', '--compressed', '-m', TIMEOUT_S, api],
                         capture_output=True, text=True)
    try:
        attrs = json.loads(res.stdout)['data']['attributes']
    except (ValueError, KeyError):
        return 'datacite-lookup-failed'
    return f"{attrs.get('state')}|{attrs.get('registered')}"


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    fh = open(OUT, 'w')
    fh.write('utc_timestamp\turl\tsources\tprobe\thttp_code\tfinal_url\n')
    fh.flush()
    for url, sources in sorted(collect_urls().items()):
        src = ','.join(sorted(set(sources)))
        for name, flags in PROBES:
            code, final = curl(url, flags)
            stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            fh.write(f'{stamp}\t{url}\t{src}\t{name}\t{code}\t{final}\n')
            fh.flush()
            print(f'{stamp}  {code:>3}  {name:<16} {url}', flush=True)
        if url.startswith('https://doi.org/'):
            doi = url[len('https://doi.org/'):]
            state = datacite_state(doi)
            stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            fh.write(f'{stamp}\t{url}\t{src}\tdatacite\t{state}\t\n')
            fh.flush()
            print(f'{stamp}       datacite         {doi} -> {state}', flush=True)
    fh.close()
    print(f'wrote {OUT}', flush=True)


if __name__ == '__main__':
    main()
