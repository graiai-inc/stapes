#!/usr/bin/env python3
"""Fetch authoritative reference metadata from CrossRef for BMC Vancouver restyling.

Parses the numbered reference list in paper/references.md, extracts each DOI, and
queries the CrossRef REST API for the canonical author list, title, journal
abbreviation, year, volume, issue, and page range. Results are written to disk
incrementally, one JSON object per line, so a network failure mid-run leaves
every reference fetched so far intact.

References without a DOI are recorded with `status: no-doi` and must be styled
by hand against their original source.
"""

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
REFS_MD = REPO / 'paper' / 'references.md'
OUT_PATH = REPO / 'build' / 'crossref_metadata.jsonl'

MAILTO = 'grey.faulkenberry@emory.edu'
REF_RE = re.compile(r'^(\d+)\.\s+(.*)$')
DOI_RE = re.compile(r'DOI:\s*(10\.\S+?)\.?$', re.IGNORECASE)


def parse_references(text: str) -> list[tuple[int, str]]:
    """Extract (number, raw text) pairs from the Formatted Reference List block."""
    marker = '## Formatted Reference List'
    block = text.split(marker, 1)[1] if marker in text else text
    refs = []
    for line in block.splitlines():
        m = REF_RE.match(line.strip())
        if m:
            refs.append((int(m.group(1)), m.group(2).strip()))
    return refs


def fetch_crossref(doi: str) -> dict:
    """Query the CrossRef REST API for one DOI."""
    url = f'https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={MAILTO}'
    req = urllib.request.Request(url, headers={'User-Agent': f'stapes-refcheck (mailto:{MAILTO})'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))['message']


def summarize(msg: dict) -> dict:
    """Reduce a CrossRef work record to the fields a Vancouver reference needs."""
    authors = []
    for a in msg.get('author', []):
        family = a.get('family', '')
        given = a.get('given', '')
        initials = ''.join(p[0] for p in re.split(r'[\s\-]+', given) if p)
        authors.append(f'{family} {initials}'.strip())
    issued = msg.get('issued', {}).get('date-parts', [[None]])[0]
    return {
        'authors': authors,
        'n_authors': len(authors),
        'title': (msg.get('title') or [''])[0],
        'container': (msg.get('container-title') or [''])[0],
        'short_container': (msg.get('short-container-title') or [''])[0],
        'year': issued[0] if issued else None,
        'volume': msg.get('volume'),
        'issue': msg.get('issue'),
        'page': msg.get('page'),
        'article_number': msg.get('article-number'),
        'type': msg.get('type'),
        'publisher': msg.get('publisher'),
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    refs = parse_references(REFS_MD.read_text(encoding='utf-8'))
    print(f'[refs] parsed {len(refs)} references from {REFS_MD}', flush=True)

    fh = open(OUT_PATH, 'w', encoding='utf-8')
    for num, raw in refs:
        m = DOI_RE.search(raw)
        record = {'n': num, 'raw': raw}
        if not m:
            record['status'] = 'no-doi'
            print(f'[{num:2d}] NO DOI -- style by hand: {raw[:70]}', flush=True)
        else:
            doi = m.group(1).rstrip('.')
            record['doi'] = doi
            try:
                record['crossref'] = summarize(fetch_crossref(doi))
                record['status'] = 'ok'
                cr = record['crossref']
                print(f'[{num:2d}] ok  {cr["n_authors"]} authors | {cr["short_container"] or cr["container"]} '
                      f'{cr["year"]};{cr["volume"] or ""}({cr["issue"] or ""}):{cr["page"] or cr["article_number"] or ""}',
                      flush=True)
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError) as exc:
                record['status'] = f'error: {exc}'
                print(f'[{num:2d}] ERROR {doi}: {exc}', flush=True)
        fh.write(json.dumps(record, ensure_ascii=False) + '\n')
        fh.flush()
        time.sleep(0.4)
    fh.close()
    print(f'[done] wrote {OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
