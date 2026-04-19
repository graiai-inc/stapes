#!/usr/bin/env python3
"""Assemble the stapes npj Digital Medicine submission package.

Reads the individual paper components and produces:
    build/stapes_submission.md   — assembled manuscript markdown
    build/stapes_manuscript.docx — pandoc-converted Word file
    build/stapes_supplementary.md — supplementary tables markdown
    build/stapes_supplementary.docx — pandoc-converted Word file
    build/stapes_cover_letter.docx — cover letter Word file

Run with the lens venv python (it has pypandoc-binary installed):
    /home/grey/dev/graiai/lens/venv/bin/python scripts/assemble_submission.py
"""

import csv
import subprocess
import sys
from pathlib import Path

import pypandoc

PAPER = Path('/home/grey/dev/graiai/stapes/paper')
BUILD = Path('/home/grey/dev/graiai/stapes/build')
BUILD.mkdir(parents=True, exist_ok=True)


def csv_to_md_table(csv_path: Path) -> str:
    """Render a CSV file as a GitHub-flavored markdown table."""
    with csv_path.open() as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return ''
    header = rows[0]
    out = ['| ' + ' | '.join(header) + ' |']
    out.append('|' + '|'.join(['---'] * len(header)) + '|')
    for row in rows[1:]:
        out.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(out)


TITLE_PAGE = """# Open Benchmark of On-Device and Cloud ASR for Clinical Conversations

**Jason Grey Faulkenberry, MD, MPH**<sup>1</sup>

<sup>1</sup> Department of Hematology and Medical Oncology, Emory University School of Medicine, Atlanta, Georgia, USA

**Corresponding author:** Jason Grey Faulkenberry, MD, MPH
Department of Hematology and Medical Oncology
Emory University School of Medicine
1929 Morrison Dr., Decatur, GA 30033, USA
Email: grey.faulkenberry@emory.edu
Phone: +1 786 571 6384

---

"""


FIGURES_BLOCK = """# Figures

![](figure1.png){width="6in"}

**Figure 1. Word error rate and clinical term recall of the best on-device ASR model versus the best cloud API, per dataset.**
**a**, Word error rate (%) of the best on-device model and the best cloud API on each of the three clinical conversation datasets (OSCE respiratory interviews, n = 272; PriMock57 primary care, n = 57; Kazi et al. psychiatric, n = 71). Lower is better. **b**, Clinical term recall (%) of the best on-device model and the best cloud API on each dataset. Higher is better. Clinical term recall was computed as the proportion of UMLS medical concept spans in the reference transcript that were correctly transcribed.

![](figure2.png){width="6in"}

**Figure 2. ROVER hypothesis fusion yields small improvements over the best single on-device model, with the best cloud API as reference.**
For each dataset, the figure shows the word error rate of the best single on-device model (left point), the best two-model ROVER fusion pair (right point, connected by a grey line to display the fusion delta), and the best cloud API as a dashed horizontal reference. Fusion provided ≤ 0.82 percentage point improvements on all three datasets. On the OSCE respiratory interview dataset, the best fused on-device pair (parakeet-tdt-0.6b-v2 + sensevoice, 11.01%) did not surpass the best cloud API (Azure, 7.70%).
"""


def read_file(path: Path) -> str:
    return path.read_text()


def strip_first_heading(text: str) -> str:
    """Drop the first top-level '## Abstract'-style heading from a file fragment."""
    lines = text.splitlines()
    out = []
    dropped = False
    for line in lines:
        if not dropped and line.startswith('## '):
            dropped = True
            continue
        out.append(line)
    return '\n'.join(out)


def build_main_manuscript() -> str:
    print('[main] assembling main manuscript', flush=True)

    # Start with the full_manuscript.md and splice in title page + tables + refs + legends.
    manuscript = read_file(PAPER / 'full_manuscript.md')

    # Prepend title page.
    out = TITLE_PAGE + manuscript.lstrip()

    # Convert CSV tables to markdown and append them as Tables 1/2/3.
    tbl1 = csv_to_md_table(PAPER / 'table1_wer.csv')
    tbl2 = csv_to_md_table(PAPER / 'table2_ctr.csv')
    tbl3 = csv_to_md_table(PAPER / 'table3_cost.csv')

    tables_block = f"""
# Tables

**Table 1. Word error rate (%) by model and dataset.** Asterisked OSCE column reflects the apostrophe-injected reference transcripts (see Methods).

{tbl1}

**Table 2. Clinical term recall (%) by model and dataset.**

{tbl2}

**Table 3. Cloud API cost for the full benchmark (~80 hours, 400 conversations).** AWS Transcribe Medical was run on representative subsets (50 OSCE, 17 PriMock57, 21 psychiatric files) due to cost; on-device inference incurs no per-encounter cost.

{tbl3}
"""

    # Build references section from references.md (numbered list at end of file).
    refs_text = read_file(PAPER / 'references.md')
    # Take only the "Formatted Reference List" block.
    marker = '## Formatted Reference List'
    if marker in refs_text:
        refs_block = refs_text.split(marker, 1)[1]
    else:
        refs_block = refs_text
    # Drop the "(NEJM AI style)" subtitle if present.
    refs_block = refs_block.replace(' (NEJM AI style)', '').strip()
    # Demote the section heading.
    refs_section = '\n# References\n\n' + refs_block + '\n'

    out = out + tables_block + refs_section + '\n' + FIGURES_BLOCK

    return out


def build_supplementary() -> str:
    print('[supp] assembling supplementary tables', flush=True)
    supp_md = read_file(PAPER / 'supplementary_tables.md')

    # S1 table is already inline as markdown in supplementary_tables.md. Append S2 and S3 as full tables.
    s2 = csv_to_md_table(PAPER / 'supplementary_table_S2_rover_pairs.csv')
    s3 = csv_to_md_table(PAPER / 'supplementary_table_S3_rover_triples.csv')

    appendix = f"""

## Table S2 data

{s2}

## Table S3 data

{s3}
"""
    return supp_md + appendix


def build_cover_letter() -> str:
    print('[cover] assembling cover letter', flush=True)
    return read_file(PAPER / 'cover_letter.md')


def write_and_convert(md_text: str, stem: str, to_docx: bool = True) -> None:
    md_path = BUILD / f'{stem}.md'
    md_path.write_text(md_text)
    print(f'[write] {md_path} ({len(md_text)} chars)', flush=True)

    if to_docx:
        docx_path = BUILD / f'{stem}.docx'
        # Pandoc converts markdown to docx. Include resource-path so image
        # references resolve against the paper/ directory.
        extra = [
            '--resource-path', str(PAPER),
            '--standalone',
        ]
        try:
            pypandoc.convert_file(
                str(md_path),
                'docx',
                outputfile=str(docx_path),
                extra_args=extra,
            )
            print(f'[docx] {docx_path}', flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f'[error] pandoc conversion failed for {stem}: {exc}', flush=True)
            raise


def main() -> None:
    main_md = build_main_manuscript()
    write_and_convert(main_md, 'stapes_manuscript')

    supp_md = build_supplementary()
    write_and_convert(supp_md, 'stapes_supplementary')

    cover_md = build_cover_letter()
    write_and_convert(cover_md, 'stapes_cover_letter')

    print('[done] submission package assembled in', BUILD, flush=True)


if __name__ == '__main__':
    sys.exit(main())
