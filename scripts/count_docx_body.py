#!/usr/bin/env python3
"""Word-equivalent body count of the rendered submission docx.

Counts words in all paragraphs between the 'Background and Significance'
heading and the 'Data Availability' heading (the JAMIA-countable main text),
matching what an editor's Word count would see. Writes result to disk.
"""

from pathlib import Path

from docx import Document

BUILD = Path('/home/grey/dev/graiai/stapes/build')
OUT = BUILD / 'qa_report.txt'

doc = Document(str(BUILD / 'stapes_manuscript.docx'))

in_body = False
words = 0
paras = 0
for p in doc.paragraphs:
    text = p.text.strip()
    if text == 'Background and Significance':
        in_body = True
    if text == 'Data Availability':
        break
    if in_body:
        words += len(p.text.split())
        paras += 1

with open(OUT, 'a') as fh:
    line = f'[docx-body] Word-equivalent body count (Background..Conclusion): {words} ({paras} paragraphs)'
    fh.write(line + '\n')
    fh.flush()
    print(line, flush=True)
