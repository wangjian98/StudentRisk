#!/usr/bin/env python3
"""Markdown → docx（提案交付件）：支持标题/表格/代码块/列表/粗体。"""
import re
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt


def add_table(doc, rows):
    header, body = rows[0], rows[1:]
    t = doc.add_table(rows=1, cols=len(header))
    t.style = 'Light Grid Accent 1'
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = str(h)
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
    for row in body:
        cells = t.add_row().cells
        for i, v in enumerate(row[:len(header)]):
            cells[i].text = str(v)
    doc.add_paragraph()


def parse_table(lines):
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip('|').split('|')]
        if all(re.fullmatch(r':?-{2,}:?', c) for c in cells):
            continue
        rows.append(cells)
    return rows


def main(md_path, out_path):
    md = open(md_path, encoding='utf-8').read().split('\n')
    doc = Document()
    st = doc.styles['Normal']
    st.font.name = 'Microsoft YaHei'
    st.font.size = Pt(10.5)

    i = 0
    in_code = False
    code_buf = []
    while i < len(md):
        ln = md[i]
        if ln.strip().startswith('```'):
            if in_code:
                p = doc.add_paragraph()
                run = p.add_run('\n'.join(code_buf))
                run.font.name = 'Consolas'
                run.font.size = Pt(9)
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(ln)
            i += 1
            continue
        if ln.strip().startswith('|') and i + 1 < len(md) and \
                md[i + 1].strip().startswith('|'):
            buf = []
            while i < len(md) and md[i].strip().startswith('|'):
                buf.append(md[i])
                i += 1
            add_table(doc, parse_table(buf))
            continue
        m = re.match(r'^(#{1,6})\s+(.*)$', ln)
        if m:
            lvl, txt = len(m.group(1)), m.group(2).strip()
            txt = re.sub(r'\*\*(.*?)\*\*', r'\1', txt)
            doc.add_heading(txt, level=min(lvl, 4))
            i += 1
            continue
        if re.match(r'^\s*[-*]\s+', ln):
            txt = re.sub(r'^\s*[-*]\s+', '', ln)
            doc.add_paragraph(re.sub(r'\*\*(.*?)\*\*', r'\1', txt), style='List Bullet')
            i += 1
            continue
        if re.match(r'^\s*\d+\.\s+', ln):
            txt = re.sub(r'^\s*\d+\.\s+', '', ln)
            doc.add_paragraph(re.sub(r'\*\*(.*?)\*\*', r'\1', txt), style='List Number')
            i += 1
            continue
        if ln.strip() == '---':
            doc.add_paragraph('―' * 30)
            i += 1
            continue
        if not ln.strip():
            i += 1
            continue
        p = doc.add_paragraph()
        parts = re.split(r'(\*\*.*?\*\*)', ln)
        for part in parts:
            if not part:
                continue
            if part.startswith('**') and part.endswith('**'):
                p.add_run(part[2:-2]).bold = True
            else:
                p.add_run(part)
        i += 1

    doc.save(out_path)
    print('written', out_path)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
