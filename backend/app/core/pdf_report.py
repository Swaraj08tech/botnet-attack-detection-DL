"""PDF security reports: two static, self-contained snapshots of ONE report record (app.core.report_record).

  render_summary_pdf   Summary report (about 2-4 pages) for students and general users: six short sections in plain language.
  render_record_pdf    Detailed report for analysts and reviewers: six main sections, then appendices with the full audit material.

Both read every value, citation id, statement and status from the record, and the Summary's sentences come from app.core.report_views,
which the web view uses too, so the two PDFs and the web view cannot disagree. Fonts are embedded (Vera TTF shipped with reportlab);
headings become PDF bookmarks; citations link to the source register; source URLs are clickable. All text from files, sources and the
LLM is escaped.
"""
from __future__ import annotations

import io
from pathlib import Path

import reportlab
from reportlab.graphics.shapes import STATE_DEFAULTS, Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import CondPageBreak, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.core.report_views import prob_pct, views

FONT_DIR = Path(reportlab.__file__).parent / 'fonts'
for name, file in (('Body', 'Vera.ttf'), ('Body-Bold', 'VeraBd.ttf'), ('Body-Italic', 'VeraIt.ttf'), ('Body-BoldItalic', 'VeraBI.ttf')):
    if name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='Body-Bold', italic='Body-Italic', boldItalic='Body-BoldItalic')
STATE_DEFAULTS['fontName'] = 'Body'                                   # charts start in the embedded font, not Times-Roman

NAVY, BRAND, MUTED, INK = colors.HexColor('#0f172a'), colors.HexColor('#2f58d6'), colors.HexColor('#64748b'), colors.HexColor('#1e293b')
SAFE, THREAT, WARN = colors.HexColor('#10b981'), colors.HexColor('#e11d48'), colors.HexColor('#d97706')
SUPPORT, OPPOSE = colors.HexColor('#c2410c'), colors.HexColor('#2563eb')
LIGHT, BORDER, AMBER_BG = colors.HexColor('#f1f5f9'), colors.HexColor('#e2e8f0'), colors.HexColor('#fffbeb')
PRIORITY = {'CRITICAL': colors.HexColor('#9f1239'), 'HIGH': THREAT, 'MODERATE': WARN, 'LOW': SAFE}
STATUS = {'available': SAFE, 'generated': SAFE, 'partial': WARN, 'pending': MUTED, 'not run': MUTED, 'failed': THREAT, 'unavailable': THREAT}
ORIGIN_LABEL = {'observed': 'observed in file', 'batch_statistic': 'batch statistic', 'model_output': 'model output', 'xai_attribution': 'XAI (TreeSHAP)',
                'retrieved_guidance': 'retrieved source', 'llm_synthesis': 'LLM, passed checks', 'analyst_proposed': 'proposed next step',
                'heuristic': 'heuristic'}
PAGE_W, PAGE_H = A4
MARGIN = 17 * mm
W = PAGE_W - 2 * MARGIN

base = getSampleStyleSheet()


def styles(size: float):
    """Paragraph styles at a base size: 8.8 pt for the Detailed report, 10 pt for the Summary."""
    k = size / 8.8
    body = ParagraphStyle(f'Body{size}', parent=base['BodyText'], fontName='Body', fontSize=size, leading=size * 1.42, textColor=INK, spaceAfter=2)
    return {
        'H1': ParagraphStyle('H1', parent=base['Heading1'], fontName='Body-Bold', fontSize=14 * k, leading=18 * k, textColor=NAVY, spaceBefore=9, spaceAfter=5),
        'H2': ParagraphStyle('H2', parent=base['Heading2'], fontName='Body-Bold', fontSize=11 * k, leading=14 * k, textColor=NAVY, spaceBefore=8, spaceAfter=3),
        'H3': ParagraphStyle(f'H3{size}', parent=base['Heading3'], fontName='Body-Bold', fontSize=9.5 * k, leading=12 * k, textColor=NAVY, spaceBefore=5,
                             spaceAfter=2),
        'BODY': body,
        'SMALL': ParagraphStyle(f'Small{size}', parent=body, fontSize=size * 0.86, leading=size * 1.18, textColor=MUTED),
        'CELL': ParagraphStyle(f'Cell{size}', parent=body, fontSize=size * 0.89, leading=size * 1.14, spaceAfter=0),
        'BULLET': ParagraphStyle(f'Bullet{size}', parent=body, leftIndent=11, bulletIndent=1, spaceAfter=4, bulletFontName='Body'),
    }


D = styles(8.8)                                                      # Detailed
S = styles(10.0)                                                     # Summary
GLYPHS = set(TTFont('Glyphs', str(FONT_DIR / 'Vera.ttf')).face.charToGlyph)
SUBSTITUTE = {'→': '->', '←': '<-', '≥': '>=', '≤': '<=', '≠': '!=', '−': '-', '‑': '-', '✓': 'yes', '✗': 'no', ' ': ' '}


def esc(text) -> str:
    """Escape untrusted text for reportlab markup and replace characters the embedded font cannot draw (no missing-glyph boxes)."""
    text = ''.join(ch if ord(ch) < 128 or ord(ch) in GLYPHS else SUBSTITUTE.get(ch, '?') for ch in str(text))
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def anchor(cid: str) -> str:
    return 'c_' + ''.join(ch if ch.isalnum() else '_' for ch in cid)


def cite_links(ids) -> str:
    return ' '.join(f'<a href="#{anchor(i)}" color="#2f58d6">[{esc(i)}]</a>' for i in ids)


def origin_tag(origin: str) -> str:
    return f"<font color='#64748b' size='6.8'>({esc(ORIGIN_LABEL.get(origin, origin))})</font>"


def pct(x, digits: int = 1) -> str:
    return '-' if x is None else f'{x * 100:.{digits}f}%'


def num(x) -> str:
    return '-' if x is None else f'{int(x):,}'


def table(rows, widths, st=D, header=True, align_right_from=None, zebra=True, keep_rows=True):
    size = st['CELL'].fontSize
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0, splitByRow=1)
    style = [('FONT', (0, 0), (-1, -1), 'Body', size), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, -1), 0.25, BORDER),
             ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3), ('LEFTPADDING', (0, 0), (-1, -1), 4)]
    if header:
        style += [('BACKGROUND', (0, 0), (-1, 0), NAVY), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('FONT', (0, 0), (-1, 0), 'Body-Bold', size)]
    if zebra:
        style += [('ROWBACKGROUNDS', (0, 1 if header else 0), (-1, -1), [colors.white, LIGHT])]
    if align_right_from is not None:
        style += [('ALIGN', (align_right_from, 0), (-1, -1), 'RIGHT')]
    t.setStyle(TableStyle(style))
    return t


def hdr(st, *cells):
    return [Paragraph(f'<font color="white"><b>{esc(c)}</b></font>', st['CELL']) for c in cells]


def box(flowables, width=W, border=BORDER, background=None, pad=6):
    t = Table([[flowables]], colWidths=[width])
    style = [('BOX', (0, 0), (-1, -1), 0.6, border), ('LEFTPADDING', (0, 0), (-1, -1), pad), ('RIGHTPADDING', (0, 0), (-1, -1), pad),
             ('TOPPADDING', (0, 0), (-1, -1), pad - 1), ('BOTTOMPADDING', (0, 0), (-1, -1), pad - 1), ('FONT', (0, 0), (-1, -1), 'Body', 8)]
    if background is not None:
        style.append(('BACKGROUND', (0, 0), (-1, -1), background))
    t.setStyle(TableStyle(style))
    return t


def status_cell(st, status: str) -> Paragraph:
    color = STATUS.get(status, MUTED).hexval()[2:]
    return Paragraph(f'<font color="#{color}"><b>{esc(status)}</b></font>', st['CELL'])


def link(url, text) -> str:
    return f'<a href="{esc(url)}" color="#2f58d6">{esc(text)}</a>' if url else esc(text)


# ------------------------------------------------------------------------------------------------------------ charts
def xai_chart(xai: dict) -> Drawing:
    """Diverging bars of the top contributions plus one bar for the omitted remainder (labelled). No base value is invented."""
    items = [(f"{x['name']} = {x['value']}", x['contribution']) for x in xai['top']]
    if xai.get('other_features'):
        items.append((f"other {xai['other_features']} features (sum)", xai['remainder_of_other_features']))
    row_h, label_w = 15, 66 * mm
    height = 18 + row_h * len(items)
    d = Drawing(W, height)
    span = max(abs(v) for _, v in items) or 1.0
    zero_x, half = label_w + (W - label_w - 40) / 2, (W - label_w - 50) / 2
    d.add(Line(zero_x, 4, zero_x, height - 12, strokeColor=MUTED, strokeWidth=0.5))
    d.add(String(zero_x - 4, height - 10, 'points away', fontName='Body', fontSize=6.5, fillColor=OPPOSE, textAnchor='end'))
    d.add(String(zero_x + 4, height - 10, f"supports {xai.get('class', '')}", fontName='Body', fontSize=6.5, fillColor=SUPPORT))
    for k, (label, v) in enumerate(items):
        y = height - 24 - k * row_h
        length = abs(v) / span * half
        other = label.startswith('other ')
        color = colors.HexColor('#94a3b8') if other else (SUPPORT if v > 0 else OPPOSE)
        d.add(Rect(zero_x if v > 0 else zero_x - length, y, length, 9, fillColor=color, strokeColor=None))
        d.add(String(4, y + 2, label[:52], fontName='Body', fontSize=7, fillColor=INK))
        d.add(String(zero_x + (length + 3 if v > 0 else -length - 3), y + 2, f'{v:+.3f}', fontName='Body', fontSize=6.5,
                     fillColor=MUTED, textAnchor='start' if v > 0 else 'end'))
    return d


def class_bars(rows: list[dict], font=7.2, row_h=13) -> Drawing:
    height = 14 + row_h * len(rows)
    d = Drawing(W, height)
    peak = max((r['count'] for r in rows), default=1) or 1
    for k, r in enumerate(rows):
        y = height - 14 - k * row_h
        length = r['count'] / peak * (W - 70 * mm)
        color = SAFE if r['class'] == 'Normal' else colors.HexColor({'Critical': '#9f1239', 'High': '#e11d48', 'Medium': '#d97706',
                                                                      'Low': '#3b82f6'}.get(r['severity_heuristic'], '#64748b'))
        d.add(String(4, y + 2, f"{r['class']}{' (predicted Normal)' if r['class'] == 'Normal' else ''}", fontName='Body', fontSize=font))
        d.add(Rect(48 * mm, y, length, row_h * 0.65, fillColor=color, strokeColor=None))
        d.add(String(48 * mm + length + 3, y + 1.5, f"{r['count']:,}", fontName='Body', fontSize=font - 0.4, fillColor=MUTED))
    return d


# ------------------------------------------------------------------------------------------------------------ document
class ReportDoc(SimpleDocTemplate):
    """Headings (H1 / H2) become PDF bookmarks (outline), so the reader can jump between parts."""

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ('H1', 'H2'):
            key = f'h{id(flowable)}'
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText()[:90], key, level=0 if flowable.style.name == 'H1' else 1)


def _decor(record, kind: str):
    prov = record['provenance']
    ident = f"analysis #{prov['analysis']['id']} · report #{record.get('report_id', '-')} · schema {record['schema_version']}"

    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - 11 * mm, PAGE_W, 11 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont('Body-Bold', 8.5)
        canvas.drawString(MARGIN, PAGE_H - 7 * mm, f'Network Security Analysis · {kind} report')
        canvas.setFont('Body', 7.5)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 7 * mm, f"{prov['analysis']['dataset_name'][:50]} · {ident}")
        canvas.setStrokeColor(BORDER)
        canvas.line(MARGIN, 11 * mm, PAGE_W - MARGIN, 11 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont('Body', 7)
        canvas.drawString(MARGIN, 7.5 * mm, f"Generated {record['generated_at'][:16].replace('T', ' ')} UTC · detector {prov['detector']['version']} · "
                                            'offline analysis of uploaded flows · predictions are not confirmed incidents')
        canvas.drawRightString(PAGE_W - MARGIN, 7.5 * mm, f'Page {doc.page}')
        canvas.restoreState()
    return draw


def _build(record, story, kind: str) -> tuple[bytes, int]:
    prov = record['provenance']
    buffer = io.BytesIO()
    doc = ReportDoc(buffer, initialFontName='Body', pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=16 * mm, bottomMargin=16 * mm,
                    title=f"{kind} security report - {prov['analysis']['dataset_name']}", author='Network Attack Detection (UNSW-NB15)',
                    subject=f"{kind} report: analysis {prov['analysis']['id']}, report {record.get('report_id')}, schema {record['schema_version']}")
    decor = _decor(record, kind)
    doc.build(story, onFirstPage=decor, onLaterPages=decor)
    return buffer.getvalue(), doc.page


def _title(record, st, kind: str, audience: str):
    prov = record['provenance']
    return [Paragraph(f'<font color="#2f58d6" size="8"><b>{kind.upper()} SECURITY REPORT</b></font> '
                      f'<font color="#64748b" size="7.5">· {esc(audience)}</font>', st['BODY']), Spacer(1, 1.5 * mm),
            Paragraph(f'<font size="17"><b>{esc(prov["analysis"]["dataset_name"])}</b></font>', ParagraphStyle('T', parent=st['BODY'], leading=22)),
            Paragraph(f"Analysis #{prov['analysis']['id']} · report #{record.get('report_id', '-')} · generated "
                      f"{esc(record['generated_at'][:16].replace('T', ' '))} UTC · detector {esc(prov['detector']['version'])} · "
                      f"report schema {record['schema_version']}", st['SMALL']), Spacer(1, 2 * mm)]


def _component_rows(record, st):
    rows = [hdr(st, 'Component', 'Status', 'Detail')]
    for name, label in (('detection', 'Detection'), ('xai', 'Explanations (TreeSHAP)'), ('group_xai', 'Group explanations'),
                        ('retrieval', 'Source retrieval (RAG)'), ('llm', 'Local LLM'), ('checks', 'Automated checks'), ('pdf', 'PDF')):
        comp = record['components'].get(name)
        if comp is None:
            continue
        detail = comp.get('detail', '') + (' — ' + '; '.join(comp.get('reasons', [])) if comp.get('reasons') else '')
        rows.append([Paragraph(label, st['CELL']), status_cell(st, 'available' if name == 'pdf' else comp.get('status', 'unknown')),
                     Paragraph(esc(detail), st['CELL'])])
    return rows


# ============================================================================================================ SUMMARY REPORT
def render_summary_pdf(record: dict) -> tuple[bytes, int]:
    """About 2-4 pages in plain language. Every sentence comes from the record or from report_views (shared with the web view)."""
    v = views(record)['summary']
    s, st = record['summary'], S
    story = _title(record, st, 'Summary', 'for students and general readers')
    story.append(box([Paragraph(f"<b>Scope.</b> {esc(s['scope'])} The results are predictions of a machine-learning model: they are not "
                                'confirmed incidents and they do not predict future attacks.', st['SMALL'])], background=LIGHT))

    # 1. overall flow summary
    story.append(Paragraph('1. Overall flow summary', st['H1']))
    tiles = [('Flows analysed', num(record['detections']['total'])), ('Predicted Normal', f"{num(s['predicted_normal'])} ({pct(s['predicted_normal_share'])})"),
             ('Predicted attack', f"{num(s['predicted_attack'])} ({pct(s['predicted_attack_share'])})"), ('Need review (confidence below 70%)', num(s['needs_review']))]
    kpi = Table([[Paragraph(f"<font color='#64748b' size='8'>{esc(k)}</font>", st['CELL']) for k, _ in tiles],
                 [Paragraph(f'<font size="13"><b>{esc(val)}</b></font>', ParagraphStyle('K', parent=st['CELL'], leading=17)) for _, val in tiles]],
                colWidths=[W / 4] * 4)
    kpi.setStyle(TableStyle([('FONT', (0, 0), (-1, -1), 'Body', 9), ('BOX', (0, 0), (-1, -1), 0.5, BORDER),
                             ('INNERGRID', (0, 0), (-1, -1), 0.25, BORDER), ('TOPPADDING', (0, 0), (-1, -1), 4)]))
    tp = s['triage_priority']
    story += [kpi, Spacer(1, 2.5 * mm),
              Paragraph(f"<b>Review priority: <font color='#{PRIORITY.get(tp['level'], MUTED).hexval()[2:]}'>{esc(tp['level'].capitalize())}</font></b>"
                        f" — a fixed rule of this application (rule: {esc(tp['rule'])}), used to decide how soon to review. It is not a measure "
                        'of damage and not a confirmed compromise.', st['BODY'])]
    sc = s.get('status_confidence')
    if sc:
        lv, tv = sc['levels'], record['detections']['confidence_levels']
        conf = [hdr(st, 'How sure is the model?', 'High (90% or more)', 'Medium (70-90%)', 'Low (below 70%)'),
                [Paragraph('<b>Attack or normal?</b>', st['CELL']), num(lv['high']), num(lv['medium']), num(lv['low'])],
                [Paragraph('<b>Exact attack type?</b>', st['CELL']), num(tv['high']), num(tv['medium']), num(tv['low'])]]
        story += [Spacer(1, 2 * mm), table(conf, [58 * mm, 38 * mm, 36 * mm, W - 132 * mm], st, align_right_from=1),
                  Paragraph('The model is usually sure whether a flow is an attack; it is less sure about the exact type, because some attack types '
                            'look the same in these measurements.', st['SMALL'])]
    tops = [c for c in record['detections']['by_class'] if c['class'] != 'Normal'][:5]
    if tops:
        story.append(class_bars(tops, font=8.4, row_h=15))

    # 2. what was detected
    det = v['detected']
    story += [Paragraph('2. What was detected', st['H1']), Paragraph(f"<b>{esc(det['question'])}</b>", st['BODY']), Paragraph(esc(det['answer']), st['BODY'])]
    if det['groups']:
        rows = [hdr(st, 'Predicted group', 'Flows', 'What it means')]
        for g in det['groups']:
            meaning = esc(g['meaning'].capitalize()) + '.'
            if g['context']:
                meaning += (f"<br/><font color='#475569'>Public guidance (a possible explanation, not proof): “{esc(g['context']['text'])}” "
                            f"{cite_links(g['context']['cites'])}</font>")
            rows.append([Paragraph(f"<b>{esc(g['class'])}</b>", st['CELL']), Paragraph(num(g['count']), st['CELL']), Paragraph(meaning, st['CELL'])])
        story += [table(rows, [30 * mm, 16 * mm, W - 46 * mm], st),
                  Paragraph(f"Group descriptions: {esc(det['groups'][0]['meaning_source'])}. Quoted guidance describes this kind of activity in "
                            'general; it is a possible explanation, not proof that it happened here.', st['SMALL'])]

    # 3. why the traffic was flagged
    story += [CondPageBreak(60 * mm), Paragraph('3. Why the traffic was flagged', st['H1'])]
    for b in v['overview']['bullets']:
        story.append(Paragraph(f"{esc(b['text'])} {cite_links(b['cites'])}", st['BULLET'], bulletText='•'))
    story.append(Paragraph('How to read this: the model looks at measurements of each flow (sizes, counts, timing, service). The reasons above '
                           'are the measurements that most often moved the model towards a group; they describe how the model decided, not what '
                           'caused the traffic.', st['SMALL']))
    if v.get('examples'):
        story.append(Paragraph('Examples: the reasoning for one flow of each main type', st['H3']))
        for ex in v['examples']:
            block = [Paragraph(f"<b>{esc(ex['class'])}</b> — example flow #{ex['flow_index']} <font color='#64748b' size='7.5'>(evidence: finding "
                               f"{esc(ex['flow_id'])} in the Detailed report)</font>", st['BODY'])]
            block += [Paragraph((f"<b>{esc(pt['title'])}:</b> " if pt.get('title') else '') + esc(pt['text'])
                                + (" <font color='#7c3aed' size='7'>(AI, checked)</font>" if pt.get('origin') == 'llm_synthesis' else ''),
                                st['BULLET'], bulletText=f'{n}.') for n, pt in enumerate(ex['points'], start=1)]
            story += [KeepTogether(block[:2])] + block[2:] + [Spacer(1, 2 * mm)]     # may continue on the next page; heading stays with point 1
        story.append(Paragraph(f"Points marked (AI, checked) were written by the local language model ({esc(v['examples'][0]['model'] or '-')}) and every "
                               'number and direction was checked against the evidence; the other points are written by code from the same evidence. '
                               'Typical values come from the training flows only. Each example explains one flow, not every flow of the type.', st['SMALL']))

    # 4. what to do
    story += [CondPageBreak(60 * mm), Paragraph('4. What to do', st['H1'])]
    if v['actions']:
        for phase, label in (('Verify', 'Verify now'), ('Contain if confirmed', 'Contain only if confirmed'), ('Recover', 'Recover and monitor')):
            group = [a for a in v['actions'] if a['phase'] == phase]
            if not group:
                continue
            rows = [hdr(st, 'What to do', 'How to do it', 'When it applies', 'How to check success')]
            for a in group:
                rows.append([Paragraph(f"{esc(a['what'])}<br/><font color='#64748b' size='7.5'>{esc(a['id'])} · {esc(a['owner'])}</font>", st['CELL']),
                             Paragraph(f"{esc(a['how'])} {cite_links([c for c in a['cites'] if c.startswith('S')])}", st['CELL']),
                             Paragraph(esc(a['when']), st['CELL']), Paragraph(esc(a['check']), st['CELL'])])
            story.append(KeepTogether([Paragraph(label, st['H3']), table(rows, [34 * mm, 58 * mm, 40 * mm, W - 132 * mm], st)]))
        story.append(Paragraph('This application never changes your network: these are suggestions for a person to decide on. The full list with '
                               'evidence is in the detailed report.', st['SMALL']))
    else:
        story.append(Paragraph('No attacks were predicted, so there are no response steps. Spot-check a few low-confidence flows.', st['BODY']))

    # 5. how to check the outcome
    o = v['outcome']
    story += [CondPageBreak(50 * mm), Paragraph('5. How to check the outcome', st['H1']),
              Paragraph(f"<b>Evidence you need.</b> {esc(o['evidence_needed'])}", st['BODY']),
              Paragraph(f"<b>Confirms the concern.</b> {esc(o['confirms'])}", st['BODY']),
              Paragraph('<b>Weakens the concern.</b> ' + esc('; '.join(o['weakens'])) + '.', st['BODY']),
              Paragraph(f"<b>Fix worked.</b> {esc(o['mitigation_success'])}", st['BODY'])]

    # 6. limits and sources
    story += [CondPageBreak(35 * mm), Paragraph('6. Limits and sources', st['H1'])]
    story += [Paragraph(esc(t), st['BULLET'], bulletText='•') for t in v['limits']]
    if v['sources']:
        story.append(Paragraph('Sources', st['H3']))
        story += [Paragraph(f'<a name="{anchor(src["id"])}"/>[{esc(src["id"])}] {esc(src["source"])}, {link(src.get("url"), src["location"])}',
                            st['BULLET'], bulletText='•') for src in v['sources']]
    cited = {c for b in v['overview']['bullets'] for c in b['cites']} | {c for g in det['groups'] if g['context'] for c in g['context']['cites']} \
        | {c for a in v['actions'] for c in a['cites'] if c.startswith('S')}
    listed = {src['id'] for src in v['sources']}
    extra = [c for c in sorted(cited) if c not in listed]
    if extra:                                                          # every link in this PDF has a target
        facts = {k: val for card in record['flows'] for k, val in card['facts'].items()}
        reg = {r['id']: r for r in record['sources']['register']}
        for c in extra:
            text = f"{reg[c]['source']}, {reg[c]['location']}" if c in reg else facts.get(c, '')
            story.append(Paragraph(f'<a name="{anchor(c)}"/>[{esc(c)}] {esc(text)}', st['BULLET'], bulletText='•'))
    story.append(Paragraph(f"Same analysis as the detailed report #{record.get('report_id', '-')} (identical counts, predictions and evidence); "
                           'the detailed report adds the full evidence, source passages and audit information.', st['SMALL']))
    return _build(record, story, 'Summary')


# ============================================================================================================ DETAILED REPORT
def render_record_pdf(record: dict) -> tuple[bytes, int]:
    """Detailed report: 1 executive assessment, 2 detections and uncertainty, 3 evidence and model explanation, 4 retrieved context,
    5 response plan, 6 limitations and provenance; appendices A-F hold the full audit material."""
    v = views(record)['summary']
    s, det, prov, st = record['summary'], record['detections'], record['provenance'], D
    P = lambda text, style='BODY': Paragraph(text, st[style])   # noqa: E731
    story = _title(record, st, 'Detailed', 'for security analysts and reviewers')
    story.append(P(f'<i>{esc(record["question"])}</i>', 'SMALL'))

    # ---------------------------------------------------------------- 1 executive assessment
    story += [P('1. Executive assessment', 'H1'), P(esc(s['scope']))]
    kpi = Table([[P('<font color="#64748b" size="7">Predicted Normal</font>', 'CELL'), P('<font color="#64748b" size="7">Predicted attack</font>', 'CELL'),
                  P('<font color="#64748b" size="7">Needs review (type confidence &lt; 70%)</font>', 'CELL'),
                  P('<font color="#64748b" size="7">Sure about attack or normal (90%+)</font>', 'CELL')],
                 [P(f"<b>{num(s['predicted_normal'])}</b> ({pct(s['predicted_normal_share'])})", 'CELL'),
                  P(f"<b>{num(s['predicted_attack'])}</b> ({pct(s['predicted_attack_share'])})", 'CELL'),
                  P(f"<b>{num(s['needs_review'])}</b>", 'CELL'),
                  P(f"<b>{num(s['status_confidence']['levels']['high'])}</b> of {num(record['detections']['total'])}" if s.get('status_confidence')
                    else f"<b>{pct(s['average_confidence'])}</b> mean", 'CELL')]], colWidths=[W / 4] * 4)
    kpi.setStyle(TableStyle([('FONT', (0, 0), (-1, -1), 'Body', 8), ('BOX', (0, 0), (-1, -1), 0.5, BORDER), ('INNERGRID', (0, 0), (-1, -1), 0.25, BORDER),
                             ('TOPPADDING', (0, 0), (-1, -1), 4)]))
    tp = s['triage_priority']
    story += [Spacer(1, 1.5 * mm), kpi, Spacer(1, 2 * mm), P('Main findings', 'H3')]
    for b in v['overview']['bullets']:
        story.append(Paragraph(f"{esc(b['text'])} {cite_links(b['cites'])} {origin_tag(b['origin'])}"
                               + (f" <font color='#64748b' size='6.8'>scope: {esc(b['scope'])}</font>" if b.get('scope') else ''), st['BULLET'], bulletText='•'))
    story.append(P(f"Same bullets as the Summary report. {esc(v['overview']['method'])}", 'SMALL'))
    nxt = [a for a in record['actions']['rows'] if a['phase'] == 'Verify'][:3]
    if nxt:
        story.append(P('Prioritised next steps', 'H3'))
        story += [Paragraph(f"<b>{esc(a['id'])}</b> {esc(a.get('title') or a['text'])} — {esc(a['scope'])}", st['BULLET'], bulletText='•') for a in nxt]
    story += [P(f"<b>Batch triage priority: <font color='#{PRIORITY.get(tp['level'], MUTED).hexval()[2:]}'>{esc(tp['level'])}</font></b> "
                f"(rule: {esc(tp['rule'])}). {esc(tp['meaning'])}", 'SMALL'),
              P(f"<b>Most important uncertainty.</b> {esc(s['most_important_uncertainty'])}"), P('Component status and coverage', 'H3'),
              table(_component_rows(record, st), [40 * mm, 22 * mm, W - 62 * mm], st)]
    cov = record['coverage']
    story.append(P(f"Explanation coverage: {cov['flows_explained']} illustrative flow cards"
                   + (f", group findings over {num(cov['group_flows_explained'])} flows" if cov.get('group_flows_explained') else '')
                   + f"; LLM explanations for {cov['attack_types_with_llm']} cards of {cov['attack_types_detected']} predicted attack types. "
                   + esc(cov.get('selection_rule', '')), 'SMALL'))

    # ---------------------------------------------------------------- 2 detections and uncertainty
    story += [CondPageBreak(70 * mm), P('2. Detection results and uncertainty', 'H1')]
    rows = [hdr(st, 'Predicted class', 'Flows', 'Share', 'Mean conf.', 'Severity (heuristic)', 'Status uncertain', 'Type uncertain')]
    for r in det['by_class']:
        rows.append([P(esc(r['class']), 'CELL'), num(r['count']), pct(r['share']), pct(r['average_confidence']),
                     '-' if r['class'] == 'Normal' else r['severity_heuristic'], num(r['status_uncertain']), num(r['type_uncertain'])])
    story += [class_bars(det['by_class']), Spacer(1, 2 * mm), table(rows, [36 * mm, 18 * mm, 18 * mm, 20 * mm, 30 * mm, 26 * mm, W - 148 * mm], st, align_right_from=1)]
    sc = s.get('status_confidence')
    if sc:
        lv, tv = sc['levels'], det['confidence_levels']
        story += [KeepTogether([P('Two confidences', 'H3'),
                                table([hdr(st, 'Confidence', 'High (>= 90%)', 'Medium (70-90%)', 'Low (< 70%)', 'Meaning'),
                                       ['Attack or normal', num(lv['high']), num(lv['medium']), num(lv['low']),
                                        P('sum of the calibrated probabilities of the nine attack classes (or of Normal)', 'CELL')],
                                       ['Exact class', num(tv['high']), num(tv['medium']), num(tv['low']),
                                        P('calibrated probability of the predicted class', 'CELL')]],
                                      [30 * mm, 24 * mm, 26 * mm, 22 * mm, W - 102 * mm], st, align_right_from=1)])]
    sets = det['sets']
    story += [KeepTogether([P('Prediction sets (90% shortlists)', 'H3'),
                            table([hdr(st, 'Shortlist', 'Flows'), ['Single class (not a guarantee)', num(sets['singleton'])], ['Several classes', num(sets['multiple'])],
                                   ['Attack STATUS uncertain: Normal and an attack type', num(sets['attack_status_uncertain'])],
                                   ['Attack TYPE uncertain: several attack types, no Normal', num(sets['attack_type_uncertain'])],
                                   ['Predicted Normal, attack type also on the shortlist', num(sets['predicted_normal_with_attack_in_set'])]],
                                  [W - 30 * mm, 30 * mm], st, align_right_from=1)])]
    story += [Paragraph(esc(n), st['BULLET'], bulletText='•') for n in det['notes'] + [x for x in det['shift_evidence'] if x]]
    if det.get('priority_flows'):
        prow = [hdr(st, 'Flow', 'Predicted class', 'Severity (heuristic)', 'Conf.', 'Shortlist', 'proto / service', 'Bytes src / dst')]
        for f in det['priority_flows']:
            prow.append([f"#{f['flow_index']}", P(esc(f['class']), 'CELL'), f['severity_heuristic'], prob_pct(f['confidence']),
                         P(esc(', '.join(f['prediction_set'])), 'CELL'), P(esc(f"{f['proto']} / {f['service']}"), 'CELL'), f"{num(f['sbytes'])} / {num(f['dbytes'])}"])
        story += [KeepTogether([P('Review order (heuristic severity, then confidence; not a list of confirmed threats)', 'H3'),
                                table(prow, [14 * mm, 26 * mm, 24 * mm, 18 * mm, 40 * mm, 26 * mm, W - 148 * mm], st)])]

    # ---------------------------------------------------------------- 3 evidence and model explanation
    story += [CondPageBreak(80 * mm), P('3. Evidence and model explanation', 'H1'),
              P('Group findings describe all flows of a predicted type; finding cards describe ONE selected flow each. Contributions (TreeSHAP) '
                'explain how the detection model scored a flow, not what caused the traffic, and they are not shares of the probability. Full '
                'attribution tables and the raw evidence given to the LLM are in Appendix A and B.', 'SMALL')]
    groups = record.get('group_findings') or []
    if groups:
        story.append(P('Group findings (per predicted attack type)', 'H2'))
        grows = [hdr(st, 'Predicted type', 'Flows explained', 'Most frequent supporting measurements (flows where it is a top-3 reason)',
                     'Most frequent opposing')]
        for g in groups:
            if g['status'] != 'available':
                grows.append([P(esc(g['class']), 'CELL'), '-', P(f"failed: {esc(g.get('reason', ''))}", 'CELL'), '-'])
                continue
            sup = '<br/>'.join(f"{esc(f['name'])}: {num(f['flows'])} of {num(g['analysed'])} ({esc(f['value_summary'])}"
                               + (f"; {esc(f['reference'])}" if f.get('reference') else '') + ')' for f in g['top_supporting'])
            opp = '<br/>'.join(f"{esc(f['name'])}: {num(f['flows'])} flows" for f in g['most_frequent_opposing']) or '-'
            grows.append([P(f"<b>{esc(g['class'])}</b>", 'CELL'), f"{num(g['analysed'])} of {num(g['population'])}", P(sup, 'CELL'), P(opp, 'CELL')])
        story += [table(grows, [24 * mm, 22 * mm, W - 84 * mm, 38 * mm], st), P(esc(groups[0]['method'].split('. ', 1)[1]), 'SMALL')]
    for card in record['flows']:
        p, xai, llm = card['prediction'], card['xai'], card['llm']
        alts = ', '.join(f"{a['class']} {prob_pct(a['probability'])}{' (on shortlist)' if a['in_set'] else ''}" for a in card.get('alternatives', []))
        head = [P(f"Finding {esc(card['id'])} · flow #{card['flow_index']}: predicted {esc(p['class'])}, probability {prob_pct(p['confidence'])}", 'H2'),
                P(f"<i>Selected as: {esc(card['selection'])}.</i>", 'SMALL'),
                P(f"Shortlist {esc(', '.join(p['prediction_set']))} · Normal on shortlist: {'yes' if p['normal_in_set'] else 'no'} · "
                  f"heuristic severity {esc(p['severity_heuristic'])}" + (f" · other classes: {esc(alts)}" if alts else ''), 'SMALL')]
        if xai.get('status') == 'available':
            head += [xai_chart(xai), P(f"Exact for the deployed score: base {xai['base_value']:+.3f} + contributions = {xai['margin']:+.3f}; "
                                       f"÷ temperature {xai['temperature']:.3f} + class offset {xai['class_offset']:+.2f} = {xai['deployed_class_score']:+.3f}.", 'SMALL')]
        else:
            head.append(P(f"<b>Explanation unavailable:</b> {esc(xai.get('reason', xai.get('status')))}", 'SMALL'))
        story.append(KeepTogether(head))
        if card.get('reasoning'):
            points = [P(f"<b>Reasoning</b> <font color='#64748b' size='7'>· {len(card['reasoning'])} points · (AI, checked) = written by the local LLM "
                        f"{esc((llm or {}).get('model') or '')} and checked against the cited evidence; the others are written by code from the same evidence"
                        + (f" · {llm['checked']} statements checked, {len(llm['removed'])} rejected" if llm else '') + '</font>', 'BODY')]
            points += [Paragraph(f"<b>{esc(r['title'])}:</b> {esc(r['text'])} {cite_links(r['cites'])}"
                                 + (" <font color='#7c3aed' size='6.8'>(AI, checked)</font>" if r['origin'] == 'llm_synthesis' else f" {origin_tag(r['origin'])}"),
                                 st['BULLET'], bulletText=f'{n}.') for n, r in enumerate(card['reasoning'], start=1)]
            story.append(box(points, background=colors.HexColor('#f5f3ff'), border=colors.HexColor('#ddd6fe')))
        elif llm and llm['explanation']:
            points = [P(f"<b>AI explanation</b> <font color='#64748b' size='7'>· local LLM {esc(llm['model'] or '')} · every number and direction "
                        f"checked against the cited evidence · {llm['checked']} statements checked, {len(llm['removed'])} rejected</font>", 'BODY')]
            points += [Paragraph(f"{esc(e['text'])} {cite_links(e['cites'])}", st['BULLET'], bulletText='•') for e in llm['explanation']]
            column_facts = {k for k, v in card['facts'].items() if v.startswith('Column ')}
            if card.get('interpretation') and not any(set(e['cites']) & column_facts for e in llm['explanation']):
                points.append(Paragraph(f"{esc(card['interpretation'])} {origin_tag('xai_attribution')} <font color='#64748b' size='6.8'>(the AI "
                                        'sentence about the measurements did not pass the checks; this line is written from the TreeSHAP '
                                        'values)</font>', st['BULLET'], bulletText='•'))
            story.append(box(points, background=colors.HexColor('#f5f3ff'), border=colors.HexColor('#ddd6fe')))
        else:
            if card.get('interpretation'):
                story.append(P(f"<b>Interpretation.</b> {esc(card['interpretation'])} {origin_tag('xai_attribution')}"))
            if llm:
                story.append(P(f"AI explanation: none accepted (status {esc(llm['status'])}; {len(llm['removed'])} rejected, see Appendix B). The TreeSHAP "
                               'evidence above stands on its own.', 'SMALL'))
        if card.get('alternatives') or card.get('missing_confirmation'):
            story.append(P("<b>Alternatives and what is missing.</b> "
                           + (f"Other classes the model did not rule out: {esc(alts)}. " if alts else 'No other class above 5%. ')
                           + esc(card.get('missing_confirmation') or ''), 'SMALL'))
        story.append(P(f"LIME: {esc(card['lime']['status'])}. Full values and contributions: Appendix A ({esc(card['id'])}).", 'SMALL'))

    # ---------------------------------------------------------------- 4 retrieved context
    story += [CondPageBreak(70 * mm), P('4. Retrieved context and source support', 'H1'),
              P('Retrieval selected passages from MITRE ATT&amp;CK, NIST SP 800-61r3, NIST SP 800-94 and the CISA playbooks (Appendix C). Kinds of '
                'statement are labelled: <b>retrieved source</b> (quoted passage), <b>LLM, passed checks</b> (written from a cited passage; the check '
                'is word overlap, not meaning), and <b>heuristic</b> (the fixed class-to-technique mapping). A technique listed here is a hypothesis '
                'for context: a predicted class is not evidence that the technique was used (for example, Generic does not mean encryption was '
                'compromised, and Backdoor does not mean persistent access exists).', 'SMALL')]
    any_attack = False
    for card in record['flows']:
        llm = card['llm']
        if not llm:
            continue
        any_attack = True
        block = [P(f"{esc(card['prediction']['class'])} — context for finding {esc(card['id'])}", 'H2')]
        maps = '; '.join(', '.join(f"{t['id']} {t['name'] or ''}".strip() for t in m['techniques']) for m in llm['mapped_techniques'] if m['predicted'])
        block.append(P(f"<b>Hypothesis (class mapping):</b> {esc(maps) or 'no ATT&CK technique mapped'} {origin_tag('heuristic')}", 'SMALL'))
        for item in llm['techniques'] + llm['context']:
            quote = (item.get('support') or {}).get('span')
            block.append(Paragraph(f"{esc(item['text'])} {cite_links(item['cites'])} {origin_tag('llm_synthesis')}"
                                   + (f"<br/><font color='#64748b' size='6.8'>Supporting sentence: “{esc(quote[:240])}”</font>" if quote else ''),
                                   st['BULLET'], bulletText='•'))
        if not (llm['techniques'] or llm['context']):
            block.append(P(f"No source statement passed the checks (status {esc(llm['status'])}); the passages are in Appendix C.", 'SMALL'))
        story.append(KeepTogether(block))
    if not any_attack:
        story.append(P('No predicted attacks: no attack context was retrieved for this batch.'))

    # ---------------------------------------------------------------- 5 response plan
    acts = record['actions']
    story += [CondPageBreak(60 * mm), P('5. Response and mitigation plan', 'H1'), P(esc(acts['note']), 'SMALL')]
    phase_names = {'Verify': 'Verify now (read-only)', 'Contain if confirmed': 'Contain only if confirmed', 'Recover': 'Recover and monitor'}
    for phase in ('Verify', 'Contain if confirmed', 'Recover'):
        group = [a for a in acts['rows'] if a['phase'] == phase]
        if not group:
            continue
        story.append(P(phase_names[phase], 'H2'))
        for a in [a for a in group if a.get('kind') != 'verify']:
            meta = [f"<b>Requires:</b> {esc(a['applicability'])}", f"<b>Owner:</b> {esc(a['owner'])}", f"<b>Possible disruption:</b> {esc(a['disruption'])}",
                    f"<b>Check:</b> {esc(a['validation'])}"] + ([f"<b>Undo:</b> {esc(a['rollback'])}"] if a.get('rollback') else [])
            story.append(KeepTogether([
                P(f"<b>{esc(a['id'])}. {esc(a.get('title') or a['phase'])}</b> <font color='#64748b' size='7'>· {esc(a['scope'])} · "
                  f"{esc(ORIGIN_LABEL.get(a['origin'], a['origin']))}</font>"),
                P(f"{esc(a['text'])} {cite_links(a['cites'])}"), P(f"<b>Why:</b> {esc(a['reason'])}", 'SMALL'),
                Paragraph(' · '.join(meta), st['SMALL']), Spacer(1, 1.5 * mm)]))
        flows = [a for a in group if a.get('kind') == 'verify']
        if flows:
            vrows = [hdr(st, 'Id', 'Flow to locate in captures or logs', 'Why (model output)', 'Evidence')]
            for a in flows:
                vrows.append([P(f"<b>{esc(a['id'])}</b>", 'CELL'), P(esc(a['text']), 'CELL'), P(esc(a['reason']), 'CELL'), P(cite_links(a['cites']), 'CELL')])
            f0 = flows[0]
            story += [table(vrows, [11 * mm, 72 * mm, 62 * mm, W - 145 * mm], st),
                      P(f"<b>For every flow above.</b> Requires: {esc(f0['applicability'])} Owner: {esc(f0['owner'])}. Disruption: {esc(f0['disruption'])} "
                        'Confirmed if logs or captures show activity matching the cited description of the predicted type; weakened if the traffic '
                        'belongs to a known service or job of your own team, other sensors show nothing for that period, or Normal is on the '
                        "flow's shortlist.", 'SMALL'), Spacer(1, 2 * mm)]
    if not acts['rows']:
        story.append(P('No actions: there are no predicted attacks in this batch.'))
    if acts['prevention']:
        story.append(P('General prevention (separate from incident response)', 'H2'))
        story += [Paragraph(f"<b>{esc(g['scope'])}:</b> {esc(g['text'][:260])} {cite_links(g['cites'])}", st['BULLET'], bulletText='•')
                  for g in acts['prevention'][:8]]

    # ---------------------------------------------------------------- 6 limitations and provenance
    lim = record['limits']
    story += [CondPageBreak(60 * mm), P('6. Limitations and provenance', 'H1'), P('What is not known', 'H3')]
    story += [Paragraph(esc(u), st['BULLET'], bulletText='•') for u in lim['unknown']]
    story += [P('Automated checks', 'H3'), P(esc(record['components'].get('checks', {}).get('method', '')), 'SMALL'),
              P(f"<b>Human review:</b> {esc(lim['human_review'])}.", 'SMALL'),
              P(f"<b>Versions:</b> detector {esc(prov['detector']['name'])} ({esc(prov['detector']['version'])}); report schema {record['schema_version']}, "
                f"template {esc(record['template_version'])}; knowledge base downloaded {esc(prov['knowledge']['downloaded'])}; LLM {esc(prov['llm']['model'])} "
                f"(local); input file sha256 {esc((prov['analysis']['input_sha256'] or 'not recorded')[:16])}…. Full hashes: Appendix D.", 'SMALL')]

    # ---------------------------------------------------------------- appendices
    story += [PageBreak(), P('Appendix A. Measured values and full attributions', 'H1'),
              P('Values observed in the upload, their position among the training flows (training rows only), and every TreeSHAP contribution '
                'of the predicted class, largest first.', 'SMALL')]
    for card in record['flows']:
        xai = card['xai']
        vals = [hdr(st, 'Measurement', 'Value', 'Compared with training flows')] + [
            [P(f"{esc(x['name'])} <font color='#64748b'>({esc(x['feature'])})</font>", 'CELL'), P(esc(x['value']), 'CELL'), P(esc(x['reference'] or '-'), 'CELL')]
            for x in card['values']]
        story += [P(f"{esc(card['id'])} · flow #{card['flow_index']} ({esc(card['prediction']['class'])})", 'H2'), table(vals, [60 * mm, 30 * mm, W - 90 * mm], st)]
        if xai.get('status') == 'available':
            allc = [hdr(st, 'Feature', 'Value', 'Contrib.', 'Feature', 'Value', 'Contrib.')]
            items = xai['all'][:24]
            half = (len(items) + 1) // 2
            for i in range(half):
                left = items[i]
                right = items[i + half] if i + half < len(items) else None
                allc.append([P(esc(left['feature']), 'CELL'), P(esc(left['value']), 'CELL'), f"{left['contribution']:+.4f}"]
                            + ([P(esc(right['feature']), 'CELL'), P(esc(right['value']), 'CELL'), f"{right['contribution']:+.4f}"] if right else ['', '', '']))
            story += [Spacer(1, 1.5 * mm), table(allc, [28 * mm, 26 * mm, 22 * mm, 28 * mm, 26 * mm, W - 130 * mm], st),
                      P(f"Largest {len(items)} of {len(xai['all'])} contributions shown; the full list is in the JSON export.", 'SMALL')]

    story += [CondPageBreak(60 * mm), P('Appendix B. Evidence given to the LLM and rejected statements', 'H1')]
    for card in record['flows']:
        llm = card['llm']
        if not llm:
            continue
        frows = [hdr(st, 'Evidence id', 'Fact (computed by code)')] + [[P(f'<a name="{anchor(k)}"/>{esc(k)}', 'CELL'), P(esc(val), 'CELL')]
                                                                       for k, val in card['facts'].items()]
        story += [P(f"{esc(card['id'])} ({esc(card['prediction']['class'])})", 'H2'), table(frows, [26 * mm, W - 26 * mm], st)]
        story += [
                  P(f"Generation: status {esc(llm['status'])}, model {esc(llm['model'] or '-')}, {llm['seconds']:.0f} s, {llm['checked']} statements checked, "
                    f"{len(llm['removed'])} rejected.", 'SMALL')]
        story += [P(f"<font color='#e11d48'>Rejected ({esc(r['part'])}, round {r.get('round', 1)}): {esc(r['reason'])}</font> — “{esc(str(r['text'])[:200])}”",
                    'SMALL') for r in llm['removed']]
    if not any(c['llm'] for c in record['flows']):
        story.append(P('No LLM step ran for this report (no predicted attacks among the selected flows).', 'SMALL'))

    story += [CondPageBreak(60 * mm), P('Appendix C. Sources', 'H1')]
    docs = record['sources'].get('documents', [])
    if docs:
        story += [P('Source documents', 'H2'),
                  table([hdr(st, 'Source', 'Publisher and title', 'Version / dates', 'Document link and hash', 'Cited')] + [
                      [P(esc(d['source']), 'CELL'), P(f"{esc(d['publisher'] or '-')}<br/>{esc(d['title'] or '-')}", 'CELL'),
                       P(f"{esc(d['version'] or '-')}{'<br/>PDF created ' + esc(d['pdf_created']) if d.get('pdf_created') else ''}<br/>retrieved "
                         f"{esc(d['retrieved'] or '-')}", 'CELL'),
                       P((link(d.get('download_url'), 'download') if d.get('download_url') else '-')
                         + f"<br/><font size='6'>sha256 {esc((d['document_sha256'] or '-')[:20])}…</font>", 'CELL'), str(d['passages'])]
                      for d in docs], [30 * mm, 52 * mm, 40 * mm, W - 137 * mm, 15 * mm], st)]
    story.append(P('Source register (passages cited in this report)', 'H2'))
    reg = [hdr(st, 'Id', 'Source and location', 'Retrieval', 'Passage (excerpt; full text in the web view and JSON)')]
    for src in record['sources']['register']:
        r = src['retrieval']
        reg.append([P(f'<a name="{anchor(src["id"])}"/><b>{esc(src["id"])}</b>', 'CELL'),
                    P(f"{esc(src['source'])}<br/>{link(src.get('url'), src['location'])}<br/><font size='6' color='#64748b'>sha256 {esc(src['content_sha256'][:16])}…</font>", 'CELL'),
                    P(f"<font size='6.5'>{esc(r['method'] or '-')}{'; score ' + str(r['score']) if r.get('score') is not None else ''}</font>", 'CELL'),
                    P(f"<font size='6.8'>{esc(src.get('excerpt') or src['text'][:320])}</font>", 'CELL')])
    story.append(table(reg, [11 * mm, 42 * mm, 30 * mm, W - 83 * mm], st) if len(reg) > 1 else P('No passages were retrieved.'))
    story.append(P(esc(record['sources']['note']), 'SMALL'))

    d = prov['detector']
    story += [CondPageBreak(60 * mm), P('Appendix D. Provenance and technical metadata', 'H1'),
              table([hdr(st, 'Item', 'Value')] + [[P(k, 'CELL'), P(esc(val), 'CELL')] for k, val in [
                  ('Detector', f"{d['name']} ({d['version']}); bundle built {d['bundle_created']} from commit {(d['bundle_source_commit'] or '-')[:10]}"),
                  *[(f'  {k}', v or '-') for k, v in d['files_sha256'].items()],
                  ('Preprocessing', f"{prov['preprocessing']['inputs']} + {', '.join(prov['preprocessing']['engineered'])}, encoded into "
                                    f"{prov['preprocessing']['model_columns']} model columns"),
                  ('Calibration', f"temperature {prov['calibration']['temperature']:.3f}; class offsets {prov['calibration']['class_offsets']}; alpha {prov['calibration']['alpha']}"),
                  ('Explanation', f"{prov['explanation']['method']}; {prov['explanation']['exactness']}"),
                  ('Knowledge base', f"downloaded {prov['knowledge']['downloaded']}; chunks sha256 {prov['knowledge']['chunks_sha256'] or '-'}"),
                  ('LLM', f"{prov['llm']['model']} ({prov['llm']['runtime']}), temperature 0, seed 42; {prov['llm']['note']}"),
                  ('Input file sha256', prov['analysis']['input_sha256'] or prov['analysis']['input_note']),
                  ('Application', f"code commit {prov['application']['code_commit'] or 'not available'}; report schema {record['schema_version']}"),
                  ('Timings (s)', ', '.join(f'{k} {val}' for k, val in record['timings'].items())),
              ]], [36 * mm, W - 36 * mm], st)]

    card = record['appendix']['model_card']
    e = card.get('evaluation')
    story += [PageBreak(), P('Appendix E. Model benchmark (official test file; not measured on this upload)', 'H1'), P(esc(record['appendix']['note']), 'SMALL')]
    if e:
        b = e['binary']
        story.append(table([hdr(st, 'Metric', 'Attack vs normal', '10-class macro', '10-class weighted'),
                            ['Accuracy', pct(b['accuracy']), pct(e['accuracy']), pct(e['accuracy'])],
                            ['Precision', pct(b['precision']), pct(e['macro']['precision']), pct(e['weighted']['precision'])],
                            ['Recall', pct(b['recall']), pct(e['macro']['recall']), pct(e['weighted']['recall'])],
                            ['F1', pct(b['f1']), pct(e['macro']['f1']), pct(e['weighted']['f1'])]], [40 * mm, 40 * mm, 40 * mm, W - 120 * mm], st, align_right_from=1))
        if e.get('ceiling'):
            story.append(P(f"Best possible on that file (a model that already knows the labels): accuracy {pct(e['ceiling']['accuracy'])}, macro F1 "
                           f"{pct(e['ceiling']['macro_f1'])}. Top-2 accuracy {pct(e.get('top2_accuracy'))}. Normal flows flagged: {pct(b['false_positive_rate'])}.", 'SMALL'))
        g = record['appendix'].get('grouped_evaluation')
        if g:
            gro = [hdr(st, 'Group', 'Precision', 'Recall', 'F1', 'Test flows')] + [
                [P(f"<b>{esc(r['class'])}</b>" if r['class'] == g['group'] else esc(r['class']), 'CELL'), pct(r['precision']), pct(r['recall']),
                 pct(r['f1']), num(r['support'])] for r in g['per_class']]
            story += [KeepTogether([P('Grouped view: DoS, Analysis and Backdoor as one group', 'H2'),
                                    table(gro, [60 * mm, 28 * mm, 28 * mm, 28 * mm, W - 144 * mm], st, align_right_from=1),
                                    P(f"Accuracy {pct(g['accuracy'])}, macro F1 {pct(g['macro_f1'])} (10 classes: {pct(e['accuracy'])} and "
                                      f"{pct(e['macro']['f1'])}). {esc(g['note'])} In UNSW-NB15 many flows of these three classes have identical "
                                      f"measurements but different labels ({num(g['within_group_confusions'])} test flows were confused within the group).", 'SMALL')])]
        cm = e['confusion_matrix']
        labels = cm['labels']
        rows = [[P('<b>True class (rows) / predicted (columns)</b>', 'CELL')] + [P(f'<b>{i + 1}</b>', 'CELL') for i in range(len(labels))]]
        for i, row in enumerate(cm['matrix']):
            rows.append([P(f'{i + 1}. {esc(labels[i])}', 'CELL')] + [f'{val:,}' for val in row])
        cmt = Table(rows, colWidths=[34 * mm] + [(W - 34 * mm) / len(labels)] * len(labels), repeatRows=1)
        style = [('FONT', (0, 0), (-1, -1), 'Body', 6.6), ('GRID', (0, 0), (-1, -1), 0.25, BORDER), ('ALIGN', (1, 1), (-1, -1), 'RIGHT')]
        for i, row in enumerate(cm['matrix']):
            total = max(sum(row), 1)
            for j, val in enumerate(row):
                if val:
                    tone = SAFE if i == j else THREAT
                    style.append(('BACKGROUND', (j + 1, i + 1), (j + 1, i + 1), colors.Color(tone.red, tone.green, tone.blue, alpha=min(0.12 + val / total * 0.8, 0.9))))
        cmt.setStyle(TableStyle(style))
        story += [KeepTogether([P('Confusion matrix (official test file)', 'H2'), cmt,
                                P('Columns use the same numbers as the rows: ' + '; '.join(f'{i + 1} = {esc(c)}' for i, c in enumerate(labels))
                                  + '. Green diagonal = correct; red = mistakes (darker = larger share of that true class).', 'SMALL')])]
    ev = record.get('evaluation')
    if ev:
        story += [CondPageBreak(50 * mm), P('Appendix F. Comparison with the labels in the uploaded file', 'H1'), P(esc(ev['scope']), 'SMALL'),
                  table([hdr(st, 'Check', 'Result'),
                         ['Attacks predicted as attacks', f"{num(ev['attacks_caught'])} of {num(ev['attacks'])} ({pct(ev['attacks_caught'] / max(ev['attacks'], 1))})"],
                         ['Normal flows predicted as attacks', f"{num(ev['normal_flagged'])} of {num(ev['normal'])} ({pct(ev['normal_flagged'] / max(ev['normal'], 1))})"],
                         ['Exact class right', f"{num(ev['exact_class_correct'])} of {num(ev['labelled_flows'])} ({pct(ev['exact_class_correct'] / max(ev['labelled_flows'], 1))})"],
                         ['True class on the 90% shortlist', f"{num(ev['truth_in_set'])} of {num(ev['labelled_flows'])}"]], [80 * mm, W - 80 * mm], st)]
    return _build(record, story, 'Detailed')
