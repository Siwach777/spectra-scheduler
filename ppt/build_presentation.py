"""Build six template-section slides using only editable text and vector diagrams.

PDF uses the same layout, not PowerPoint export. Requires python-pptx, reportlab,
and Noto Sans fonts. Running this replaces the generated PPTX and PDF.
"""
from math import atan2, cos, sin
from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Pt
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
INK, BLUE, MUTED = '142B40', '245CDE', '637487'
LINE, WHITE, GREEN, AMBER = 'D9E2ED', 'FFFFFF', '078276', 'B36D1B'
for name, suffix in [('Regular', 'Regular'), ('Bold', 'Bold')]:
    pdfmetrics.registerFont(TTFont(name, f'/usr/share/fonts/noto/NotoSans-{suffix}.ttf'))
prs = Presentation(ROOT / 'SIH2026-IDEA-Presentation-Format.pptx')
prs.slide_width, prs.slide_height = Pt(960), Pt(540)
while len(prs.slides) > 6:
    item = prs.slides._sldIdLst[-1]
    prs.part.drop_rel(item.rId)
    prs.slides._sldIdLst.remove(item)
pdf = canvas.Canvas(str(ROOT / 'Spectra-Scheduler-SIH2026.pdf'), pagesize=(960, 540), invariant=1)
pdf.setTitle('Spectra Scheduler | SIH26055')
pdf.setAuthor('Spectra Scheduler')
slide = None


def box(x, y, w, h, fill=WHITE, border=None, rounded=False, ellipse=False):
    kind = MSO_SHAPE.OVAL if ellipse else (MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE)
    s = slide.shapes.add_shape(kind, Pt(x), Pt(y), Pt(w), Pt(h))
    if rounded:
        s.adjustments[0] = .12
    s.fill.solid()
    s.fill.fore_color.rgb = RGBColor.from_string(fill)
    if border:
        s.line.color.rgb = RGBColor.from_string(border)
        s.line.width = Pt(1)
    else:
        s.line.fill.background()
    pdf.setFillColor(HexColor('#'+fill))
    pdf.setStrokeColor(HexColor('#'+(border or fill)))
    pdf.setLineWidth(1)
    if ellipse:
        pdf.ellipse(x, 540-y-h, x+w, 540-y, fill=1, stroke=bool(border))
    elif rounded:
        pdf.roundRect(x, 540-y-h, w, h, 7, fill=1, stroke=bool(border))
    else:
        pdf.rect(x, 540-y-h, w, h, fill=1, stroke=bool(border))


def txt(x, y, w, value, size=15, color=INK, bold=False, url=None):
    # Noto Sans does not include these arrow glyphs on this build host.
    # Directional relationships use vector connectors, not font-dependent symbols.
    value = value.replace(' → ', ' / ').replace(' ↔ ', ' / ').replace(' →', '')
    lines, font = value.split('\n'), 'Bold' if bold else 'Regular'
    for content in lines:
        if pdfmetrics.stringWidth(content, font, size) > w:
            raise ValueError(f'Text exceeds width {w}: {content}')
    h = len(lines)*size*1.3+5
    assert x >= 0 and x+w <= 960 and y >= 0 and y+h <= 540, value
    tf = slide.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h)).text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, content in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = p.space_after = Pt(0)
        p.line_spacing = Pt(size*1.3)
        r = p.add_run()
        r.text = content
        r.font.name, r.font.size, r.font.bold = 'Noto Sans', Pt(size), bold
        r.font.color.rgb = RGBColor.from_string(color)
        if url:
            r.hyperlink.address = url
        pdf.setFont(font, size)
        pdf.setFillColor(HexColor('#'+color))
        pdf.drawString(x, 540-y-size-i*size*1.3, content)
    if url:
        pdf.linkURL(url, (x, 540-y-h, x+w, 540-y))


def line(x1, y1, x2, y2, color=LINE, width=1.4, arrow=False, dashed=False):
    s = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Pt(x1), Pt(y1), Pt(x2), Pt(y2))
    s.line.color.rgb = RGBColor.from_string(color)
    s.line.width = Pt(width)
    if dashed:
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        s.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if arrow:
        end = OxmlElement('a:tailEnd')
        end.set('type', 'triangle')
        s.line._get_or_add_ln().append(end)
    pdf.setStrokeColor(HexColor('#'+color))
    pdf.setLineWidth(width)
    pdf.setDash(4, 3) if dashed else pdf.setDash()
    pdf.line(x1, 540-y1, x2, 540-y2)
    pdf.setDash()
    if arrow:
        angle = atan2(y2-y1, x2-x1)
        p = pdf.beginPath()
        p.moveTo(x2, 540-y2)
        for a in (angle+.42, angle-.42):
            p.lineTo(x2-8*cos(a), 540-(y2-8*sin(a)))
        p.close()
        pdf.setFillColor(HexColor('#'+color))
        pdf.drawPath(p, fill=1, stroke=0)


def begin(n, section, title=None, sub=None):
    global slide
    slide = prs.slides[n-1]
    for shape in list(slide.shapes):
        shape._element.getparent().remove(shape._element)
    box(0, 0, 960, 540)
    txt(44, 24, 725, section, 10, BLUE, True)
    txt(820, 24, 105, 'SIH 2026', 11, INK, True)
    if title:
        txt(44, 69, 880, title, 32, INK, True)
    if sub:
        txt(44, 119, 875, sub, 13, MUTED)
    line(44, 506, 916, 506, LINE, .7)
    txt(44, 516, 450, 'SPECTRA SCHEDULER   /   SIH26055', 8, MUTED, True)
    txt(879, 515, 40, f'0{n} / 06', 8, MUTED)


def end(notes):
    slide.notes_slide.notes_text_frame.text = notes
    pdf.showPage()


def node(x, y, w, title, detail, accent=BLUE):
    box(x, y, w, 79, WHITE, LINE, rounded=True)
    box(x, y+18, 3, 42, accent)
    txt(x+17, y+13, w-30, title, 17, INK, True)
    txt(x+17, y+43, w-30, detail, 12, MUTED)


begin(1, 'TITLE PAGE  /  SMART SCAN STRATEGY')
txt(44, 101, 500, 'Spectra\nScheduler', 56, INK, True)
txt(47, 262, 458, 'Smarter listening.\nUnder real receiver constraints.', 23, MUTED)
txt(47, 350, 458, 'Smart Scan strategy for Electronic Warfare', 15, INK, True)
txt(47, 382, 450, 'SIH26055  •  DRDO  •  Software', 13, BLUE)
txt(47, 432, 480, 'Team name: __________________    Team ID: __________', 10, MUTED)
txt(47, 455, 460, 'Theme: pending official confirmation', 10, MUTED)
txt(566, 107, 345, 'WIDE SPECTRUM', 11, MUTED, True)
box(629, 139, 76, 112, 'EEF3FF')
for i, height in enumerate([22, 42, 29, 67, 92, 48, 28, 56, 78, 33, 45, 65, 24, 38]):
    box(567+i*23, 238-height, 9, height, BLUE if 3 <= i <= 5 else 'D9E3EF')
line(565, 249, 898, 249, MUTED, 1)
line(629, 139, 629, 251, BLUE, 1)
line(705, 139, 705, 251, BLUE, 1)
txt(710, 258, 205, 'Frequency →', 11, MUTED)
line(667, 262, 667, 306, BLUE, 1.8, arrow=True)
node(560, 322, 337, 'Limited-bandwidth receiver', 'Select band  →  dwell  →  revisit')
txt(569, 426, 333, 'One observation window.\nMany signals outside it.', 19, INK, True)
end('Spectrum is conceptual, not measured. Passive scheduling under partial observations; no live RF hardware. '
    'Theme needs official confirmation. Team fields intentionally blank.')

begin(2, 'IDEA TITLE', 'A closed loop, not a fixed sweep.',
      'Proposed solution: adapt the observation schedule as evidence changes.')
node(75, 192, 305, '01  Observe', 'Hits, misses, signal measurements')
node(580, 192, 305, '02  Estimate', 'Activity history + track confidence')
node(580, 365, 305, '03  Schedule', 'Next band + listening dwell')
node(75, 365, 305, '04  Retune & listen', 'Respect hardware observation cost')
line(380, 232, 578, 232, BLUE, 1.8, arrow=True)
txt(425, 209, 150, 'feedback', 11, MUTED)
line(733, 273, 733, 363, BLUE, 1.8, arrow=True)
txt(747, 308, 150, 'decision', 11, MUTED)
line(578, 405, 382, 405, BLUE, 1.8, arrow=True)
txt(422, 381, 150, 'action', 11, MUTED)
line(226, 363, 226, 273, BLUE, 1.8, arrow=True)
txt(90, 308, 126, 'new observation', 11, MUTED)
txt(366, 294, 236, 'Search ↔ Revisit', 22, INK, True)
txt(375, 329, 224, 'Coverage + switching cost', 12, MUTED)
txt(76, 474, 815, 'DISTINCTIVE DESIGN   Observation-only inference • Drift response • Explicit receiver constraints', 12, BLUE)
end('System-level control loop. Bayesian, track-aware and fitted-hit schedulers are separate baselines, '
    'not a fully fused controller. Policies do not receive emitter IDs, future transmissions or false-alarm '
    'truth labels. Dwell and coverage safeguards balance exploration and exploitation. No proven algorithmic novelty is claimed.')

begin(3, 'TECHNICAL APPROACH', 'Two pipelines. One evaluation discipline.',
      'Python / NumPy  •  h5py / HDF5  •  scikit-learn  •  JSON model artifacts')
txt(44, 177, 870, 'ONLINE SIMULATION', 11, BLUE, True)
for x, title, detail in [(44, 'RF simulator', 'Dynamic emitters + noise'),
                        (344, 'Policy inference', 'History → band selection'),
                        (644, 'Paired evaluation', 'Interception + latency')]:
    node(x, 205, 272, title, detail)
line(317, 245, 341, 245, BLUE, 1.5, arrow=True)
line(617, 245, 641, 245, BLUE, 1.5, arrow=True)
txt(344, 292, 555, '13 baselines + logistic hit predictor; truth hidden from policies', 10.5, MUTED)
txt(44, 329, 870, 'OFFLINE PULSE ASSOCIATION', 11, GREEN, True)
for x, title, detail in [(44, 'TSRD pulse data', 'Validate → stream → sample'),
                        (344, 'Feature + cluster', 'Scaling → HDBSCAN'),
                        (644, 'File-local scoring', 'V-measure + pairwise F1')]:
    node(x, 355, 272, title, detail, GREEN)
line(317, 395, 341, 395, GREEN, 1.5, arrow=True)
line(617, 395, 641, 395, GREEN, 1.5, arrow=True)
line(58, 436, 58, 471, AMBER, 1.3, dashed=True)
line(58, 471, 331, 471, AMBER, 1.3, arrow=True, dashed=True)
txt(344, 456, 572, 'PLANNED: dataset-derived simulation calibration', 13, AMBER, True)
txt(344, 478, 572, 'Recorded scans do not reveal every alternative tuning outcome.', 10, MUTED)
end('Implemented independent pipelines; dashed annotation marks an unimplemented calibration bridge. '
    'PDW: pulse descriptor word. Signature preprocessing scales frequency/log1p pulse width/amplitude, '
    'encodes AoA with sine/cosine, excludes absolute ToA. HDBSCAN is per-file unsupervised clustering. '
    'The logistic predictor uses past observations to predict hits including false alarms; not RL. '
    'Rust/C++ is unimplemented. Process parallelism supports existing comparisons and dataset files; '
    'learning evaluation is sequential. Dataset amplitude dB is not assumed to be simulator dBm.')

begin(4, 'FEASIBILITY AND VIABILITY', 'Evidence first. Claims kept in scope.',
      'Implemented ingestion and association benchmarks; learned scheduling remains experimental.')
txt(44, 178, 385, '233.2M', 58, BLUE, True)
txt(47, 259, 360, 'scan-training pulses validated', 18, INK, True)
txt(47, 297, 360, '2,500 files  •  chunked HDF5 processing\nBounded samples; no full-dataset RAM load', 12, MUTED)
txt(475, 179, 440, 'ASSOCIATION / PAIRWISE F1', 11, MUTED, True)
for y, title, score, color in [(219, 'Raw features', .1815, 'B7C6D9'),
                             (272, 'Scaled signatures', .8121, BLUE)]:
    txt(475, y, 260, title, 14, INK)
    box(475, y+26, 350*score, 13, color)
    txt(847, y+13, 68, f'{score:.3f}', 20, INK, True)
txt(475, 328, 440, '10 training files • 100k sampled pulses • not held-out', 10.5, MUTED)
line(44, 368, 916, 368, LINE)
for x, title, body in [(44, 'Incomplete observations', 'Truth-separated evaluation\nRetuning excluded from targets'),
                       (344, 'Scale & domain shift', 'Chunked I/O + bounded sampling\nUnit calibration remains open'),
                       (644, 'Unproven policy gain', 'Constant-model ablation\nExisting defaults retained')]:
    txt(x, 390, 275, title, 16, INK, True)
    txt(x, 424, 275, body, 12, MUTED)
txt(44, 480, 873, 'Learning result: the fitted policy does not consistently outperform the strongest existing baseline.', 11, AMBER)
end('Sources: docs/dataset-workflow.md, docs/learning-workflow.md and generated JSON reports. '
    'Audit: 233172417 pulses, 2500 scan training files, eight empty, no read/schema errors. '
    'Association comparison: first ten lexicographic training files, 10000 sampled pulses/file, seed 0, '
    'minimum cluster size 20, minimum samples 10. Scores are file-macro with noise treated as one cluster; '
    'not held-out and not scheduler performance. Fitted model used 32664 listening examples. '
    'Held-out simulation: 50 seeds from 10000. Learned/best-existing interception: acquisition 9.71/11.17, '
    'change 13.00/34.20, mixed 6.59/8.55, tracking 8.40/13.47 percent. Crowded 8.57 versus 6.91 percent, '
    'but constant-model also 8.57; learning cannot claim that improvement.')

begin(5, 'IMPACT AND BENEFITS', 'Optimise the trade-off. Not a single metric.',
      'Intended users: receiver-algorithm researchers and spectrum-system evaluation teams.')
points = [(258, 212), (95, 403), (421, 403)]
for a, b in [(0, 1), (1, 2), (2, 0)]:
    line(*points[a], *points[b], LINE, 2)
for x, y in points:
    box(x-6, y-6, 12, 12, BLUE, ellipse=True)
txt(192, 165, 230, 'Interception', 20, INK, True)
txt(44, 427, 180, 'Discovery', 20, INK, True)
txt(350, 427, 165, 'Retuning cost', 20, INK, True)
txt(186, 320, 210, 'Receiver budget', 17, BLUE, True)
txt(541, 192, 375, 'Earlier detection', 23, INK, True)
txt(541, 229, 375, 'Measure first-detection and reacquisition delay.', 12, MUTED)
txt(541, 280, 375, 'Repeatable experimentation', 23, INK, True)
txt(541, 317, 375, 'Compare policies under identical seeded conditions.', 12, MUTED)
txt(541, 368, 375, 'Controlled compute demand', 23, INK, True)
txt(541, 405, 375, 'Stream data; profile memory and decision latency.', 12, MUTED)
txt(44, 479, 872, 'PLANNED DELIVERY   Calibration bridge  →  broader validation  →  experiment API / dashboard', 11, BLUE)
end('Triangle is a conceptual trade-off, not a measured Pareto frontier. Benefits are objectives, '
    'not operational savings. Tests include unit, integration, deterministic regression and split-boundary '
    'checks; broader structural generalisation remains unverified. API, GUI, calibration and hardware '
    'integration are not implemented. Delivery line is a proposal, not an implemented pipeline.')

begin(6, 'RESEARCH AND REFERENCES', 'Grounded in research. Tested independently.',
      'References inform the methods; published scores are not presented as project results.')
refs = [
    ('01', 'Benchmark foundation', 'The Turing Synthetic Radar Dataset', 'Gunn et al.  /  arXiv:2602.03856',
     'https://arxiv.org/abs/2602.03856'),
    ('02', 'Data & evaluation contract', 'TSRD dataset card + challenge implementation', 'PDW schema, receiver modes, splits and scoring',
     'https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset'),
    ('03', 'Representation-learning research', 'Radar Pulse Deinterleaving with Transformer', 'Based Deep Metric Learning  /  arXiv:2503.13476',
     'https://arxiv.org/abs/2503.13476')]
for i, (number, role, title, detail, url) in enumerate(refs):
    y = 180+i*96
    txt(44, y, 80, number, 28, BLUE, True)
    txt(110, y, 295, role, 15, INK, True)
    line(410, y+12, 455, y+12, LINE, 1.3, arrow=True)
    txt(476, y, 440, title, 15, INK, True, url=url)
    txt(476, y+30, 440, detail, 12, MUTED, url=url)
    if i<2:
        line(110, y+75, 916, y+75, LINE, .7)
txt(44, 472, 875, 'PROJECT EVIDENCE   dataset-workflow.md  /  learning-workflow.md  /  reproducible JSON reports', 11, MUTED)
end('Primary references linked on slide. Challenge code: '
    'https://github.com/alan-turing-institute/turing-deinterleaving-challenge . '
    'Transformer metric learning is research, not implemented. Dataset card and paper versions differ '
    'in aggregate counts; local audit counts are used in this deck. Official theme needs confirmation. '
    'The supplied template six content sections are retained; instructions slide omitted.')

prs.core_properties.title = 'Spectra Scheduler — Smart Scan Strategy'
prs.core_properties.author = 'Spectra Scheduler'
prs.core_properties.subject = 'SIH26055 | adaptive receiver scheduling'
prs.save(ROOT / 'Spectra-Scheduler-SIH2026.pptx')
pdf.save()
assert len(prs.slides) == 6
assert not any(s.shape_type == 13 for page in prs.slides for s in page.shapes)
print('Built six slides: editable vector diagrams only; no picture shapes.')
