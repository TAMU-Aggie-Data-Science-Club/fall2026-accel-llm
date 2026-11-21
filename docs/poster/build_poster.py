"""Mock-up of the ADSC Canva template (48 x 36 in) filled from poster_text.md, to check fit before pasting into Canva.

    python make_figures.py && python build_poster.py   ->  AccelLLM_poster_draft.pptx
Layout per the template: left Intro/Objective/Methods, center Results (widest), right Discussion/Conclusion/References/QR.
Sizes per the template: title ~70 pt, section bars ~40 pt lavender on dark, body ~35 pt black, subheads bold #1f163e.
"""
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).parent
BG, BAR, LAV, SUB = RGBColor(0x0d, 0x0a, 0x1a), RGBColor(0x2b, 0x1f, 0x52), RGBColor(0xe4, 0xdf, 0xef), RGBColor(0x1f, 0x16, 0x3e)
WHITE, BLACK = RGBColor(0xff, 0xff, 0xff), RGBColor(0, 0, 0)
BODY, HEAD, TITLE = 34, 40, 70
COLS = [  # (x, width, sections)
    (0.8, 13.6, ["Introduction", "Objective", "Methods"]),
    (15.2, 17.6, ["Results"]),
    (33.6, 13.6, ["Discussion", "Conclusion", "References", "Contact"]),
]
SIZE = {"References": 24}
FIGS = {
    "Methods": [("fig1_architecture.png", "Figure 1: System architecture, edge draft to HPRC verifier")],
    "Results": [("fig2_hprc_wallclock.png", "Figure 2: Wall-clock time on HPRC, cloud-only vs. distributed"),
                ("fig3_measured_runs.png", "Figure 3: Acceptance and speedup by prompt type (local Qwen2.5 0.5B → 1.5B, k = 5)")],
}
TOP, BOTTOM = 6.4, 34.6


def parse(md):
    head, *parts = re.split(r"\n## ", md)
    sections = {}
    for part in parts:
        title, _, body = part.partition("\n")
        sections[title.strip()] = [l.strip() for l in body.strip().splitlines() if l.strip()]
    return [l.lstrip("# ") for l in head.strip().splitlines()], sections


def add_runs(p, line, size, color):
    for chunk in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", line):
        if chunk:
            r = p.add_run()
            r.text = chunk.strip("*")
            bold = chunk.startswith("**")
            r.font.size, r.font.name, r.font.bold = Pt(size), "Arial", bold
            r.font.italic = chunk.startswith("*") and not bold
            r.font.color.rgb = SUB if bold and color == BLACK else color


def rect(s, x, y, w, h, fill):
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    r.fill.solid(); r.fill.fore_color.rgb = fill; r.line.fill.background()
    return r


def text_height(lines, w, size):
    cpl = (w - 0.5) * 72 / (size * 0.5)
    return sum((len(re.sub(r"\*", "", l)) // cpl + 1) * size * 1.2 / 72 + size * 0.3 / 72 for l in lines) + 0.4


def textbox(s, x, y, w, lines, size):
    h = text_height(lines, w, size)
    rect(s, x, y, w, h, WHITE)
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.25)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        add_runs(p, ("•  " + line[2:]) if line.startswith("- ") else line, size, BLACK)
        p.space_after = Pt(size * 0.3)
    return h


def picture(s, name, x, y, w, caption):
    """White panel, figure, centered numbered caption below (template rule)."""
    from PIL import Image
    iw, ih = Image.open(HERE / name).size
    ph = (w - 0.6) * ih / iw + 0.4
    ch = text_height([caption], w, 24) - 0.2
    rect(s, x, y, w, ph + ch, WHITE)
    s.shapes.add_picture(str(HERE / name), Inches(x + 0.3), Inches(y + 0.2), width=Inches(w - 0.6))
    cap = s.shapes.add_textbox(Inches(x), Inches(y + ph), Inches(w), Inches(ch)).text_frame
    cap.word_wrap = True
    add_runs(cap.paragraphs[0], f"**{caption}**", 24, BLACK)
    cap.paragraphs[0].alignment = PP_ALIGN.CENTER
    return ph + ch


head, sections = parse((HERE / "poster_text.md").read_text())
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(48), Inches(36)
s = prs.slides.add_slide(prs.slide_layouts[6])
rect(s, 0, 0, 48, 36, BG)
tf = s.shapes.add_textbox(Inches(5), Inches(0.6), Inches(38), Inches(5)).text_frame
tf.word_wrap = True
for i, (line, size) in enumerate(zip(head, (TITLE, 44))):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    add_runs(p, f"**{line}**" if i == 0 else line, size, LAV)
    p.alignment = PP_ALIGN.CENTER
for x, label in [(0.8, "project\nlogo"), (43.4, "ADSC\nlogo")]:
    o = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(0.6), Inches(3.8), Inches(3.8))
    o.fill.background(); o.line.color.rgb = LAV
    add_runs(o.text_frame.paragraphs[0], label, 24, LAV)
foot = s.shapes.add_textbox(Inches(0.8), Inches(34.9), Inches(20), Inches(0.8)).text_frame
add_runs(foot.paragraphs[0], "**AGGIE DATA SCIENCE CLUB**", 28, LAV)

for x, w, names in COLS:
    y = TOP
    for name in names:
        bar = rect(s, x, y, w, 1.0, BAR)
        add_runs(bar.text_frame.paragraphs[0], f"**{'CONTACT US BY SCANNING THE QR CODE' if name == 'Contact' else name.upper()}**",
                 HEAD if name != "Contact" else 28, LAV)
        bar.text_frame.paragraphs[0].runs[0].font.color.rgb = LAV
        bar.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        y += 1.15
        y += textbox(s, x, y, w, sections[name], SIZE.get(name, BODY)) + 0.15
        for fig, cap in FIGS.get(name, []):
            y += picture(s, fig, x, y, w, cap) + 0.15
        y += 0.35
    flag = "  <-- OVERFLOWS, trim text" if y > BOTTOM else ""
    print(f"column at x={x:>4}: content ends at {y:.1f} in (limit {BOTTOM}){flag}")

out = HERE / "AccelLLM_poster_draft.pptx"
prs.save(out)
print("wrote", out)
