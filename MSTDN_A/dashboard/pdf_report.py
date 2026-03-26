"""PDF report generator for MSTDN-A session results — redesigned."""
from __future__ import annotations
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle, HRFlowable,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Wedge, Circle, Line
from reportlab.graphics import renderPDF
from reportlab.platypus.flowables import Flowable

# ── palette ───────────────────────────────────────────────────────────────────
BG      = colors.HexColor("#0d1117")
CARD    = colors.HexColor("#161b22")
CARD2   = colors.HexColor("#1c2128")
BORDER  = colors.HexColor("#30363d")
TEXT    = colors.HexColor("#c9d1d9")
MUTED   = colors.HexColor("#8b949e")
ACCENT  = colors.HexColor("#58a6ff")
GREEN   = colors.HexColor("#3fb950")
YELLOW  = colors.HexColor("#d29922")
RED     = colors.HexColor("#f85149")
PURPLE  = colors.HexColor("#8957e5")
WHITE   = colors.white

ZONE_GREEN  = colors.HexColor("#0a1f0e")
ZONE_YELLOW = colors.HexColor("#1f1800")
ZONE_RED    = colors.HexColor("#1f0808")

EMO_COLORS = [
    colors.HexColor("#f85149"), colors.HexColor("#3fb950"),
    colors.HexColor("#8957e5"), colors.HexColor("#ffd700"),
    colors.HexColor("#c9d1d9"), colors.HexColor("#58a6ff"),
    colors.HexColor("#ff9800"), colors.HexColor("#e57373"),
    colors.HexColor("#ffb300"), colors.HexColor("#7986cb"),
    colors.HexColor("#ab47bc"),
]
EMOTIONS = ["Anger","Disgust","Fear","Happiness","Neutral","Sadness",
            "Surprise","Contempt","Anxiety","Helplessness","Disappointment"]


def _stress_label(v: float) -> tuple[str, object]:
    if v > 0.7:  return "High",   RED
    if v > 0.45: return "Medium", YELLOW
    return "Low", GREEN


# ── page background ───────────────────────────────────────────────────────────
def _add_page_bg(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(BG)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    # header band
    canvas.setFillColor(CARD)
    canvas.rect(0, h - 20*mm, w, 20*mm, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, h - 20*mm, 3*mm, 20*mm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(10*mm, h - 12*mm, "MSTDN-A")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(27*mm, h - 12*mm, "Emotion Intelligence Report")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(w - 10*mm, h - 12*mm,
                           f"Generated {datetime.now().strftime('%d %b %Y  %H:%M')}")
    # footer
    canvas.setFillColor(CARD)
    canvas.rect(0, 0, w, 9*mm, fill=1, stroke=0)
    canvas.setFillColor(BORDER)
    canvas.rect(0, 9*mm, w, 0.3, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 6.5)
    canvas.drawCentredString(w / 2, 3*mm,
                             f"Page {doc.page}  \u00b7  MSTDN-A Multimodal Emotion Detection")
    canvas.restoreState()


# ── styles ────────────────────────────────────────────────────────────────────
def _styles():
    base = getSampleStyleSheet()
    def P(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=base[parent], **kw)
    return {
        "title":    P("title",  fontSize=24, fontName="Helvetica-Bold",
                      textColor=WHITE, spaceAfter=3, leading=28),
        "body":     P("body",   fontSize=8.5, textColor=TEXT, leading=13, spaceAfter=4),
        "body_sm":  P("bsm",   fontSize=7.5, textColor=MUTED, leading=11),
        "label":    P("lbl",   fontSize=6.5, textColor=MUTED, fontName="Helvetica-Bold",
                      spaceAfter=1, textTransform="uppercase", letterSpacing=0.5),
        "section":  P("sec",   fontSize=10, fontName="Helvetica-Bold",
                      textColor=ACCENT, spaceBefore=6, spaceAfter=3),
        "p_name":   P("pn",    fontSize=12, fontName="Helvetica-Bold",
                      textColor=WHITE, spaceAfter=1),
        "caption":  P("cap",   fontSize=6.5, textColor=MUTED,
                      alignment=TA_CENTER, fontName="Helvetica-Oblique"),
        "bullet":   P("bul",   fontSize=8.5, textColor=TEXT, leading=13,
                      leftIndent=10, spaceAfter=3),
        "kv_key":   P("kvk",   fontSize=6, textColor=MUTED,
                      fontName="Helvetica-Bold", textTransform="uppercase",
                      spaceAfter=1, leading=8),
        "kv_val":   P("kvv",   fontSize=9, textColor=TEXT,
                      fontName="Helvetica-Bold", leading=11),
    }


# ── stat card ─────────────────────────────────────────────────────────────────
def _stat_card(label: str, value: str, value_color=ACCENT, width=34*mm) -> Table:
    t = Table(
        [[Paragraph(value, ParagraphStyle("sv", fontSize=15, fontName="Helvetica-Bold",
                    textColor=value_color, alignment=TA_CENTER))],
         [Paragraph(label, ParagraphStyle("sl", fontSize=6, textColor=MUTED,
                    alignment=TA_CENTER, fontName="Helvetica-Bold",
                    textTransform="uppercase"))]],
        colWidths=[width],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), CARD),
        ("BOX",           (0,0), (-1,-1), 0.5, BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 3),
        ("RIGHTPADDING",  (0,0), (-1,-1), 3),
    ]))
    return t


# ── horizontal emotion bars ───────────────────────────────────────────────────
class EmotionBars(Flowable):
    """Horizontal bar chart for emotion distribution — easier to read."""
    BAR_H = 9
    GAP   = 4

    def __init__(self, counts: dict, width: float):
        super().__init__()
        total = sum(counts.values()) or 1
        self.items = sorted(
            [(e, counts.get(e, 0), counts.get(e, 0) / total)
             for e in EMOTIONS if counts.get(e, 0) > 0],
            key=lambda x: -x[1]
        )
        self.width  = width
        self.height = len(self.items) * (self.BAR_H + self.GAP) + 4
        self._lw    = 26 * mm   # label column width
        self._pw    = 12 * mm   # percent column width

    def draw(self):
        w   = float(self.width)
        bar = w - self._lw - self._pw - 3
        y   = float(self.height) - self.BAR_H - 2

        for emo, cnt, pct in self.items:
            ei  = EMOTIONS.index(emo) if emo in EMOTIONS else 0
            col = EMO_COLORS[ei]
            bw  = bar * pct

            # label
            self.canv.setFont("Helvetica", 7)
            self.canv.setFillColor(TEXT)
            self.canv.drawString(0, y + 1.5, emo)

            # track background
            self.canv.setFillColor(CARD)
            self.canv.roundRect(self._lw, y, bar, self.BAR_H, 2, fill=1, stroke=0)

            # filled portion
            if bw > 2:
                self.canv.setFillColor(col)
                self.canv.roundRect(self._lw, y, bw, self.BAR_H, 2, fill=1, stroke=0)

            # count + percent
            self.canv.setFont("Helvetica-Bold", 6.5)
            self.canv.setFillColor(TEXT)
            self.canv.drawString(self._lw + bar + 3, y + 1.5,
                                 f"{pct*100:.0f}%  ({cnt})")

            y -= (self.BAR_H + self.GAP)


# ── stress timeline (annotated) ───────────────────────────────────────────────
class StressTimeline(Flowable):
    """Stress timeline with labeled zone bands, Y-axis, and threshold lines."""

    def __init__(self, stresses: list[float], width: float, height: float = 38*mm):
        super().__init__()
        self.stresses = stresses
        self.width    = width
        self.height   = height

    def draw(self):
        w, h   = float(self.width), float(self.height)
        ss     = self.stresses[-160:]
        n      = max(len(ss), 1)
        lpad   = 24    # Y-axis label space
        rpad   = 18    # right space for threshold labels
        bpad   = 14    # bottom space (X label)
        tpad   = 4
        cx     = lpad
        cy     = bpad
        cw     = w - lpad - rpad
        ch     = h - bpad - tpad

        # ── zone backgrounds ──────────────────────────────────────────────────
        zones = [
            (0.00, 0.45, ZONE_GREEN),
            (0.45, 0.70, ZONE_YELLOW),
            (0.70, 1.00, ZONE_RED),
        ]
        for lo, hi, bg in zones:
            y0 = cy + lo * ch
            y1 = cy + hi * ch
            self.canv.setFillColor(bg)
            self.canv.rect(cx, y0, cw, y1 - y0, fill=1, stroke=0)

        # ── zone border lines ─────────────────────────────────────────────────
        self.canv.setLineWidth(0.3)
        for thresh, col in [(0.45, YELLOW), (0.70, RED)]:
            y = cy + thresh * ch
            self.canv.setStrokeColor(col)
            self.canv.setDash([3, 3])
            self.canv.line(cx, y, cx + cw, y)
            self.canv.setDash([])
            # label on right
            self.canv.setFillColor(col)
            self.canv.setFont("Helvetica-Bold", 5.5)
            lbl = "MED" if thresh == 0.45 else "HIGH"
            self.canv.drawString(cx + cw + 2, y - 2, lbl)

        # ── bars ──────────────────────────────────────────────────────────────
        bw = cw / n
        for i, v in enumerate(ss):
            col = RED if v > 0.7 else YELLOW if v > 0.45 else GREEN
            bh  = max(1.5, ch * float(v))
            self.canv.setFillColor(col)
            self.canv.rect(cx + i * bw, cy, max(bw - 0.4, 0.4), bh, fill=1, stroke=0)

        # ── Y-axis ────────────────────────────────────────────────────────────
        self.canv.setStrokeColor(BORDER)
        self.canv.setLineWidth(0.4)
        self.canv.line(cx, cy, cx, cy + ch)
        self.canv.setLineWidth(0.2)
        for pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = cy + pct * ch
            self.canv.setStrokeColor(BORDER)
            self.canv.line(cx, y, cx + cw, y)
            self.canv.setFillColor(MUTED)
            self.canv.setFont("Helvetica", 5.5)
            self.canv.drawRightString(cx - 2, y - 2, f"{int(pct*100)}%")

        # ── X-axis bottom border ──────────────────────────────────────────────
        self.canv.setStrokeColor(BORDER)
        self.canv.setLineWidth(0.4)
        self.canv.line(cx, cy, cx + cw, cy)

        # ── X label ───────────────────────────────────────────────────────────
        self.canv.setFillColor(MUTED)
        self.canv.setFont("Helvetica", 5.5)
        self.canv.drawString(cx, 2, "Start")
        self.canv.drawRightString(cx + cw, 2, "End")
        self.canv.drawCentredString(cx + cw / 2, 2, f"{len(ss)} readings")

        # ── legend ────────────────────────────────────────────────────────────
        items = [("Low  (0–45%)", GREEN), ("Medium  (45–70%)", YELLOW), ("High  (>70%)", RED)]
        lx = cx
        ly = cy + ch + 2
        for lbl, col in items:
            self.canv.setFillColor(col)
            self.canv.rect(lx, ly, 7, 5, fill=1, stroke=0)
            self.canv.setFillColor(MUTED)
            self.canv.setFont("Helvetica", 5.5)
            self.canv.drawString(lx + 9, ly, lbl)
            lx += cw / 3


# ── meeting review (structured) ───────────────────────────────────────────────
def _build_review_table(summary: dict, participants: list[dict],
                        p_summaries: dict, S: dict, content_w: float) -> Table:
    dom  = summary.get("dominant_emotion", "Neutral")
    ms   = summary.get("mean_stress", 0)
    mv   = summary.get("mean_valence", 0)
    hsp  = summary.get("high_stress_pct", 0)
    n    = summary.get("total_readings", 0)
    dur  = summary.get("duration_min", 0)
    sl, sc = _stress_label(ms)

    mood_word = {
        "Happiness":      "positive and energetic",
        "Neutral":        "calm and composed",
        "Anger":          "tense and confrontational",
        "Sadness":        "subdued and low-energy",
        "Anxiety":        "anxious and unsettled",
        "Fear":           "fearful or uncertain",
        "Surprise":       "alert and reactive",
        "Disgust":        "uncomfortable or dissatisfied",
        "Contempt":       "dismissive in tone",
        "Helplessness":   "disengaged or overwhelmed",
        "Disappointment": "disappointed or deflated",
    }.get(dom, "mixed")

    valence_word = ("broadly positive" if mv > 0.1
                    else "broadly negative" if mv < -0.1 else "neutral")
    stress_desc  = ("high stress detected throughout" if ms > 0.7
                    else "moderate stress observed"   if ms > 0.45
                    else "group remained largely relaxed")

    bp = ParagraphStyle("bp", fontSize=8.5, textColor=TEXT, leading=13)
    bph = ParagraphStyle("bph", fontSize=8.5, textColor=ACCENT,
                         fontName="Helvetica-Bold", leading=13)

    def row(icon, text_para):
        return [Paragraph(icon, ParagraphStyle("ic", fontSize=9, textColor=ACCENT,
                                               leading=13, alignment=TA_CENTER)),
                text_para]

    names = ", ".join(p["name"] for p in participants) if participants else "—"

    rows = [
        row(">>",
            Paragraph(
                f"<b>Session:</b>  {len(participants)} participant(s) \u2014 {names}.  "
                f"{n} affective readings over approx. <b>{dur:.0f} min</b>.", bp)),
        row(">>",
            Paragraph(
                f"<b>Mood:</b>  Overall atmosphere was <b>{mood_word}</b> with dominant "
                f"emotion <b>{dom}</b>.  Valence {mv:+.2f} \u2014 {valence_word} tone.", bp)),
        row(">>",
            Paragraph(
                f"<b>Stress:</b>  Mean stress index <b>{ms:.0%}</b> \u2014 {stress_desc}.  "
                f"Stress level: <font color='{'#f85149' if sl=='High' else '#d29922' if sl=='Medium' else '#3fb950'}'>"
                f"<b>{sl}</b></font>.", bp)),
    ]
    if hsp > 25:
        rows.append(row("!",
            Paragraph(
                f"<b>Alert:</b>  <b>{hsp:.0f}%</b> of readings exceeded the high-stress "
                f"threshold (70%).  Consider workload review or follow-up.", bp)))

    highlights = []
    for p in participants:
        ps = p_summaries.get(p["id"])
        if ps:
            highlights.append(
                f"<b>{p['name']}</b>: dominant <i>{ps.get('dominant_emotion','—')}</i>, "
                f"stress {ps.get('mean_stress',0):.0%}")
    if highlights:
        rows.append(row(">>",
            Paragraph("<b>Individuals:</b>  " + "  \u00b7  ".join(highlights) + ".", bp)))

    col_w = [6*mm, content_w - 6*mm]
    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), CARD),
        ("BOX",           (0,0), (-1,-1), 0.5, BORDER),
        ("LINEBEFORE",    (0,0), (0,-1),  2.5, ACCENT),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (0,-1),  5),
        ("LEFTPADDING",   (1,0), (1,-1),  4),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("LINEBELOW",     (0,0), (-1,-2), 0.3, BORDER),
    ]))
    return t


# ── main export ───────────────────────────────────────────────────────────────
def export_pdf(session_id: int, out_path: str) -> None:
    from dashboard.database import get_session, get_participants, get_readings, \
        get_alerts, session_summary

    sess   = get_session(session_id)
    if not sess:
        raise ValueError(f"Session {session_id} not found")
    parts  = get_participants(session_id)
    summ   = session_summary(session_id)
    alerts = get_alerts(session_id)
    all_r  = get_readings(session_id, limit=10000)

    try:
        t0 = datetime.fromisoformat(sess["started_at"])
        t1 = datetime.fromisoformat(sess["ended_at"]) if sess.get("ended_at") else datetime.now()
        dur_min = (t1 - t0).total_seconds() / 60
    except Exception:
        dur_min = 0
    summ["duration_min"] = dur_min

    p_summaries: dict[int, dict] = {}
    p_readings:  dict[int, list] = {}
    for p in parts:
        pr = get_readings(session_id, p["id"], limit=10000)
        p_readings[p["id"]] = list(reversed(pr))
        if pr:
            sts = [r["stress"]  for r in pr]
            vls = [r["valence"] for r in pr]
            cnt = Counter(r["emotion"] for r in pr)
            p_summaries[p["id"]] = {
                "dominant_emotion": cnt.most_common(1)[0][0],
                "emotion_counts":   dict(cnt),
                "mean_stress":      sum(sts) / len(sts),
                "mean_valence":     sum(vls) / len(vls),
                "high_stress_pct":  sum(s > 0.7 for s in sts) / len(sts) * 100,
                "readings":         len(pr),
                "stresses":         sts,
            }

    S = _styles()
    w_page, h_page = A4
    margin    = 15 * mm
    content_w = w_page - 2 * margin

    doc = BaseDocTemplate(
        out_path, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=26*mm, bottomMargin=15*mm,
    )
    frame = Frame(margin, 15*mm, content_w, h_page - 41*mm, id="main")
    doc.addPageTemplates([PageTemplate(id="bg", frames=[frame], onPage=_add_page_bg)])

    story = []

    # ── TITLE ─────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 4*mm),
        Paragraph("Emotion Intelligence", S["label"]),
        Paragraph(sess["name"], S["title"]),
        HRFlowable(width="100%", thickness=0.5, color=ACCENT, spaceAfter=4),
    ]

    # ── META INFO ROW (6 cells across full width) ─────────────────────────────
    cell_w = content_w / 6
    meta = [
        ("Location",     sess.get("location") or "—"),
        ("Date",         t0.strftime("%d %b %Y") if dur_min else "—"),
        ("Start Time",   t0.strftime("%H:%M")    if dur_min else "—"),
        ("Duration",     f"{dur_min:.0f} min"),
        ("Participants", str(len(parts))),
        ("Readings",     str(summ.get("total_readings", 0))),
    ]

    def _mc(k, v):
        return Table(
            [[Paragraph(k, S["kv_key"])], [Paragraph(v, S["kv_val"])]],
            colWidths=[cell_w - 4],
        )

    meta_row = Table(
        [[_mc(k, v) for k, v in meta]],
        colWidths=[cell_w] * 6,
    )
    meta_row.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), CARD),
        ("BOX",           (0,0), (-1,-1), 0.5, BORDER),
        ("LINEAFTER",     (0,0), (-2,-1), 0.3, BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 5),
        ("RIGHTPADDING",  (0,0), (-1,-1), 5),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    story += [meta_row, Spacer(1, 4*mm)]

    # ── SESSION KPI CARDS ─────────────────────────────────────────────────────
    dom_emo = summ.get("dominant_emotion", "—")
    ms      = summ.get("mean_stress", 0)
    mv      = summ.get("mean_valence", 0)
    hsp     = summ.get("high_stress_pct", 0)
    sl, sc  = _stress_label(ms)
    cw5     = content_w / 5

    stat_row = Table([[
        _stat_card("Dominant Emotion", dom_emo,        ACCENT,                    cw5 - 3),
        _stat_card("Mean Stress",       f"{ms:.0%}",    sc,                        cw5 - 3),
        _stat_card("High-Stress %",     f"{hsp:.0f}%",  RED if hsp > 30 else MUTED, cw5 - 3),
        _stat_card("Mean Valence",      f"{mv:+.2f}",   GREEN if mv > 0 else RED,   cw5 - 3),
        _stat_card("Stress Level",      sl,             sc,                        cw5 - 3),
    ]], colWidths=[cw5] * 5)
    stat_row.setStyle(TableStyle([
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("LEFTPADDING",  (0,0), (-1,-1), 2),
        ("RIGHTPADDING", (0,0), (-1,-1), 2),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
    ]))
    story += [stat_row, Spacer(1, 5*mm)]

    # ── MEETING REVIEW ────────────────────────────────────────────────────────
    story.append(Paragraph("Meeting Review", S["section"]))
    story.append(_build_review_table(summ, parts, p_summaries, S, content_w))
    story.append(Spacer(1, 5*mm))

    # ── EMOTION DISTRIBUTION (horizontal bars) ────────────────────────────────
    story.append(Paragraph("Emotion Distribution", S["section"]))
    all_counts = Counter(r["emotion"] for r in all_r)
    if all_counts:
        story.append(EmotionBars(all_counts, width=content_w))
    else:
        story.append(Paragraph("No emotion data recorded.", S["body_sm"]))
    story.append(Spacer(1, 5*mm))

    # ── STRESS TIMELINE ───────────────────────────────────────────────────────
    story.append(Paragraph("Stress Timeline", S["section"]))
    all_stresses = [r["stress"] for r in reversed(all_r)]
    if all_stresses:
        story.append(StressTimeline(all_stresses, width=content_w, height=40*mm))
    else:
        story.append(Paragraph("No stress data recorded.", S["body_sm"]))
    story.append(Spacer(1, 5*mm))

    # ── ALERTS ────────────────────────────────────────────────────────────────
    if alerts:
        story.append(Paragraph(f"Stress Alerts  ({len(alerts)})", S["section"]))
        p_names = {p["id"]: p["name"] for p in parts}
        alert_rows = [[
            Paragraph("Time",        S["label"]),
            Paragraph("Participant", S["label"]),
            Paragraph("Message",     S["label"]),
        ]]
        for a in alerts[:20]:
            ts = a["timestamp"][:19].replace("T", "  ")
            alert_rows.append([
                Paragraph(ts,                                      S["body_sm"]),
                Paragraph(p_names.get(a["participant_id"],"Group"),S["body_sm"]),
                Paragraph(a["message"],                            S["body_sm"]),
            ])
        at = Table(alert_rows, colWidths=[38*mm, 38*mm, content_w - 76*mm])
        at.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1, 0), CARD),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.transparent,
                                               colors.HexColor("#160a0a")]),
            ("BOX",           (0,0), (-1,-1), 0.5, BORDER),
            ("LINEBELOW",     (0,0), (-1,-1), 0.3, BORDER),
            ("LINEBEFORE",    (0,0), (0,-1),  2.5, RED),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ]))
        story += [at, Spacer(1, 5*mm)]

    # ── PER-PARTICIPANT SECTIONS ──────────────────────────────────────────────
    if parts:
        story += [PageBreak(), Paragraph("Individual Analysis", S["section"]),
                  HRFlowable(width="100%", thickness=0.5, color=ACCENT, spaceAfter=5)]

    for p in parts:
        ps  = p_summaries.get(p["id"])
        pr  = p_readings.get(p["id"], [])

        pdom = ps["dominant_emotion"] if ps else "—"
        pms  = ps["mean_stress"]      if ps else 0.0
        pmv  = ps["mean_valence"]     if ps else 0.0
        phsp = ps["high_stress_pct"]  if ps else 0.0
        psl, psc = _stress_label(pms)
        pei  = EMOTIONS.index(pdom) if pdom in EMOTIONS else 0

        # participant name band
        name_band = Table(
            [[Paragraph(p["name"], S["p_name"]),
              Paragraph(
                  (p.get("role") or "Participant") +
                  (f"  \u00b7  {p.get('department','')}" if p.get("department") else ""),
                  ParagraphStyle("prole", fontSize=7.5, textColor=MUTED,
                                 leading=10, alignment=TA_RIGHT)
              )]],
            colWidths=[content_w * 0.6, content_w * 0.4],
        )
        name_band.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), CARD2),
            ("BOX",           (0,0), (-1,-1), 0.5, BORDER),
            ("LINEBEFORE",    (0,0), (0,-1),  3, EMO_COLORS[pei]),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (0,-1),  8),
            ("RIGHTPADDING",  (1,0), (1,-1),  8),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        story += [name_band, Spacer(1, 3*mm)]

        if not ps:
            story.append(Paragraph("No readings recorded for this participant.", S["body_sm"]))
        else:
            # KPI cards
            pstat = Table([[
                _stat_card("Dominant",    pdom,            EMO_COLORS[pei], cw5 - 3),
                _stat_card("Mean Stress", f"{pms:.0%}",    psc,             cw5 - 3),
                _stat_card("High Stress", f"{phsp:.0f}%",  RED if phsp > 30 else MUTED, cw5 - 3),
                _stat_card("Valence",     f"{pmv:+.2f}",   GREEN if pmv > 0 else RED, cw5 - 3),
                _stat_card("Readings",    str(ps["readings"]), ACCENT, cw5 - 3),
            ]], colWidths=[cw5] * 5)
            pstat.setStyle(TableStyle([
                ("ALIGN",        (0,0), (-1,-1), "CENTER"),
                ("LEFTPADDING",  (0,0), (-1,-1), 2),
                ("RIGHTPADDING", (0,0), (-1,-1), 2),
                ("TOPPADDING",   (0,0), (-1,-1), 0),
                ("BOTTOMPADDING",(0,0), (-1,-1), 0),
            ]))
            story += [pstat, Spacer(1, 4*mm)]

            # personal stress timeline
            if ps["stresses"]:
                story.append(Paragraph("Stress Timeline", S["section"]))
                story.append(StressTimeline(ps["stresses"], width=content_w, height=30*mm))
                story.append(Spacer(1, 4*mm))

            # emotion breakdown
            story.append(Paragraph("Emotion Breakdown", S["section"]))
            story.append(EmotionBars(ps["emotion_counts"], width=content_w))
            story.append(Spacer(1, 4*mm))

            # recent readings table
            story.append(Paragraph("Recent Readings  (last 15)", S["section"]))
            rr_rows = [[
                Paragraph("Time",    S["label"]),
                Paragraph("Emotion", S["label"]),
                Paragraph("Stress",  S["label"]),
                Paragraph("Valence", S["label"]),
                Paragraph("Arousal", S["label"]),
            ]]
            for r in pr[-15:]:
                ts2     = r["timestamp"][11:19]
                sl2, sc2 = _stress_label(r["stress"])
                rr_rows.append([
                    Paragraph(ts2,                    S["body_sm"]),
                    Paragraph(r["emotion"],            S["body_sm"]),
                    Paragraph(f"{r['stress']:.0%}",
                              ParagraphStyle("sv2", fontSize=7.5, textColor=sc2)),
                    Paragraph(f"{r['valence']:+.2f}",  S["body_sm"]),
                    Paragraph(f"{r['arousal']:.2f}",   S["body_sm"]),
                ])
            rrt = Table(rr_rows,
                        colWidths=[28*mm, 38*mm, 24*mm, 24*mm, content_w - 114*mm])
            rrt.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1, 0), CARD),
                ("ROWBACKGROUNDS",(0,1), (-1,-1),
                 [colors.transparent, colors.HexColor("#161b22")]),
                ("BOX",           (0,0), (-1,-1), 0.5, BORDER),
                ("LINEBELOW",     (0,0), (-1,-1), 0.3, BORDER),
                ("TOPPADDING",    (0,0), (-1,-1), 3),
                ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ]))
            story += [rrt, Spacer(1, 4*mm)]

        story.append(HRFlowable(width="100%", thickness=0.3,
                                color=BORDER, spaceAfter=5, spaceBefore=3))

    doc.build(story)
