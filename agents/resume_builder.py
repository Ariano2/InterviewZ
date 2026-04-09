"""
agents/resume_builder.py
Generates a clean, ATS-friendly single-page PDF resume from structured data.
Uses reportlab — pure Python, zero system dependencies.
"""
import copy
import io
from difflib import SequenceMatcher
from typing import Dict, List

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Unicode → cp1252 sanitizer ────────────────────────────────────────────────
# Helvetica in reportlab uses WinAnsiEncoding (cp1252). Any character outside
# that range renders as a black rectangle. Map the common offenders first,
# then hard-drop anything else that still won't encode.
_UNICODE_REPLACEMENTS = {
    "\u2014": "-",    # em dash
    "\u2013": "-",    # en dash
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
    "\u2022": "-",    # bullet (we prepend our own)
    "\u25cf": "-",    # black circle bullet
    "\u25aa": "-",    # black small square
    "\u25ba": ">",    # right-pointing pointer
    "\u25b8": ">",    # small right-pointing triangle
    "\u2026": "...",  # ellipsis
    "\u00a0": " ",    # non-breaking space
    "\u2192": "->",   # right arrow
    "\u2190": "<-",   # left arrow
    "\u00ae": "(R)",  # registered trademark
    "\u2122": "(TM)", # trademark
    "\u00a9": "(C)",  # copyright
    "\u2665": "",     # heart
    "\u00b7": ".",    # middle dot
    "\u0026": "&amp;",# ampersand — must be XML-escaped for Paragraph
    "\u003c": "&lt;", # less-than  — XML-escape
    "\u003e": "&gt;", # greater-than — XML-escape
}


def _safe(text: str) -> str:
    """
    Sanitize a string for use in a reportlab Paragraph with Helvetica.
    1. Replaces known problematic Unicode chars with ASCII equivalents.
    2. Drops anything still outside cp1252.
    """
    if not text:
        return ""
    for char, rep in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, rep)
    # Drop remaining chars that can't encode to cp1252
    text = text.encode("cp1252", errors="ignore").decode("cp1252")
    return text

# ── Colours ───────────────────────────────────────────────────────────────────
C_DARK = HexColor("#1a1a2e")
C_BLUE = HexColor("#4f6ef7")
C_GRAY = HexColor("#4a4e6a")
C_LGRAY = HexColor("#8890a4")
C_LINE = HexColor("#d0d4e8")
C_BODY = HexColor("#2c2c40")


# ── Style factory ─────────────────────────────────────────────────────────────
def _styles() -> Dict[str, ParagraphStyle]:
    base = dict(fontName="Helvetica", fontSize=9, textColor=C_BODY, leading=13)
    return {
        "name": ParagraphStyle(
            "name",
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=26,          # explicit — prevents squishing into contact row
            textColor=C_DARK,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "contact": ParagraphStyle(
            "contact",
            fontName="Helvetica",
            fontSize=8.5,
            leading=14,
            textColor=C_GRAY,
            alignment=TA_CENTER,
            spaceBefore=0,
            spaceAfter=3,
        ),
        "section": ParagraphStyle(
            "section",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=C_DARK,
            spaceBefore=6,
            spaceAfter=2,
            tracking=60,        # letter-spacing ~= LaTeX feel
        ),
        "item_title": ParagraphStyle(
            "item_title",
            fontName="Helvetica-Bold",
            fontSize=9.5,
            textColor=C_DARK,
            spaceAfter=0,
            leading=13,
        ),
        "item_sub": ParagraphStyle(
            "item_sub",
            fontName="Helvetica-Oblique",
            fontSize=9,
            textColor=C_GRAY,
            spaceAfter=2,
            leading=12,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            **base,
            leftIndent=10,
            spaceAfter=2,
        ),
        "skills_row": ParagraphStyle(
            "skills_row",
            **base,
            leftIndent=4,
            spaceAfter=2.5,
        ),
        "summary": ParagraphStyle(
            "summary",
            fontSize=9.5,
            textColor=C_GRAY,
            spaceAfter=4,
            leading=14,
        ),
        "date_right": ParagraphStyle(
            "date_right",
            fontSize=9,
            textColor=C_GRAY,
            alignment=TA_RIGHT,
            leading=13,
        ),
    }


# ── Fuzzy bullet matching ─────────────────────────────────────────────────────
def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _apply_rewrites(data: dict, rewrites: List[Dict]) -> dict:
    """
    Substitute or remove bullets based on rewrite action.
    action="rewrite" → replace with improved text
    action="remove"  → drop bullet entirely
    """
    data = copy.deepcopy(data)
    for rw in rewrites:
        original = (rw.get("original") or "").strip()
        action   = (rw.get("action") or "rewrite").lower()
        improved = (rw.get("improved") or "").strip()
        if not original:
            continue

        for section in ("experience", "projects", "education"):
            for item in data.get(section, []):
                bullets = item.get("bullets") or []
                new_bullets = []
                for b in bullets:
                    if _similarity(b, original) >= 0.55:
                        if action == "remove":
                            pass  # drop it
                        else:
                            new_bullets.append(improved or b)
                    else:
                        new_bullets.append(b)
                item["bullets"] = new_bullets

        new_ach = []
        for a in data.get("achievements") or []:
            if _similarity(a, original) >= 0.55:
                if action != "remove":
                    new_ach.append(improved or a)
            else:
                new_ach.append(a)
        data["achievements"] = new_ach

    return data


# ── PDF builder ───────────────────────────────────────────────────────────────
def build_resume_pdf(resume_data: dict, rewrites: List[Dict] = None) -> bytes:
    """
    Build a professional PDF resume.
    rewrites: list of {original, improved} dicts from bullet_rewriter.
    Returns raw PDF bytes.
    """
    data = _apply_rewrites(resume_data, rewrites or [])
    S = _styles()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    name = _safe((data.get("name") or "Your Name").strip())
    story.append(Paragraph(name, S["name"]))

    # Row 1: email · phone · location
    row1 = [_safe(data.get(k) or "") for k in ("email", "phone", "location")]
    row1 = [v for v in row1 if v]
    # Row 2: linkedin · github · website
    row2 = [_safe(data.get(k) or "") for k in ("linkedin", "github", "website")]
    row2 = [v for v in row2 if v]

    sep = "  |  "
    if row1:
        story.append(Paragraph(sep.join(row1), S["contact"]))
    if row2:
        story.append(Paragraph(sep.join(row2), S["contact"]))

    story.append(
        HRFlowable(width="100%", thickness=1.5, color=C_BLUE, spaceAfter=5)
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    if data.get("summary"):
        story.append(Paragraph(_safe(data["summary"]), S["summary"]))
        story.append(
            HRFlowable(width="100%", thickness=0.5, color=C_LINE, spaceAfter=4)
        )

    # ── Helpers ───────────────────────────────────────────────────────────────
    _tbl_style = TableStyle(
        [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]
    )

    def _section_hdr(title: str):
        story.append(Paragraph(_safe(title.upper()), S["section"]))
        story.append(
            HRFlowable(width="100%", thickness=0.75, color=C_LINE, spaceAfter=3)
        )

    def _title_row(left_html: str, right_text: str):
        # left_html already contains <b> tags — sanitize only the text nodes
        tbl = Table(
            [
                [
                    Paragraph(left_html, S["item_title"]),
                    Paragraph(_safe(right_text), S["date_right"]),
                ]
            ],
            colWidths=["68%", "32%"],
        )
        tbl.setStyle(_tbl_style)
        story.append(tbl)

    def _bullets(items):
        for b in (items or []):
            b = _safe((b or "").strip()).lstrip("*->) ")
            # strip any residual bullet-like leading chars after sanitization
            b = b.lstrip(".-># ")
            if b:
                story.append(Paragraph(f"- {_safe(b)}", S["bullet"]))

    # ── Education ─────────────────────────────────────────────────────────────
    edu_list = data.get("education") or []
    if edu_list:
        _section_hdr("Education")
        for edu in edu_list:
            degree = _safe(edu.get("degree") or "")
            inst   = _safe(edu.get("institution") or "")
            loc    = _safe(edu.get("location") or "")
            dates  = _safe(edu.get("dates") or "")
            gpa    = _safe(edu.get("gpa") or "")

            right_parts = [p for p in [dates, f"GPA: {gpa}" if gpa else ""] if p]
            _title_row(f"<b>{degree}</b>", "  |  ".join(right_parts))

            sub_parts = [p for p in [inst, loc] if p]
            if sub_parts:
                story.append(Paragraph(", ".join(sub_parts), S["item_sub"]))
            _bullets(edu.get("bullets"))
            story.append(Spacer(1, 3))

    # ── Experience ────────────────────────────────────────────────────────────
    exp_list = data.get("experience") or []
    if exp_list:
        _section_hdr("Experience")
        for exp in exp_list:
            title   = _safe(exp.get("title") or "")
            company = _safe(exp.get("company") or "")
            loc     = _safe(exp.get("location") or "")
            dates   = _safe(exp.get("dates") or "")

            _title_row(f"<b>{title}</b>", dates)
            sub_parts = [p for p in [company, loc] if p]
            if sub_parts:
                story.append(Paragraph(", ".join(sub_parts), S["item_sub"]))
            _bullets(exp.get("bullets"))
            story.append(Spacer(1, 3))

    # ── Projects ──────────────────────────────────────────────────────────────
    proj_list = data.get("projects") or []
    if proj_list:
        _section_hdr("Projects")
        for proj in proj_list:
            pname = _safe(proj.get("name") or "")
            tech  = _safe(proj.get("tech") or "")
            dates = _safe(proj.get("dates") or "")
            link  = _safe(proj.get("link") or "")

            tech_html = (
                f' <font color="#4f6ef7">| {tech}</font>' if tech else ""
            )
            right_parts = [p for p in [dates, link] if p]
            _title_row(
                f"<b>{pname}</b>{tech_html}",
                "  |  ".join(right_parts),
            )
            _bullets(proj.get("bullets"))
            story.append(Spacer(1, 3))

    # ── Skills ────────────────────────────────────────────────────────────────
    skills = data.get("skills") or {}
    skill_labels = {
        "languages": "Languages",
        "frameworks": "Frameworks &amp; Libraries",
        "tools": "Tools &amp; Platforms",
        "other": "Other",
    }
    skill_rows = [
        (label, [_safe(i) for i in skills[key]])
        for key, label in skill_labels.items()
        if skills.get(key)
    ]
    if skill_rows:
        _section_hdr("Technical Skills")
        for label, items in skill_rows:
            story.append(
                Paragraph(
                    f"<b>{label}:</b>  {',  '.join(items)}",
                    S["skills_row"],
                )
            )

    # ── Certifications ────────────────────────────────────────────────────────
    certs = data.get("certifications") or []
    if certs:
        _section_hdr("Certifications")
        for c in certs:
            story.append(Paragraph(f"- {_safe(c)}", S["bullet"]))

    # ── Achievements ──────────────────────────────────────────────────────────
    achievements = data.get("achievements") or []
    if achievements:
        _section_hdr("Achievements &amp; Awards")
        for a in achievements:
            story.append(Paragraph(f"- {_safe(a)}", S["bullet"]))

    # ── Extra sections ────────────────────────────────────────────────────────
    for section in data.get("extra_sections") or []:
        sec_title = _safe(section.get("title") or "Other")
        items = section.get("items") or []
        if items:
            _section_hdr(sec_title)
            for item in items:
                story.append(Paragraph(f"- {_safe(item)}", S["bullet"]))

    doc.build(story)
    return buf.getvalue()
