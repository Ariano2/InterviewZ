"""
agents/resume_maker.py
AI assistance for the Make My Resume builder.
  - enhance_bullets  : rough notes → 3 strong resume bullets
  - generate_summary : name + role + skills → professional summary
  - render_resume_html : resume data dict → styled HTML string for live preview
"""
import json
import re
from html import escape
from groq import Groq

GROQ_MODEL = "openai/gpt-oss-120b"


# ── AI: Bullet Enhancement ────────────────────────────────────────────────────

def enhance_bullets(
    role: str,
    company_or_project: str,
    description: str,
    client: Groq,
    model: str = GROQ_MODEL,
) -> list:
    """
    Takes rough notes about a role/project and returns 3 strong resume bullets.
    Each bullet: strong action verb + what + measurable impact.
    Raises RuntimeError on failure so the caller can display the real reason.
    """
    prompt = f"""You are an expert resume writer for software/tech roles in India.

Role / Project: {role}
Company / Tech Stack: {company_or_project}
What I did (rough notes): {description}

Write exactly 3 strong, ATS-optimised resume bullet points.
Rules:
- Start each with a DISTINCT strong action verb (e.g. Engineered, Reduced, Designed, Automated, Shipped)
- Weave in the technologies or tools mentioned
- Add realistic quantified impact (%, x faster, N users, ms latency, etc.) where sensible
- Keep each bullet under 20 words
- No bullet symbols or numbering — plain text only

Return ONLY a valid JSON array of exactly 3 strings. No markdown, no explanation.
Example: ["Built X that achieved Y using Z", "Reduced latency by 40% via cache layer", "Led migration of A to B cutting infra cost by 30%"]"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=600,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())
        # Extract JSON array even when model wraps it in prose
        match = re.search(r'\[[\s\S]*\]', raw)
        if match:
            raw = match.group(0)
        bullets = json.loads(raw)
        if isinstance(bullets, list):
            return [str(b).strip() for b in bullets if b]
        raise ValueError("Response was not a list")
    except Exception as e:
        raise RuntimeError(f"Bullet generation failed: {e}") from e


# ── AI: Summary Generation ────────────────────────────────────────────────────

def generate_summary(
    name: str,
    target_role: str,
    experience_titles: list,
    skills_flat: list,
    client: Groq,
    model: str = GROQ_MODEL,
) -> str:
    """
    Generates a 2-3 sentence professional summary for the resume header.
    Raises RuntimeError on failure.
    """
    exp_text    = ", ".join(experience_titles[:3]) if experience_titles else "various technical projects"
    skills_text = ", ".join(skills_flat[:12])      if skills_flat        else "multiple technologies"

    prompt = f"""Write a professional resume summary for {name or 'a candidate'} targeting a {target_role or 'software engineer'} role.

Recent experience: {exp_text}
Key skills: {skills_text}

Requirements:
- 2-3 sentences only, no more
- Third-person tone — do NOT use "I"
- Highlight expertise, key technologies, and impact
- End with the value they bring to a team

Return ONLY the summary paragraph. No quotes, no labels."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=180,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Summary generation failed: {e}") from e


# ── HTML Preview Renderer ─────────────────────────────────────────────────────

def render_resume_html(data: dict) -> str:
    """
    Renders the resume data dict as a styled, self-contained HTML document
    suitable for embedding in a Streamlit components.html() iframe.
    Empty slots are skipped. A placeholder is shown if the resume is blank.
    """

    def _e(text) -> str:
        return escape(str(text or "").strip())

    def _sep(items, sep=" · ") -> str:
        return sep.join(i for i in items if i)

    parts = []

    # ── Header ────────────────────────────────────────────────────────────
    name_val = _e(data.get("name"))
    if name_val:
        parts.append(f'<div class="name">{name_val}</div>')
    else:
        parts.append('<div class="name" style="color:#b0b4c4;">Your Name</div>')

    r1 = _sep([_e(data.get("email")), _e(data.get("phone")), _e(data.get("location"))])
    if r1:
        parts.append(f'<div class="contact-row">{r1}</div>')

    r2 = _sep([_e(data.get("linkedin")), _e(data.get("github"))])
    if r2:
        parts.append(f'<div class="contact-row">{r2}</div>')

    parts.append('<hr class="rule-blue">')

    # ── Summary ───────────────────────────────────────────────────────────
    if (data.get("summary") or "").strip():
        parts.append(f'<div class="summary">{_e(data["summary"])}</div>')
        parts.append('<hr class="rule-light">')

    # ── Education ─────────────────────────────────────────────────────────
    edu_list = [e for e in (data.get("education") or []) if e.get("degree") or e.get("institution")]
    if edu_list:
        parts.append('<div class="section-hdr">Education</div><hr class="rule-light">')
        for edu in edu_list:
            deg   = _e(edu.get("degree"))
            inst  = _e(edu.get("institution"))
            loc   = _e(edu.get("location"))
            dates = _e(edu.get("dates"))
            gpa   = _e(edu.get("gpa"))
            right = _sep([dates, f"GPA: {gpa}" if gpa else ""])
            left  = deg or inst
            parts.append(f'<div class="entry"><div class="entry-top"><span class="entry-title">{left}</span><span class="entry-date">{right}</span></div>')
            sub = _sep([inst if deg else "", loc], ", ")
            if sub:
                parts.append(f'<div class="entry-sub">{sub}</div>')
            for ach_line in (edu.get("achievements") or "").split("\n"):
                if ach_line.strip():
                    parts.append(f'<div class="bullet">{_e(ach_line.strip())}</div>')
            parts.append('</div>')

    # ── Experience ────────────────────────────────────────────────────────
    exp_list = [e for e in (data.get("experience") or []) if e.get("title")]
    if exp_list:
        parts.append('<div class="section-hdr">Experience</div><hr class="rule-light">')
        for exp in exp_list:
            title   = _e(exp.get("title"))
            company = _e(exp.get("company"))
            loc     = _e(exp.get("location"))
            dates   = _e(exp.get("dates"))
            parts.append(f'<div class="entry"><div class="entry-top"><span class="entry-title">{title}</span><span class="entry-date">{dates}</span></div>')
            sub = _sep([company, loc], ", ")
            if sub:
                parts.append(f'<div class="entry-sub">{sub}</div>')
            for b in (exp.get("bullets") or []):
                if b.strip():
                    parts.append(f'<div class="bullet">{_e(b)}</div>')
            parts.append('</div>')

    # ── Projects ──────────────────────────────────────────────────────────
    proj_list = [p for p in (data.get("projects") or []) if p.get("name")]
    if proj_list:
        parts.append('<div class="section-hdr">Projects</div><hr class="rule-light">')
        for proj in proj_list:
            pname = _e(proj.get("name"))
            tech  = _e(proj.get("tech"))
            dates = _e(proj.get("dates"))
            link  = _e(proj.get("link"))
            tech_html = f' <span class="tech">| {tech}</span>' if tech else ""
            right = _sep([dates, link])
            parts.append(f'<div class="entry"><div class="entry-top"><span class="entry-title">{pname}{tech_html}</span><span class="entry-date">{right}</span></div>')
            for b in (proj.get("bullets") or []):
                if b.strip():
                    parts.append(f'<div class="bullet">{_e(b)}</div>')
            parts.append('</div>')

    # ── Skills ────────────────────────────────────────────────────────────
    sk = data.get("skills") or {}
    sk_rows = [
        ("Languages",             sk.get("languages", "")),
        ("Frameworks & Libraries", sk.get("frameworks", "")),
        ("Tools & Platforms",     sk.get("tools", "")),
        ("Other",                 sk.get("other", "")),
    ]
    sk_rows = [(lbl, val) for lbl, val in sk_rows if (val or "").strip()]
    if sk_rows:
        parts.append('<div class="section-hdr">Technical Skills</div><hr class="rule-light">')
        for lbl, val in sk_rows:
            parts.append(f'<div class="skills-row"><span class="sk-label">{lbl}:</span>  {_e(val)}</div>')

    # ── Certifications ────────────────────────────────────────────────────
    certs = (data.get("certifications") or "").strip()
    if certs:
        parts.append('<div class="section-hdr">Certifications</div><hr class="rule-light">')
        for c in certs.split("\n"):
            if c.strip():
                parts.append(f'<div class="bullet">{_e(c.strip())}</div>')

    # ── Achievements ──────────────────────────────────────────────────────
    ach = (data.get("achievements") or "").strip()
    if ach:
        parts.append('<div class="section-hdr">Achievements &amp; Awards</div><hr class="rule-light">')
        for a in ach.split("\n"):
            if a.strip():
                parts.append(f'<div class="bullet">{_e(a.strip())}</div>')

    # ── Placeholder if empty ──────────────────────────────────────────────
    has_content = edu_list or exp_list or proj_list or sk_rows or certs or ach or (data.get("summary") or "").strip()
    if not has_content:
        parts.append('<div class="placeholder-block">👈 Fill in your details on the left to see your resume here live</div>')

    content = "\n".join(parts)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  background: #eef0f7;
  padding: 16px 10px 24px;
}}
.page {{
  background: white;
  max-width: 640px;
  margin: 0 auto;
  padding: 28px 32px 28px;
  color: #1a1a2e;
  font-size: 10.5px;
  line-height: 1.55;
  box-shadow: 0 4px 24px rgba(0,0,0,0.10);
  border-radius: 2px;
}}
.name {{
  text-align: center;
  font-size: 21px;
  font-weight: 700;
  color: #1a1a2e;
  letter-spacing: -0.3px;
  margin-bottom: 4px;
}}
.contact-row {{
  text-align: center;
  color: #4a4e6a;
  font-size: 9px;
  margin-bottom: 2px;
}}
.rule-blue {{
  height: 1.5px;
  background: #4f6ef7;
  border: none;
  margin: 8px 0 6px;
}}
.rule-light {{
  height: 0.75px;
  background: #d0d4e8;
  border: none;
  margin: 3px 0 5px;
}}
.summary {{
  color: #4a4e6a;
  font-size: 9.5px;
  margin-bottom: 6px;
  line-height: 1.65;
}}
.section-hdr {{
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: #1a1a2e;
  margin-top: 10px;
  margin-bottom: 1px;
}}
.entry {{ margin-bottom: 7px; }}
.entry-top {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
}}
.entry-title {{ font-size: 10px; font-weight: 700; color: #1a1a2e; flex: 1; }}
.entry-date {{ font-size: 9px; color: #4a4e6a; white-space: nowrap; }}
.entry-sub {{ font-size: 9px; color: #4a4e6a; font-style: italic; margin-bottom: 2px; }}
.tech {{ color: #4f6ef7; font-weight: 400; font-style: normal; font-size: 9px; }}
.bullet {{
  font-size: 9.5px;
  color: #2c2c40;
  padding-left: 14px;
  margin-bottom: 1.5px;
  position: relative;
}}
.bullet::before {{ content: "–  "; color: #4f6ef7; font-weight: 700; }}
.skills-row {{ font-size: 9.5px; color: #2c2c40; margin-bottom: 2.5px; padding-left: 4px; }}
.sk-label {{ font-weight: 700; color: #1a1a2e; }}
.placeholder-block {{
  background: #f5f6fa;
  border: 1.5px dashed #d0d4e8;
  border-radius: 6px;
  padding: 24px;
  text-align: center;
  color: #8890a4;
  font-size: 11px;
  margin: 24px 0;
  line-height: 1.8;
}}
</style>
</head>
<body>
<div class="page">
{content}
</div>
</body>
</html>"""
