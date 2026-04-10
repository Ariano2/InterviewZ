"""
agents/portfolio_generator.py
Generates a portfolio website from resume structure.

Pipeline:
1. Detect missing sections → inject dummy placeholders + flag them
2. Single LLM call → polished tagline, about paragraph, enhanced project descriptions
3. Python fills HTML template placeholders — LLM never touches HTML structure
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from groq import Groq

GROQ_MODEL     = "openai/gpt-oss-120b"
TEMPLATES_DIR  = Path(__file__).parent.parent / "templates" / "portfolio"

# ── Dummy placeholder data ─────────────────────────────────────────────────────

_DUMMY_PROJECT = {
    "name":    "Your Project Title",
    "tech":    "Python · React · PostgreSQL",
    "dates":   "2024",
    "link":    None,
    "bullets": [
        "Describe what you built and the problem it solved",
        "Quantify your impact — users reached, performance gains, time saved",
    ],
    "_is_dummy": True,
}

_DUMMY_EXPERIENCE = {
    "title":   "Your Role / Internship Title",
    "company": "Company Name",
    "location": "City / Remote",
    "dates":   "Month Year – Month Year",
    "bullets": [
        "Describe your key responsibilities and what you owned",
        "Include a metric or outcome wherever possible",
    ],
    "_is_dummy": True,
}

_DUMMY_SKILLS = {
    "languages":  ["Python", "JavaScript"],
    "frameworks": ["React", "Node.js"],
    "tools":      ["Git", "Docker", "Linux"],
    "other":      ["REST APIs", "SQL"],
    "_is_dummy": True,
}


# ── Section detection ──────────────────────────────────────────────────────────

def _detect_and_fill_dummies(structure: dict) -> Tuple[dict, List[str]]:
    """
    Checks each key section. Injects dummy data where empty.
    Returns (enriched_structure, list_of_dummy_section_names).
    """
    import copy
    s = copy.deepcopy(structure)
    dummy_sections: List[str] = []

    if not s.get("projects"):
        s["projects"] = [_DUMMY_PROJECT]
        dummy_sections.append("Projects")

    if not s.get("experience"):
        s["experience"] = [_DUMMY_EXPERIENCE]
        dummy_sections.append("Experience")

    skills = s.get("skills") or {}
    all_skills = (
        (skills.get("languages") or []) +
        (skills.get("frameworks") or []) +
        (skills.get("tools") or []) +
        (skills.get("other") or [])
    )
    if not all_skills:
        s["skills"] = _DUMMY_SKILLS
        dummy_sections.append("Skills")

    if not s.get("name"):
        s["name"] = "Your Name"
        dummy_sections.append("Name")

    return s, dummy_sections


# ── LLM content enhancement ────────────────────────────────────────────────────

def _enhance_content(structure: dict, target_role: str, client: Groq) -> dict:
    """
    Single LLM call. Generates:
    - tagline: punchy 6-10 word phrase
    - about: 3-sentence professional bio
    - typing_roles: 3 role alternatives for the typing animation
    - project_enhancements: {original_name: enhanced_description} for real (non-dummy) projects
    """
    name = structure.get("name") or "the candidate"
    edu  = structure.get("education") or []
    edu_str = ", ".join(
        f"{e.get('degree','?')} at {e.get('institution','?')}"
        for e in edu[:2]
    )

    real_projects = [
        p for p in (structure.get("projects") or [])
        if not p.get("_is_dummy")
    ]
    proj_str = "\n".join(
        f"- {p.get('name','?')} ({p.get('tech','')}): {'; '.join((p.get('bullets') or [])[:2])}"
        for p in real_projects[:3]
    )

    exp = structure.get("experience") or []
    exp_str = ", ".join(
        f"{e.get('title','?')} at {e.get('company','?')}"
        for e in exp[:2]
        if not e.get("_is_dummy")
    )

    skills = structure.get("skills") or {}
    top_skills = ", ".join(
        (skills.get("languages") or [])[:4] +
        (skills.get("frameworks") or [])[:3]
    )

    prompt = f"""Create portfolio website content for {name}.
Target role: {target_role or "Software Developer"}
Education: {edu_str or "Not specified"}
Experience: {exp_str or "Not specified"}
Top skills: {top_skills or "Not specified"}
Projects:
{proj_str or "None listed"}

Return ONLY valid JSON:
{{
  "tagline": "<punchy 6-10 word phrase describing what they build/do>",
  "about": "<exactly 3 sentences: who they are, what they build, what they're looking for. Grounded in the facts above. Professional but human tone.>",
  "typing_roles": ["<role 1>", "<role 2>", "<role 3>"],
  "project_enhancements": {{
    "<exact project name>": "<1 punchy sentence enhancement with impact/metric>"
  }}
}}"""

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=800,
        )
        raw = resp.choices[0].message.content or ""
        raw = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```\s*$", "", raw).strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass

    # Fallback
    return {
        "tagline": f"Building great software as a {target_role or 'Developer'}",
        "about": f"{name} is a developer with experience in {top_skills or 'software development'}. They build reliable, user-focused applications. Currently seeking opportunities as a {target_role or 'Software Developer'}.",
        "typing_roles": [target_role or "Software Developer", "Problem Solver", "Builder"],
        "project_enhancements": {},
    }


# ── HTML snippet generators ────────────────────────────────────────────────────

def _skill_chips_html(skills: dict) -> str:
    is_dummy = skills.get("_is_dummy", False)
    all_skills = (
        (skills.get("languages") or []) +
        (skills.get("frameworks") or []) +
        (skills.get("tools") or []) +
        (skills.get("other") or [])
    )
    chips = "\n".join(
        f'<span class="skill-chip">{s}</span>'
        for s in all_skills if s
    )
    dummy_attr = ' data-dummy="true"' if is_dummy else ""
    return f'<div class="skills-chips"{dummy_attr}>\n{chips}\n</div>'


def _projects_html(projects: list, enhancements: dict) -> str:
    cards = []
    for p in projects[:6]:
        name      = p.get("name", "")
        tech      = p.get("tech", "")
        link      = p.get("link") or ""
        bullets   = p.get("bullets") or []
        is_dummy  = p.get("_is_dummy", False)

        # Use LLM enhancement if available, else first bullet
        description = enhancements.get(name) or (bullets[0] if bullets else "")

        link_btn = (
            f'<a href="{link}" target="_blank" class="project-link">View Project →</a>'
            if link else ""
        )
        dummy_attr  = ' data-dummy="true"' if is_dummy else ""
        tech_tags   = "".join(
            f'<span class="tech-tag">{t.strip()}</span>'
            for t in tech.replace("·", ",").split(",") if t.strip()
        )

        cards.append(f"""
<div class="project-card"{dummy_attr}>
  <div class="project-header">
    <h3 class="project-name">{name}</h3>
    {link_btn}
  </div>
  <p class="project-desc">{description}</p>
  <div class="project-tags">{tech_tags}</div>
</div>""")
    return "\n".join(cards)


def _experience_html(experience: list) -> str:
    items = []
    for e in experience:
        title    = e.get("title", "")
        company  = e.get("company", "")
        dates    = e.get("dates", "")
        loc      = e.get("location", "")
        bullets  = e.get("bullets") or []
        is_dummy = e.get("_is_dummy", False)

        bullet_html = "\n".join(
            f"<li>{b}</li>" for b in bullets if b
        )
        dummy_attr = ' data-dummy="true"' if is_dummy else ""

        items.append(f"""
<div class="timeline-item"{dummy_attr}>
  <div class="timeline-dot"></div>
  <div class="timeline-content">
    <div class="timeline-header">
      <div>
        <h3 class="exp-title">{title}</h3>
        <p class="exp-company">{company}{' · ' + loc if loc else ''}</p>
      </div>
      <span class="exp-dates">{dates}</span>
    </div>
    <ul class="exp-bullets">{bullet_html}</ul>
  </div>
</div>""")
    return "\n".join(items)


def _education_html(education: list) -> str:
    cards = []
    for e in education:
        degree  = e.get("degree", "")
        inst    = e.get("institution", "")
        dates   = e.get("dates", "")
        gpa     = e.get("gpa", "")
        gpa_str = f'<span class="edu-gpa">GPA: {gpa}</span>' if gpa else ""

        cards.append(f"""
<div class="edu-card">
  <div class="edu-header">
    <div>
      <h3 class="edu-degree">{degree}</h3>
      <p class="edu-inst">{inst}</p>
    </div>
    <div class="edu-meta">
      <span class="edu-dates">{dates}</span>
      {gpa_str}
    </div>
  </div>
</div>""")
    return "\n".join(cards)


def _stats_html(structure: dict) -> str:
    n_projects   = len([p for p in (structure.get("projects") or []) if not p.get("_is_dummy")])
    n_skills     = sum(
        len(v) for k, v in (structure.get("skills") or {}).items()
        if k != "_is_dummy" and isinstance(v, list)
    )
    n_exp        = len([e for e in (structure.get("experience") or []) if not e.get("_is_dummy")])

    stats = [
        (str(n_projects), "Projects Built"),
        (str(n_skills),   "Technologies"),
        (str(n_exp),      "Roles / Internships"),
    ]
    items = "\n".join(
        f'<div class="stat-item"><span class="stat-num" data-target="{val}">{val}</span><span class="stat-label">{label}</span></div>'
        for val, label in stats
    )
    return f'<div class="stats-inner">{items}</div>'


def _contact_html(structure: dict) -> str:
    links = []
    if structure.get("email"):
        links.append(f'<a href="mailto:{structure["email"]}" class="contact-link">✉ {structure["email"]}</a>')
    linkedin = structure.get("linkedin") or ""
    if linkedin:
        url = linkedin if linkedin.startswith("http") else f"https://{linkedin}"
        links.append(f'<a href="{url}" target="_blank" class="contact-link">in LinkedIn</a>')
    github = structure.get("github") or ""
    if github:
        url = github if github.startswith("http") else f"https://{github}"
        handle = github.replace("github.com/", "").strip("/")
        links.append(f'<a href="{url}" target="_blank" class="contact-link">⌥ {handle}</a>')
    return "\n".join(links)


def _dummy_banner_script(dummy_sections: List[str]) -> str:
    if not dummy_sections:
        return ""
    sections_str = ", ".join(dummy_sections)
    return f"""<script>
// Dummy section warning — injected by PrepSense AI
document.addEventListener("DOMContentLoaded", () => {{
  const banner = document.createElement("div");
  banner.style.cssText = "position:fixed;bottom:0;left:0;right:0;background:#f59e0b;color:#1a1a2e;padding:10px 20px;font-size:0.85rem;font-family:sans-serif;z-index:9999;text-align:center;";
  banner.innerHTML = "⚠️ <strong>Placeholder content detected</strong> in: {sections_str}. Update your resume and regenerate to replace these sections.";
  document.body.appendChild(banner);
}});
</script>"""


# ── Main public function ───────────────────────────────────────────────────────

def generate_portfolio(
    resume_structure: dict,
    target_role: str,
    template_name: str,           # "luminary" or "noir"
    client: Groq,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Builds a complete portfolio website from resume structure.

    Returns:
        files:          {"index.html": str, "style.css": str, "script.js": str}
        dummy_sections: list of section names that used placeholder content
    """
    template_dir = TEMPLATES_DIR / template_name
    if not template_dir.exists():
        raise FileNotFoundError(f"Template '{template_name}' not found at {template_dir}")

    # 1. Detect missing sections + inject dummies
    structure, dummy_sections = _detect_and_fill_dummies(resume_structure)

    # 2. LLM enhancement
    enhanced = _enhance_content(structure, target_role, client)

    tagline      = enhanced.get("tagline", "Building things that matter")
    about        = enhanced.get("about", "")
    typing_roles = enhanced.get("typing_roles") or [target_role or "Developer"]
    enhancements = enhanced.get("project_enhancements") or {}

    # 3. Build HTML snippets
    skills_html     = _skill_chips_html(structure.get("skills") or {})
    projects_html   = _projects_html(structure.get("projects") or [], enhancements)
    experience_html = _experience_html(structure.get("experience") or [])
    education_html  = _education_html(structure.get("education") or [])
    stats_html      = _stats_html(structure)
    contact_html    = _contact_html(structure)
    dummy_script    = _dummy_banner_script(dummy_sections)

    name     = structure.get("name") or "Developer"
    initials = "".join(w[0].upper() for w in name.split()[:2]) or "P"
    github   = structure.get("github") or ""
    linkedin = structure.get("linkedin") or ""
    email    = structure.get("email") or ""
    typing_json = json.dumps(typing_roles)

    # 4. Fill template placeholders
    replacements = {
        "{{FULL_NAME}}":       name,
        "{{INITIALS}}":        initials,
        "{{TAGLINE}}":         tagline,
        "{{ABOUT_PARAGRAPH}}": about,
        "{{TYPING_ROLES_JSON}}": typing_json,
        "{{SKILLS_HTML}}":     skills_html,
        "{{PROJECTS_HTML}}":   projects_html,
        "{{EXPERIENCE_HTML}}": experience_html,
        "{{EDUCATION_HTML}}":  education_html,
        "{{STATS_HTML}}":      stats_html,
        "{{CONTACT_HTML}}":    contact_html,
        "{{DUMMY_BANNER_SCRIPT}}": dummy_script,
        "{{NAV_YEAR}}":        "2025",
        "{{EMAIL}}":           email,
        "{{GITHUB_URL}}":      (github if github.startswith("http") else f"https://{github}") if github else "#",
        "{{LINKEDIN_URL}}":    (linkedin if linkedin.startswith("http") else f"https://{linkedin}") if linkedin else "#",
    }

    files: Dict[str, str] = {}
    for filename in ("index.html", "style.css", "script.js"):
        path = template_dir / filename
        content = path.read_text(encoding="utf-8")
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)
        files[filename] = content

    return files, dummy_sections
