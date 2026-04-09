"""
agents/ats_analyzer.py
Rubric-based ATS analysis. Score is computed from weighted sub-scores,
not guessed by the LLM. Supports optional JD text for precise keyword matching.

Sub-score weights:
  Keyword Match     40%  — computed: matched / total required keywords
  Quantification    25%  — computed: bullets with numbers / total bullets
  Action Verbs      15%  — LLM-rated
  Section Complete  10%  — computed: required sections present
  ATS Formatting    10%  — LLM-rated
"""

import json
import re
from typing import TypedDict

from groq import Groq

GROQ_MODEL = "openai/gpt-oss-120b"
MAX_RESUME_CHARS = 7000
MAX_JD_CHARS = 4000

WEIGHTS = {
    "keyword":        0.40,
    "quantification": 0.25,
    "action_verb":    0.15,
    "sections":       0.10,
    "formatting":     0.10,
}

STRONG_VERBS = {
    "led", "built", "designed", "developed", "engineered", "implemented",
    "architected", "deployed", "optimised", "optimized", "reduced", "increased",
    "improved", "launched", "created", "established", "managed", "directed",
    "spearheaded", "delivered", "achieved", "secured", "automated", "migrated",
    "refactored", "scaled", "integrated", "researched", "published", "trained",
    "mentored", "coordinated", "negotiated", "resolved", "diagnosed", "built",
    "shipped", "rewrote", "restructured", "accelerated", "streamlined", "drove",
    "generated", "closed", "recruited", "onboarded", "formulated", "proposed",
}

REQUIRED_SECTIONS = ["experience", "education", "skill", "project"]


class ATSResult(TypedDict):
    ats_score: int
    # Sub-scores (0-100 each)
    keyword_score: int
    quantification_score: int
    action_verb_score: int
    section_score: int
    formatting_score: int
    # Keywords
    matched_keywords: list
    missing_keywords: list
    keyword_match_pct: int
    # Qualitative
    strong_areas: list
    weak_areas: list
    summary: str
    # Meta
    used_jd: bool
    quant_detail: str


_FALLBACK: ATSResult = {
    "ats_score": 0,
    "keyword_score": 0,
    "quantification_score": 0,
    "action_verb_score": 0,
    "section_score": 0,
    "formatting_score": 0,
    "matched_keywords": [],
    "missing_keywords": [],
    "keyword_match_pct": 0,
    "strong_areas": [],
    "weak_areas": [],
    "summary": "Analysis could not be completed. Please try again.",
    "used_jd": False,
    "quant_detail": "",
}


# ── Programmatic checks ───────────────────────────────────────────────────────

# Common abbreviation pairs (full form → short form, both directions checked)
_ABBREV = {
    "machine learning": "ml",
    "artificial intelligence": "ai",
    "natural language processing": "nlp",
    "large language model": "llm",
    "retrieval augmented generation": "rag",
    "continuous integration": "ci",
    "continuous deployment": "cd",
    "object oriented programming": "oop",
    "application programming interface": "api",
    "user interface": "ui",
    "user experience": "ux",
    "sql": "structured query language",
}


def _word_present(word: str, text: str) -> bool:
    """True if `word` appears as a whole token in `text` (word-boundary aware)."""
    pattern = r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])"
    return bool(re.search(pattern, text))


def _check_keywords(resume_text: str, keywords: list) -> tuple:
    """
    Check each keyword against resume text (case-insensitive).
    Also catches common abbreviations: 'machine learning' ↔ 'ML', etc.
    Returns (matched, missing, pct).
    """
    text_lower = resume_text.lower()
    matched, missing = [], []

    for kw in keywords:
        kw_lower = kw.lower().strip()
        found = _word_present(kw_lower, text_lower)
        if not found:
            abbrev = _ABBREV.get(kw_lower)
            if abbrev and _word_present(abbrev, text_lower):
                found = True
            else:
                for full, short in _ABBREV.items():
                    if kw_lower == short and _word_present(full, text_lower):
                        found = True
                        break
        (matched if found else missing).append(kw)

    total = len(keywords)
    pct = round(len(matched) / total * 100) if total else 0
    return matched, missing, pct


def _quantification_rate(resume_text: str) -> tuple:
    """
    Count bullet lines that contain at least one number.
    Returns (score_0_to_100, detail_string).
    """
    lines = resume_text.split("\n")

    # Primary: lines that start with a bullet character
    bullets = [
        l.strip() for l in lines
        if re.match(r"^\s*[-•*▪▸●✓]\s+\S", l)
    ]
    # Fallback: longer content lines that look like achievements
    if len(bullets) < 3:
        bullets = [l.strip() for l in lines if len(l.strip()) > 45]

    if not bullets:
        return 50, "Could not detect bullet lines"

    quantified = sum(1 for b in bullets if re.search(r"\d+", b))
    pct = round(quantified / len(bullets) * 100)
    detail = f"{quantified} / {len(bullets)} bullets contain a number or metric"
    return pct, detail


def _section_completeness(resume_text: str) -> int:
    """Returns 0-100 based on how many standard sections are present."""
    text_lower = resume_text.lower()
    found = sum(1 for s in REQUIRED_SECTIONS if s in text_lower)
    return round(found / len(REQUIRED_SECTIONS) * 100)


def _programmatic_action_verb_score(resume_text: str) -> int:
    """
    Fallback action verb score computed without the LLM.
    Returns 0-100.
    """
    lines = resume_text.split("\n")
    bullets = [l.strip() for l in lines if re.match(r"^\s*[-•*▪▸●]\s+\S", l)]
    if not bullets:
        return 55

    strong = 0
    for b in bullets:
        text = re.sub(r"^[-•*▪▸●]\s*", "", b).strip()
        first = text.split()[0].lower().rstrip(",.;:") if text.split() else ""
        if first in STRONG_VERBS:
            strong += 1

    return round(strong / len(bullets) * 100)


# ── LLM call ──────────────────────────────────────────────────────────────────

def _llm_analysis(
    resume_text: str,
    target_role: str,
    client: Groq,
    jd_text: str = "",
) -> dict:
    """
    Single LLM call that handles:
      - Keyword extraction (from JD if provided, else from role name)
      - Action verb quality rating
      - ATS formatting rating
      - Strong / weak areas and summary
    """
    if jd_text.strip():
        context_block = f'Job Description:\n"""\n{jd_text.strip()[:MAX_JD_CHARS]}\n"""'
        kw_instruction = (
            "Extract 15-25 specific technical skills, tools, frameworks, and domain keywords "
            "that this JD explicitly requires or strongly implies. Pull exact terms from the JD text."
        )
    else:
        context_block = f'Target Role: "{target_role}"'
        kw_instruction = (
            f"List 15-20 keywords and skills typically required for {target_role} roles "
            f"in India in 2025-2026. Include technical skills, tools, and domain-relevant terms."
        )

    prompt = f"""You are a senior ATS expert and technical recruiter in India.

{context_block}

Resume:
\"\"\"
{resume_text[:MAX_RESUME_CHARS]}
\"\"\"

Return ONLY a valid JSON object. No markdown fences, no extra text.

{{
  "required_keywords": [
    // {kw_instruction}
  ],
  "action_verb_rating": <integer 0-100: how consistently are strong action verbs used at the start of bullet points? 100=every bullet starts with a strong verb, 0=all passive/weak>,
  "formatting_rating": <integer 0-100: ATS formatting quality. Deduct for: tables, multi-column layouts, headers/footers, graphics, unusual unicode bullets. Award for: clean section headings, plain text, standard fonts implied>,
  "strong_areas": [<3-5 specific, honest strengths of this resume for the role>],
  "weak_areas": [<3-5 specific gaps or weaknesses — be brutal and specific, not generic>],
  "summary": "<2-3 sentences: direct, specific assessment. If JD was provided, reference it. Don't pad.>"
}}

Return ONLY the JSON object."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15,
        max_tokens=1500,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    return json.loads(raw.strip())


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_ats(
    resume_text: str,
    target_role: str,
    client: Groq,
    jd_text: str = "",
) -> ATSResult:
    """
    Rubric-based ATS analysis.
    jd_text: optional job description for precise keyword extraction.
    Never raises — returns _FALLBACK on any failure.
    """
    try:
        # Step 1 — LLM: keywords + qualitative ratings
        llm = _llm_analysis(resume_text, target_role, client, jd_text)

        required_kws = [str(k) for k in (llm.get("required_keywords") or []) if k]

        # Step 2 — Programmatic sub-scores
        matched, missing, kw_pct = _check_keywords(resume_text, required_kws)
        quant_score, quant_detail = _quantification_rate(resume_text)
        section_score = _section_completeness(resume_text)

        # Step 3 — LLM-rated sub-scores (clamped)
        action_score = max(0, min(100, int(llm.get("action_verb_rating") or 0)
                                  or _programmatic_action_verb_score(resume_text)))
        fmt_score    = max(0, min(100, int(llm.get("formatting_rating") or 70)))

        # Step 4 — Weighted final score
        final = round(
            WEIGHTS["keyword"]        * kw_pct        +
            WEIGHTS["quantification"] * quant_score   +
            WEIGHTS["action_verb"]    * action_score  +
            WEIGHTS["sections"]       * section_score +
            WEIGHTS["formatting"]     * fmt_score
        )
        final = max(0, min(100, final))

        return ATSResult(
            ats_score=final,
            keyword_score=kw_pct,
            quantification_score=quant_score,
            action_verb_score=action_score,
            section_score=section_score,
            formatting_score=fmt_score,
            matched_keywords=matched,
            missing_keywords=missing,
            keyword_match_pct=kw_pct,
            strong_areas=list(llm.get("strong_areas") or []),
            weak_areas=list(llm.get("weak_areas") or []),
            summary=str(llm.get("summary") or ""),
            used_jd=bool(jd_text.strip()),
            quant_detail=quant_detail,
        )

    except Exception:
        return _FALLBACK
