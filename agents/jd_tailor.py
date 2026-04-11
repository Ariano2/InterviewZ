"""
agents/jd_tailor.py
Tailors resume bullets to a specific JD and generates a matching cover letter.
Two focused LLM calls — one for bullet tailoring, one for the cover letter.
"""
import json
import re
from typing import Dict, List

from groq import Groq

GROQ_MODEL = "openai/gpt-oss-120b"
MAX_RESUME_CHARS = 7000
MAX_JD_CHARS = 4000


def tailor_resume(
    resume_text: str,
    jd_text: str,
    target_role: str,
    client: Groq,
    model: str = GROQ_MODEL,
) -> Dict:
    """
    Rewrites bullets to inject missing JD keywords naturally.

    Returns:
      {
        "rewrites":  [{"original": str, "improved": str, "keyword_added": str}],
        "added_keywords": [str],
      }
    """
    prompt = f"""You are an expert resume tailor. Edit the resume so it passes ATS screening for the job description below — minimum changes, only natural edits.

Job Description:
\"\"\"
{jd_text.strip()[:MAX_JD_CHARS]}
\"\"\"

Resume:
\"\"\"
{resume_text[:MAX_RESUME_CHARS]}
\"\"\"

Target Role: {target_role}

Instructions:
1. Find the 8-12 most important keywords/skills/tools from the JD that are MISSING from the resume.
2. For each missing keyword, find the single most appropriate existing bullet point to inject it NATURALLY. If no suitable home exists, skip it.
3. Rewrite ONLY those bullets. Keep everything else exactly as-is.
4. Never fabricate experience. Only inject a keyword if the underlying work is clearly related.

Return ONLY a valid JSON array — no markdown fences, no explanation:

[
  {{
    "original": "exact verbatim bullet from resume",
    "improved": "rewritten bullet with keyword naturally integrated",
    "keyword_added": "the specific keyword injected"
  }}
]

Return ONLY the JSON array."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=6000,
            top_p=0.9,
        )
        raw = resp.choices[0].message.content
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())
        items = json.loads(raw)
        if not isinstance(items, list):
            return {"rewrites": [], "added_keywords": []}
        rewrites = [
            {
                "original":      str(r.get("original") or ""),
                "improved":      str(r.get("improved") or ""),
                "keyword_added": str(r.get("keyword_added") or ""),
            }
            for r in items
            if isinstance(r, dict) and r.get("original") and r.get("improved")
        ]
        added = [r["keyword_added"] for r in rewrites if r["keyword_added"]]
        return {"rewrites": rewrites, "added_keywords": added}
    except Exception as e:
        return {"rewrites": [], "added_keywords": []}


def generate_cover_letter(
    resume_text: str,
    jd_text: str,
    target_role: str,
    client: Groq,
    model: str = GROQ_MODEL,
) -> str:
    """
    Generates a concise, role-specific cover letter grounded in the resume.
    Returns plain text (no markdown).
    """
    prompt = f"""Write a professional cover letter for the job below based on the candidate's resume.

Job Description:
\"\"\"
{jd_text.strip()[:MAX_JD_CHARS]}
\"\"\"

Candidate Resume:
\"\"\"
{resume_text[:MAX_RESUME_CHARS]}
\"\"\"

Target Role: {target_role}

Requirements:
- 3 paragraphs, ~250-300 words total
- Para 1: Role applied for, where found, 1 punchy sentence on why this company/role specifically
- Para 2: 2-3 concrete achievements from the resume that directly match JD requirements — use specific numbers/results where available
- Para 3: Cultural fit + enthusiasm + clear CTA (interview request)
- Tone: confident, professional, not sycophantic — no "I am excited to apply" openers
- India market style — formal but direct
- Leave [Company Name] as a placeholder if company name is not in the JD
- Sign off: "Regards," then a blank line for the candidate's name

Return ONLY the cover letter text. No commentary, no subject line, no markdown."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=800,
        )
        content = resp.choices[0].message.content
        if not content or not content.strip():
            return "Error: model returned an empty response. Please try again."
        return content.strip()
    except Exception as e:
        return f"Error generating cover letter: {e}"
