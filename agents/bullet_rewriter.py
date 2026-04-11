"""
agents/bullet_rewriter.py
Resume bullet rewriter — works for any career field in India.
Returns a structured list of {original, improved, why} dicts.
"""
import json
import re
from typing import List, Dict

from groq import Groq

GROQ_MODEL = "openai/gpt-oss-120b"
MAX_RESUME_CHARS = 7500


def rewrite_bullets(resume_text: str, target_role: str, client: Groq, model: str = GROQ_MODEL) -> List[Dict]:
    """
    Identifies weak bullets and rewrites them.
    Returns list of dicts: [{original, improved, why}, ...]
    Returns an empty list on failure.
    """
    role_context = (
        f"Target role: {target_role}"
        if target_role.strip()
        else "Target role not specified — make improvements generally strong and ATS-friendly for India."
    )

    prompt = f"""You are an expert Indian resume coach helping candidates land better roles across all fields.

Resume:
\"\"\"
{resume_text[:MAX_RESUME_CHARS]}
\"\"\"

{role_context}

Task:
Scan every bullet point / description line in the resume. For each one decide:

A) REMOVE — if the bullet is purely redundant or zero-value. Remove when:
   - It just restates the tech stack already listed in the project/job title (e.g. "Used React for frontend and Node for backend" under a project already titled "React + Node App")
   - It's an obvious filler with no information ("Worked on the project", "Helped the team", "Was part of development")
   - It duplicates information already said in another bullet of the same item

B) REWRITE — if the bullet has real underlying content but is written weakly:
   - Vague / passive language ("assisted", "responsible for", "helped with", "involved in")
   - No numbers or measurable impact
   - Duty-focused instead of achievement-focused
   Rewrite rules:
   - Start with a strong action verb (Led, Designed, Reduced, Built, Deployed, Secured, etc.)
   - PAR / STAR: context + action + result / impact
   - Quantify wherever possible; if absent, add conservative realistic estimates marked ~
   - Natural field-relevant keywords — never force tech keywords into non-tech bullets
   - 1–2 lines max

C) KEEP — strong bullets that are already good. Do NOT include these in the output at all.

Return ONLY a valid JSON array — no markdown fences, no explanation:

[
  {{
    "action": "remove",
    "original": "exact verbatim bullet from resume",
    "improved": null,
    "why": "1 sentence: why this bullet adds no value"
  }},
  {{
    "action": "rewrite",
    "original": "exact verbatim bullet from resume",
    "improved": "your rewritten bullet",
    "why": "1–2 sentences: what you fixed and why it helps shortlisting"
  }}
]

Only include bullets you are removing or rewriting. Return ONLY the JSON array."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.55,
            max_tokens=5000,
            top_p=0.9,
        )
        raw = response.choices[0].message.content.strip()
        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())
        pairs = json.loads(raw)
        # Validate shape
        if isinstance(pairs, list):
            return [
                {
                    "action":   str(p.get("action") or "rewrite").lower(),
                    "original": str(p.get("original") or ""),
                    "improved": str(p.get("improved") or "") if p.get("improved") else "",
                    "why":      str(p.get("why") or ""),
                }
                for p in pairs
                if isinstance(p, dict) and p.get("original")
            ]
        return []

    except Exception as e:
        return [{"original": "", "improved": "", "why": f"Error: {e}"}]
