"""
agents/skill_gap.py
Categorises matched/missing ATS keywords into radar chart axes.
Single focused LLM call — piggybacks on keyword data already extracted by ats_analyzer.
"""
import json
import re
from typing import Dict, List

from groq import Groq

GROQ_MODEL = "openai/gpt-oss-120b"


def analyze_skill_gap(
    matched_keywords: List[str],
    missing_keywords: List[str],
    target_role: str,
    client: Groq,
    model: str = GROQ_MODEL,
) -> Dict:
    """
    Groups matched and missing JD keywords into 5-6 skill categories and
    scores each category 0-10 for resume coverage vs JD requirement level.

    Returns:
    {
        "categories": ["Programming Languages", "ML Frameworks", ...],
        "resume_scores": [8, 4, ...],   # 0-10: how well resume covers this category
        "jd_scores":     [9, 9, ...],   # 0-10: how heavily the JD demands this category
    }
    Returns empty lists on failure — caller should handle gracefully.
    """
    if not matched_keywords and not missing_keywords:
        return {"categories": [], "resume_scores": [], "jd_scores": []}

    # Trim to keep prompt short
    matched_str = ", ".join(matched_keywords[:15])
    missing_str = ", ".join(missing_keywords[:15])

    prompt = f"""Role: {target_role}
Matched skills (in resume): {matched_str}
Missing skills (not in resume): {missing_str}

Group these into 5 skill categories for this role. Score each 0-10:
- resume_score: coverage in resume (high if mostly matched, low if mostly missing)
- jd_score: JD demand level (high if many skills in this category appear)

Return ONLY JSON:
{{"categories":["A","B","C","D","E"],"resume_scores":[7,4,8,6,3],"jd_scores":[9,8,7,6,8]}}"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
            max_tokens=1500,
        )
        content = resp.choices[0].message.content
        if not content:
            return {"categories": [], "resume_scores": [], "jd_scores": [], "error": f"Model returned empty content. Finish reason: {resp.choices[0].finish_reason}"}
        raw = content.strip()
        raw_original = raw  # keep for error reporting
        raw = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"```\s*$", "", raw).strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return {"categories": [], "resume_scores": [], "jd_scores": [], "error": f"No JSON found. Raw response: {raw_original[:400]}"}

        data = json.loads(m.group(0))
        cats     = [str(c) for c in (data.get("categories") or [])]
        # Use float() first to handle scores returned as "7.0" or 7.5
        r_scores = [int(float(s)) for s in (data.get("resume_scores") or [])]
        j_scores = [int(float(s)) for s in (data.get("jd_scores")     or [])]

        # Ensure all three arrays are the same length and scores are clamped 0-10
        n = min(len(cats), len(r_scores), len(j_scores))
        return {
            "categories":    cats[:n],
            "resume_scores": [max(0, min(10, s)) for s in r_scores[:n]],
            "jd_scores":     [max(0, min(10, s)) for s in j_scores[:n]],
        }
    except Exception as e:
        return {"categories": [], "resume_scores": [], "jd_scores": [], "error": str(e)}
