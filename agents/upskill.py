"""
agents/upskill.py
Recommends skills based on resume gaps and generates structured learning plans
with YouTube search links for resources.
"""
import json
import re
from typing import Dict, List
from urllib.parse import quote_plus

from groq import Groq

GROQ_MODEL = "openai/gpt-oss-120b"


def recommend_skills(
    resume_text: str,
    target_role: str,
    missing_keywords: List[str],
    client: Groq,
    model: str = GROQ_MODEL,
) -> List[Dict]:
    """
    Returns top 5 skills the candidate should learn next.
    [
      {
        "skill": str,
        "priority": "High" | "Medium" | "Low",
        "reason": str
      }
    ]
    Returns empty list on failure.
    """
    missing_str = ", ".join(missing_keywords[:20]) if missing_keywords else "not available"

    prompt = f"""Based on this candidate's resume and target role, recommend the top 5 skills they should learn next to become more competitive.

Target Role: {target_role}
Skills missing from resume (ATS gap): {missing_str}
Resume excerpt:
{resume_text[:1800]}

Consider: industry demand for the role, candidate's existing tech stack, learning synergy, and career impact.
Priority: High = critical for role, Medium = beneficial, Low = nice to have.

Return ONLY a JSON array (no markdown):
[
  {{"skill": "Docker", "priority": "High", "reason": "Container skills are expected for most SDE roles and pair well with your existing backend stack."}},
  {{"skill": "System Design", "priority": "High", "reason": "Required for senior interviews; your experience scales well but needs a structured design framework."}}
]"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
        content = (resp.choices[0].message.content or "").strip()
        content = re.sub(r"```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"```\s*$", "", content).strip()
        m = re.search(r"\[[\s\S]*\]", content)
        if not m:
            return []
        return json.loads(m.group(0))
    except Exception:
        return []


def generate_learning_plan(skill: str, target_role: str, client: Groq, model: str = GROQ_MODEL) -> Dict:
    """
    Generates a 4-week structured learning plan for a skill.
    Returns:
    {
      "overview": str,
      "weeks": [
        {"week": 1, "goal": str, "topics": [str, ...]},
        ...
      ],
      "resources": [
        {"title": str, "type": "Video"|"Course"|"Docs"|"Practice", "search_query": str},
        ...
      ]
    }
    Returns empty dict on failure.
    """
    prompt = f"""Create a focused 4-week learning roadmap for "{skill}" targeting a {target_role} role.

Return ONLY a JSON object (no markdown):
{{
  "overview": "2-sentence overview: why this skill matters for the role and the approach to learning it.",
  "weeks": [
    {{"week": 1, "goal": "Understand fundamentals", "topics": ["Core concept A", "Core concept B", "Hands-on: Hello World project"]}},
    {{"week": 2, "goal": "Intermediate patterns", "topics": ["Pattern X", "Pattern Y"]}},
    {{"week": 3, "goal": "Real-world application", "topics": ["Build project", "Integration with other tools"]}},
    {{"week": 4, "goal": "Interview readiness", "topics": ["Common interview patterns", "Mock problems", "Portfolio project"]}}
  ],
  "resources": [
    {{"title": "Resource name (e.g. 'Traversy Media - Docker Crash Course')", "type": "Video", "search_query": "Docker crash course for beginners traversy media"}},
    {{"title": "FreeCodeCamp full course name", "type": "Course", "search_query": "freecodecamp docker full course"}},
    {{"title": "Official Docs / Guide name", "type": "Docs", "search_query": "docker getting started tutorial documentation"}},
    {{"title": "Practice platform or project idea", "type": "Practice", "search_query": "docker hands on project tutorial"}}
  ]
}}

Include 4-6 resources. Use realistic, well-known resource names (channels like Fireship, Traversy Media, FreeCodeCamp, Academind, The Net Ninja, etc. are good choices). search_query should be something you'd actually type into YouTube."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1800,
        )
        content = (resp.choices[0].message.content or "").strip()
        content = re.sub(r"```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"```\s*$", "", content).strip()
        m = re.search(r"\{[\s\S]*\}", content)
        if not m:
            return {}
        return json.loads(m.group(0))
    except Exception:
        return {}


def yt_search_url(query: str) -> str:
    """Returns a YouTube search URL for the given query string."""
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"
