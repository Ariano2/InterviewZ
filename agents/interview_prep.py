"""
agents/interview_prep.py
Generates resume-centric interview Q&As grouped by difficulty.
Questions are grounded in the candidate's actual projects, tech choices, and experience.
"""
import json
import re
from typing import Dict, List

from groq import Groq

GROQ_MODEL = "openai/gpt-oss-120b"


def generate_qna(
    resume_text: str,
    target_role: str,
    client: Groq,
) -> Dict[str, List[Dict]]:
    """
    Reads the resume and generates interview Q&As in three difficulty tiers.

    Returns:
    {
      "easy":   [{"question": str, "answer": str, "example": str|None}, ...],  # ~12 Qs
      "medium": [...],                                                           # ~12 Qs
      "hard":   [...]                                                            # ~10 Qs
    }

    Easy   — "What is X?" definitions for every lib/tool/language on the resume.
    Medium — Project-specific: "Why did you use X in <project>?", "How did your
             <feature> work?", "Walk me through the architecture of <project>."
    Hard   — Trade-offs, scale, redesign: "How would you scale <project>?",
             "What would you change about <design choice>?", "What's the
             bottleneck in <system>?"

    All answers must be laconic:
      Easy/Medium → 2-3 sentences max.
      Hard        → 3-5 sentences max if truly needed, never more.
    Returns {"easy": [], "medium": [], "hard": []} on failure.
    """
    prompt = f"""You are a senior technical interviewer preparing questions specifically for THIS candidate.

Target Role: {target_role}

Candidate's Resume:
{resume_text[:3500]}

Generate interview Q&As in three tiers. Read the resume carefully — reference actual project names, actual tech choices, actual experience listed.

EASY (~12 questions):
  - "What is X?" — X must be a language, tool, library, or concept the candidate explicitly lists.
  - Answer: 1-2 tight sentences. The kind a confident candidate says without hesitation.
  - Example: short code snippet or analogy only if it genuinely clarifies. Otherwise null.

MEDIUM (~12 questions):
  - Project/experience questions: "Why did you choose <tech> for <project on resume>?",
    "How did you implement <feature they built>?", "What does <specific thing they did> do?"
  - Answer: 2-3 sentences. Direct, no filler.
  - Example: null unless a 1-2 line snippet makes it sharper.

HARD (~10 questions):
  - Trade-offs, redesigns, scale: "What's the bottleneck in <their architecture>?",
    "How would you scale <their project>?", "What would you redesign and why?",
    "What's a limitation of <their tech choice>?"
  - Answer: 3-5 sentences. Show depth, not length.
  - Example: null unless critical.

RULES:
  - Every question must reference something real from the resume. No generic textbook Qs.
  - Answers are for the candidate to study — clear, confident, interview-ready.
  - No essays. No bullet-point answers. Prose only, 2-5 sentences strictly.
  - If the resume uses WebSockets — ask about WebSockets. If they used Redis — ask about Redis.
    If they built a recommendation engine — ask how it works. You get the idea.

Return ONLY valid JSON (no markdown):
{{
  "easy": [
    {{"question": "What is a REST API?", "answer": "A REST API is a web interface that communicates via standard HTTP methods (GET, POST, PUT, DELETE) using stateless requests. Resources are represented as URLs and responses are typically JSON.", "example": null}},
    ...
  ],
  "medium": [
    {{"question": "Why did you choose FastAPI over Flask for your backend in <project>?", "answer": "FastAPI gives automatic request validation via Pydantic and async support out of the box, which mattered because the project needed to handle concurrent requests efficiently. Flask would have required extra libraries to achieve the same.", "example": null}},
    ...
  ],
  "hard": [
    {{"question": "Your <project> uses a single Postgres instance — how would you scale it under 100x load?", "answer": "I'd introduce read replicas to offload SELECT-heavy traffic and add a caching layer (Redis) in front of frequently-hit queries. For write scaling, horizontal sharding by user ID or adding a connection pooler like PgBouncer would help. Beyond that, moving analytics queries to a separate OLAP store keeps the primary DB lean.", "example": null}},
    ...
  ]
}}"""

    empty = {"easy": [], "medium": [], "hard": []}
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=6000,
        )
        content = (resp.choices[0].message.content or "").strip()
        content = re.sub(r"```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"```\s*$", "", content).strip()
        m = re.search(r"\{[\s\S]*\}", content)
        if not m:
            return empty
        data = json.loads(m.group(0))
        return {
            "easy":   data.get("easy")   or [],
            "medium": data.get("medium") or [],
            "hard":   data.get("hard")   or [],
        }
    except Exception:
        return empty
