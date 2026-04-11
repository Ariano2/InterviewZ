"""
agents/resume_structurer.py
Extracts a structured JSON representation from raw resume text via Groq.
"""
import json
import re

from groq import Groq

GROQ_MODEL = "openai/gpt-oss-120b"
MAX_CHARS = 8000


def structure_resume(resume_text: str, client: Groq, model: str = GROQ_MODEL) -> dict:
    """
    Parses resume text into a structured dict.

    Returns keys: name, email, phone, linkedin, github, website, location,
    summary, education, experience, projects, skills, certifications,
    achievements, extra_sections.
    On failure returns a minimal safe dict.
    """
    prompt = f"""Extract the resume below into a structured JSON object.
Return ONLY valid JSON — no markdown fences, no explanation, no extra text.

Resume:
\"\"\"
{resume_text[:MAX_CHARS]}
\"\"\"

Use exactly this schema (omit keys that have no data, use null for unknown values):

{{
  "name": "Full Name",
  "email": "email@example.com",
  "phone": "+91-XXXXXXXXXX",
  "linkedin": "linkedin.com/in/handle  or  null",
  "github": "github.com/handle  or  null",
  "website": "url  or  null",
  "location": "City, State  or  null",
  "summary": "professional summary paragraph  or  null",
  "education": [
    {{
      "degree": "B.Tech Computer Science",
      "institution": "University Name",
      "location": "City",
      "dates": "2022 – 2026",
      "gpa": "8.5 / 10  or  null",
      "bullets": ["relevant coursework, honours, etc. — one string per item"]
    }}
  ],
  "experience": [
    {{
      "title": "Job Title",
      "company": "Company Name",
      "location": "City  or  Remote",
      "dates": "Jun 2024 – Aug 2024",
      "bullets": ["exact bullet text from resume — one string per bullet"]
    }}
  ],
  "projects": [
    {{
      "name": "Project Name",
      "tech": "Python, React, etc.",
      "dates": "Jan 2024  or  null",
      "link": "github link  or  null",
      "bullets": ["exact bullet text — one string per bullet"]
    }}
  ],
  "skills": {{
    "languages": ["Python", "Java"],
    "frameworks": ["React", "Django"],
    "tools": ["Git", "Docker"],
    "other": ["any other skills"]
  }},
  "certifications": ["Cert Name — Issuer  (Year)"],
  "achievements": ["achievement text"],
  "extra_sections": [
    {{
      "title": "Section Title",
      "items": ["item text"]
    }}
  ]
}}

Return ONLY the JSON object."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=3500,
        )
        raw = response.choices[0].message.content.strip()
        # Strip any accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.strip())
        return json.loads(raw)

    except Exception as e:
        return {
            "error": str(e),
            "name": "Resume",
            "education": [],
            "experience": [],
            "projects": [],
            "skills": {},
            "certifications": [],
            "achievements": [],
            "extra_sections": [],
        }
