"""
agents/job_search.py
Live job search via JSearch API (aggregates LinkedIn + Indeed + Glassdoor).
Returns a formatted text block ready to inject into the LLM context.
"""
import requests
from typing import List, Dict

JSEARCH_HOST  = "jsearch.p.rapidapi.com"
JSEARCH_URL   = "https://jsearch.p.rapidapi.com/search"
REQUEST_TIMEOUT = 25


def search_jobs(
    query: str,
    location: str = "India",
    employment_type: str = "FULLTIME",
    num_results: int = 5,
    rapidapi_key: str = "",
) -> List[Dict]:
    """
    Calls JSearch API and returns a list of clean job dicts.
    Raises on HTTP / network errors so the caller can handle gracefully.
    """
    params = {
        "query":            f"{query} in {location}",
        "page":             "1",
        "num_pages":        "1",
        "employment_types": employment_type,
        "date_posted":      "month",
    }
    headers = {
        "X-RapidAPI-Key":  rapidapi_key,
        "X-RapidAPI-Host": JSEARCH_HOST,
    }

    resp = requests.get(JSEARCH_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json().get("data") or []

    jobs = []
    for job in data[:num_results]:
        city    = job.get("job_city") or ""
        country = job.get("job_country") or ""
        loc     = f"{city}, {country}".strip(", ") if city else country

        jobs.append({
            "title":       job.get("job_title", "N/A"),
            "company":     job.get("employer_name", "N/A"),
            "location":    loc or "N/A",
            "type":        job.get("job_employment_type", "").replace("_", " ").title(),
            "source":      job.get("job_publisher", "N/A"),
            "posted":      (job.get("job_posted_at_datetime_utc") or "")[:10],
            "apply_link":  job.get("job_apply_link", ""),
            "description": (job.get("job_description") or "")[:1500].strip(),
            "snippet":     (job.get("job_description") or "")[:250].strip(),
        })

    return jobs


def format_jobs_for_llm(jobs: List[Dict], query: str, location: str) -> str:
    """
    Converts job list to a clean text block for LLM context injection.
    """
    if not jobs:
        return f"[Job search for '{query}' in {location} returned no results. Advise the user to broaden their search or check back later.]"

    lines = [
        f"=== LIVE JOB LISTINGS — '{query}' in {location} (via LinkedIn / Indeed / Glassdoor) ===",
    ]
    for i, j in enumerate(jobs, 1):
        lines.append(
            f"\n{i}. {j['title']} — {j['company']}\n"
            f"   Location: {j['location']} | Type: {j['type']} | Source: {j['source']} | Posted: {j['posted']}\n"
            f"   Apply: {j['apply_link']}\n"
            f"   About: {j['snippet']}..."
        )
    lines.append("\n=== END JOB LISTINGS ===")
    lines.append(
        "Present these as a clean table (Title | Company | Location | Source | Apply Link). "
        "Then briefly advise which 1-2 best match the candidate's resume and why."
    )
    return "\n".join(lines)
