"""
agents/job_matcher.py
Semantic job matching — ranks live job listings by cosine similarity
to the user's resume using BGE-small-en-v1.5 embeddings.

Pipeline:
  1. Embed resume text (truncated to 4000 chars, ~512 tokens)
  2. Embed each job's title + company + description
  3. Cosine similarity = dot product on unit-normalised vectors
  4. Return jobs sorted by match_score descending
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

# Shared BGE singleton — same instance as ats_analyzer uses (~130 MB saved)
from utils.embed_cache import get_bge_model as _get_model


def match_jobs_to_resume(resume_text: str, jobs: list[dict], resume_embedding=None) -> list[dict]:
    """
    Semantically ranks jobs against the resume.

    Each returned dict is a copy of the input job dict with one added key:
      match_score — int 0-100, percentage cosine similarity to resume

    Jobs with no description fall back to title+company only — scores
    will be lower but still valid for ranking.
    """
    if not jobs or not resume_text:
        return jobs

    import numpy as np

    model = _get_model()

    # Use pre-computed embedding if provided — avoids re-encoding same resume text
    # on repeated job searches within the same session.
    if resume_embedding is not None:
        resume_vec = resume_embedding
    else:
        # Embed resume — truncate to 4000 chars (BGE-small handles ~512 tokens,
        # 4000 ASCII chars ≈ 700-800 tokens, model silently truncates to 512)
        resume_vec = model.encode(
            resume_text[:4000],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    # Build a text representation for each job
    jd_texts = []
    for j in jobs:
        description = j.get("description") or j.get("snippet") or ""
        text = f"{j.get('title', '')} at {j.get('company', '')}\n{description}"
        jd_texts.append(text[:2000])  # cap at 2000 chars per job

    jd_vecs = model.encode(
        jd_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=8,
    )

    # Cosine similarity = dot product for unit vectors → shape (n_jobs,)
    scores = np.dot(jd_vecs, resume_vec)

    results = []
    for job, score in zip(jobs, scores):
        pct = int(round(float(score) * 100))
        pct = max(0, min(100, pct))
        results.append({**job, "match_score": pct})

    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results
