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
import os
import re
import warnings
from typing import TypedDict

from groq import Groq

# ── Sentence-transformer + KeyBERT models (lazy-loaded, cached across calls) ──
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_EMBED_MODEL   = None
_KEYBERT_MODEL = None


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        warnings.filterwarnings("ignore")
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _EMBED_MODEL


def _get_keybert():
    global _KEYBERT_MODEL
    if _KEYBERT_MODEL is None:
        from keybert import KeyBERT
        # Reuse the already-loaded BGE model — no extra download
        _KEYBERT_MODEL = KeyBERT(model=_get_embed_model())
    return _KEYBERT_MODEL


def extract_keywords_keybert(text: str, top_n: int = 15) -> list:
    """
    Extract keyphrases from `text` using KeyBERT + BGE-small embeddings.

    Uses Maximal Marginal Relevance (MMR) to balance relevance against
    redundancy — so you get diverse keywords, not ten synonyms of the same idea.

    keyphrase_ngram_range=(1,2): single words AND two-word phrases.
    diversity=0.5: midpoint between max-relevance and max-diversity.

    Returns a list of keyword strings (no scores).
    """
    try:
        kw_model = _get_keybert()
        results  = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n,
            use_mmr=True,
            diversity=0.5,
        )
        return [kw for kw, _score in results]
    except Exception:
        return []


def compute_similarity_metrics(resume_text: str, jd_text: str) -> dict:
    """
    Encodes resume and JD with all-MiniLM-L6-v2 and computes four metrics.

    All four use the same two embedding vectors — no extra model calls.

    Metrics:
      cosine     — cos(θ) = A·B  (= dot product since vectors are unit-normalised).
                   Measures angle; magnitude-independent.

      euclidean  — sqrt(Σ(Ai−Bi)²).  For unit vectors, d ∈ [0, 2], so
                   euclidean_sim = 1 − d/2  maps it back to [0, 1].
                   Equivalent to cosine for normalised vectors (d = sqrt(2*(1−cos))),
                   but on a distance scale — different presentation, same geometry.

      manhattan  — Σ|Ai−Bi|  (L1 norm).  Less sensitive than L2 to a few
                   dominant dimensions blowing up the score.
                   Converted: manhattan_sim = 1 / (1 + d/sqrt(dims)).

      pearson    — mean-centred cosine: corr(A−Ā, B−B̄).
                   Asks "do dimensions above average in the resume also tend to
                   be above average in the JD?" — captures co-variance, not just
                   alignment.

    Returns dict with keys: cosine, euclidean, manhattan, pearson.
    Values are floats in [0.0, 1.0].  Returns all -1.0 on failure.
    """
    _fail = {"cosine": -1.0, "euclidean": -1.0, "manhattan": -1.0, "pearson": -1.0}
    try:
        import numpy as np
        from sentence_transformers import util

        model = _get_embed_model()
        a = model.encode(resume_text[:4000], normalize_embeddings=True)  # shape (384,)
        b = model.encode(jd_text[:4000],     normalize_embeddings=True)

        # Cosine (dot product on unit vectors)
        cosine = float(np.clip(np.dot(a, b), 0.0, 1.0))

        # Euclidean similarity  (distance → similarity)
        euc_dist = float(np.linalg.norm(a - b))          # ∈ [0, 2] for unit vecs
        euclidean = round(float(np.clip(1.0 - euc_dist / 2.0, 0.0, 1.0)), 3)

        # Manhattan similarity
        man_dist  = float(np.sum(np.abs(a - b)))
        dims      = float(len(a))
        manhattan = round(float(np.clip(1.0 / (1.0 + man_dist / (dims ** 0.5)), 0.0, 1.0)), 3)

        # Pearson correlation
        a_c = a - a.mean();  b_c = b - b.mean()
        denom = (np.linalg.norm(a_c) * np.linalg.norm(b_c))
        pearson = round(float(np.clip(np.dot(a_c, b_c) / denom, 0.0, 1.0)), 3) if denom > 1e-9 else 0.0

        return {
            "cosine":    round(cosine,    3),
            "euclidean": euclidean,
            "manhattan": manhattan,
            "pearson":   pearson,
        }
    except Exception:
        return _fail


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
    semantic_score: float   # cosine similarity vs JD; -1.0 when no JD or failure
    similarity_metrics: dict  # {cosine, euclidean, manhattan, pearson}; empty when no JD
    # KeyBERT
    keybert_resume_kws: list  # BERT-extracted keyphrases from resume
    keybert_jd_kws: list      # BERT-extracted keyphrases from JD ([] when no JD)
    keybert_overlap: list     # keyphrases appearing in both


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
    "semantic_score": -1.0,
    "similarity_metrics": {},
    "keybert_resume_kws": [],
    "keybert_jd_kws": [],
    "keybert_overlap": [],
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
        max_tokens=4000,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Model returned empty response")
    raw = re.sub(r"```(?:json)?\s*", "", content, flags=re.IGNORECASE).strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError(f"No JSON object found in response: {raw[:200]}")
    return json.loads(m.group(0))


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
        # Step 1 — Model: embedding-based similarity metrics (no LLM involved)
        if jd_text.strip():
            sim_metrics    = compute_similarity_metrics(resume_text, jd_text)
            semantic_score = sim_metrics.get("cosine", -1.0)
        else:
            sim_metrics    = {}
            semantic_score = -1.0

        # Step 1b — KeyBERT: unsupervised BERT keyword extraction (no LLM)
        kb_resume_kws = extract_keywords_keybert(resume_text[:MAX_RESUME_CHARS], top_n=15)
        kb_jd_kws     = extract_keywords_keybert(jd_text[:MAX_JD_CHARS], top_n=15) if jd_text.strip() else []
        # Overlap: JD keyword appears (substring) in any resume keyword or vice versa
        resume_kws_lower = " ".join(kb_resume_kws).lower()
        kb_overlap = [kw for kw in kb_jd_kws if kw.lower() in resume_kws_lower]

        # Step 2 — LLM: keywords + qualitative ratings
        llm = _llm_analysis(resume_text, target_role, client, jd_text)

        required_kws = [str(k) for k in (llm.get("required_keywords") or []) if k]

        # Step 3 — Programmatic sub-scores
        matched, missing, kw_pct = _check_keywords(resume_text, required_kws)
        quant_score, quant_detail = _quantification_rate(resume_text)
        section_score = _section_completeness(resume_text)

        # Step 4 — LLM-rated sub-scores (clamped)
        action_score = max(0, min(100, int(llm.get("action_verb_rating") or 0)
                                  or _programmatic_action_verb_score(resume_text)))
        fmt_score    = max(0, min(100, int(llm.get("formatting_rating") or 70)))

        # Step 5 — Weighted final score
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
            semantic_score=semantic_score,
            similarity_metrics=sim_metrics,
            keybert_resume_kws=kb_resume_kws,
            keybert_jd_kws=kb_jd_kws,
            keybert_overlap=kb_overlap,
        )

    except Exception:
        return _FALLBACK
