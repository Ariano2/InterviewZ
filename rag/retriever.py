"""
rag/retriever.py
Hybrid BM25 + dense (BGE) retrieval with score fusion.

Pipeline per query:
  1. Dense  — Chroma cosine similarity, fetch TOP_CANDIDATES candidates
  2. BM25   — Okapi BM25 over the full corpus, fetch TOP_CANDIDATES candidates
  3. Union  — merge candidate sets; non-overlapping candidates get score=0 on the missing side
  4. Normalise — each score list independently min-max normalised to [0, 1]
  5. Fuse   — hybrid = ALPHA * dense_norm + (1 - ALPHA) * bm25_norm
  6. Filter — drop chunks below MIN_HYBRID_SCORE (irrelevant queries inject nothing)
  7. Return — top-k by hybrid score, capped at k

Viva defence notes:
  - BM25 (Okapi BM25, Robertson & Sparck Jones 1994) is a probabilistic TF-IDF variant.
    It handles exact proper-noun matches (e.g. "CERT-IN", "MERN") that dense embedding
    geometry may not rank highly.
  - ALPHA=0.6 biases toward semantic understanding while retaining lexical precision.
    This is consistent with production hybrid systems (Elasticsearch, Pinecone Hybrid).
  - MIN_HYBRID_SCORE=0.15 is a soft gate: only inject context when at least one signal
    agrees the chunk is relevant. Prevents padding the prompt with noise.
"""

import json
import re
from pathlib import Path

from rag.ingest import BM25_CORPUS_PATH, load_vectorstore

# ── Tunable constants ─────────────────────────────────────────────────────────
ALPHA            = 0.6   # weight for dense score  (1-ALPHA goes to BM25)
TOP_CANDIDATES   = 10    # candidates fetched from each retriever before fusion
MIN_HYBRID_SCORE = 0.15  # chunks below this are dropped (noise filter)
RAW_DENSE_FLOOR  = 0.20  # minimum raw cosine similarity before any retrieval happens
                         # guards against injecting resume chunks for greetings/small talk


# ── BM25 helpers ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric. Keeps hyphenated tokens intact."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def _load_bm25():
    """
    Load corpus from disk and build a BM25Okapi index.
    Returns (index, corpus_list) or (None, []) if corpus missing.
    """
    path = Path(BM25_CORPUS_PATH)
    if not path.exists():
        return None, []
    try:
        from rank_bm25 import BM25Okapi
        corpus = json.loads(path.read_text(encoding="utf-8"))
        tokenized = [_tokenize(entry["text"]) for entry in corpus]
        return BM25Okapi(tokenized), corpus
    except Exception:
        return None, []


def _bm25_scores_for_query(query: str) -> dict[int, float]:
    """
    Returns {chunk_index: raw_bm25_score} for every chunk in the corpus.
    Returns {} if BM25 index cannot be built.
    """
    bm25, corpus = _load_bm25()
    if bm25 is None or not corpus:
        return {}
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)          # ndarray, one score per corpus entry
    return {entry["chunk_index"]: float(scores[i]) for i, entry in enumerate(corpus)}


# ── Normalisation ─────────────────────────────────────────────────────────────

def _minmax(scores: dict[int, float]) -> dict[int, float]:
    """
    Min-max normalise a {chunk_index: score} dict to [0, 1].

    Special cases:
      all zeros  → 0.0 for all  (nothing matched — do NOT inflate to 1.0)
      all equal and non-zero → 1.0 for all (rare, e.g. identical chunks)
    """
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == 0.0:
        # Every score is zero — no signal from this retriever
        return {k: 0.0 for k in scores}
    if hi == lo:
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve_with_scores(
    query: str,
    k: int = 4,
    vectorstore=None,
) -> list[dict]:
    """
    Hybrid retrieval. Returns up to k chunks, sorted by hybrid score descending.

    Each dict:
      text         — chunk text
      chunk_index  — position in original split
      dense_score  — cosine similarity from BGE (0–1)
      bm25_score   — normalised BM25 Okapi score (0–1)
      hybrid_score — ALPHA * dense + (1-ALPHA) * bm25  (0–1)

    Accepts an optional pre-loaded vectorstore to avoid reloading the embedding
    model on every call (pass st.session_state.vectorstore from the app).
    """
    try:
        vs = vectorstore or load_vectorstore()

        # ── 1. Dense retrieval — fetch more candidates than we'll return ──────
        raw = vs.similarity_search_with_score(query, k=TOP_CANDIDATES)
        # Chroma returns squared L2; convert to cosine for normalised vectors
        dense_raw: dict[int, float] = {}
        dense_texts: dict[int, str] = {}
        for doc, l2_sq in raw:
            idx = doc.metadata.get("chunk_index", -1)
            cosine = max(0.0, min(1.0, 1.0 - l2_sq / 2.0))
            dense_raw[idx] = cosine
            dense_texts[idx] = doc.page_content

        # Early exit: if even the best raw cosine is below floor, the query is
        # conversational / off-topic (e.g. "Hello", "thanks"). Don't inject noise.
        if not dense_raw or max(dense_raw.values()) < RAW_DENSE_FLOOR:
            return []

        # ── 2. BM25 retrieval — full corpus scores ────────────────────────────
        bm25_raw = _bm25_scores_for_query(query)

        # ── 3. Union of candidate indices ─────────────────────────────────────
        # Take top-TOP_CANDIDATES from BM25 by raw score to keep candidate set bounded
        top_bm25_indices = set(
            sorted(bm25_raw, key=bm25_raw.get, reverse=True)[:TOP_CANDIDATES]
        )
        all_indices = set(dense_raw.keys()) | top_bm25_indices

        # Fill in missing scores with 0 so both dicts cover all candidates
        for idx in all_indices:
            dense_raw.setdefault(idx, 0.0)
            bm25_raw.setdefault(idx, 0.0)

        # ── 4. Normalise independently ────────────────────────────────────────
        dense_norm = _minmax(dense_raw)
        bm25_norm  = _minmax(bm25_raw)

        # ── 5. Fuse ───────────────────────────────────────────────────────────
        fused = {
            idx: round(ALPHA * dense_norm[idx] + (1 - ALPHA) * bm25_norm[idx], 4)
            for idx in all_indices
        }

        # ── 6. Resolve texts for BM25-only candidates (not in Chroma results) ─
        # Load corpus once if needed
        _, corpus = _load_bm25()
        corpus_map: dict[int, str] = {e["chunk_index"]: e["text"] for e in corpus}
        for idx in all_indices:
            if idx not in dense_texts and idx in corpus_map:
                dense_texts[idx] = corpus_map[idx]

        # ── 7. Filter + sort + cap at k ───────────────────────────────────────
        chunks = [
            {
                "text":         dense_texts.get(idx, ""),
                "chunk_index":  idx,
                "dense_score":  round(dense_norm.get(idx, 0.0), 4),
                "bm25_score":   round(bm25_norm.get(idx, 0.0), 4),
                "hybrid_score": fused[idx],
            }
            for idx in all_indices
            if fused[idx] >= MIN_HYBRID_SCORE and dense_texts.get(idx, "").strip()
        ]
        chunks.sort(key=lambda c: c["hybrid_score"], reverse=True)
        return chunks[:k]

    except Exception:
        return []


def retrieve_context(query: str, k: int = 4, vectorstore=None) -> str:
    """
    Convenience wrapper — returns a concatenated context string.
    Used by agents that don't need per-chunk scores.
    """
    chunks = retrieve_with_scores(query, k=k, vectorstore=vectorstore)
    if not chunks:
        return ""
    return "\n\n---\n\n".join(c["text"] for c in chunks)
