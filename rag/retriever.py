"""
rag/retriever.py
Hybrid BM25 + dense (BGE via Supabase pgvector) retrieval with score fusion.

Pipeline per query:
  1. Dense  — Supabase match_resume_chunks RPC (cosine similarity), TOP_CANDIDATES results
  2. BM25   — Okapi BM25 over the in-memory corpus (passed from session state), TOP_CANDIDATES
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

import contextlib
import logging
import os
import re
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

for _logger in ("transformers", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(_logger).setLevel(logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()

from langchain_huggingface import HuggingFaceEmbeddings
from supabase import create_client, Client

# ── Tunable constants ─────────────────────────────────────────────────────────
ALPHA            = 0.6   # weight for dense score  (1-ALPHA goes to BM25)
TOP_CANDIDATES   = 10    # candidates fetched from each retriever before fusion
MIN_HYBRID_SCORE = 0.15  # chunks below this are dropped (noise filter)
RAW_DENSE_FLOOR  = 0.20  # minimum raw cosine similarity before any retrieval happens
                         # guards against injecting resume chunks for greetings/small talk

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


_EMBED_MODEL_CACHE: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Returns a cached HuggingFaceEmbeddings instance — loads once per process."""
    global _EMBED_MODEL_CACHE
    if _EMBED_MODEL_CACHE is None:
        with _quiet():
            _EMBED_MODEL_CACHE = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    return _EMBED_MODEL_CACHE


def _get_supabase(access_token: str) -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_ANON_KEY"]
    client = create_client(url, key)
    client.postgrest.auth(access_token)
    return client


# ── BM25 helpers ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric. Keeps hyphenated tokens intact."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def _bm25_scores_for_query(query: str, corpus: list[dict]) -> dict[int, float]:
    """
    Returns {chunk_index: raw_bm25_score} over the in-memory corpus.
    Returns {} if corpus is empty or rank_bm25 unavailable.
    """
    if not corpus:
        return {}
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [_tokenize(entry["text"]) for entry in corpus]
        bm25 = BM25Okapi(tokenized)
        tokens = _tokenize(query)
        scores = bm25.get_scores(tokens)
        return {entry["chunk_index"]: float(scores[i]) for i, entry in enumerate(corpus)}
    except Exception:
        return {}


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
        return {k: 0.0 for k in scores}
    if hi == lo:
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


# ── Dense retrieval via Supabase pgvector ─────────────────────────────────────

def _dense_scores(
    query: str,
    access_token: str,
) -> tuple[dict[int, float], dict[int, str]]:
    """
    Embed query, call match_resume_chunks RPC, return (scores, texts) dicts.
    RLS on the table ensures only the authenticated user's chunks are returned.
    """
    embeddings_model = _get_embeddings()
    with _quiet():
        q_vec = embeddings_model.embed_query(query)

    supabase = _get_supabase(access_token)
    result = supabase.rpc(
        "match_resume_chunks",
        {"query_embedding": q_vec, "match_count": TOP_CANDIDATES},
    ).execute()

    scores: dict[int, float] = {}
    texts:  dict[int, str]   = {}
    for row in (result.data or []):
        idx = row["chunk_index"]
        scores[idx] = float(row["similarity"])
        texts[idx]  = row["content"]

    return scores, texts


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve_with_scores(
    query: str,
    k: int = 4,
    corpus: list[dict] | None = None,
    access_token: str = "",
) -> list[dict]:
    """
    Hybrid retrieval. Returns up to k chunks, sorted by hybrid score descending.

    Each dict:
      text         — chunk text
      chunk_index  — position in original split
      dense_score  — cosine similarity from BGE pgvector (0–1)
      bm25_score   — normalised BM25 Okapi score (0–1)
      hybrid_score — ALPHA * dense + (1-ALPHA) * bm25  (0–1)

    corpus  — in-memory list[{chunk_index, text}] built at ingest time (for BM25).
              Pass st.session_state.vectorstore from the app.
    access_token — Supabase JWT for the authenticated user (for dense RPC).
    """
    if not access_token:
        return []
    corpus = corpus or []

    try:
        # ── 1. Dense retrieval ────────────────────────────────────────────────
        dense_raw, dense_texts = _dense_scores(query, access_token)

        # Early exit: if even the best raw cosine is below floor, the query is
        # conversational / off-topic (e.g. "Hello", "thanks"). Don't inject noise.
        if not dense_raw or max(dense_raw.values()) < RAW_DENSE_FLOOR:
            return []

        # ── 2. BM25 retrieval — full corpus scores ────────────────────────────
        bm25_raw = _bm25_scores_for_query(query, corpus)

        # ── 3. Union of candidate indices ─────────────────────────────────────
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

        # ── 6. Resolve texts for BM25-only candidates ─────────────────────────
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


def retrieve_context(
    query: str,
    k: int = 4,
    corpus: list[dict] | None = None,
    access_token: str = "",
) -> str:
    """
    Convenience wrapper — returns a concatenated context string.
    Used by agents that don't need per-chunk scores.
    """
    chunks = retrieve_with_scores(query, k=k, corpus=corpus, access_token=access_token)
    if not chunks:
        return ""
    return "\n\n---\n\n".join(c["text"] for c in chunks)
