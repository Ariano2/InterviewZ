"""
rag/ingest.py
Chunks resume text, embeds with sentence-transformers, persists to Supabase pgvector.
Also returns the plain-text corpus for in-memory BM25 hybrid retrieval.
"""
import contextlib
import logging
import os
import warnings

# Suppress noisy HF / transformer logs before any heavy imports
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

for _logger in ("transformers", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(_logger).setLevel(logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client, Client

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@contextlib.contextmanager
def _quiet():
    """Redirect stdout/stderr to /dev/null — catches any print-based chatter."""
    with open(os.devnull, "w") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


def _get_embeddings() -> HuggingFaceEmbeddings:
    with _quiet():
        return HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


def _get_supabase(access_token: str) -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_ANON_KEY"]
    client = create_client(url, key)
    client.postgrest.auth(access_token)
    return client


def ingest_resume(text: str, user_id: str, access_token: str) -> dict:
    """
    Split → embed → upsert into Supabase pgvector.

    Returns a dict with:
      corpus  — list[{chunk_index, text}]  for BM25 hybrid retrieval
      vectors — list[list[float]]           for PCA visualisation (pre-computed, free)
    """
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_text(text)

    if not chunks:
        raise ValueError("Resume text is empty after splitting. Check the uploaded file.")

    embeddings_model = _get_embeddings()
    with _quiet():
        vectors = embeddings_model.embed_documents(chunks)

    supabase = _get_supabase(access_token)

    # Delete stale chunks for this user so re-uploads don't mix data
    supabase.table("resume_chunks").delete().eq("user_id", user_id).execute()

    # Insert new chunks with their vectors
    rows = [
        {
            "user_id":     user_id,
            "chunk_index": i,
            "content":     chunk,
            "embedding":   vector,
        }
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    supabase.table("resume_chunks").insert(rows).execute()

    corpus = [{"text": c, "chunk_index": i} for i, c in enumerate(chunks)]
    return {"corpus": corpus, "vectors": vectors}
