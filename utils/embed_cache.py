"""
utils/embed_cache.py
Process-level singleton for BAAI/bge-small-en-v1.5.

Importing this module from ats_analyzer.py AND job_matcher.py means
Python only ever holds ONE SentenceTransformer instance for the whole
process — saving ~130 MB of duplicate model weights in RAM.

Usage:
    from utils.embed_cache import get_bge_model
    model = get_bge_model()   # fast no-op after first call
"""
import os
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

_BGE_MODEL = None


def get_bge_model():
    """
    Returns the shared BAAI/bge-small-en-v1.5 SentenceTransformer.
    Loads on first call; subsequent calls return the cached instance.
    Thread-safe in CPython (GIL protects the assignment).
    """
    global _BGE_MODEL
    if _BGE_MODEL is None:
        warnings.filterwarnings("ignore")
        from sentence_transformers import SentenceTransformer
        _BGE_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _BGE_MODEL
