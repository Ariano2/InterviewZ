"""
rag/ingest.py
Chunks resume text, embeds with sentence-transformers, persists to Chroma.
"""
import warnings
warnings.filterwarnings("ignore")

import os
# ── Environment setup to prevent repeated downloads ──
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

# Suppress ALL logs and warnings from transformers and sentence-transformers
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)  # fixes BertModel LOAD REPORT
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Use transformers' own internal verbosity API — most reliable suppression
import transformers
transformers.logging.set_verbosity_error()

# Suppress tqdm progress bars
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Persistent directory for Chroma
CHROMA_PERSIST_DIR = "./chroma_db"

# Small, fast, completely local embedding model
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Shared collection name
COLLECTION_NAME = "resume_chunks"


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance with all warnings suppressed."""
    import sys

    # Re-apply transformers verbosity here in case it was reset during import
    transformers.logging.set_verbosity_error()
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    # Redirect both stdout and stderr to devnull to catch any print-based messages
    from contextlib import contextmanager

    @contextmanager
    def suppress_output():
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    with suppress_output():
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return embeddings


def ingest_resume(text: str) -> Chroma:
    """
    Split → embed → upsert into Chroma.
    Returns the Chroma vectorstore instance.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError("Resume text appears empty after splitting. Check the file.")

    metadatas = [{"source": "resume", "chunk_index": i} for i in range(len(chunks))]
    embeddings = _get_embeddings()

    # Delete existing collection to avoid stale data
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load the persisted Chroma collection for querying."""
    embeddings = _get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )