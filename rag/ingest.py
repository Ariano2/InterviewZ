"""
rag/ingest.py
Chunks resume text, embeds with sentence-transformers, persists to Chroma.
Also saves a plain-text BM25 corpus (JSON) alongside Chroma for hybrid retrieval.
"""
import contextlib
import io
import json
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

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PERSIST_DIR = "./chroma_db"
EMBED_MODEL_NAME   = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME    = "resume_chunks"
BM25_CORPUS_PATH   = "./chroma_db/bm25_corpus.json"


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


def ingest_resume(text: str) -> Chroma:
    """Split → embed → upsert into Chroma. Returns the vectorstore."""
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_text(text)

    if not chunks:
        raise ValueError("Resume text is empty after splitting. Check the uploaded file.")

    metadatas  = [{"source": "resume", "chunk_index": i} for i in range(len(chunks))]
    embeddings = _get_embeddings()

    # Drop stale collection so re-uploads don't mix data
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    with contextlib.suppress(Exception):
        client.delete_collection(COLLECTION_NAME)

    vs = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )

    # Persist BM25 corpus — plain text list for lexical retrieval
    corpus = [{"text": c, "chunk_index": i} for i, c in enumerate(chunks)]
    with open(BM25_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    return vs


def load_vectorstore() -> Chroma:
    """Load the persisted Chroma collection for querying."""
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=_get_embeddings(),
        collection_name=COLLECTION_NAME,
    )
