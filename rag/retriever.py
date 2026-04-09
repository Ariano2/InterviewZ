"""
rag/retriever.py
Retrieves top-k relevant chunks from Chroma for a given query string.
"""

from rag.ingest import load_vectorstore


def retrieve_context(query: str, k: int = 4) -> str:
    """
    Returns a concatenated string of the top-k most relevant resume chunks.
    Returns empty string if the index doesn't exist yet.
    """
    try:
        vs = load_vectorstore()
        docs = vs.similarity_search(query, k=k)
        if not docs:
            return ""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    except Exception:
        # Collection not built yet or Chroma error — graceful fallback
        return ""
