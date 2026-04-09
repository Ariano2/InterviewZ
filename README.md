# 🎯 PrepSense AI — Resume Reviewer

RAG-powered resume intelligence built with **Streamlit + Groq + LangChain + Chroma**.

## Stack

| Layer | Technology |
|---|---|
| UI | Streamlit ≥ 1.34 |
| LLM (primary) | Groq API — `llama-3.3-70b-versatile` |
| LLM (optional) | Ollama — local via OpenAI-compatible endpoint |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` — runs fully locally |
| Vector Store | Chroma — persistent local DB |
| PDF parsing | PyMuPDF (`fitz`) |
| DOCX parsing | python-docx |
| Env management | python-dotenv |

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key (free at console.groq.com)
cp .env.example .env
# Edit .env → add your key

# 3. Launch
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Features

### 🏆 ATS Analysis
- Upload PDF or DOCX resume
- Groq scores ATS compatibility 0–100 for your target role
- Shows matched vs missing keywords as visual chips
- Lists 3–5 strong areas and 3–5 weak gaps

### ✍️ Bullet Point Rewriter
- Identifies 5–7 weak bullets (vague, passive, no metrics)
- Rewrites each with: power verb + quantified impact + role keywords
- Shows "Why it's stronger" for each rewrite
- Download all rewrites as `.txt`

### 💬 RAG-Powered Resume Chat
- Multi-turn conversation — ask anything about your resume
- Every reply is grounded via Chroma semantic search (top-4 chunks)
- Supports Groq (cloud, fast) or Ollama (local, private) via sidebar toggle
- Full conversation history kept in session state

### 📄 Raw Text View
- Inspect exactly how your resume was parsed
- Download as `.txt`

---

## Project Structure

```
resume_reviewer/
├── app.py                      ← Streamlit app (entry point)
├── requirements.txt
├── .env.example
│
├── utils/
│   └── file_parser.py          ← PDF / DOCX → plain text
│
├── rag/
│   ├── ingest.py               ← Chunk → embed → Chroma upsert
│   └── retriever.py            ← Similarity search → context string
│
└── agents/
    ├── ats_analyzer.py         ← ATS score + keywords (Groq, JSON output)
    ├── bullet_rewriter.py      ← Bullet improvement (Groq)
    └── chat_agent.py           ← RAG chat (Groq or Ollama)
```

---

## RAG Pipeline (for viva explanation)

```
Resume PDF/DOCX
      │
      ▼
file_parser.py  →  raw text string
      │
      ▼
RecursiveCharacterTextSplitter  (chunk_size=500, overlap=80)
      │
      ▼
HuggingFaceEmbeddings  (all-MiniLM-L6-v2, local, free, ~22MB)
      │
      ▼
Chroma.from_texts()  →  persisted to ./chroma_db/
      │
  [on query]
      │
      ▼
vectorstore.similarity_search(query, k=4)
      │
      ▼
top-4 chunks → injected into Groq system prompt → answer
```

## Viva Talking Points

1. **Why RAG instead of just pasting the resume?**  
   Chroma retrieves only the *relevant* sections per query. For a 3-page resume this matters less, but it demonstrates production-ready architecture and avoids context-window bloat for longer documents.

2. **Why `all-MiniLM-L6-v2` for embeddings?**  
   22MB model, runs in-process on CPU, zero API cost, zero network calls. Cosine similarity in ~5ms per query.

3. **Why Chroma?**  
   Zero server setup, persists to disk automatically, works offline, and has first-class LangChain integration.

4. **Why Groq over OpenAI?**  
   ~300 tokens/second on free tier, same REST API shape, `llama-3.3-70b` is competitive quality. Perfect for live demos.

5. **Ollama toggle?**  
   Shows multi-LLM awareness. Same interface, just swaps to `localhost:11434/v1`. Useful for offline/private use cases — relevant for enterprise deployments.

6. **ATS JSON output reliability?**  
   The prompt enforces JSON-only output. Post-processing strips any markdown fences with regex before `json.loads()`. Typed as `ATSResult` TypedDict. Falls back to a safe default dict on any exception — never crashes the UI.
