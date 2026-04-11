# PrepSense AI — Technical Project Report

**Project Title:** PrepSense AI — An Intelligent Resume Analysis and Career Preparation Platform  
**Technology Stack:** Python · Streamlit · Groq LLM API · Supabase (pgvector + Auth) · BGE Embeddings · BM25 Hybrid RAG  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Authentication & User Management](#3-authentication--user-management)
4. [RAG Subsystem — Ingestion](#4-rag-subsystem--ingestion)
5. [RAG Subsystem — Hybrid Retrieval](#5-rag-subsystem--hybrid-retrieval)
6. [Agent: ATS Analyzer](#6-agent-ats-analyzer)
7. [Agent: Bullet Rewriter](#7-agent-bullet-rewriter)
8. [Agent: JD Tailor & Cover Letter](#8-agent-jd-tailor--cover-letter)
9. [Agent: Resume Chat](#9-agent-resume-chat)
10. [Agent: Skill Gap Analyzer](#10-agent-skill-gap-analyzer)
11. [Agent: Resume Structurer](#11-agent-resume-structurer)
12. [Agent: PDF Resume Builder](#12-agent-pdf-resume-builder)
13. [Agent: Portfolio Generator](#13-agent-portfolio-generator)
14. [Agent: GitHub Publisher](#14-agent-github-publisher)
15. [Agent: Interview Prep](#15-agent-interview-prep)
16. [Agent: Upskill Recommender](#16-agent-upskill-recommender)
17. [Agent: Job Search](#17-agent-job-search)
18. [File Parser Utility](#18-file-parser-utility)
19. [Application Entry Point & Session Management](#19-application-entry-point--session-management)
20. [Model Switching & Comparison Mode](#20-model-switching--comparison-mode)
21. [PCA Embedding Visualization](#21-pca-embedding-visualization)
22. [Design Decisions & Architectural Trade-offs](#22-design-decisions--architectural-trade-offs)
23. [Constants & Configuration Reference](#23-constants--configuration-reference)
24. [Data Flow Diagrams](#24-data-flow-diagrams)
25. [Error Handling & Fallback Strategy](#25-error-handling--fallback-strategy)
26. [Performance Characteristics](#26-performance-characteristics)
27. [Security Considerations](#27-security-considerations)
28. [Dependencies & External Services](#28-dependencies--external-services)

---

## 1. Project Overview

PrepSense AI is a full-stack AI-powered career preparation platform built for the Indian job market. It takes a candidate's resume as input and provides a suite of intelligent tools: ATS scoring, bullet rewriting, job-description tailoring, RAG-powered career chat, portfolio website generation, and interview preparation.

The system is built on Streamlit for the frontend, Groq for LLM inference, Supabase for authentication and persistent vector storage, and a locally run BGE embedding model for semantic search. All major analysis tasks are delegated to discrete agents, each with a well-defined input/output contract, a carefully tuned prompt, and a safe fallback path.

### 1.1 Feature Set

| Feature | Agent | Output |
|---------|-------|--------|
| ATS Scoring | `ats_analyzer` | 0-100 rubric score with sub-scores |
| Bullet Rewriting | `bullet_rewriter` | Before/after pairs + downloadable PDF |
| JD Tailoring | `jd_tailor` | Keyword-injected resume + cover letter |
| Resume Chat | `chat_agent` | Multi-turn RAG-grounded conversation |
| Skill Gap Radar | `skill_gap` | Radar chart (5 categories) |
| Portfolio Builder | `portfolio_generator` | Full HTML/CSS/JS portfolio website |
| GitHub Publishing | `github_publisher` | Live GitHub Pages URL |
| Interview Prep | `interview_prep` | Q&A in Easy / Medium / Hard tiers |
| Upskill Planner | `upskill` | 4-week learning plan per skill |
| Live Job Search | `job_search` | LinkedIn / Indeed / Glassdoor results |
| Model Comparison | app.py inline | Two-model ATS comparison side-by-side |

### 1.2 Target User Profile

The system targets Indian job seekers at any experience level, across all domains (software, finance, law, medicine, design, civil services). The LLM system prompts are calibrated for the Indian job market — referencing off-campus hiring, Naukri/LinkedIn, tier-1 vs tier-2/3 companies, metro vs tier-2 cities, and startup vs MNC vs PSU trade-offs.

---

## 2. System Architecture

### 2.1 High-Level Component Map

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit Frontend (app.py)            │
│   Auth Gate → Sidebar (model picker) → 7 Tabs              │
└──────────────┬──────────────────────┬───────────────────────┘
               │                      │
    ┌──────────▼──────────┐  ┌────────▼──────────────────────┐
    │   Supabase Platform  │  │        Groq LLM API            │
    │  - Auth (JWT)        │  │  - openai/gpt-oss-120b         │
    │  - pgvector table    │  │  - llama-3.3-70b-versatile     │
    │  - Row Level Security│  │  - llama-3.1-8b-instant        │
    └──────────┬──────────┘  │  - gemma2-9b-it                │
               │              └────────┬──────────────────────┘
    ┌──────────▼──────────┐           │
    │   RAG Subsystem      │  ┌────────▼──────────────────────┐
    │  rag/ingest.py       │  │        Agent Layer             │
    │  rag/retriever.py    │  │  11 discrete agents            │
    │  BGE-small-en-v1.5   │  │  Each: prompt + call + parse   │
    │  BM25 Okapi (in-mem) │  │  + fallback                    │
    └──────────────────────┘  └───────────────────────────────┘
```

### 2.2 Request Lifecycle (Upload → Chat)

1. **User uploads resume** (PDF or DOCX)
2. `utils/file_parser.py` extracts plain text
3. `rag/ingest.py` chunks → embeds (BGE-small) → upserts to Supabase pgvector
4. Vectors returned to app.py → PCA computed → stored in session state
5. User sends a chat message
6. `rag/retriever.py` embeds the query, calls Supabase RPC for dense results, runs BM25 on in-memory corpus, fuses scores
7. Top-4 chunks injected into system prompt
8. `chat_agent.py` calls Groq with optional tool (job search)
9. Reply returned, session summary updated if history is long

### 2.3 Agent Isolation Pattern

Every agent follows the same contract:

```
Input:  resume_text (str) + task-specific params + groq_client + model (str)
Output: typed dict / list / string
Errors: always caught internally → return empty fallback, never crash Streamlit
```

No agent holds state between calls. All state lives in `st.session_state` in app.py.

### 2.4 Technology Choices Summary

| Component | Technology | Why |
|-----------|------------|-----|
| UI Framework | Streamlit | Rapid prototyping, reactive session state |
| LLM Inference | Groq API | 300 tokens/sec, free tier, drop-in OpenAI compat |
| Vector DB | Supabase pgvector | Postgres extension, RLS for multi-tenancy, integrated auth |
| Auth | Supabase Auth | JWT-based, email/password, free tier |
| Embeddings | BGE-small-en-v1.5 | 384-dim, 22MB, best quality/size ratio |
| Lexical Retrieval | BM25 Okapi (rank-bm25) | Handles acronyms and proper nouns dense models miss |
| PDF Parse | PyMuPDF (fitz) | Fast, handles scanned PDFs better than pdfminer |
| PDF Build | reportlab | Zero system dependencies, pure Python |
| DOCX Parse | python-docx | Official OOXML library |
| Keyword Extract | KeyBERT + MMR | BERT-based, diversity-aware, no separate API call |
| Visualization | Plotly | Interactive charts in Streamlit |
| PCA | scikit-learn | Lightweight, only 2D projection needed |
| Job Search | JSearch (RapidAPI) | Aggregates LinkedIn + Indeed + Glassdoor |
| GitHub Auth | Device Flow OAuth | No redirect URL required (works in any Streamlit env) |

---

## 3. Authentication & User Management

### 3.1 Supabase Auth Integration

Authentication uses Supabase's built-in email/password auth. On startup, before rendering any application content, `app.py` checks for a valid session:

```python
if not st.session_state.supabase_user_id:
    _show_auth_page()
    st.stop()
```

If no user is authenticated, the auth page is shown and `st.stop()` halts all further execution — preventing any rendering of application tabs, sidebar features, or session data.

### 3.2 Auth Page Structure

The auth page renders two tabs: Login and Sign Up.

**Login flow:**
```python
res = sb.auth.sign_in_with_password({"email": email, "password": password})
st.session_state.supabase_user_id      = res.user.id
st.session_state.supabase_access_token = res.session.access_token
st.session_state.supabase_email        = res.user.email
st.rerun()
```

**Sign up flow:**
```python
res = sb.auth.sign_up({"email": email, "password": password})
# Email confirmation can be disabled in Supabase dashboard for dev/demo
```

On successful login, three values are stored in session state:
- `supabase_user_id` — UUID of the authenticated user (used for RLS scoping)
- `supabase_access_token` — JWT passed to all Supabase table queries and RPC calls
- `supabase_email` — displayed in sidebar

### 3.3 Logout

The sidebar contains a logout button that clears all three auth keys and reruns:

```python
if st.button("Logout"):
    for k in ("supabase_user_id", "supabase_access_token", "supabase_email"):
        st.session_state[k] = ""
    st.rerun()
```

This triggers the auth gate check, which immediately shows the login page.

### 3.4 Row-Level Security

The Supabase `resume_chunks` table has RLS enabled with a single policy:

```sql
create policy "own_chunks" on public.resume_chunks
    for all using (auth.uid() = user_id);
```

This means no query — regardless of application code — can access another user's chunks. The JWT is verified by Supabase and `auth.uid()` is resolved server-side.

### 3.5 Database Schema

```sql
create extension if not exists vector with schema extensions;

create table public.resume_chunks (
    id           bigserial primary key,
    user_id      uuid references auth.users(id) on delete cascade not null,
    chunk_index  int not null,
    content      text not null,
    embedding    vector(384),       -- BGE-small output dimension
    created_at   timestamptz default now()
);

alter table public.resume_chunks enable row level security;

create index on public.resume_chunks
    using ivfflat (embedding vector_cosine_ops) with (lists = 50);
```

The `ivfflat` index clusters vectors into 50 lists for approximate nearest-neighbor search. For a typical user (10-40 chunks), exact search would be equally fast, but IVFFlat is the production-standard choice.

### 3.6 Supabase RPC Function

```sql
create or replace function match_resume_chunks(
    query_embedding vector(384),
    match_count     int
)
returns table (chunk_index int, content text, similarity float)
language sql stable
as $$
    select
        chunk_index,
        content,
        1 - (embedding <=> query_embedding) as similarity
    from public.resume_chunks
    where user_id = auth.uid()
    order by embedding <=> query_embedding
    limit match_count;
$$;
```

The `<=>` operator is pgvector's cosine distance. `1 - cosine_distance = cosine_similarity`. The function is `stable` (read-only, no side effects), which allows PostgreSQL to cache results within a transaction. `auth.uid()` ensures RLS is applied even inside the function body.

---

## 4. RAG Subsystem — Ingestion

### 4.1 Overview

`rag/ingest.py` handles the full pipeline from raw resume text to stored dense vectors. It is called once per resume upload and replaces any previously stored chunks for that user.

### 4.2 Text Chunking

```python
RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

**Why these parameters:**

- `chunk_size=700`: Approximately 175-225 tokens at typical English tokenization ratios. This leaves headroom under the BGE-small context window (512 tokens) while providing meaningful semantic units.
- `chunk_overlap=50`: A 50-character overlap ensures that sentences or bullet points split at chunk boundaries are still represented in adjacent chunks. Without overlap, a keyword spanning a boundary would be missed by retrieval.
- `separators`: The list is tried in order. The splitter first tries `\n\n` (paragraph boundary), then `\n` (line break), then `. ` (sentence), then ` ` (word), and finally `""` (character) as a last resort. This hierarchy means chunks break at semantically meaningful boundaries wherever possible.

**Typical output:** A 500-word resume produces 8-15 chunks. A 1000-word resume produces 15-30 chunks.

### 4.3 Embedding Model

```python
HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

**BAAI/bge-small-en-v1.5** is a 22MB sentence transformer model from Beijing Academy of Artificial Intelligence. It produces 384-dimensional dense vectors. Key properties:

- **L2-normalized output** (enforced via `normalize_embeddings=True`): Cosine similarity between two normalized vectors equals their dot product, which is faster to compute. pgvector's `<=>` operator (cosine distance) works correctly on normalized vectors.
- **MTEB score:** Consistently outperforms `all-MiniLM-L6-v2` of equal size on retrieval benchmarks, with particular strength on asymmetric retrieval tasks (short query vs. long passage — exactly the RAG use case).
- **Local inference:** No API call, no latency, no cost. Runs on CPU in 5-10ms per chunk.

### 4.4 Supabase Upsert

```python
# 1. Delete old chunks for this user (idempotent re-upload)
supabase.table("resume_chunks").delete().eq("user_id", user_id).execute()

# 2. Insert new chunks
rows = [
    {
        "user_id":     user_id,
        "chunk_index": i,
        "content":     chunk,
        "embedding":   vector,   # list[float], length 384
    }
    for i, (chunk, vector) in enumerate(zip(chunks, vectors))
]
supabase.table("resume_chunks").insert(rows).execute()
```

The delete-before-insert pattern ensures a re-upload replaces rather than appends. `chunk_index` is the position in the original split — preserved across uploads to maintain consistent indexing in the sidebar and PCA visualization.

### 4.5 Return Value

```python
return {
    "corpus":  [{"text": c, "chunk_index": i} for i, c in enumerate(chunks)],
    "vectors": vectors,  # list[list[float]]
}
```

The corpus is kept in memory (stored in `st.session_state.vectorstore`) for BM25 retrieval, which needs the raw text. The vectors are returned so app.py can compute PCA immediately after ingest, without a second embedding pass.

### 4.6 Noise Suppression

The module suppresses verbose output from HuggingFace transformers using multiple strategies:

```python
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")
for _logger in ("transformers", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(_logger).setLevel(logging.ERROR)

@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
```

This prevents model loading progress bars and download messages from appearing in the Streamlit UI.

---

## 5. RAG Subsystem — Hybrid Retrieval

### 5.1 Design Philosophy

A naive dense-only retrieval system will miss exact keyword matches. If a user asks "Did I use Docker?" and the resume chunk says "Containerized the API with Docker Compose", pure semantic search works well. But if the user asks about "CERT-IN certification" and the resume uses that exact acronym, a dense embedding may not rank it highly because the embedding space doesn't strongly separate uncommon acronyms.

BM25 (Okapi BM25, Robertson & Sparck Jones 1994) is a probabilistic TF-IDF variant that handles this case: it rewards rare term matches, exact phrase matches, and proper nouns. The hybrid approach combines the strengths of both.

### 5.2 Retrieval Constants

```python
ALPHA            = 0.6   # Weight for dense score (1-ALPHA = 0.4 for BM25)
TOP_CANDIDATES   = 10    # Candidates fetched from each method before fusion
MIN_HYBRID_SCORE = 0.15  # Minimum fused score to include a chunk
RAW_DENSE_FLOOR  = 0.20  # If max cosine < 0.20, query is off-topic — return nothing
```

**ALPHA = 0.6 rationale:** Semantic understanding (dense) is the primary signal; BM25 is a correction signal for lexical gaps. A 60/40 split is consistent with production hybrid systems (Elasticsearch BM25+dense, Pinecone Hybrid Search, Vespa). Values closer to 0.5 treat both signals equally (less semantic bias); values closer to 0.7-0.8 over-rely on dense (weakens acronym handling).

**RAW_DENSE_FLOOR = 0.20 rationale:** When a user types "hello", "thanks", or "how are you?", the query embedding has essentially no overlap with any resume chunk. Returning random low-score chunks as "context" would inject noise into the system prompt and degrade response quality. The floor acts as an early exit before any expensive operations.

### 5.3 Dense Retrieval

```python
def _dense_scores(query: str, access_token: str) -> tuple[dict[int, float], dict[int, str]]:
    q_vec = embeddings_model.embed_query(query)   # 384-dim, normalized
    
    result = supabase.rpc(
        "match_resume_chunks",
        {"query_embedding": q_vec, "match_count": TOP_CANDIDATES},
    ).execute()
    
    scores = {}
    texts  = {}
    for row in result.data:
        idx          = row["chunk_index"]
        scores[idx]  = float(row["similarity"])  # cosine similarity [0, 1]
        texts[idx]   = row["content"]
    
    return scores, texts
```

The Supabase RPC handles:
1. Cosine distance computation (`<=>` operator)
2. Ordering by distance (ascending = most similar first)
3. Limiting to `TOP_CANDIDATES` results
4. RLS filtering (only current user's chunks)

### 5.4 BM25 Retrieval

```python
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())

def _bm25_scores_for_query(query: str, corpus: list[dict]) -> dict[int, float]:
    from rank_bm25 import BM25Okapi
    
    tokenized = [_tokenize(entry["text"]) for entry in corpus]
    bm25      = BM25Okapi(tokenized)
    tokens    = _tokenize(query)
    scores    = bm25.get_scores(tokens)
    
    return {entry["chunk_index"]: float(scores[i]) for i, entry in enumerate(corpus)}
```

**Tokenizer design:** The regex `[a-z0-9]+(?:-[a-z0-9]+)*` lowercases and splits on non-alphanumeric characters, but preserves hyphenated tokens intact (e.g., "CERT-IN", "fine-tuning", "full-stack"). This means `_tokenize("CERT-IN certified")` returns `["cert-in", "certified"]`, and a query for "CERT-IN" matches the exact token.

**BM25Okapi default parameters:** `k1=1.5` (term frequency saturation: after enough occurrences, additional mentions add diminishing weight), `b=0.75` (length normalization: shorter documents are rewarded). These are the standard Okapi BM25 defaults and are not tuned further.

### 5.5 Score Fusion

```python
# 1. Union: combine top-10 dense and top-10 BM25 candidates
all_indices = set(dense_raw.keys()) | top_bm25_indices

# 2. Fill zeros: candidates only in one method get 0 in the other
for idx in all_indices:
    dense_raw.setdefault(idx, 0.0)
    bm25_raw.setdefault(idx, 0.0)

# 3. Normalise each independently to [0, 1]
dense_norm = _minmax(dense_raw)
bm25_norm  = _minmax(bm25_raw)

# 4. Fuse
fused = {
    idx: round(ALPHA * dense_norm[idx] + (1 - ALPHA) * bm25_norm[idx], 4)
    for idx in all_indices
}
```

**Min-max normalisation:**
```python
def _minmax(scores: dict[int, float]) -> dict[int, float]:
    lo, hi = min(scores.values()), max(scores.values())
    if hi == 0.0:   return {k: 0.0 for k in scores}  # no signal
    if hi == lo:    return {k: 1.0 for k in scores}  # uniform
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}
```

The normalisation handles a subtle edge case: when all BM25 scores are zero (query has no matching terms), it returns 0.0 for all rather than the usual `(v - lo) / span` which would inflate non-matches to 1.0. This is the `hi == 0.0` guard.

### 5.6 Filtering and Output

```python
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
return chunks[:k]   # k=4 by default
```

The UI uses the per-chunk scores to highlight chunks in the sidebar: retrieved chunks glow green with score bars showing dense vs BM25 contributions.

---

## 6. Agent: ATS Analyzer

### 6.1 Function Signature

```python
def analyze_ats(
    resume_text: str,
    target_role: str,
    client: Groq,
    jd_text: str = "",
    model: str = GROQ_MODEL,
) -> ATSResult:
```

This is the most complex agent in the system. It combines four distinct computation paths: programmatic scoring, embedding similarity, non-LLM keyword extraction (KeyBERT), and a single LLM call for subjective dimensions.

### 6.2 Scoring Rubric

```python
WEIGHTS = {
    "keyword":        0.40,
    "quantification": 0.25,
    "action_verb":    0.15,
    "sections":       0.10,
    "formatting":     0.10,
}
```

Final score:
```
ats_score = 0.40 × keyword% + 0.25 × quant% + 0.15 × action_score + 0.10 × section_score + 0.10 × format_score
```

All sub-scores are on a 0-100 scale before weighting.

### 6.3 Keyword Matching

The LLM extracts 15-25 required keywords from the job description (or infers them from the role name if no JD is provided). These keywords are then checked against the resume programmatically:

```python
_ABBREV = {
    "machine learning": "ml",
    "artificial intelligence": "ai",
    "natural language processing": "nlp",
    "large language model": "llm",
    "application programming interface": "api",
    "continuous integration": "ci",
    "continuous deployment": "cd",
    "version control": "git",
}

def _word_present(word: str, text: str) -> bool:
    # Normalise both sides through abbreviation map
    # Check with word-boundary regex: (?<![a-z0-9])word(?![a-z0-9])
    # Returns True if word OR its abbreviation is found
```

This normalization means "ML" in a resume matches a keyword "machine learning" from the JD, and vice versa. The word-boundary regex prevents "html" from matching inside "Xhtml" or a URL.

**keyword_score = (matched / total) × 100**

### 6.4 Quantification Rate

```python
def _quantification_rate(resume_text: str) -> tuple[int, str]:
    bullets = [l for l in lines if re.match(r"^\s*[-•*▪▸●✓]\s+\S", l)]
    
    if len(bullets) < 3:
        # Fallback: lines over 45 chars are likely bullet-style content
        bullets = [l for l in lines if len(l.strip()) > 45]
    
    quantified = sum(1 for b in bullets if re.search(r"\d+", b))
    rate       = int(quantified / len(bullets) * 100) if bullets else 0
    detail     = f"{quantified} / {len(bullets)} bullets"
    return rate, detail
```

Any digit in a bullet counts as quantification — "50 engineers", "3.2× speedup", "2024", "$40K ARR" all qualify. This is intentionally broad: the point is to detect whether the candidate uses concrete numbers at all.

### 6.5 Section Completeness

```python
REQUIRED_SECTIONS = {"experience", "education", "skill", "project"}
```

Each required keyword is checked against the full resume text (case-insensitive). Score = 25 per section found (4 × 25 = 100 max). This is a simple heuristic but reliable for standard resume formats.

### 6.6 Similarity Metrics (JD Required)

When a job description is provided, four embedding-based similarity metrics are computed between the full resume vector and the full JD vector:

```python
def compute_similarity_metrics(resume_text: str, jd_text: str) -> dict:
    model = _get_embed_model()   # SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    rv = model.encode(resume_text[:4000], normalize_embeddings=True)  # 384-dim
    jv = model.encode(jd_text[:4000],    normalize_embeddings=True)
    
    cosine    = float(np.dot(rv, jv))                    # = cos(θ) for normalized
    euclidean = float(1 - np.linalg.norm(rv - jv) / 2)  # normalized to [0, 1]
    manhattan = float(1 / (1 + np.sum(np.abs(rv - jv)) / np.sqrt(len(rv))))
    pearson   = float(np.corrcoef(rv, jv)[0, 1])
    
    return {"cosine": cosine, "euclidean": euclidean, "manhattan": manhattan, "pearson": pearson}
```

These four metrics capture different geometric relationships between the embedding vectors:
- **Cosine:** Angle between vectors (standard for semantic similarity)
- **Euclidean:** Euclidean distance converted to similarity
- **Manhattan:** L1 distance (less sensitive to outlier dimensions)
- **Pearson:** Linear correlation (accounts for mean shift between vectors)

### 6.7 KeyBERT Extraction

```python
def extract_keywords_keybert(text: str, top_n: int = 15) -> list:
    kw_model = KeyBERT(model=_get_embed_model())
    results  = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),  # single + bigram
        top_n=15,
        use_mmr=True,                   # Maximal Marginal Relevance
        diversity=0.5,
    )
    return [kw for kw, score in results]
```

Maximal Marginal Relevance (MMR) iteratively selects keywords that are relevant to the text but diverse from each other. With `diversity=0.5`, it balances 50% relevance, 50% diversity. This prevents returning "Python", "Python 3", "Python 3.10" as three keywords when they are semantically identical.

**KeyBERT output is used for:**
1. `keybert_resume_kws` — BERT-extracted keywords from the resume
2. `keybert_jd_kws` — BERT-extracted keywords from the JD
3. `keybert_overlap` — intersection (shown in UI as "BERT-matched terms")

This is separate from the LLM-extracted keyword list and serves as a complementary, model-agnostic signal.

### 6.8 LLM Call

All subjective assessments are made in a single LLM call with temperature 0.15 (near-deterministic):

```python
client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are an ATS scoring engine. Return only JSON."},
        {"role": "user",   "content": prompt},
    ],
    temperature=0.15,
    max_tokens=4000,
    top_p=0.9,
)
```

The LLM is asked to return:
- `required_keywords`: list of 15-25 keywords the role demands
- `action_verb_score`: 0-100 rating of action verb quality
- `formatting_score`: 0-100 ATS compliance rating
- `strong_areas`: list of 3-5 strengths
- `weak_areas`: list of 3-5 weaknesses
- `summary`: 2-3 sentence overall assessment

### 6.9 JSON Extraction Pipeline

To handle LLM output that may or may not include markdown fences:

```python
# Strip markdown fences
raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

# Extract outermost JSON object
match = re.search(r"\{[\s\S]*\}", raw)
if not match:
    return _FALLBACK

parsed = json.loads(match.group(0))
```

All numeric fields are clamped to [0, 100]:
```python
def _safe_int(val, lo=0, hi=100) -> int:
    try:    return max(lo, min(hi, int(float(val))))
    except: return 0
```

### 6.10 Return Type

`ATSResult` is a TypedDict with 18 fields covering every score, keyword list, similarity metric, and summary text. The UI renders different parts of this object in different sections of the ATS tab.

---

## 7. Agent: Bullet Rewriter

### 7.1 Function Signature

```python
def rewrite_bullets(
    resume_text: str,      # Truncated to 7500 chars
    target_role: str,
    client: Groq,
    model: str = GROQ_MODEL,
) -> List[Dict]:
```

### 7.2 Output Schema

```python
[
    {
        "action":   "rewrite" | "remove",
        "original": "exact verbatim text from resume",
        "improved": "rewritten bullet",
        "why":      "1-2 sentence explanation",
    },
    ...
]
```

### 7.3 Rewriting Rules (from system prompt)

The LLM is instructed to:
- **REWRITE** bullets that are vague, passive, duty-focused, or lack metrics
- **REMOVE** bullets that restate the job title, are filler, or duplicate other bullets
- **OMIT** bullets that are already strong — only weak bullets appear in output
- **Apply STAR format** where possible (Situation, Task, Action, Result) but condense to one strong sentence
- **Inject metrics** by inferring plausible numbers from context where the original is vague (flagged in `why` field)

### 7.4 LLM Parameters

```python
temperature = 0.55   # Higher than ATS; creativity needed for rewrites
top_p       = 0.9
max_tokens  = 5000   # Large budget for a full resume worth of bullets
```

Temperature 0.55 is deliberately higher than the ATS agent (0.15) because bullet rewriting benefits from stylistic variation. A deterministic model tends to produce formulaic rewrites; slight randomness produces more natural-sounding bullets.

---

## 8. Agent: JD Tailor & Cover Letter

### 8.1 `tailor_resume`

```python
def tailor_resume(
    resume_text: str,
    jd_text: str,
    target_role: str,
    client: Groq,
    model: str = GROQ_MODEL,
) -> Dict:
```

**Strategy:** The LLM identifies 8-12 keywords from the JD that are absent from the resume, then rewrites the most relevant existing bullet to naturally incorporate each missing keyword. The goal is to pass automated ATS keyword filters without fabricating experience.

**Output:**
```python
{
    "rewrites": [
        {
            "original":       "exact bullet from resume",
            "improved":       "rewritten bullet with keyword injected",
            "keyword_added":  "Kubernetes",
        }
    ],
    "added_keywords": ["Kubernetes", "Terraform", ...]
}
```

**Temperature:** 0.3 — conservative, avoids over-creative injection that changes meaning.

### 8.2 `generate_cover_letter`

```python
def generate_cover_letter(
    resume_text: str,
    jd_text: str,
    target_role: str,
    client: Groq,
    model: str = GROQ_MODEL,
) -> str:
```

Generates a 3-paragraph cover letter (~250-300 words):
1. Role enthusiasm + company mention + why you fit
2. Two specific achievements from the resume that match JD requirements
3. Forward-looking close + call to action

Tone calibration: Indian professional context — formal but not stiff, specific not generic. Avoids clichés like "I am passionate about..." which most ATS pre-screens filter.

**Temperature:** 0.4 (slight warmth for readable prose).

---

## 9. Agent: Resume Chat

### 9.1 Function Signature

```python
def chat_with_resume(
    user_message: str,
    chat_history: list[dict],
    groq_client: Groq,
    target_role: str = "",
    session_summary: str = "",
    rapidapi_key: str = "",
    corpus: list[dict] | None = None,
    access_token: str = "",
    model: str = GROQ_MODEL,
) -> tuple[str, str, list[dict]]:
```

Returns `(reply, updated_summary, retrieved_chunks)`.

### 9.2 Context Injection

The system prompt is constructed per-turn:

```
You are a practical Indian career mentor...

=== PRIOR CONVERSATION SUMMARY ===
{session_summary}
=== END SUMMARY ===

=== USER RESUME / CAREER CONTEXT ===
{top_4_resume_chunks}
=== END CONTEXT ===

Target Role: {target_role}
```

The resume context is the top-4 hybrid-retrieved chunks, formatted as plain text with `---` separators between chunks.

### 9.3 Groq Tool Calling

The agent registers one tool: `search_jobs`. The LLM decides autonomously whether to call it.

```python
TOOLS = [{
    "type": "function",
    "function": {
        "name":        "search_jobs",
        "description": "Search live jobs on LinkedIn/Indeed/Glassdoor",
        "parameters": {
            "type": "object",
            "properties": {
                "query":           {"type": "string"},
                "location":        {"type": "string"},
                "employment_type": {"enum": ["FULLTIME", "PARTTIME", "INTERN", "CONTRACTOR"]},
            },
            "required": ["query", "location"],
        },
    }
}]
```

**Two-pass flow:**
1. First call: `tool_choice="auto"` — model either replies or calls `search_jobs`
2. If tool called: execute search, format results, second call without tools
3. If no tool: return first reply directly

**LLM instruction on tool use:**
> "Call search_jobs ONLY when the user asks to find/search/browse job openings. Do NOT call it for general advice, resume feedback, salary questions, or interview prep."

### 9.4 Session Summary Compression

```python
MAX_HISTORY_TURNS = 6  # Keep last 12 messages (6 user + 6 assistant)

def _maybe_compress_summary(existing_summary, chat_history, latest_reply, client, model):
    if len(chat_history) < MAX_HISTORY_TURNS * 2:
        return existing_summary
    
    # Compress oldest turns into rolling summary
    turns_to_compress = chat_history[:-MAX_HISTORY_TURNS * 2]
    # LLM prompt: "Summarise in 120 words or less. Include names, skills, companies, goals."
    # Returns compressed summary replacing existing_summary
```

This prevents context bloat across long conversations while preserving the most relevant facts (skills mentioned, companies of interest, goals stated).

### 9.5 Markdown Post-Processing

```python
def _clean_markdown(text: str) -> str:
    # Unicode punctuation → ASCII
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    
    # Strip over-bolding (>40 chars bolded = probably wrong)
    text = re.sub(r"\*\*(.{41,}?)\*\*", r"\1", text)
    
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()
```

This ensures the Streamlit markdown renderer handles all output correctly regardless of model (different models have different markdown habits).

### 9.6 Chat History Structure

```python
# In session state
chat_history = [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "...", "chunks": [...]},
    ...
]
```

The `chunks` field on assistant messages stores the retrieval metadata used to generate that response. This enables the PCA visualization to highlight which chunks were used in the most recent query.

---

## 10. Agent: Skill Gap Analyzer

### 10.1 Function Signature

```python
def analyze_skill_gap(
    matched_keywords: List[str],
    missing_keywords: List[str],
    target_role: str,
    client: Groq,
    model: str = GROQ_MODEL,
) -> Dict:
```

### 10.2 Output Schema

```python
{
    "categories":     ["Programming Languages", "ML Frameworks", "DevOps", "Cloud", "Databases"],
    "resume_scores":  [8, 4, 6, 3, 7],   # 0-10, current level per category
    "jd_scores":      [9, 9, 8, 7, 8],   # 0-10, JD requirement per category
}
```

The LLM groups matched keywords into the appropriate category for resume_scores, and missing keywords into JD requirement levels. The Plotly radar chart renders this as two overlaid polygons: "Your Resume" (teal) vs "JD Requirement" (red). The gap between the two polygons visualizes skill deficiency.

**Temperature:** 0.2 — needs to be consistent and JSON-safe.

---

## 11. Agent: Resume Structurer

### 11.1 Purpose

`structure_resume` parses free-form resume text into a structured JSON dict. This structured representation is used by:
1. `resume_builder` (PDF generation)
2. `portfolio_generator` (portfolio website generation)
3. Both re-use `st.session_state.resume_structure` if already computed

### 11.2 Output Schema (abbreviated)

```python
{
    "name":           str,
    "email":          str,
    "phone":          str,
    "linkedin":       str,
    "github":         str,
    "website":        str,
    "location":       str,
    "summary":        str,
    "education":      [{"degree", "institution", "location", "dates", "gpa", "bullets"}],
    "experience":     [{"title", "company", "location", "dates", "bullets"}],
    "projects":       [{"name", "dates", "tech", "bullets"}],
    "skills":         {"languages", "frameworks", "tools", "other"},
    "certifications": [str],
    "achievements":   [str],
    "extra_sections": [{"title", "items"}],
}
```

**Temperature:** 0.1 — minimum randomness. The task is extraction, not generation. Higher temperature risks hallucinating email addresses or dates.

**Fallback:** Returns a minimal dict with all arrays empty and all strings as `""`. Both `resume_builder` and `portfolio_generator` handle missing fields gracefully.

---

## 12. Agent: PDF Resume Builder

### 12.1 Technology

`reportlab` is a pure-Python PDF generation library. It requires no system dependencies (no LaTeX, no Pandoc, no wkhtmltopdf), making it reliable across operating systems and cloud environments.

### 12.2 Unicode Sanitization

Helvetica (the chosen font) uses cp1252 encoding and cannot render Unicode characters. All text goes through a two-stage sanitization:

```python
_UNICODE_REPLACEMENTS = {
    "\u2014": "-",      # em dash → hyphen
    "\u2013": "-",      # en dash → hyphen
    "\u2018": "'",      # left single quote
    "\u2019": "'",      # right single quote
    "\u201c": '"',      # left double quote
    "\u201d": '"',      # right double quote
    "\u2022": "-",      # bullet
    "\u2192": "->",     # right arrow
    "\u00e9": "e",      # é (accented)
    # ... 8 more
}

text = text.encode("cp1252", errors="ignore").decode("cp1252")
```

This ensures that LLM outputs (which frequently use Unicode punctuation) render correctly in the PDF.

### 12.3 Bullet Substitution

When rewrites are provided, the builder substitutes improved bullets using fuzzy matching:

```python
from difflib import SequenceMatcher

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()
```

If `_similarity(original, resume_bullet) >= 0.55`, the resume bullet is replaced with the improved version. If `action == "remove"`, the bullet is omitted entirely. This threshold tolerates minor formatting differences between the raw LLM output and the actual resume text.

### 12.4 Page Layout

```
Page size:  letter (8.5" × 11" = 215.9mm × 279.4mm)
Margins:    0.55" all sides
Font:       Helvetica (system font, no embedding needed)
Colors:
  Headers:  #1a1a2e (dark navy)
  Accent:   #4f6ef7 (blue)
  Subtext:  #4a4e6a (gray)
  Body:     #2c2c40 (near-black)
```

Section headers have a 0.75pt horizontal rule below them. Bullet points use a dash (-) prefix and are indented 12pt from the left margin.

---

## 13. Agent: Portfolio Generator

### 13.1 Templates

Two HTML templates are bundled in `templates/`:
- **Luminary:** Light theme, card-based layout, blue accent
- **Noir:** Dark theme, gradient header, modern aesthetic

Both templates use `{{PLACEHOLDER}}` substitution — no JavaScript templating engine, no Jinja2, just `str.replace()` on 15 named slots.

### 13.2 Pipeline

```
1. _detect_and_fill_dummies(structure)
      → Identify missing sections
      → Inject placeholder content (marked _is_dummy: True)
      → Return (enriched_structure, dummy_section_names)

2. _enhance_content(structure, target_role, client, model)
      → Single LLM call (temp=0.4, max_tokens=800)
      → Returns: {tagline, about, typing_roles, project_enhancements}

3. HTML snippet generators (pure Python, no LLM)
      → _skill_chips_html()
      → _projects_html()
      → _experience_html()
      → _education_html()
      → _stats_html()
      → _contact_html()

4. Template substitution (str.replace)
5. _dummy_banner_script(dummy_sections)
      → Injects sticky JS banner warning about placeholder content
```

### 13.3 LLM Enhancement Details

The `_enhance_content` function generates four items in one call:

- **tagline:** 6-10 word punchy personal brand statement
- **about:** 3-sentence professional bio (written in third person)
- **typing_roles:** 3 role alternatives for the animated typing effect in the header (e.g., "Machine Learning Engineer", "AI Researcher", "Data Scientist")
- **project_enhancements:** {original_project_name: one-sentence impact-focused re-description}

### 13.4 Dummy Content Detection

```python
def _detect_and_fill_dummies(structure: dict) -> tuple[dict, list[str]]:
    dummy_sections = []
    
    if not structure.get("projects"):
        structure["projects"] = [
            {"name": "Sample Project", "_is_dummy": True, ...}
        ]
        dummy_sections.append("Projects")
    
    # Similar for experience, skills, name
```

Dummy items carry `_is_dummy: True` in their dict. The HTML generators check this flag and add `data-dummy="true"` to rendered elements. The JavaScript banner counts all dummy elements and shows a persistent warning: "X section(s) use placeholder data — update your resume and regenerate."

---

## 14. Agent: GitHub Publisher

### 14.1 OAuth: Device Flow

GitHub's Device Flow OAuth works without a callback URL:

1. App requests a device code from GitHub
2. App shows the user a code (e.g., `ABCD-1234`) and the URL `github.com/login/device`
3. User visits URL, enters code, authorizes the app
4. App polls GitHub every 5 seconds for the access token
5. On authorization, GitHub returns the token

```python
def poll_for_token(client_id, client_secret, device_code, interval=5, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = requests.post("https://github.com/login/oauth/access_token", data={...})
        
        if "access_token" in data:  return data["access_token"], ""
        if data.get("error") == "authorization_pending": time.sleep(interval); continue
        if data.get("error") == "slow_down":             interval += 5; continue
        if data.get("error") == "expired_token":         return "", "Code expired"
        if data.get("error") == "access_denied":         return "", "User denied"
    
    return "", "Timed out"
```

### 14.2 Publishing Steps

```
1. DELETE /repos/{username}/{repo_name}    — remove old deployment (idempotent)
   sleep(2)                                — wait for GitHub cleanup

2. POST /user/repos                        — create new repo (auto_init=True)
   sleep(2)                                — wait for main branch creation

3. PUT /repos/{username}/{repo_name}/contents/{file}   (×3: index.html, style.css, script.js)
   — base64-encode file content
   — fetch existing SHA if file exists (required for updates)
   — PUT with message, content, sha

4. POST /repos/{username}/{repo_name}/pages           — enable GitHub Pages
   — source: {branch: "main", path: "/"}
   sleep(2)                                           — propagation buffer

5. Return: "https://{username}.github.io/{repo_name}"
```

### 14.3 File Encoding

```python
encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
```

GitHub's Contents API requires base64-encoded file content. The file is first encoded to bytes (UTF-8), then base64-encoded to a string (ASCII-safe), then sent in the JSON payload.

---

## 15. Agent: Interview Prep

### 15.1 Function Signature

```python
def generate_qna(
    resume_text: str,
    target_role: str,
    client: Groq,
    model: str = GROQ_MODEL,
) -> Dict[str, List[Dict]]:
```

### 15.2 Question Tiers

**Easy (~12 questions):** Definitions for every language, library, tool, and framework on the resume. "What is FastAPI?", "What is the difference between a list and a tuple in Python?"

**Medium (~12 questions):** Project-specific design questions. "Why did you choose PostgreSQL over MongoDB for this project?", "How does your recommendation system handle cold start?" Each question references something real from the resume.

**Hard (~10 questions):** Trade-offs, scale, failure modes. "If your microservice needed to handle 10× traffic, what would you change?", "What were the bottlenecks you encountered, and how did you address them?"

**Key constraint in prompt:** "Every single question must reference something real and specific from the resume. Do not generate generic textbook questions."

### 15.3 Output per Question

```python
{
    "question": "Why did you use BGE-small instead of all-MiniLM for your RAG system?",
    "answer":   "BGE-small-en-v1.5 outperforms all-MiniLM-L6-v2 on MTEB retrieval benchmarks at equal model size. It was also released more recently (2024) and is specifically optimized for asymmetric retrieval — short queries against long passages — which matches our use case exactly.",
    "example":  None,
}
```

---

## 16. Agent: Upskill Recommender

### 16.1 `recommend_skills`

```python
def recommend_skills(
    resume_text: str,
    target_role: str,
    missing_keywords: List[str],  # From ATS analysis
    client: Groq,
    model: str = GROQ_MODEL,
) -> List[Dict]:
```

Returns top 5 skills to learn, prioritized by market demand for the target role:

```python
[
    {
        "skill":    "Docker",
        "priority": "High",
        "reason":   "Container skills expected in 90% of SDE job postings. Your resume shows none.",
    },
    ...
]
```

`missing_keywords` from the ATS analysis is injected into the prompt as context, so recommendations are grounded in actual gaps rather than generic advice.

### 16.2 `generate_learning_plan`

```python
def generate_learning_plan(
    skill: str,
    target_role: str,
    client: Groq,
    model: str = GROQ_MODEL,
) -> Dict:
```

Returns a 4-week structured plan:

```python
{
    "overview": "Docker enables containerized deployment...",
    "weeks": [
        {
            "week":   1,
            "goal":   "Understand containers and run your first image",
            "topics": ["What is Docker?", "docker run / pull / ps", "Hello World container"],
        },
        ...  # weeks 2, 3, 4
    ],
    "resources": [
        {
            "title":        "Traversy Media — Docker Crash Course",
            "type":         "Video",
            "search_query": "traversy media docker crash course",
        },
        ...
    ],
}
```

YouTube links are generated via:
```python
def yt_search_url(query: str) -> str:
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}"
```

This avoids hardcoding URLs that may go stale; search results are always current.

---

## 17. Agent: Job Search

### 17.1 JSearch API

The JSearch API (via RapidAPI) aggregates job listings from LinkedIn, Indeed, and Glassdoor. It returns structured JSON without requiring individual authentication tokens for each platform.

```python
headers = {
    "X-RapidAPI-Key":  rapidapi_key,
    "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
}
params = {
    "query":            f"{query} in {location}",
    "page":             "1",
    "num_pages":        "1",
    "employment_types": employment_type,
    "date_posted":      "month",
}
response = requests.get(url, headers=headers, params=params, timeout=10)
```

### 17.2 Output Formatting

Results are formatted for injection into the LLM system prompt:

```
Found 5 jobs for "ML Engineer" in "Bangalore":

1. ML Engineer | Google | Bangalore, IN | FULLTIME | Indeed | 2024-01-15
   Apply: https://...
   "We are looking for..."

[LLM instruction: Present as a clean table. Advise which 1-2 best match resume & why.]
```

The LLM then synthesizes this raw job data into a clean, opinionated response.

---

## 18. File Parser Utility

### 18.1 Supported Formats

```python
def parse_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    
    if suffix == ".pdf":   return _extract_pdf(tmp_path)
    if suffix in (".docx", ".doc"):  return _extract_docx(tmp_path)
    raise ValueError(f"Unsupported format: {suffix}")
```

### 18.2 PDF Extraction

```python
import fitz  # PyMuPDF

def _extract_pdf(path: str) -> str:
    doc   = fitz.open(path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n\n".join(pages).strip()
```

PyMuPDF (`fitz`) is chosen over `pdfminer` and `pypdf` because it handles:
- Multi-column layouts
- Text boxes and tables
- Embedded fonts
- Scanned PDFs (when OCR layer exists)

The `"text"` extraction mode returns raw text preserving word order; `"blocks"` would return bounding-box coordinates, which are useful for layout analysis but unnecessary here.

### 18.3 DOCX Extraction

```python
from docx import Document

def _extract_docx(path: str) -> str:
    doc        = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()
```

`python-docx` iterates over OOXML paragraph elements. Tables and text boxes in DOCX files may not be captured by this approach, but standard resume DOCX files (linear paragraph structure) extract correctly.

### 18.4 Temp File Handling

Streamlit's `UploadedFile` is an in-memory buffer, not a file path. PyMuPDF and python-docx require file paths. The solution:

```python
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded_file.read())
    tmp_path = tmp.name
try:
    text = extractor(tmp_path)
finally:
    os.unlink(tmp_path)  # Always clean up, even on exception
```

---

## 19. Application Entry Point & Session Management

### 19.1 Session State Defaults

`app.py` initializes 23 session state keys on first run:

```python
_DEFAULTS = {
    "resume_text":           None,
    "resume_indexed":        False,
    "loaded_filename":       "",
    "ats_result":            None,
    "ats_result_prev":       None,
    "jd_text":               "",
    "bullets_result":        None,
    "resume_structure":      None,
    "resume_pdf_bytes":      None,
    "jd_tailor_result":      None,
    "cover_letter":          None,
    "tailored_pdf_bytes":    None,
    "skill_gap_result":      None,
    "github_token":          "",
    "github_username":       "",
    "github_device_code":    "",
    "github_user_code":      "",
    "portfolio_files":       None,
    "portfolio_dummy_sections": [],
    "portfolio_pages_url":   "",
    "chat_history":          [],
    "session_summary":       "",
    "groq_client":           None,
    "groq_model":            "openai/gpt-oss-120b",
    "rapidapi_key":          os.getenv("RAPIDAPI_KEY", "").strip(),
    "interview_qna":         None,
    "upskill_recommended":   None,
    "upskill_plan":          None,
    "upskill_selected_skill": "",
    "resume_chunks":         [],
    "pca_coords":            [],
    "pca_variance":          [0.0, 0.0],
    "vectorstore":           None,   # Stores BM25 corpus after ingest
    "last_retrieved":        [],
    "_active_groq_key":      "",
    "supabase_user_id":      "",
    "supabase_access_token": "",
    "supabase_email":        "",
}
```

### 19.2 Groq Client Management

```python
active_key = st.session_state.get("groq_api_key_input", "")

if active_key != st.session_state._active_groq_key:
    st.session_state.groq_client    = Groq(api_key=active_key) if active_key else None
    st.session_state._active_groq_key = active_key
```

The Groq client is only recreated when the API key actually changes. Streamlit reruns the entire script on every user interaction; naive client initialization would create a new HTTP connection on every keypress in the API key input box.

### 19.3 Upload Detection

```python
if uploaded_file and uploaded_file.name != st.session_state.loaded_filename:
    # New file — process
```

Comparing filenames (not file content) is a deliberate choice. Content hashing would add overhead and is unnecessary: if the user uploads a file with the same name, it is reasonable to assume it is the same file. If they want to re-index, they can clear and re-upload.

### 19.4 State Reset on Upload

```python
st.session_state.update({
    "resume_text":       text,
    "loaded_filename":   uploaded_file.name,
    "ats_result":        None,
    "ats_result_prev":   None,
    "bullets_result":    None,
    "resume_structure":  None,
    # ... all derivative state cleared
})
```

All analysis results are cleared on upload because they are derived from the previous resume text. Retaining stale results would be confusing.

---

## 20. Model Switching & Comparison Mode

### 20.1 Sidebar Model Selector

```python
_GROQ_MODELS = {
    "GPT-OSS 120B (default)": "openai/gpt-oss-120b",
    "Llama 3.3 70B":          "llama-3.3-70b-versatile",
    "Llama 3.1 8B (fast)":    "llama-3.1-8b-instant",
    "Gemma 2 9B":             "gemma2-9b-it",
}
_selected = st.selectbox("🤖 Model", list(_GROQ_MODELS.keys()), ...)
st.session_state.groq_model = _GROQ_MODELS[_selected]
```

The selected model ID is stored in session state and passed as the `model` parameter to every agent call. The model parameter was added to all 11 agent functions as `model: str = GROQ_MODEL`, defaulting to GPT-OSS 120B if not specified. This design ensures backward compatibility: any call that omits `model` gets the default.

### 20.2 Comparison Mode

The ATS tab contains a collapsible "Compare two models side-by-side" expander. When expanded:

1. User selects Model A and Model B from independent dropdowns
2. Clicking "Run Comparison" calls `analyze_ats` twice — once per model
3. Results are displayed in two columns with score, strong areas, weak areas, and summary

```python
_r1 = analyze_ats(..., model=_m1_id)
_r2 = analyze_ats(..., model=_m2_id)

_col1, _col2 = st.columns(2)
for _col, _res, _mname in [(_col1, _r1, _m1_name), (_col2, _r2, _m2_name)]:
    with _col:
        st.markdown(f"**{_mname}** — Score: {_res['ats_score']}/100")
        st.caption("Strong: " + "; ".join(_res["strong_areas"][:3]))
        st.caption("Weak:   " + "; ".join(_res["weak_areas"][:3]))
```

This feature directly demonstrates the impact of model choice on output quality, which is useful both for end users choosing a model and for academic evaluation of model performance differences.

---

## 21. PCA Embedding Visualization

### 21.1 Purpose

PCA (Principal Component Analysis) reduces the 384-dimensional BGE embeddings to 2D for visualization. The resulting scatter plot shows:
- How resume chunks cluster semantically
- Which chunks are similar to each other (proximity = semantic similarity)
- Which chunks were used in the most recent chat query (highlighted in the scatter plot)

### 21.2 Computation

```python
from sklearn.decomposition import PCA
import numpy as np

_arr = np.array(_vectors, dtype=float)  # Shape: (n_chunks, 384)
_pca = PCA(n_components=2)
_xy  = _pca.fit_transform(_arr)         # Shape: (n_chunks, 2)

st.session_state.pca_coords = [
    {
        "chunk_index": _corpus[i]["chunk_index"],
        "x":    float(_xy[i, 0]),
        "y":    float(_xy[i, 1]),
        "text": _docs[i][:120],
    }
    for i in range(len(_docs))
]
st.session_state.pca_variance = _pca.explained_variance_ratio_.tolist()
```

The explained variance ratio is displayed in the chart subtitle — e.g., "PC1: 34.2%, PC2: 21.7% variance explained." This contextualizes the projection for users who understand dimensionality reduction.

### 21.3 Why Pre-compute at Upload

PCA `fit_transform` on all chunks is performed once at upload time using the vectors already computed during ingest. There is no additional encoding cost: the vectors are returned from `ingest_resume` and immediately used for PCA. Computing PCA on every chat turn would add 50-100ms overhead per query for no benefit (the chunk space doesn't change between queries).

### 21.4 Retrieved Chunk Highlighting

After each chat query, `st.session_state.last_retrieved` contains the chunk indices used in context. The PCA scatter plot splits chunks into two traces:

- **Base trace:** All chunks not retrieved (small gray dots)
- **Highlight trace:** Retrieved chunks (larger colored star markers with ★ labels)

This gives users visual feedback on which parts of their resume the system consulted when answering a question.

---

## 22. Design Decisions & Architectural Trade-offs

### 22.1 Why Hybrid RAG Over Dense-Only

Dense embedding models excel at semantic similarity but underperform on exact keyword and acronym matching. Consider a resume with "MERN stack" and a query asking about "MongoDB" — the semantic distance between "MERN" and "MongoDB" may not be close enough in embedding space to retrieve the chunk, despite MongoDB being M in MERN. BM25 handles this case exactly because it matches the token "mongodb" (or "mern") directly.

The 0.6/0.4 dense/BM25 weighting was chosen to bias toward semantic understanding (the primary use case is career advice, not keyword lookup) while preserving lexical precision for technical terms. This ratio is consistent with what production hybrid search systems (Elasticsearch, Pinecone Hybrid, Vespa) report as effective starting points.

### 22.2 Why Rubric-Based ATS Scoring Instead of Pure LLM

A pure LLM-assigned score (e.g., "Rate this resume 0-100") is not defensible because:
1. The score is non-reproducible — the same resume may score 72 or 68 on two calls
2. The LLM cannot explain exactly which keywords it checked
3. Calibration varies across models (llama-8b and gpt-4o give different baselines)

The rubric-based approach grounds 75% of the score (keywords + quantification + sections) in programmatic computation. Only the subjective 25% (action verbs + formatting) is delegated to the LLM. This makes the score auditable, reproducible (deterministic computation), and explainable.

### 22.3 Why Supabase Over Local Chroma

Chroma (the original vector DB) stored vectors locally in a `chroma_db/` directory. This had two problems:
1. **No persistence across server restarts** on cloud deployments
2. **No user isolation** — all users' resume chunks were in the same collection

Supabase pgvector solves both: data persists in a managed Postgres instance, and Row-Level Security ensures user isolation at the database layer. The trade-off is network latency for each vector operation (~50-100ms per RPC call), which is acceptable for interactive use.

### 22.4 Why Session State for Corpus (BM25) Instead of Supabase

BM25 operates on in-memory text. Fetching all chunks from Supabase before each query would add network latency on every chat turn. The BM25 corpus (list of {chunk_index, text} dicts) is small (typically 2-10KB for a full resume) and is loaded once at upload time into `st.session_state.vectorstore`. Since Streamlit sessions are user-specific, there is no cross-user leakage.

### 22.5 Why Temperature Varies by Agent

| Agent | Temperature | Reason |
|-------|-------------|--------|
| Resume Structurer | 0.1 | Extraction task; hallucination risk high |
| ATS Analyzer | 0.15 | Scoring rubric; consistency required |
| Skill Gap | 0.2 | Category assignment; must be stable |
| JD Tailor | 0.3 | Keyword injection; accuracy > creativity |
| Cover Letter | 0.4 | Professional prose; slight warmth OK |
| Portfolio Enhancement | 0.4 | Copywriting; needs personality |
| Interview Prep | 0.4 | Questions need variety but stay grounded |
| Bullet Rewriter | 0.55 | Creative rewrites need stylistic range |
| Chat Agent | 0.65 | Conversational; natural variation expected |

### 22.6 Why Device Flow OAuth for GitHub

Standard OAuth requires a pre-registered redirect URI. For a Streamlit app deployed at changing URLs (local dev: localhost:8501, cloud: various), maintaining a static redirect URI is brittle. GitHub Device Flow requires no redirect — the user visits github.com directly and enters a code. The app polls for the token. This works correctly regardless of where Streamlit is running.

### 22.7 Why BGE-small Over Larger Models

| Model | Size | MTEB Retrieval | Notes |
|-------|------|----------------|-------|
| BGE-small-en-v1.5 | 22MB | 0.516 | Chosen |
| all-MiniLM-L6-v2 | 22MB | 0.493 | Slightly weaker |
| BGE-base-en-v1.5 | 110MB | 0.536 | 5MB better, 5× larger |
| text-embedding-3-small | API | 0.618 | Best quality, API cost |

BGE-small was chosen as the best quality/size trade-off for local inference. BGE-base would improve retrieval quality by ~4% but requires 5× more memory and load time, which is noticeable in Streamlit's cold-start environment.

---

## 23. Constants & Configuration Reference

### 23.1 ATS Scoring

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_RESUME_CHARS` | 7000 | LLM token budget for resume text |
| `MAX_JD_CHARS` | 4000 | LLM token budget for JD text |
| `WEIGHTS["keyword"]` | 0.40 | Keyword match score weight |
| `WEIGHTS["quantification"]` | 0.25 | Quantification score weight |
| `WEIGHTS["action_verb"]` | 0.15 | Action verb score weight |
| `WEIGHTS["sections"]` | 0.10 | Section completeness weight |
| `WEIGHTS["formatting"]` | 0.10 | Formatting score weight |
| `REQUIRED_SECTIONS` | 4 | experience, education, skill, project |
| `len(STRONG_VERBS)` | 55 | Power action verb list size |
| `_ABBREV` entries | 8 | Abbreviation normalization pairs |

### 23.2 RAG Retrieval

| Constant | Value | Purpose |
|----------|-------|---------|
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `CHUNK_SIZE` | 700 | Max characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between adjacent chunks |
| `ALPHA` | 0.6 | Dense weight in hybrid fusion |
| `TOP_CANDIDATES` | 10 | Candidates fetched per retriever |
| `MIN_HYBRID_SCORE` | 0.15 | Minimum score to include a chunk |
| `RAW_DENSE_FLOOR` | 0.20 | Early-exit threshold (off-topic guard) |

### 23.3 Chat Agent

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_HISTORY_TURNS` | 6 | Message pairs kept before compression |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Default inference model |

### 23.4 Available Groq Models

| Display Name | Model ID | Characteristics |
|-------------|----------|----------------|
| GPT-OSS 120B (default) | openai/gpt-oss-120b | Best quality, ~1 sec latency |
| Llama 3.3 70B | llama-3.3-70b-versatile | High quality, Llama family |
| Llama 3.1 8B (fast) | llama-3.1-8b-instant | Fast, lower quality |
| Gemma 2 9B | gemma2-9b-it | Google architecture |

### 23.5 UI Colors

| Variable | Hex | Usage |
|----------|-----|-------|
| Primary Blue | `#4f6ef7` | Buttons, links, accents |
| Dark Navy | `#1a1a2e` | Headers, logo |
| Gray | `#8890a4` | Secondary text |
| Green | `#27ae60` | Matched keywords, success states |
| Red | `#e74c3c` | Missing keywords, errors |
| Orange | `#e67e22` | Warnings, moderate scores |
| Teal | `#1abc9c` | "Your resume" radar trace |

### 23.6 External API Limits

| API | Limit | Notes |
|-----|-------|-------|
| Groq free tier | ~1000 tokens/min | Burst-friendly, resets per minute |
| JSearch (RapidAPI) | 100 requests/month (free tier) | Each search = 1 request |
| GitHub REST API | 5000 requests/hour (authenticated) | Well above typical usage |
| Supabase free tier | 500MB storage, 2GB bandwidth | ~500 users within budget |

---

## 24. Data Flow Diagrams

### 24.1 Upload Pipeline

```
User selects file (PDF/DOCX)
         │
         ▼
utils/file_parser.parse_uploaded_file()
  ├── PDF: PyMuPDF fitz.open() → page.get_text("text") per page
  └── DOCX: python-docx Document() → paragraph.text join
         │
         ▼ plain text string
         │
rag/ingest.ingest_resume(text, user_id, access_token)
  ├── RecursiveCharacterTextSplitter(700, 50) → [chunk₀, chunk₁, ..., chunkₙ]
  ├── HuggingFaceEmbeddings.embed_documents([chunks]) → [vec₀, ..., vecₙ]  (384-dim each)
  ├── supabase.table("resume_chunks").delete().eq("user_id", uid)  → clear old data
  └── supabase.table("resume_chunks").insert([{user_id, chunk_index, content, embedding}])
         │
         ▼ {"corpus": [...], "vectors": [...]}
         │
app.py post-ingest
  ├── st.session_state.vectorstore    = corpus       (for BM25)
  ├── st.session_state.resume_chunks  = [{chunk_index, text}]  (for sidebar)
  └── PCA(n_components=2).fit_transform(vectors) → pca_coords, pca_variance
```

### 24.2 Chat Pipeline

```
User types message → st.session_state.chat_history.append({role: user, content})
         │
         ▼
rag/retriever.retrieve_with_scores(query, k=4, corpus, access_token)
  ├── Dense: embed_query(query) → [384-dim vector]
  │         supabase.rpc("match_resume_chunks", {embedding, match_count=10})
  │         → {chunk_index: cosine_similarity, ...}
  │
  ├── BM25: BM25Okapi(tokenized_corpus).get_scores(tokenize(query))
  │         → {chunk_index: bm25_score, ...}
  │
  ├── Union all candidate indices
  ├── Min-max normalise both score dicts
  ├── Fuse: 0.6×dense_norm + 0.4×bm25_norm
  └── Filter (>=0.15) → sort desc → top 4
         │
         ▼ retrieved_chunks: [{text, chunk_index, dense, bm25, hybrid}]
         │
agents/chat_agent.chat_with_resume(...)
  ├── Build system prompt with RAG context + session_summary + target_role
  ├── groq.chat.completions.create(model, messages, tools=TOOLS, tool_choice="auto")
  │   ├── Model replies directly → clean_markdown(reply)
  │   └── Model calls search_jobs(query, location, type)
  │       ├── job_search.search_jobs(query, location, type, rapidapi_key)
  │       ├── format_jobs_for_llm(jobs)
  │       └── Second groq call (no tools) → clean_markdown(reply)
  │
  └── _maybe_compress_summary() if history >= 12 turns
         │
         ▼ (reply, updated_summary, chunks)
         │
app.py
  ├── chat_history.append({role: assistant, content: reply, chunks: chunks})
  ├── session_summary = updated_summary
  └── last_retrieved  = [{chunk_index, dense, bm25, hybrid}]   (for PCA highlight)
```

### 24.3 ATS Analysis Pipeline

```
User clicks "Run ATS Analysis"
         │
         ▼
agents/ats_analyzer.analyze_ats(resume_text, target_role, client, jd_text, model)
  │
  ├── [if jd_text] compute_similarity_metrics(resume_text, jd_text)
  │     └── BGE-small.encode(resume), BGE-small.encode(jd)
  │         → {cosine, euclidean, manhattan, pearson}
  │
  ├── extract_keywords_keybert(resume_text)  → keybert_resume_kws
  ├── [if jd_text] extract_keywords_keybert(jd_text) → keybert_jd_kws
  │
  ├── _programmatic_action_verb_score(resume_text) → programmatic fallback
  ├── _section_completeness(resume_text) → section_score (0/25/50/75/100)
  ├── _quantification_rate(resume_text) → (quant_score, quant_detail)
  │
  ├── _llm_analysis(resume_text, target_role, client, jd_text, model)
  │     Single Groq call (temp=0.15, max_tokens=4000)
  │     → {required_keywords, action_verb_score, formatting_score,
  │        strong_areas, weak_areas, summary}
  │
  ├── _check_keywords(resume_text, required_keywords)
  │     → (matched_keywords, missing_keywords, keyword_score)
  │
  └── Compute final score:
        ats_score = 0.40×keyword + 0.25×quant + 0.15×action + 0.10×section + 0.10×fmt
         │
         ▼ ATSResult TypedDict (18 fields)
         │
app.py
  └── [if jd_text] analyze_skill_gap(matched_kws, missing_kws, target_role, client, model)
        Single Groq call → {categories, resume_scores, jd_scores}
        → Plotly radar chart
```

---

## 25. Error Handling & Fallback Strategy

The system follows a consistent error handling philosophy across all agents: **never crash the Streamlit UI; always return a safe default.**

```python
# Pattern used in every agent
try:
    # ... main logic ...
except Exception:
    return _FALLBACK
```

| Component | Failure Cause | Fallback Response |
|-----------|---------------|-------------------|
| ATS Analyzer | LLM timeout, JSON parse fail | `_FALLBACK` dict (all zeros, empty lists) |
| Bullet Rewriter | LLM error, malformed JSON | `[]` empty list (no rewrites shown) |
| JD Tailor | LLM error | `{"rewrites": [], "added_keywords": []}` |
| Cover Letter | LLM error | `""` empty string (UI shows warning) |
| Resume Structurer | LLM error | Minimal dict (all arrays empty, strings `""`) |
| RAG Retrieval | Supabase timeout, BM25 import fail | `[]` (chat proceeds with no resume context) |
| RAG Ingest | Embedding failure | Exception propagated → app.py shows error |
| Job Search | JSearch API down, timeout | `[]` empty list, model told search failed |
| Portfolio Generator | LLM enhancement fail | Template fills with raw structure data |
| GitHub Publisher | Auth fail, API error | `("", error_message)` (UI shows error) |
| PCA Computation | Numpy error, <2 chunks | Skip (no PCA chart shown) |
| PDF Build | reportlab error | Exception propagated → download button hidden |

**JSON extraction robustness pattern:**

```python
# 1. Strip markdown fences
raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

# 2. Extract first JSON object
match = re.search(r"\{[\s\S]*\}", raw)
if not match:
    return fallback

# 3. Parse
parsed = json.loads(match.group(0))

# 4. Clamp/validate all fields
score = max(0, min(100, int(float(parsed.get("score", 0)))))
```

---

## 26. Performance Characteristics

### 26.1 Latency per Operation

| Operation | Typical Latency | Primary Bottleneck |
|-----------|----------------|-------------------|
| PDF parse | 1-2s | File size, page count |
| BGE-small embed (10 chunks) | 100-200ms | Local CPU (no GPU) |
| Supabase pgvector insert (20 rows) | 300-500ms | Network RTT |
| Supabase RPC (dense retrieval) | 150-300ms | Network RTT + pgvector |
| BM25 score (20 chunks) | 5-10ms | In-memory, pure Python |
| PCA (20 chunks × 384 dim) | 50-100ms | sklearn numpy ops |
| ATS analysis (LLM call) | 3-5s | Groq API (4000 tokens) |
| Bullet rewrite (LLM call) | 4-7s | Groq API (5000 tokens) |
| Chat turn (retrieve + LLM) | 2-4s | Groq API (2200 tokens) |
| Portfolio generate (LLM + HTML) | 5-8s | Groq API + template fill |
| GitHub publish (API calls) | 8-12s | GitHub API rate + network |

### 26.2 Memory Footprint

| Component | Memory |
|-----------|--------|
| BGE-small model (loaded) | ~90MB |
| KeyBERT model (shares BGE) | +5MB overhead |
| Chroma (removed) | — |
| BM25 corpus (20 chunks) | ~50KB |
| Session state (typical) | ~1-5MB |

### 26.3 Groq Token Budget per Operation

| Operation | Prompt Tokens | Completion Tokens | Total |
|-----------|--------------|-------------------|-------|
| ATS (resume 500 words) | ~1200 | ~800 | ~2000 |
| ATS with JD | ~1800 | ~800 | ~2600 |
| Bullet rewrite | ~1000 | ~2000 | ~3000 |
| Chat turn (k=4 chunks) | ~900 | ~600 | ~1500 |
| Cover letter | ~800 | ~400 | ~1200 |
| Interview QNA | ~700 | ~3000 | ~3700 |

Groq free tier: ~1000 tokens/minute. Three consecutive ATS analyses would approach the rate limit.

---

## 27. Security Considerations

### 27.1 Authentication

All application state is gated behind Supabase authentication. `st.stop()` is called if no valid session exists, ensuring zero application logic runs for unauthenticated users. The JWT from Supabase is passed to all database operations; Supabase verifies it server-side before executing any query.

### 27.2 Row-Level Security

Database-layer isolation via Supabase RLS means a compromised application layer cannot access other users' data. Even if application code had a bug that passed the wrong user_id, the database policy `auth.uid() = user_id` would reject the query.

### 27.3 API Key Storage

API keys (Groq, RapidAPI, GitHub) are stored in `.env` and loaded via `python-dotenv`. They live in `st.session_state` during runtime and are never rendered directly in the UI. In production, Streamlit Secrets or environment variables injected by the hosting platform should be used instead of `.env`.

### 27.4 File Upload Safety

Uploaded files are written to a temporary file with a fixed suffix (`.pdf` or `.docx`), processed, and immediately deleted. The file is never executed or interpreted as code. File size limits are enforced by Streamlit's upload widget configuration.

### 27.5 LLM Output Safety

LLM outputs are treated as untrusted text. In the resume builder, all LLM-generated content is passed through Unicode sanitization and cp1252 encoding (which strips any non-printable characters). In the portfolio generator, content is inserted into HTML templates via `str.replace()` — not `eval()` or template rendering with code execution.

---

## 28. Dependencies & External Services

### 28.1 Python Packages

```
streamlit>=1.34.0        — UI framework
supabase>=2.3.0          — Auth + pgvector client
groq>=1.0.0              — LLM inference API
langchain-text-splitters>=0.3.0  — RecursiveCharacterTextSplitter
langchain-huggingface>=0.1.0     — HuggingFaceEmbeddings wrapper
sentence-transformers>=3.0.0     — BGE-small-en-v1.5 model
PyMuPDF>=1.25.0          — PDF parsing (fitz)
python-docx>=1.1.0       — DOCX parsing
python-dotenv>=1.0.0     — .env loading
reportlab>=4.0.0         — PDF generation
plotly>=5.0.0            — Interactive charts
requests>=2.31.0         — HTTP for GitHub + JSearch APIs
keybert>=0.8.0           — BERT keyword extraction
scikit-learn>=1.3.0      — PCA
rank-bm25>=0.2.2         — BM25 Okapi algorithm
```

### 28.2 External Services

| Service | Purpose | Auth Method | Free Tier |
|---------|---------|-------------|-----------|
| Supabase | Auth + pgvector | API key + JWT | 500MB storage, 2GB bandwidth |
| Groq API | LLM inference | API key | ~1000 tokens/minute |
| JSearch (RapidAPI) | Job listings | API key | 100 requests/month |
| GitHub REST API | Portfolio publish | Device Flow OAuth | 5000 req/hour |
| Hugging Face Hub | Model download (once) | Optional HF token | Public models free |

### 28.3 Local Model

`BAAI/bge-small-en-v1.5` is downloaded from Hugging Face Hub on first run and cached in the HuggingFace cache directory (`~/.cache/huggingface/`). Subsequent runs load from cache with no network request. Model size: 22MB. The model is loaded into RAM and shared between `rag/ingest.py`, `rag/retriever.py`, and `agents/ats_analyzer.py` (all use the same `_get_embed_model()` cached instance from ats_analyzer).

---

*This document covers all architectural decisions, implementation details, and design trade-offs for PrepSense AI. The ablation study results (BM25 vs dense-only vs hybrid), ATS weight regression analysis, and model comparison benchmarks will be added in a subsequent section.*
