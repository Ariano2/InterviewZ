# PrepSense AI

An AI-powered career platform built with Streamlit, Groq, LangChain, and Chroma. Upload a resume and get ATS scoring, bullet rewrites, JD-based tailoring, RAG-powered chat, interview prep, skill gap analysis, semantic job matching, portfolio generation, and a from-scratch resume builder — all in one app.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit 1.34+ |
| LLM | Groq API — GPT-OSS 120B / Llama 3.3 70B / Llama 3.1 8B / Gemma 2 9B |
| Embeddings | `BAAI/bge-small-en-v1.5` via SentenceTransformers (384-dim, local CPU) |
| RAG Retrieval | LangChain + Chroma (dense) + BM25 (sparse) — hybrid fusion |
| Auth | Supabase (email/password, JWT session tokens) |
| Job Search | RapidAPI → JSearch (LinkedIn / Indeed / Glassdoor) |
| GitHub Integration | GitHub OAuth Device Flow — publishes portfolio to GitHub Pages |
| PDF Parsing | PyMuPDF (`fitz`) + python-docx |
| PDF Generation | WeasyPrint |
| Dimensionality Reduction | scikit-learn PCA (chunk embedding visualization) |
| Env | python-dotenv |

---

## Project Structure

```
app.py                      — Orchestrator: page config, auth, sidebar, upload, tab dispatch
styles.css                  — All UI styles and utility classes (no inline CSS in Python)
requirements.txt

agents/
  ats_analyzer.py           — ATS scoring: keyword extraction, semantic similarity, weighted composite score
  bullet_rewriter.py        — Identifies weak bullets, rewrites with power verbs + metrics via Groq
  chat_agent.py             — Agentic RAG chat: tool-calling loop for ATS + job search within chat
  jd_tailor.py              — Rewrites resume bullets to match a JD; generates cover letter
  skill_gap.py              — Radar chart skill gap analysis between resume and JD
  resume_structurer.py      — Parses resume into structured JSON (name, education, experience, etc.)
  resume_builder.py         — Renders structured JSON → styled PDF via WeasyPrint
  resume_maker.py           — Enhances bullets and generates summary for the resume builder form
  portfolio_generator.py    — Generates Luminary/Noir portfolio HTML+CSS+JS from resume structure
  github_publisher.py       — Creates GitHub repo, pushes portfolio files, enables Pages via API
  interview_prep.py         — Generates Easy/Medium/Hard Q&A sets grounded in the resume
  upskill.py                — Recommends skills based on gaps; generates 4-week learning roadmap
  job_matcher.py            — Encodes job descriptions with BGE-small, ranks by cosine similarity to resume
  job_search.py             — RapidAPI JSearch wrapper — fetches live jobs by role, location, type

rag/
  ingest.py                 — Chunks text (RecursiveCharacterTextSplitter), embeds with BGE-small,
                              upserts to Chroma, returns corpus + raw vectors for PCA
  retriever.py              — Hybrid retrieval: dense (Chroma cosine) + BM25 sparse, score fusion,
                              returns top-k chunks with per-chunk dense/BM25/hybrid scores

tabs/
  ats.py                    — ATS Score tab UI: JD input, score breakdown, keyword chips, delta vs prev run
  bullets.py                — Bullet Rewriter tab UI: before/after cards, download rewrites
  jd_tailor.py              — JD Tailor + Cover Letter tab UI
  chat.py                   — Resume Chat tab UI: message thread, PCA scatter plot, retrieved chunk debug
  portfolio.py              — Portfolio Generator tab UI: template picker, GitHub OAuth, publish flow
  interview.py              — Interview Prep + Upskill tab UI: MCQ expanders, skill cards, week plan
  job_match.py              — Job Match tab UI: search form, job cards with match %, ATS/Tailor shortcuts
  raw_text.py               — Raw parsed text view with download
  resume_maker.py           — Make My Resume tab UI: multi-section form, live HTML preview, PDF export

ui/
  components.py             — Shared render helpers: require_resume, score_bar, chip_list,
                              bullet_diff, job_card, score_color, alert_green/blue, section_heading

utils/
  file_parser.py            — PDF (PyMuPDF) and DOCX (python-docx) → clean plain text
  embed_cache.py            — Process-level BGE-small singleton (loaded once, shared by all agents)
```

---

## User Flow

1. **Auth** — User signs up / logs in via Supabase email auth. JWT access token stored in session state. All subsequent Supabase calls (RAG upsert) use this token for row-level security.

2. **Upload** — PDF or DOCX parsed to plain text. Text is chunked (500 chars, 80 overlap), embedded with BGE-small, and upserted to Chroma. BM25 index built from the same corpus. PCA run on chunk vectors for visualization. Resume embedding (full text, 384-dim) pre-computed and cached in session state for reuse across all features.

3. **ATS Score** — Optional JD paste. Groq extracts keywords and scores 5 dimensions (semantic similarity 50%, keyword coverage 20%, formatting 15%, section completeness 10%, quantified impact 5%). BGE-small computes cosine similarity between resume and JD vectors. Composite weighted score with delta comparison against previous run.

4. **Bullet Rewriter** — Groq identifies 5–7 weak bullets, rewrites each with a power verb, quantified impact, and role keywords. Before/after diff cards with reasoning.

5. **JD Tailor** — Rewrites resume bullets specifically for a pasted JD. Highlights added keywords. Optionally generates a tailored cover letter. PDF export via WeasyPrint.

6. **Resume Chat** — Multi-turn RAG chat. Each query hits the hybrid retriever (dense + BM25 fusion), injects top-k chunks into Groq context. Agentic mode: Groq can invoke `analyze_ats` or `search_jobs` as tools within the conversation. Retrieved chunk indices and scores shown in sidebar with score bars.

7. **Portfolio** — Chooses Luminary (light) or Noir (dark) template. GitHub OAuth device flow (no stored credentials — user approves in browser). Groq generates portfolio content from resume structure. HTML/CSS/JS files pushed to a new GitHub repo; Pages enabled via API. Live URL returned.

8. **Interview Prep** — Groq generates ~34 Q&A pairs (Easy/Medium/Hard) grounded in the user's specific resume — projects, tech choices, trade-offs. Upskill tab recommends missing skills and generates a 4-week learning roadmap with YouTube resource links.

9. **Job Match** — RapidAPI fetches live jobs. Each job description encoded with BGE-small. Cosine similarity against pre-computed resume embedding ranks jobs. One-click to load any JD into ATS or Tailor tabs.

10. **Resume Builder** — Form-based builder (personal info, education, experience, projects, skills, certifications). Groq enhances bullet points and generates a professional summary. Renders to styled HTML preview and downloadable PDF.

---

## Session & State Management

All runtime state lives in `st.session_state`. Key entries: `resume_text`, `resume_embedding` (numpy array, pre-computed on upload), `vectorstore` (BM25 corpus), `resume_chunks`, `pca_coords`, `groq_client` (recreated only when API key changes), `chat_history`, `last_retrieved` (chunk indices + scores from last query). No database is used for session data — Supabase is used only for auth and RAG vector persistence.

---

## Quickstart

```bash
pip install -r requirements.txt

# Copy and fill in keys
cp .env.example .env
# Required: GROQ_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY
# Optional: RAPIDAPI_KEY (job search), GITHUB_CLIENT_ID + GITHUB_CLIENT_SECRET (portfolio)

streamlit run app.py
```
