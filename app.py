"""
app.py  —  PrepSense AI · Resume Reviewer
Run with:  streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from supabase import create_client

from rag.ingest import ingest_resume
from utils.file_parser import parse_uploaded_file

import tabs.ats as tab_ats_mod
import tabs.bullets as tab_bullets_mod
import tabs.jd_tailor as tab_jd_mod
import tabs.chat as tab_chat_mod
import tabs.portfolio as tab_portfolio_mod
import tabs.interview as tab_interview_mod
import tabs.raw_text as tab_raw_mod
import tabs.resume_maker as tab_maker_mod
import tabs.job_match as tab_jobs_mod

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PrepSense AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (cached — disk read once per server start) ────────────────────────────
_css_path = os.path.join(os.path.dirname(__file__), "styles.css")


@st.cache_data
def _load_css(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


st.markdown(f"<style>{_load_css(_css_path)}</style>", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "resume_text": None,
    "resume_indexed": False,
    "loaded_filename": "",
    "ats_result": None,
    "ats_result_prev": None,
    "jd_text": "",
    "bullets_result": None,
    "resume_structure": None,
    "resume_pdf_bytes": None,
    "jd_tailor_result": None,
    "cover_letter": None,
    "tailored_pdf_bytes": None,
    "skill_gap_result": None,
    "github_token": "",
    "github_username": "",
    "github_device_code": "",
    "github_user_code": "",
    "portfolio_files": None,
    "portfolio_dummy_sections": [],
    "portfolio_pages_url": "",
    "chat_history": [],
    "session_summary": "",
    "groq_client": None,
    "groq_model": "openai/gpt-oss-120b",
    "rapidapi_key": os.getenv("RAPIDAPI_KEY", "").strip(),
    "interview_qna": None,
    "upskill_recommended": None,
    "upskill_plan": None,
    "upskill_selected_skill": "",
    "resume_chunks": [],
    "pca_coords": [],
    "pca_variance": [0.0, 0.0],
    "vectorstore": None,
    "last_retrieved": [],
    "_active_groq_key": "",
    "maker_data": {
        "name": "", "email": "", "phone": "",
        "linkedin": "", "github": "", "location": "",
        "summary": "",
        "education":  [],
        "experience": [],
        "projects":   [],
        "skills": {"languages": "", "frameworks": "", "tools": "", "other": ""},
        "certifications": "",
        "achievements": "",
    },
    "maker_pdf_bytes": None,
    "job_match_results": [],
    "job_match_query": "",
    "resume_embedding": None,
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ── Supabase auth state ───────────────────────────────────────────────────────
for _k, _v in {"supabase_user_id": "", "supabase_access_token": "", "supabase_email": ""}.items():
    st.session_state.setdefault(_k, _v)


def _get_supabase_anon():
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])


def _show_auth_page() -> None:
    st.markdown(
        '<h1 style="font-family:Inter,sans-serif;font-weight:700;color:#1a1a2e;">🎯 PrepSense AI</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("**AI-powered resume reviewer, interview coach, and career mentor.**")
    st.divider()

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        email    = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", use_container_width=True, type="primary"):
            try:
                sb  = _get_supabase_anon()
                res = sb.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.supabase_user_id      = res.user.id
                st.session_state.supabase_access_token = res.session.access_token
                st.session_state.supabase_email        = res.user.email
                st.rerun()
            except Exception as _e:
                st.error(f"Login failed: {_e}")

    with tab_signup:
        email    = st.text_input("Email", key="signup_email")
        password = st.text_input("Password (min 6 chars)", type="password", key="signup_password")
        if st.button("Create Account", use_container_width=True):
            try:
                sb  = _get_supabase_anon()
                res = sb.auth.sign_up({"email": email, "password": password})
                if res.user:
                    st.success("Account created! Check your email to confirm, then log in.")
                else:
                    st.error("Sign up failed — try a different email.")
            except Exception as _e:
                st.error(f"Sign up failed: {_e}")


if not st.session_state.supabase_user_id:
    _show_auth_page()
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-family:Inter,sans-serif;font-weight:700;font-size:1.2rem;'
        'color:#1a1a2e;letter-spacing:-0.3px;margin-bottom:2px;">🎯 PrepSense AI</p>',
        unsafe_allow_html=True,
    )
    st.caption("RESUME · RAG · INTERVIEW PREP")
    st.caption(f"👤 {st.session_state.supabase_email}")
    if st.button("Logout", use_container_width=True):
        for _k in ("supabase_user_id", "supabase_access_token", "supabase_email"):
            st.session_state[_k] = ""
        st.rerun()
    st.divider()

    target_role = st.text_input(
        "🎯 Target Role",
        value="Software Development Engineer",
        placeholder="e.g. ML Engineer at Google",
    )

    _GROQ_MODELS = {
        "GPT-OSS 120B (default)": "openai/gpt-oss-120b",
        "Llama 3.3 70B":          "llama-3.3-70b-versatile",
        "Llama 3.1 8B (fast)":    "llama-3.1-8b-instant",
        "Gemma 2 9B":             "gemma2-9b-it",
    }
    _selected_model_name = st.selectbox(
        "🤖 Model",
        list(_GROQ_MODELS.keys()),
        index=list(_GROQ_MODELS.values()).index(
            st.session_state.groq_model
            if st.session_state.groq_model in _GROQ_MODELS.values()
            else "openai/gpt-oss-120b"
        ),
    )
    st.session_state.groq_model = _GROQ_MODELS[_selected_model_name]

    st.divider()
    st.markdown('<p class="section-label">Status</p>', unsafe_allow_html=True)

    if st.session_state.resume_text:
        wc = len(st.session_state.resume_text.split())
        st.markdown(f'<p class="label-text text-ok">✓ Resume loaded ({wc} words)</p>', unsafe_allow_html=True)
        if st.session_state.resume_indexed:
            st.markdown('<p class="label-text text-ok">✓ RAG index ready</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="label-text text-warn">⏳ Indexing…</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="label-text text-muted">○ No resume uploaded</p>', unsafe_allow_html=True)

    if st.session_state.session_summary:
        st.divider()
        st.markdown('<p class="section-label">🧠 Session Memory</p>', unsafe_allow_html=True)
        _preview = st.session_state.session_summary[:200] + ("…" if len(st.session_state.session_summary) > 200 else "")
        st.markdown(f'<p class="meta-text">{_preview}</p>', unsafe_allow_html=True)

    _chunks = st.session_state.resume_chunks
    if _chunks:
        st.divider()
        st.markdown('<p class="section-label">📄 Resume Chunks</p>', unsafe_allow_html=True)

        _retrieved_map: dict[int, dict] = {
            r["chunk_index"]: r for r in st.session_state.last_retrieved
        }
        _n_retrieved = len(_retrieved_map)
        if _n_retrieved:
            st.caption(f"🔍 {_n_retrieved} chunk{'s' if _n_retrieved > 1 else ''} used in last query")

        with st.expander(f"View all {len(_chunks)} indexed chunks", expanded=False):
            st.caption(f"{len(_chunks)} chunks · BGE-small + BM25 hybrid index")
            for _c in sorted(_chunks, key=lambda x: x["chunk_index"]):
                _idx   = _c["chunk_index"]
                _hit   = _retrieved_map.get(_idx)
                _bc    = "#27ae60" if _hit else "#e0e4f0"
                _lc    = "#27ae60" if _hit else "#3452c7"
                _badge = " ✦ retrieved" if _hit else ""

                st.markdown(
                    f'<p style="font-size:0.72rem;font-weight:600;color:{_lc};margin:4px 0 2px;">'
                    f'Chunk #{_idx}{_badge}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p style="font-size:0.72rem;color:#555;line-height:1.5;margin:0 0 4px;'
                    f'border-left:3px solid {_bc};padding-left:8px;">'
                    f'{_c["text"][:280]}{"…" if len(_c["text"]) > 280 else ""}</p>',
                    unsafe_allow_html=True,
                )
                if _hit:
                    _d = _hit["dense_score"]
                    _b = _hit["bm25_score"]
                    _h = _hit["hybrid_score"]
                    st.markdown(
                        f'<div class="chunk-scores">'
                        f'<span style="color:#3452c7;">■</span> Dense&nbsp;<b>{_d:.2f}</b>&nbsp;&nbsp;'
                        f'<span style="color:#e67e22;">■</span> BM25&nbsp;<b>{_b:.2f}</b>&nbsp;&nbsp;'
                        f'<span style="color:#27ae60;">■</span> Hybrid&nbsp;<b>{_h:.2f}</b>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div style="display:flex;gap:3px;margin:0 0 10px 10px;height:5px;">'
                        f'<div style="width:{int(_d*80)}px;background:#3452c7;border-radius:2px;"></div>'
                        f'<div style="width:{int(_b*80)}px;background:#e67e22;border-radius:2px;"></div>'
                        f'<div style="width:{int(_h*80)}px;background:#27ae60;border-radius:2px;"></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('<div style="margin-bottom:8px;"></div>', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="brand">Prep<span>Sense</span> AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub">ATS Scorer · Bullet Rewriter · RAG Chat · Powered by Groq</p>', unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ── Groq API Key ──────────────────────────────────────────────────────────────
_env_key              = os.getenv("GROQ_API_KEY", "").strip()
_github_client_id     = os.getenv("GITHUB_CLIENT_ID", "").strip()
_github_client_secret = os.getenv("GITHUB_CLIENT_SECRET", "").strip()

if _env_key:
    active_key = _env_key
    st.success("✓ Groq API key loaded from .env", icon="✅")
else:
    col_key, col_status = st.columns([3, 1])
    with col_key:
        active_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            placeholder="gsk_...   (get free at console.groq.com)",
        ).strip()
    with col_status:
        st.write("")
        if active_key:
            st.success("Connected", icon="✅")
        else:
            st.warning("Required", icon="🔑")

if active_key != st.session_state._active_groq_key:
    st.session_state.groq_client      = Groq(api_key=active_key) if active_key else None
    st.session_state._active_groq_key = active_key

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ── File Upload ───────────────────────────────────────────────────────────────
col_up, col_info = st.columns([3, 2])

with col_up:
    uploaded_file = st.file_uploader("Drop your resume here (PDF or DOCX)", type=["pdf", "docx"])

with col_info:
    st.markdown(
        '<div class="card-blue">'
        '<div class="section-label">How it works</div>'
        '<p class="body-text text-primary" style="line-height:1.8;margin:0;">'
        '1️⃣ Upload PDF or DOCX<br>'
        '2️⃣ Auto-chunked → embedded → Chroma RAG<br>'
        '3️⃣ Run ATS analysis or bullet rewrites<br>'
        '4️⃣ Chat with your resume via RAG context'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Process upload ────────────────────────────────────────────────────────────
if uploaded_file and uploaded_file.name != st.session_state.loaded_filename:
    with st.spinner("📄 Parsing resume…"):
        try:
            text = parse_uploaded_file(uploaded_file)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    st.session_state.update({
        "resume_text":          text,
        "loaded_filename":      uploaded_file.name,
        "ats_result":           None,
        "ats_result_prev":      None,
        "bullets_result":       None,
        "resume_structure":     None,
        "resume_pdf_bytes":     None,
        "jd_tailor_result":     None,
        "cover_letter":         None,
        "tailored_pdf_bytes":   None,
        "skill_gap_result":     None,
        "resume_indexed":       False,
        "resume_chunks":        [],
        "pca_coords":           [],
        "pca_variance":         [0.0, 0.0],
        "vectorstore":          None,
        "last_retrieved":       [],
        "chat_history":         [],
        "session_summary":      "",
        "interview_qna":        None,
        "upskill_recommended":  None,
        "upskill_plan":         None,
        "upskill_selected_skill": "",
        "resume_embedding":     None,
    })

    with st.spinner("🗂️ Building RAG index…"):
        try:
            _ingest_result = ingest_resume(
                text,
                st.session_state.supabase_user_id,
                st.session_state.supabase_access_token,
            )
            _corpus  = _ingest_result["corpus"]
            _vectors = _ingest_result["vectors"]
            st.session_state.resume_indexed = True
        except Exception as e:
            st.error(f"RAG indexing failed: {e}")
            _corpus, _vectors = [], []

    if st.session_state.resume_indexed:
        st.session_state.vectorstore   = _corpus
        st.session_state.resume_chunks = [
            {"chunk_index": c["chunk_index"], "text": c["text"]} for c in _corpus
        ]

        # PCA from ingest vectors
        _docs = [c["text"] for c in _corpus]
        if len(_docs) >= 2:
            try:
                import numpy as np
                from sklearn.decomposition import PCA
                _arr = np.array(_vectors, dtype=float)
                _pca = PCA(n_components=2)
                _xy  = _pca.fit_transform(_arr)
                st.session_state.pca_coords = [
                    {"chunk_index": _corpus[i]["chunk_index"],
                     "x": float(_xy[i, 0]), "y": float(_xy[i, 1]),
                     "text": _docs[i][:120]}
                    for i in range(len(_docs))
                ]
                st.session_state.pca_variance = _pca.explained_variance_ratio_.tolist()
            except Exception as _e:
                st.warning(f"PCA computation failed: {_e}")

        # Pre-compute resume embedding (reused by ATS + Job Match + Chat)
        try:
            from utils.embed_cache import get_bge_model as _get_bge
            st.session_state.resume_embedding = _get_bge().encode(
                text[:4000], normalize_embeddings=True, show_progress_bar=False
            )
        except Exception:
            pass  # non-fatal — features fall back to encoding on demand

        st.success(f"✓ **{uploaded_file.name}** parsed & indexed", icon="🗂️")
        st.rerun()

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
(
    _tab_ats, _tab_bullets, _tab_jd, _tab_chat,
    _tab_portfolio, _tab_interview, _tab_raw, _tab_maker, _tab_jobs,
) = st.tabs([
    "🏆  ATS Score", "✍️  Bullet Rewriter", "🎯  JD Tailor", "💬  Resume Chat",
    "🌐  Portfolio", "🎓  Interview Prep", "📄  Raw Text", "🛠️  Make My Resume", "🔍  Job Match",
])

with _tab_ats:
    tab_ats_mod.render(target_role)

with _tab_bullets:
    tab_bullets_mod.render(target_role)

with _tab_jd:
    tab_jd_mod.render(target_role)

with _tab_chat:
    tab_chat_mod.render(target_role, rapidapi_key=st.session_state.rapidapi_key)

with _tab_portfolio:
    tab_portfolio_mod.render(
        target_role,
        github_client_id=_github_client_id,
        github_client_secret=_github_client_secret,
    )

with _tab_interview:
    tab_interview_mod.render(target_role)

with _tab_raw:
    tab_raw_mod.render()

with _tab_maker:
    tab_maker_mod.render(target_role)

with _tab_jobs:
    tab_jobs_mod.render(target_role, rapidapi_key=st.session_state.rapidapi_key)
