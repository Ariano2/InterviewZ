"""
app.py  —  PrepSense AI · Resume Reviewer
Run with:  streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from agents.ats_analyzer import analyze_ats
from agents.bullet_rewriter import rewrite_bullets
from agents.chat_agent import chat_with_resume
from agents.jd_tailor import tailor_resume, generate_cover_letter
from agents.skill_gap import analyze_skill_gap
from agents.resume_structurer import structure_resume
from agents.resume_builder import build_resume_pdf
from agents.portfolio_generator import generate_portfolio
from agents.github_publisher import request_device_code, poll_for_token, get_github_username, publish_portfolio
from agents.interview_prep import generate_qna
from agents.upskill import recommend_skills, generate_learning_plan, yt_search_url
from rag.ingest import ingest_resume
from utils.file_parser import parse_uploaded_file

load_dotenv()

# ── Page config ────────────────────────────────────
st.set_page_config(
    page_title="PrepSense AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS from external file ────────────────────────────────────────────────
_css_path = os.path.join(os.path.dirname(__file__), "styles.css")
with open(_css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "resume_text": None,
    "resume_indexed": False,
    "loaded_filename": "",
    "ats_result": None,
    "ats_result_prev": None,      # Previous run — for delta comparison
    "jd_text": "",                # Optional pasted job description
    "bullets_result": None,       # List[Dict] of {original, improved, why}
    "resume_structure": None,     # Structured JSON from resume_structurer
    "resume_pdf_bytes": None,     # Built PDF bytes
    "jd_tailor_result": None,     # {rewrites, added_keywords}
    "cover_letter": None,         # Plain-text cover letter
    "tailored_pdf_bytes": None,   # PDF with JD-tailored bullets
    "skill_gap_result": None,     # {categories, resume_scores, jd_scores}
    "github_token": "",           # Device flow access token
    "github_username": "",        # resolved after auth
    "github_device_code": "",     # pending device auth
    "github_user_code": "",       # shown to user
    "portfolio_files": None,      # {"index.html":..., "style.css":..., "script.js":...}
    "portfolio_dummy_sections": [],
    "portfolio_pages_url": "",
    "chat_history": [],
    "session_summary": "",
    "groq_client": None,
    "rapidapi_key": os.getenv("RAPIDAPI_KEY", "").strip(),
    # Interview Prep
    "interview_qna": None,           # Dict {easy, medium, hard} from generate_qna
    # Upskill
    "upskill_recommended": None,     # List[Dict] from recommend_skills
    "upskill_plan": None,            # Dict from generate_learning_plan
    "upskill_selected_skill": "",    # skill currently showing a plan for
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ── Sidebar ───────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-family:Inter,sans-serif;font-weight:700;font-size:1.2rem;'
        'color:#1a1a2e;letter-spacing:-0.3px;margin-bottom:2px;">🎯 PrepSense AI</p>',
        unsafe_allow_html=True,
    )
    st.caption("RESUME · RAG · INTERVIEW PREP")
    st.divider()

    target_role = st.text_input(
        "🎯 Target Role",
        value="Software Development Engineer",
        placeholder="e.g. ML Engineer at Google",
    )

    st.divider()
    st.markdown('<p class="section-label">Status</p>', unsafe_allow_html=True)

    if st.session_state.resume_text:
        wc = len(st.session_state.resume_text.split())
        st.markdown(f'<p style="font-size:0.88rem;color:#1a7a45;font-weight:500;">✓ Resume loaded ({wc} words)</p>', unsafe_allow_html=True)
        if st.session_state.resume_indexed:
            st.markdown('<p style="font-size:0.88rem;color:#1a7a45;font-weight:500;">✓ RAG index ready</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="font-size:0.88rem;color:#e67e22;">⏳ Indexing…</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:0.88rem;color:#8890a4;">○ No resume uploaded</p>', unsafe_allow_html=True)

    summary = st.session_state.session_summary
    if summary:
        st.divider()
        st.markdown('<p class="section-label">🧠 Session Memory</p>', unsafe_allow_html=True)
        preview = summary[:200] + ("…" if len(summary) > 200 else "")
        st.markdown(f'<p style="font-size:0.78rem;color:#8890a4;line-height:1.5;">{preview}</p>', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="brand">Prep<span>Sense</span> AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub">ATS Scorer · Bullet Rewriter · RAG Chat · Powered by Groq</p>', unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ── API Key ───────────────────────────────────────────────────────────────────────
_env_key           = os.getenv("GROQ_API_KEY", "").strip()
_github_client_id  = os.getenv("GITHUB_CLIENT_ID", "").strip()
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

st.session_state.groq_client = Groq(api_key=active_key) if active_key else None
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ── File Upload ───────────────────────────────────────────────────────────────────
col_up, col_info = st.columns([3, 2])

with col_up:
    uploaded_file = st.file_uploader("Drop your resume here (PDF or DOCX)", type=["pdf", "docx"])

with col_info:
    st.markdown("""
<div class="card-blue">
  <div class="section-label">How it works</div>
  <p style="font-size:0.88rem;color:#3452c7;line-height:1.8;margin:0;">
    1️⃣ Upload PDF or DOCX<br>
    2️⃣ Auto-chunked → embedded → Chroma RAG<br>
    3️⃣ Run ATS analysis or bullet rewrites<br>
    4️⃣ Chat with your resume via RAG context
  </p>
</div>
""", unsafe_allow_html=True)

# ── Process upload ────────────────────────────────────────────────────────────────
if uploaded_file and uploaded_file.name != st.session_state.loaded_filename:
    with st.spinner("📄 Parsing resume…"):
        try:
            text = parse_uploaded_file(uploaded_file)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    st.session_state.update({
        "resume_text": text,
        "loaded_filename": uploaded_file.name,
        "ats_result": None,
        "ats_result_prev": None,
        "bullets_result": None,
        "resume_structure": None,
        "resume_pdf_bytes": None,
        "jd_tailor_result": None,
        "cover_letter": None,
        "tailored_pdf_bytes": None,
        "skill_gap_result": None,
        "resume_indexed": False,
        "chat_history": [],
        "session_summary": "",
        "interview_qna": None,
        "upskill_recommended": None,
        "upskill_plan": None,
        "upskill_selected_skill": "",
    })

    with st.spinner("🗂️ Building RAG index…"):
        try:
            ingest_resume(text)
            st.session_state.resume_indexed = True
        except Exception as e:
            st.error(f"RAG indexing failed: {e}")

    if st.session_state.resume_indexed:
        st.success(f"✓ **{uploaded_file.name}** parsed & indexed", icon="🗂️")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────────
tab_ats, tab_bullets, tab_jd, tab_chat, tab_portfolio, tab_interview, tab_raw = st.tabs([
    "🏆  ATS Score", "✍️  Bullet Rewriter", "🎯  JD Tailor", "💬  Resume Chat", "🌐  Portfolio", "🎓  Interview Prep", "📄  Raw Text"
])

# ─────────────────────────────────────────────────────────────────────────────────
# TAB 1 — ATS Analysis
# ─────────────────────────────────────────────────────────────────────────────────
with tab_ats:
    if not st.session_state.resume_text:
        st.info("Upload a resume above to run ATS analysis.", icon="👆")
    else:
        # ── JD input ──────────────────────────────────────────────────────────
        with st.expander("📋  Paste a Job Description for precise scoring (recommended)", expanded=False):
            st.markdown(
                '<p style="font-size:0.85rem;color:#4a4e6a;margin-bottom:6px;">'
                'Without a JD the scorer uses generic role keywords. With a JD it matches '
                'exact requirements — scores will be more accurate and actionable.</p>',
                unsafe_allow_html=True,
            )
            jd_input = st.text_area(
                "Job Description",
                value=st.session_state.jd_text,
                height=200,
                placeholder="Paste the full job description here…",
                label_visibility="collapsed",
            )
            if jd_input != st.session_state.jd_text:
                st.session_state.jd_text = jd_input

        jd_badge = "🎯 JD-matched" if st.session_state.jd_text.strip() else "🔎 Role-based"
        if st.button(f"▶  Run ATS Analysis  ({jd_badge})", key="run_ats"):
            if not st.session_state.groq_client:
                st.error("Please provide a Groq API key first.")
            else:
                # Save previous result for delta comparison
                if st.session_state.ats_result:
                    st.session_state.ats_result_prev = st.session_state.ats_result

                with st.spinner(f"Analysing for **{target_role}**…"):
                    st.session_state.ats_result = analyze_ats(
                        st.session_state.resume_text,
                        target_role,
                        st.session_state.groq_client,
                        jd_text=st.session_state.jd_text,
                    )

                if st.session_state.jd_text.strip() and st.session_state.ats_result:
                    with st.spinner("📊  Mapping skill gap radar…"):
                        st.session_state.skill_gap_result = analyze_skill_gap(
                            st.session_state.ats_result.get("matched_keywords", []),
                            st.session_state.ats_result.get("missing_keywords", []),
                            target_role,
                            st.session_state.groq_client,
                        )

        if r := st.session_state.ats_result:
            prev = st.session_state.ats_result_prev
            score = max(0, min(100, r["ats_score"]))

            if score >= 70:
                color, band = "#27ae60", "Strong match"
            elif score >= 45:
                color, band = "#e67e22", "Moderate match"
            else:
                color, band = "#e74c3c", "Weak match"

            bg = "#e8f8ef" if score >= 70 else "#fef3e8" if score >= 45 else "#fef0f0"

            # ── Main score gauge — split into columns to avoid HTML parser issues ──
            g_col, t_col = st.columns([1, 5])
            with g_col:
                st.markdown(
                    f'<div style="width:90px;height:90px;border-radius:50%;'
                    f'background:conic-gradient({color} {score}%,#eef0f7 0);'
                    f'display:flex;align-items:center;justify-content:center;">'
                    f'<div style="width:68px;height:68px;border-radius:50%;background:#fff;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:1.35rem;font-weight:700;color:{color};">{score}</div></div>',
                    unsafe_allow_html=True,
                )
            with t_col:
                jd_badge = " · JD-matched" if r.get("used_jd") else ""
                st.markdown(
                    f'<div style="padding-top:0.6rem;font-size:1.05rem;font-weight:700;color:#1a1a2e;">'
                    f'ATS Score'
                    f'<span style="font-size:0.8rem;font-weight:600;background:{bg};color:{color};'
                    f'padding:2px 10px;border-radius:100px;margin-left:8px;">{band}{jd_badge}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Summary in its own call — LLM text never touches the badge HTML
                st.caption(r["summary"])
            st.progress(score / 100)
            st.write("")

            # ── Sub-score breakdown ────────────────────────────────────────────
            st.markdown('<div class="section-label" style="margin-bottom:0.6rem;">Score Breakdown</div>', unsafe_allow_html=True)

            prev_keys = [
                "keyword_score", "quantification_score",
                "action_verb_score", "section_score", "formatting_score",
            ]
            sub_scores = [
                ("🔑 Keyword Match",    "wt 40%", r["keyword_score"],        prev_keys[0]),
                ("📊 Quantification",   "wt 25%", r["quantification_score"], prev_keys[1]),
                ("⚡ Action Verbs",     "wt 15%", r["action_verb_score"],     prev_keys[2]),
                ("📋 Sections",         "wt 10%", r["section_score"],         prev_keys[3]),
                ("🧹 ATS Format",       "wt 10%", r["formatting_score"],      prev_keys[4]),
            ]

            # Row of metrics (st.metric handles delta natively — no raw HTML needed)
            metric_cols = st.columns(5)
            for col, (label, wt_label, val, pkey) in zip(metric_cols, sub_scores):
                delta = None
                if prev and prev.get(pkey) is not None:
                    diff = val - prev[pkey]
                    delta = diff if diff != 0 else None
                col.metric(label=f"{label}\n{wt_label}", value=val, delta=delta)

            # Progress bars — one flat div per score, no nested spans
            bar_html = ""
            for _, _, val, _ in sub_scores:
                bar_color = "#27ae60" if val >= 70 else "#e67e22" if val >= 45 else "#e74c3c"
                bar_html += (
                    f'<div style="flex:1;background:#eef0f7;border-radius:4px;height:7px;margin:0 3px;">'
                    f'<div style="background:{bar_color};width:{val}%;height:7px;border-radius:4px;"></div>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="display:flex;gap:0;margin-top:-0.6rem;margin-bottom:0.4rem;">{bar_html}</div>',
                unsafe_allow_html=True,
            )

            # Quantification detail line
            if r.get("quant_detail"):
                st.markdown(
                    f'<p style="font-size:0.8rem;color:#8890a4;margin-top:0.2rem;">'
                    f'Quantification detail: {r["quant_detail"]}</p>',
                    unsafe_allow_html=True,
                )

            if prev:
                prev_total = prev.get("ats_score", score)
                total_diff = score - prev_total
                if total_diff != 0:
                    d_color = "#27ae60" if total_diff > 0 else "#e74c3c"
                    d_sign  = "+" if total_diff > 0 else ""
                    st.markdown(
                        f'<p style="font-size:0.85rem;color:{d_color};font-weight:600;">'
                        f'vs previous run: {d_sign}{total_diff} pts overall</p>',
                        unsafe_allow_html=True,
                    )

            st.write("")

            # ── Keywords ───────────────────────────────────────────────────────
            col_matched, col_missing = st.columns(2)
            with col_matched:
                st.markdown(
                    f'<div class="card"><div class="section-label">✅ Matched Keywords '
                    f'<span style="color:#27ae60;">({len(r["matched_keywords"])})</span></div>',
                    unsafe_allow_html=True,
                )
                if r["matched_keywords"]:
                    chips = "".join(f'<span class="chip chip-green">{k}</span>' for k in r["matched_keywords"])
                    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="font-size:0.88rem;color:#8890a4;">None detected.</p>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_missing:
                st.markdown(
                    f'<div class="card"><div class="section-label">❌ Missing Keywords '
                    f'<span style="color:#e74c3c;">({len(r["missing_keywords"])})</span></div>',
                    unsafe_allow_html=True,
                )
                if r["missing_keywords"]:
                    chips = "".join(f'<span class="chip chip-red">{k}</span>' for k in r["missing_keywords"])
                    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="font-size:0.88rem;color:#27ae60;font-weight:500;">Full coverage — none missing!</p>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Strong / Weak areas ────────────────────────────────────────────
            col_strong, col_weak = st.columns(2)
            with col_strong:
                st.markdown('<div class="card"><div class="section-label">💪 Strong Areas</div>', unsafe_allow_html=True)
                for item in r["strong_areas"]:
                    st.markdown(f'<p style="font-size:0.92rem;color:#1a4a2e;margin:4px 0;">▸ {item}</p>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_weak:
                st.markdown('<div class="card"><div class="section-label">⚠️ Weak Areas / Gaps</div>', unsafe_allow_html=True)
                for item in r["weak_areas"]:
                    st.markdown(f'<p style="font-size:0.92rem;color:#7a2020;margin:4px 0;">▸ {item}</p>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Skill Gap Radar ────────────────────────────────────────────────
            st.write("")
            st.markdown(
                '<div class="section-label" style="margin-bottom:0.5rem;">📊 Skill Gap Radar</div>',
                unsafe_allow_html=True,
            )

            if not r.get("used_jd"):
                st.info(
                    "Paste a Job Description above and re-run ATS analysis to see your personalised skill gap radar chart.",
                    icon="💡",
                )
            else:
                sg = st.session_state.skill_gap_result
                if sg and sg.get("categories"):
                    import plotly.graph_objects as go

                    cats     = sg["categories"]
                    r_scores = sg["resume_scores"]
                    j_scores = sg["jd_scores"]

                    # Close the polygon loop for radar
                    cats_loop  = cats     + [cats[0]]
                    r_loop     = r_scores + [r_scores[0]]
                    j_loop     = j_scores + [j_scores[0]]

                    fig = go.Figure()

                    # JD Required — solid orange, strong fill drawn first (background layer)
                    fig.add_trace(go.Scatterpolar(
                        r=j_loop,
                        theta=cats_loop,
                        fill="toself",
                        fillcolor="rgba(230, 126, 34, 0.35)",
                        line=dict(color="#e67e22", width=3),
                        name="JD Required",
                    ))

                    # Your Resume — solid teal, strong fill drawn on top
                    fig.add_trace(go.Scatterpolar(
                        r=r_loop,
                        theta=cats_loop,
                        fill="toself",
                        fillcolor="rgba(26, 188, 156, 0.45)",
                        line=dict(color="#1abc9c", width=3),
                        name="Your Resume",
                    ))

                    fig.update_layout(
                        polar=dict(
                            bgcolor="rgba(248, 249, 252, 1)",
                            radialaxis=dict(
                                visible=True,
                                range=[0, 10],
                                tickvals=[2, 4, 6, 8, 10],
                                tickfont=dict(size=10, color="#555"),
                                gridcolor="#d8dce8",
                                linecolor="#d8dce8",
                            ),
                            angularaxis=dict(
                                tickfont=dict(size=12, color="#1a1a2e"),
                                linecolor="#d8dce8",
                                gridcolor="#d8dce8",
                            ),
                        ),
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom", y=1.06,
                            xanchor="center", x=0.5,
                            font=dict(size=13),
                        ),
                        margin=dict(l=50, r=50, t=70, b=30),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=440,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "Teal = your resume coverage · Orange = JD requirement level · Scale 0–10 per category · Gap between shapes = areas to improve"
                    )
                else:
                    err = sg.get("error") if sg else None
                    if err:
                        st.warning(f"Skill gap agent error: {err}", icon="⚠️")
                    else:
                        st.info("Skill gap data unavailable — re-run ATS with a job description.", icon="💡")

# ─────────────────────────────────────────────────────────────────────────────────
# TAB 2 — Bullet Rewriter + PDF Resume Builder
# ─────────────────────────────────────────────────────────────────────────────────
with tab_bullets:
    if not st.session_state.resume_text:
        st.info("Upload a resume above to rewrite bullet points.", icon="👆")
    else:
        st.markdown("""
<div class="card-blue" style="margin-bottom:1rem;">
  <div class="section-label">AI Bullet Rewriter + PDF Resume Generator</div>
  <p style="font-size:0.9rem;color:#3452c7;line-height:1.65;margin:0;">
    Rewrites your weakest bullets with strong action verbs &amp; metrics,
    then rebuilds your entire resume as a clean, ATS-friendly PDF — ready to download.
  </p>
</div>
""", unsafe_allow_html=True)

        if st.button("▶  Rewrite Bullets & Build PDF Resume", key="run_bullets"):
            if not st.session_state.groq_client:
                st.error("Please provide a Groq API key first.")
            else:
                # Step 1 — rewrite bullets (structured JSON)
                with st.spinner("✍️  Rewriting weak bullets with strong action verbs & metrics…"):
                    pairs = rewrite_bullets(
                        st.session_state.resume_text,
                        target_role,
                        st.session_state.groq_client,
                    )
                    st.session_state.bullets_result = pairs

                # Step 2 — extract resume structure
                with st.spinner("🗂️  Parsing resume structure…"):
                    structure = structure_resume(
                        st.session_state.resume_text,
                        st.session_state.groq_client,
                    )
                    st.session_state.resume_structure = structure

                # Step 3 — build PDF with rewritten bullets substituted in
                with st.spinner("📄  Building your PDF resume…"):
                    try:
                        pdf_bytes = build_resume_pdf(structure, pairs)
                        st.session_state.resume_pdf_bytes = pdf_bytes
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
                        st.session_state.resume_pdf_bytes = None

        # ── PDF download (persists across reruns once built) ──────────────────
        if st.session_state.resume_pdf_bytes:
            st.success("✅ PDF resume built successfully!", icon="📄")
            st.download_button(
                label="⬇  Download Rewritten Resume (.pdf)",
                data=st.session_state.resume_pdf_bytes,
                file_name="rewritten_resume.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.write("")

        # ── Before / After cards ──────────────────────────────────────────────
        pairs = st.session_state.bullets_result
        if pairs:
            rewrites = [p for p in pairs if p.get("action") != "remove"]
            removals = [p for p in pairs if p.get("action") == "remove"]

            summary_parts = []
            if rewrites:
                summary_parts.append(f"{len(rewrites)} rewritten")
            if removals:
                summary_parts.append(f"{len(removals)} removed")
            st.markdown(
                f'<div class="section-label" style="margin-bottom:0.6rem;">'
                f'{" · ".join(summary_parts)}</div>',
                unsafe_allow_html=True,
            )

            for p in rewrites:
                original = (p.get("original") or "").strip()
                improved = (p.get("improved") or "").strip()
                why      = (p.get("why")      or "").strip()
                if not original:
                    continue
                with st.container(border=True):
                    st.markdown("**ORIGINAL**")
                    st.write(original)
                    st.markdown("**IMPROVED**")
                    st.write(improved)
                    st.markdown("**WHY BETTER**")
                    st.caption(why)

            if removals:
                st.markdown("**Removed — redundant / zero-value**")
            for p in removals:
                original = (p.get("original") or "").strip()
                why      = (p.get("why")      or "").strip()
                if not original:
                    continue
                with st.container(border=True):
                    st.write(f"~~{original}~~")
                    st.caption(f"Removed: {why}")

# ─────────────────────────────────────────────────────────────────────────────────
# TAB 3 — JD Tailor + Cover Letter
# ─────────────────────────────────────────────────────────────────────────────────
with tab_jd:
    if not st.session_state.resume_text:
        st.info("Upload a resume above to tailor it to a job description.", icon="👆")
    else:
        st.markdown("""
<div class="card-blue" style="margin-bottom:1rem;">
  <div class="section-label">JD Tailor + Cover Letter Generator</div>
  <p style="font-size:0.9rem;color:#3452c7;line-height:1.65;margin:0;">
    Paste a job description — the AI injects missing keywords into your existing bullets
    with minimum edits, then generates a tailored cover letter and a downloadable PDF.
  </p>
</div>
""", unsafe_allow_html=True)

        jd_tailor_input = st.text_area(
            "Job Description",
            value=st.session_state.jd_text,
            height=220,
            placeholder="Paste the full job description here (required)…",
        )
        if jd_tailor_input != st.session_state.jd_text:
            st.session_state.jd_text = jd_tailor_input

        if st.button("▶  Tailor Resume & Generate Cover Letter", key="run_jd_tailor"):
            if not st.session_state.groq_client:
                st.error("Please provide a Groq API key first.")
            elif not st.session_state.jd_text.strip():
                st.warning("Paste a job description above before running.", icon="⚠️")
            else:
                # Step 1 — keyword injection
                with st.spinner("🎯  Injecting missing JD keywords into your bullets…"):
                    tailor_result = tailor_resume(
                        st.session_state.resume_text,
                        st.session_state.jd_text,
                        target_role,
                        st.session_state.groq_client,
                    )
                    st.session_state.jd_tailor_result = tailor_result

                # Step 2 — cover letter
                with st.spinner("✉️  Writing cover letter…"):
                    cl = generate_cover_letter(
                        st.session_state.resume_text,
                        st.session_state.jd_text,
                        target_role,
                        st.session_state.groq_client,
                    )
                    st.session_state.cover_letter = cl

                # Step 3 — build tailored PDF
                with st.spinner("📄  Building tailored PDF resume…"):
                    try:
                        structure = st.session_state.resume_structure or structure_resume(
                            st.session_state.resume_text,
                            st.session_state.groq_client,
                        )
                        st.session_state.resume_structure = structure
                        pdf_bytes = build_resume_pdf(
                            structure,
                            tailor_result.get("rewrites") or [],
                        )
                        st.session_state.tailored_pdf_bytes = pdf_bytes
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
                        st.session_state.tailored_pdf_bytes = None

        # ── Results ───────────────────────────────────────────────────────────
        tr = st.session_state.jd_tailor_result
        cl = st.session_state.cover_letter

        if tr or cl:
            st.write("")
            col_dl1, col_dl2 = st.columns(2)

            if st.session_state.tailored_pdf_bytes:
                with col_dl1:
                    st.download_button(
                        label="⬇  Download Tailored Resume (.pdf)",
                        data=st.session_state.tailored_pdf_bytes,
                        file_name="tailored_resume.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

            if cl and not cl.startswith("Error"):
                with col_dl2:
                    st.download_button(
                        label="⬇  Download Cover Letter (.txt)",
                        data=cl.encode("utf-8"),
                        file_name="cover_letter.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

            st.write("")

            # Cover letter display
            if cl:
                st.markdown('<div class="section-label" style="margin-bottom:0.4rem;">✉️ Cover Letter</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.write(cl)

            st.write("")

            # Keyword injection summary
            if tr:
                rewrites = tr.get("rewrites") or []
                added    = tr.get("added_keywords") or []

                st.markdown(
                    f'<div class="section-label" style="margin-bottom:0.4rem;">'
                    f'🔑 Keywords Injected ({len(added)})</div>',
                    unsafe_allow_html=True,
                )
                if added:
                    chips = "".join(f'<span class="chip chip-green">{k}</span>' for k in added)
                    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

                st.write("")

                if rewrites:
                    st.markdown(
                        f'<div class="section-label" style="margin-bottom:0.4rem;">'
                        f'✍️ Bullet Changes ({len(rewrites)})</div>',
                        unsafe_allow_html=True,
                    )
                    for rw in rewrites:
                        original = (rw.get("original") or "").strip()
                        improved = (rw.get("improved") or "").strip()
                        kw       = (rw.get("keyword_added") or "").strip()
                        if not original:
                            continue
                        with st.container(border=True):
                            st.markdown("**ORIGINAL**")
                            st.write(original)
                            st.markdown("**IMPROVED**")
                            st.write(improved)
                            if kw:
                                st.caption(f"Keyword added: {kw}")


# ─────────────────────────────────────────────────────────────────────────────────
# TAB 4 — RAG Chat
# ─────────────────────────────────────────────────────────────────────────────────
with tab_chat:
    _job_search_active = bool(st.session_state.rapidapi_key)
    _job_badge = (
        '🔍 <strong>Live job search active</strong> (LinkedIn · Indeed · Glassdoor) — try <em>"Find ML engineer jobs in Bangalore"</em>'
        if _job_search_active
        else 'Add <code>RAPIDAPI_KEY</code> to .env to enable live job search'
    )
    st.markdown(f"""
<div class="card-blue" style="margin-bottom:1rem;">
  <div class="section-label">RAG-Powered Resume Chat</div>
  <p style="font-size:0.9rem;color:#3452c7;line-height:1.65;margin:0;">
    Every answer is grounded in your actual resume via vector search.<br>
    Try: <strong>"What are my strongest skills?"</strong> · <strong>"Suggest improvements for my projects section"</strong><br>
    <span style="font-size:0.85rem;">{_job_badge}</span>
  </p>
</div>
""", unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="bubble-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown(f"🤖 {msg['content']}")

    if user_msg := st.chat_input("Ask anything about your resume or interview prep…"):
        if not st.session_state.groq_client:
            st.error("Please provide a Groq API key first.")
        elif not st.session_state.resume_indexed:
            st.warning("Upload & index a resume first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})

            with st.spinner("Retrieving context & generating…"):
                reply, updated_summary = chat_with_resume(
                    user_message=user_msg,
                    chat_history=st.session_state.chat_history[:-1],
                    groq_client=st.session_state.groq_client,
                    target_role=target_role,
                    session_summary=st.session_state.session_summary,
                    rapidapi_key=st.session_state.rapidapi_key,
                )

            st.session_state.session_summary = updated_summary
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑 Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.session_summary = ""
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────
# TAB 5 — Portfolio Generator
# ─────────────────────────────────────────────────────────────────────────────────
with tab_portfolio:
    if not st.session_state.resume_text:
        st.info("Upload a resume above to generate your portfolio.", icon="👆")
    else:
        st.markdown("""
<div class="card-blue" style="margin-bottom:1rem;">
  <div class="section-label">🌐 One-Click Portfolio Generator</div>
  <p style="font-size:0.9rem;color:#3452c7;line-height:1.65;margin:0;">
    Generates a fully animated personal website from your resume — then publishes it
    to GitHub Pages automatically. Live URL in under 60 seconds.
  </p>
</div>
""", unsafe_allow_html=True)

        # ── Step 1: Template picker ───────────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-bottom:0.5rem;">① Choose a Template</div>', unsafe_allow_html=True)
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            luminary_selected = st.button(
                "☀️  Luminary — Clean & Minimal",
                use_container_width=True,
                key="tmpl_luminary",
                type="primary" if st.session_state.get("_tmpl") == "luminary" else "secondary",
            )
            st.caption("White background · Gradient accents · Dot-grid hero · Card layout")
        with col_t2:
            noir_selected = st.button(
                "🌑  Noir — Dark Pro",
                use_container_width=True,
                key="tmpl_noir",
                type="primary" if st.session_state.get("_tmpl") == "noir" else "secondary",
            )
            st.caption("Dark bg · Cyan neon glow · Animated orbs · Terminal about card")

        if luminary_selected:
            st.session_state["_tmpl"] = "luminary"
            st.rerun()
        if noir_selected:
            st.session_state["_tmpl"] = "noir"
            st.rerun()

        chosen_template = st.session_state.get("_tmpl", "luminary")
        st.markdown(
            f'<p style="font-size:0.85rem;color:#3452c7;margin-top:4px;">Selected: <strong>{"Luminary" if chosen_template == "luminary" else "Noir"}</strong></p>',
            unsafe_allow_html=True,
        )

        # ── Step 2: Repo name ─────────────────────────────────────────────────
        st.write("")
        st.markdown('<div class="section-label" style="margin-bottom:0.5rem;">② Repo & Site Name</div>', unsafe_allow_html=True)
        repo_name = st.text_input(
            "GitHub repo name",
            value="portfolio",
            placeholder="e.g. portfolio  →  username.github.io/portfolio",
            label_visibility="collapsed",
        )
        if repo_name == f"{st.session_state.github_username}.github.io":
            st.warning("This will overwrite your main GitHub Pages site. Rename unless that's intentional.", icon="⚠️")

        # ── Step 3: GitHub OAuth ──────────────────────────────────────────────
        st.write("")
        st.markdown('<div class="section-label" style="margin-bottom:0.5rem;">③ Connect GitHub</div>', unsafe_allow_html=True)

        if st.session_state.github_token:
            st.success(f"✓ Connected as **{st.session_state.github_username}**", icon="✅")
            if st.button("Disconnect", key="gh_disconnect"):
                st.session_state.github_token    = ""
                st.session_state.github_username = ""
                st.session_state.github_device_code = ""
                st.session_state.github_user_code    = ""
                st.rerun()

        elif st.session_state.github_user_code:
            # Device code issued — waiting for user to authorise
            st.markdown(
                f"""<div style="background:#f0fdf4;border:1.5px solid #27ae60;border-radius:10px;padding:18px 22px;">
                <p style="font-size:0.9rem;color:#1a4a2e;margin:0 0 10px 0;">
                  <strong>Step 1</strong> — Open
                  <a href="https://github.com/login/device" target="_blank" style="color:#27ae60;">
                    github.com/login/device
                  </a> in a new tab
                </p>
                <p style="font-size:0.9rem;color:#1a4a2e;margin:0 0 10px 0;">
                  <strong>Step 2</strong> — Enter this code:
                </p>
                <p style="font-family:monospace;font-size:1.6rem;font-weight:800;
                           letter-spacing:0.15em;color:#1a1a2e;margin:0 0 12px 0;">
                  {st.session_state.github_user_code}
                </p>
                <p style="font-size:0.82rem;color:#4a4e6a;margin:0;">
                  Then click the button below once you've approved.
                </p>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button("✅  I've approved — complete connection", key="gh_poll"):
                with st.spinner("🔐 Checking GitHub authorisation…"):
                    _token, _err = poll_for_token(
                        _github_client_id,
                        _github_client_secret,
                        st.session_state.github_device_code,
                        interval=2,
                        timeout=60,
                    )
                if _token:
                    st.session_state.github_token    = _token
                    st.session_state.github_username = get_github_username(_token)
                    st.session_state.github_device_code = ""
                    st.session_state.github_user_code    = ""
                    st.rerun()
                else:
                    st.error(f"Auth failed: {_err}. Click 'Connect GitHub' to try again.")
                    st.session_state.github_device_code = ""
                    st.session_state.github_user_code    = ""

        else:
            if _github_client_id:
                if st.button("🐙  Connect GitHub", key="gh_connect"):
                    with st.spinner("Requesting device code from GitHub…"):
                        _dc_data, _dc_err = request_device_code(_github_client_id)
                    if _dc_err:
                        st.error(f"GitHub error: {_dc_err}")
                    else:
                        st.session_state.github_device_code = _dc_data["device_code"]
                        st.session_state.github_user_code    = _dc_data["user_code"]
                        st.rerun()
                st.caption("Grants repo-only scope. We create one repo on your behalf — nothing else.")
            else:
                st.warning("Add `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET` to your `.env` to enable GitHub publishing.", icon="⚙️")

        # ── Generate + Publish ────────────────────────────────────────────────
        st.write("")
        st.markdown('<div class="section-label" style="margin-bottom:0.5rem;">④ Generate &amp; Publish</div>', unsafe_allow_html=True)

        can_publish = bool(st.session_state.github_token and st.session_state.groq_client)
        if st.button(
            "🚀  Generate Portfolio & Publish to GitHub Pages",
            key="run_portfolio",
            disabled=not can_publish,
            use_container_width=True,
        ):
            # Ensure resume is structured
            with st.spinner("🗂️  Parsing resume structure…"):
                structure = st.session_state.resume_structure or structure_resume(
                    st.session_state.resume_text,
                    st.session_state.groq_client,
                )
                st.session_state.resume_structure = structure

            with st.spinner("✨  Generating portfolio content…"):
                files, dummy_sections = generate_portfolio(
                    structure,
                    target_role,
                    chosen_template,
                    st.session_state.groq_client,
                )
                st.session_state.portfolio_files           = files
                st.session_state.portfolio_dummy_sections  = dummy_sections

            with st.spinner("🐙  Publishing to GitHub Pages…"):
                pages_url, pub_err = publish_portfolio(
                    token=st.session_state.github_token,
                    files=files,
                    repo_name=repo_name.strip() or "portfolio",
                    description=f"Portfolio of {st.session_state.resume_structure.get('name','') or 'Developer'} — built with PrepSense AI",
                )
                st.session_state.portfolio_pages_url = pages_url
                if pub_err and not pages_url:
                    st.error(f"Publishing failed: {pub_err}")

        if not can_publish and st.session_state.groq_client:
            st.caption("Connect GitHub above to enable publishing.")
        elif not can_publish:
            st.caption("Provide Groq API key and connect GitHub to publish.")

        # ── Results ───────────────────────────────────────────────────────────
        pages_url      = st.session_state.portfolio_pages_url
        dummy_sections = st.session_state.portfolio_dummy_sections
        files          = st.session_state.portfolio_files

        if pages_url:
            st.write("")
            st.success("🎉 Portfolio published!", icon="🌐")
            st.markdown(
                f'<a href="{pages_url}" target="_blank" style="display:inline-block;padding:12px 28px;'
                f'background:linear-gradient(135deg,#27ae60,#1abc9c);color:#fff;border-radius:10px;'
                f'font-weight:700;font-size:1rem;text-decoration:none;">🔗  Visit Live Site →</a>',
                unsafe_allow_html=True,
            )
            st.caption(f"URL: {pages_url}  ·  GitHub Pages takes ~30-60 seconds to go live after first publish.")

        if dummy_sections:
            st.write("")
            st.warning(
                f"**Placeholder content** was used for: **{', '.join(dummy_sections)}**.\n\n"
                f"These sections show a yellow ⚠️ banner on your live site. "
                f"Add the missing content to your resume and regenerate to replace them.",
                icon="⚠️",
            )

        if files:
            st.write("")
            st.markdown('<div class="section-label" style="margin-bottom:0.4rem;">⬇ Download Files</div>', unsafe_allow_html=True)
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            with dl_col1:
                st.download_button("⬇ index.html", data=files["index.html"], file_name="index.html", mime="text/html", use_container_width=True)
            with dl_col2:
                st.download_button("⬇ style.css",  data=files["style.css"],  file_name="style.css",  mime="text/css",  use_container_width=True)
            with dl_col3:
                st.download_button("⬇ script.js",  data=files["script.js"],  file_name="script.js",  mime="text/javascript", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────────
# TAB 6 — Interview Prep (Mock Interview + Upskill)
# ─────────────────────────────────────────────────────────────────────────────────
with tab_interview:
    if not st.session_state.resume_text:
        st.info("Upload a resume above to use Interview Prep.", icon="👆")
    else:
        st.markdown("""
<div class="card-blue" style="margin-bottom:1rem;">
  <div class="section-label">🎓 Interview Prep &amp; Upskill Hub</div>
  <p style="font-size:0.9rem;color:#3452c7;line-height:1.65;margin:0;">
    Practice with resume-based MCQs and concept Q&amp;As, then build a personalised learning plan
    for any skill you want to add to your arsenal.
  </p>
</div>
""", unsafe_allow_html=True)

        prep_tab, upskill_tab = st.tabs(["🧠  Mock Interview", "🚀  Upskill"])

        # ── MOCK INTERVIEW ─────────────────────────────────────────────────────
        with prep_tab:
            st.markdown(
                '<p style="font-size:0.9rem;color:#4a4e6a;margin-bottom:0.8rem;">'
                'Resume-specific questions — your projects, your tech choices, your trade-offs. '
                '~34 questions across Easy / Medium / Hard.</p>',
                unsafe_allow_html=True,
            )

            if st.button("▶  Generate Interview Questions", key="run_interview"):
                if not st.session_state.groq_client:
                    st.error("Please provide a Groq API key first.")
                else:
                    with st.spinner("🧠  Reading your resume and generating questions…"):
                        st.session_state.interview_qna = generate_qna(
                            st.session_state.resume_text,
                            target_role,
                            st.session_state.groq_client,
                        )

            qna = st.session_state.interview_qna
            if not isinstance(qna, dict):
                st.session_state.interview_qna = None
                qna = None

            if qna and (qna.get("easy") or qna.get("medium") or qna.get("hard")):
                easy_qs   = qna.get("easy",   [])
                medium_qs = qna.get("medium", [])
                hard_qs   = qna.get("hard",   [])

                # Summary counts
                total = len(easy_qs) + len(medium_qs) + len(hard_qs)
                st.markdown(
                    f'<p style="font-size:0.82rem;color:#8890a4;margin:0.4rem 0 0.8rem;">'
                    f'{total} questions — {len(easy_qs)} Easy · {len(medium_qs)} Medium · {len(hard_qs)} Hard</p>',
                    unsafe_allow_html=True,
                )

                diff_easy, diff_med, diff_hard = st.tabs([
                    f"🟢  Easy ({len(easy_qs)})",
                    f"🟡  Medium ({len(medium_qs)})",
                    f"🔴  Hard ({len(hard_qs)})",
                ])

                def _render_qna_list(questions: list, prefix: str) -> None:
                    if not questions:
                        st.info("No questions generated for this tier.", icon="ℹ️")
                        return
                    for i, item in enumerate(questions):
                        question = item.get("question", "")
                        answer   = item.get("answer", "")
                        example  = item.get("example")
                        with st.expander(f"**{prefix}{i+1}.** {question}", expanded=False):
                            st.markdown(
                                f'<p style="font-size:0.92rem;color:#1a1a2e;line-height:1.75;margin:0 0 4px 0;">'
                                f'{answer}</p>',
                                unsafe_allow_html=True,
                            )
                            if example:
                                lang = (
                                    "python" if ("def " in example or "import " in example)
                                    else "javascript" if ("=>" in example or "const " in example or "function" in example)
                                    else "text"
                                )
                                st.code(example, language=lang)

                with diff_easy:
                    _render_qna_list(easy_qs, "E")
                with diff_med:
                    _render_qna_list(medium_qs, "M")
                with diff_hard:
                    _render_qna_list(hard_qs, "H")

        # ── UPSKILL ────────────────────────────────────────────────────────────
        with upskill_tab:
            st.markdown(
                '<p style="font-size:0.9rem;color:#4a4e6a;margin-bottom:0.8rem;">'
                'Enter a skill you want to learn, or let us recommend based on your resume gaps. '
                'Get a 4-week roadmap + curated YouTube resources.</p>',
                unsafe_allow_html=True,
            )

            # ── Skill input row ───────────────────────────────────────────────
            col_skill_in, col_skill_btn = st.columns([3, 1])
            with col_skill_in:
                custom_skill = st.text_input(
                    "Skill to learn",
                    placeholder="e.g. Docker, System Design, TypeScript, GraphQL…",
                    label_visibility="collapsed",
                    key="upskill_custom_input",
                )
            with col_skill_btn:
                run_custom = st.button("▶  Get Plan", key="run_upskill_custom", use_container_width=True)

            if run_custom and custom_skill.strip():
                if not st.session_state.groq_client:
                    st.error("Please provide a Groq API key first.")
                else:
                    st.session_state.upskill_selected_skill = custom_skill.strip()
                    with st.spinner(f"📅  Building learning plan for **{custom_skill.strip()}**…"):
                        st.session_state.upskill_plan = generate_learning_plan(
                            custom_skill.strip(),
                            target_role,
                            st.session_state.groq_client,
                        )

            st.divider()

            # ── AI Recommendations ────────────────────────────────────────────
            st.markdown(
                '<div class="section-label" style="margin-bottom:0.5rem;">✨ AI-Recommended Skills</div>',
                unsafe_allow_html=True,
            )
            st.caption("Based on your resume gaps and target role — click any skill to generate its learning plan.")

            if st.button("🔍  Get Skill Recommendations", key="run_skill_recs"):
                if not st.session_state.groq_client:
                    st.error("Please provide a Groq API key first.")
                else:
                    missing_kw = (st.session_state.ats_result or {}).get("missing_keywords", [])
                    with st.spinner("✨  Analysing skill gaps…"):
                        st.session_state.upskill_recommended = recommend_skills(
                            st.session_state.resume_text,
                            target_role,
                            missing_kw,
                            st.session_state.groq_client,
                        )

            recs = st.session_state.upskill_recommended
            if recs:
                priority_color = {"High": "#e74c3c", "Medium": "#e67e22", "Low": "#27ae60"}
                rec_cols = st.columns(min(len(recs), 3))
                for idx, rec in enumerate(recs):
                    skill_name = rec.get("skill", "")
                    priority   = rec.get("priority", "Medium")
                    reason     = rec.get("reason", "")
                    p_color    = priority_color.get(priority, "#8890a4")

                    with rec_cols[idx % 3]:
                        st.markdown(
                            f'<div class="card" style="padding:14px 16px;margin-bottom:0.5rem;">'
                            f'<div style="font-weight:700;font-size:0.95rem;color:#1a1a2e;">{skill_name}</div>'
                            f'<span style="font-size:0.75rem;font-weight:600;color:{p_color};'
                            f'background:{p_color}18;padding:2px 8px;border-radius:100px;">{priority} Priority</span>'
                            f'<p style="font-size:0.82rem;color:#4a4e6a;margin:6px 0 0 0;line-height:1.5;">{reason}</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if st.button(f"📅 Plan for {skill_name}", key=f"plan_{idx}", use_container_width=True):
                            if not st.session_state.groq_client:
                                st.error("Please provide a Groq API key first.")
                            else:
                                st.session_state.upskill_selected_skill = skill_name
                                with st.spinner(f"📅  Building plan for **{skill_name}**…"):
                                    st.session_state.upskill_plan = generate_learning_plan(
                                        skill_name,
                                        target_role,
                                        st.session_state.groq_client,
                                    )

            # ── Learning Plan ─────────────────────────────────────────────────
            plan = st.session_state.upskill_plan
            sel  = st.session_state.upskill_selected_skill

            if plan and sel:
                st.write("")
                st.markdown(
                    f'<div class="section-label" style="margin-bottom:0.6rem;">📅 4-Week Plan — {sel}</div>',
                    unsafe_allow_html=True,
                )

                overview = plan.get("overview", "")
                if overview:
                    st.markdown(
                        f'<div class="card-blue" style="margin-bottom:1rem;">'
                        f'<p style="font-size:0.9rem;color:#3452c7;margin:0;line-height:1.65;">{overview}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Week cards
                weeks = plan.get("weeks", [])
                if weeks:
                    week_cols = st.columns(len(weeks))
                    week_colors = ["#3452c7", "#27ae60", "#e67e22", "#8e44ad"]
                    for wi, week in enumerate(weeks):
                        wnum   = week.get("week", wi + 1)
                        goal   = week.get("goal", "")
                        topics = week.get("topics", [])
                        wcolor = week_colors[wi % len(week_colors)]
                        topics_html = "".join(
                            f'<li style="font-size:0.82rem;color:#4a4e6a;margin:3px 0;">{t}</li>'
                            for t in topics
                        )
                        with week_cols[wi]:
                            st.markdown(
                                f'<div class="card" style="padding:14px;height:100%;">'
                                f'<div style="font-size:0.75rem;font-weight:700;color:{wcolor};'
                                f'text-transform:uppercase;letter-spacing:0.5px;">Week {wnum}</div>'
                                f'<div style="font-weight:600;font-size:0.88rem;color:#1a1a2e;'
                                f'margin:4px 0 8px;">{goal}</div>'
                                f'<ul style="margin:0;padding-left:1rem;">{topics_html}</ul>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                # Resources
                resources = plan.get("resources", [])
                if resources:
                    st.write("")
                    st.markdown(
                        '<div class="section-label" style="margin-bottom:0.6rem;">🎬 Resources &amp; YouTube Playlists</div>',
                        unsafe_allow_html=True,
                    )
                    type_icon = {"Video": "▶️", "Course": "🎓", "Docs": "📖", "Practice": "💻"}
                    type_color = {"Video": "#e74c3c", "Course": "#3452c7", "Docs": "#27ae60", "Practice": "#8e44ad"}

                    for res in resources:
                        title  = res.get("title", "Resource")
                        rtype  = res.get("type", "Video")
                        query  = res.get("search_query", title)
                        url    = yt_search_url(query)
                        icon   = type_icon.get(rtype, "🔗")
                        color  = type_color.get(rtype, "#4a4e6a")

                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:12px;padding:10px 14px;'
                            f'border-radius:8px;background:#f8f9fc;border:1px solid #e8eaf4;margin-bottom:6px;">'
                            f'<span style="font-size:1.1rem;">{icon}</span>'
                            f'<div style="flex:1;">'
                            f'<span style="font-size:0.88rem;font-weight:600;color:#1a1a2e;">{title}</span>'
                            f'<span style="font-size:0.75rem;font-weight:600;color:{color};'
                            f'background:{color}18;padding:1px 7px;border-radius:100px;margin-left:8px;">{rtype}</span>'
                            f'</div>'
                            f'<a href="{url}" target="_blank" style="font-size:0.8rem;font-weight:600;'
                            f'color:#3452c7;text-decoration:none;white-space:nowrap;">Search on YouTube →</a>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

# ─────────────────────────────────────────────────────────────────────────────────
# TAB 7 — Raw Resume Text
# ─────────────────────────────────────────────────────────────────────────────────
with tab_raw:
    if not st.session_state.resume_text:
        st.info("No resume loaded yet. Upload one above.", icon="👆")
    else:
        text = st.session_state.resume_text
        st.markdown(
            f'<div class="section-label">Parsed Text · {len(text.split())} words · {len(text)} chars</div>',
            unsafe_allow_html=True,
        )
        preview = text[:6000] + ("\n…[truncated — first 6,000 chars shown]" if len(text) > 6000 else "")
        st.markdown(f"""
<div class="card">
  <pre style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;color:#4a4e6a;
              white-space:pre-wrap;line-height:1.75;max-height:520px;overflow-y:auto;margin:0;">{preview}</pre>
</div>
""", unsafe_allow_html=True)

        st.download_button(
            label="⬇ Download parsed text (.txt)",
            data=text,
            file_name="parsed_resume.txt",
            mime="text/plain",
        )