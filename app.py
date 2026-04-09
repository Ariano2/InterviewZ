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
from agents.resume_structurer import structure_resume
from agents.resume_builder import build_resume_pdf
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
    "chat_history": [],
    "session_summary": "",
    "groq_client": None,
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
_env_key = os.getenv("GROQ_API_KEY", "").strip()

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
        "resume_indexed": False,
        "chat_history": [],
        "session_summary": "",
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
tab_ats, tab_bullets, tab_chat, tab_raw = st.tabs([
    "🏆  ATS Score", "✍️  Bullet Rewriter", "💬  Resume Chat", "📄  Raw Text"
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
# TAB 3 — RAG Chat
# ─────────────────────────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown("""
<div class="card-blue" style="margin-bottom:1rem;">
  <div class="section-label">RAG-Powered Resume Chat</div>
  <p style="font-size:0.9rem;color:#3452c7;line-height:1.65;margin:0;">
    Every answer is grounded in your actual resume via vector search.<br>
    Try: <strong>"What are my strongest skills?"</strong> · <strong>"Suggest improvements for my projects section"</strong>
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
# TAB 4 — Raw Resume Text
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