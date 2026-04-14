"""tabs/ats.py — ATS Score tab."""
import streamlit as st

from agents.ats_analyzer import analyze_ats
from agents.skill_gap import analyze_skill_gap
from ui.components import require_resume, section_heading, chip_list, score_color


def render(target_role: str) -> None:
    if not require_resume("Upload a resume above to run ATS analysis."):
        return

    # ── JD input ──────────────────────────────────────────────────────────────
    with st.expander("📋  Paste a Job Description for precise scoring (recommended)", expanded=False):
        st.markdown(
            '<p class="label-text">Without a JD the scorer uses generic role keywords. '
            "With a JD it matches exact requirements — scores will be more accurate and actionable.</p>",
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
            if st.session_state.ats_result:
                st.session_state.ats_result_prev = st.session_state.ats_result
            with st.spinner(f"Analysing for **{target_role}**…"):
                st.session_state.ats_result = analyze_ats(
                    st.session_state.resume_text,
                    target_role,
                    st.session_state.groq_client,
                    jd_text=st.session_state.jd_text,
                    model=st.session_state.groq_model,
                    resume_embedding=st.session_state.resume_embedding,
                )
            if st.session_state.jd_text.strip() and st.session_state.ats_result:
                with st.spinner("📊  Mapping skill gap radar…"):
                    st.session_state.skill_gap_result = analyze_skill_gap(
                        st.session_state.ats_result.get("matched_keywords", []),
                        st.session_state.ats_result.get("missing_keywords", []),
                        target_role,
                        st.session_state.groq_client,
                        model=st.session_state.groq_model,
                    )

    # ── Model comparison ──────────────────────────────────────────────────────
    with st.expander("⚖️  Compare two models side-by-side", expanded=False):
        _CMP_MODELS = {
            "GPT-OSS 120B": "openai/gpt-oss-120b",
            "Llama 3.3 70B": "llama-3.3-70b-versatile",
            "Llama 3.1 8B (fast)": "llama-3.1-8b-instant",
            "Gemma 2 9B": "gemma2-9b-it",
        }
        _c1, _c2 = st.columns(2)
        with _c1:
            _m1_name = st.selectbox("Model A", list(_CMP_MODELS.keys()), index=0, key="cmp_m1")
        with _c2:
            _m2_name = st.selectbox("Model B", list(_CMP_MODELS.keys()), index=1, key="cmp_m2")

        if st.button("▶  Run Comparison", key="run_comparison"):
            if not st.session_state.groq_client:
                st.error("Groq API key required.")
            else:
                with st.spinner(f"Running {_m1_name} vs {_m2_name}…"):
                    _r1 = analyze_ats(
                        st.session_state.resume_text, target_role,
                        st.session_state.groq_client,
                        jd_text=st.session_state.jd_text, model=_CMP_MODELS[_m1_name],
                        resume_embedding=st.session_state.resume_embedding,
                    )
                    _r2 = analyze_ats(
                        st.session_state.resume_text, target_role,
                        st.session_state.groq_client,
                        jd_text=st.session_state.jd_text, model=_CMP_MODELS[_m2_name],
                        resume_embedding=st.session_state.resume_embedding,
                    )
                _col1, _col2 = st.columns(2)
                for _col, _res, _mname in [(_col1, _r1, _m1_name), (_col2, _r2, _m2_name)]:
                    with _col:
                        _sc = max(0, min(100, _res.get("ats_score", 0)))
                        _clr, _, _ = score_color(_sc)
                        st.markdown(
                            f"**{_mname}**  \n"
                            f'<span style="font-size:2rem;font-weight:700;color:{_clr};">{_sc}</span>/100',
                            unsafe_allow_html=True,
                        )
                        st.caption("**Strong areas:** " + "; ".join(_res.get("strong_areas", [])[:3]))
                        st.caption("**Weak areas:** " + "; ".join(_res.get("weak_areas", [])[:3]))
                        st.caption("**Summary:** " + _res.get("summary", "—"))

    # ── Results ───────────────────────────────────────────────────────────────
    r = st.session_state.ats_result
    if not r:
        return

    prev  = st.session_state.ats_result_prev
    score = max(0, min(100, r["ats_score"]))
    color, bg, band = score_color(score)

    # Gauge + summary
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
        jd_badge_text = " · JD-matched" if r.get("used_jd") else ""
        st.markdown(
            f'<div style="padding-top:0.6rem;font-size:1.05rem;font-weight:700;color:#1a1a2e;">'
            f'ATS Score'
            f'<span style="font-size:0.8rem;font-weight:600;background:{bg};color:{color};'
            f'padding:2px 10px;border-radius:100px;margin-left:8px;">{band}{jd_badge_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(r["summary"])
    st.progress(score / 100)

    # Embedding similarity metrics
    sim_metrics = r.get("similarity_metrics", {})
    if sim_metrics and sim_metrics.get("cosine", -1.0) >= 0:
        st.markdown(
            '<div class="section-label" style="margin:0.7rem 0 0.4rem;">'
            '🤖 Embedding Similarity Metrics'
            '<span class="text-muted" style="font-size:0.72rem;font-weight:400;margin-left:8px;">'
            'BAAI/bge-small-en-v1.5 · no LLM involved</span></div>',
            unsafe_allow_html=True,
        )
        _metric_defs = [
            ("Cosine",    "cos(θ) = A·B — angle between vectors.",            sim_metrics.get("cosine",    0.0)),
            ("Euclidean", "1 − d/2 where d=‖A−B‖₂.",                          sim_metrics.get("euclidean", 0.0)),
            ("Manhattan", "L1 norm similarity.",                                sim_metrics.get("manhattan", 0.0)),
            ("Pearson",   "Mean-centred cosine — co-variance of dims.",         sim_metrics.get("pearson",   0.0)),
        ]
        sim_cols = st.columns(4)
        for col, (label, tooltip, val) in zip(sim_cols, _metric_defs):
            val_color, _, _ = score_color(int(val * 100))
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value" style="color:{val_color};">{val:.3f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            col.caption(tooltip)

    st.write("")
    # Sub-score breakdown
    section_heading("Score Breakdown")
    sub_scores = [
        ("🔑 Keyword Match",  "wt 50%", r["keyword_score"],        "keyword_score"),
        ("⚡ Action Verbs",   "wt 20%", r["action_verb_score"],     "action_verb_score"),
        ("📊 Quantification", "wt 15%", r["quantification_score"],  "quantification_score"),
        ("📋 Sections",       "wt 10%", r["section_score"],         "section_score"),
        ("🧹 ATS Format",     "wt  5%", r["formatting_score"],      "formatting_score"),
    ]
    metric_cols = st.columns(5)
    for col, (label, wt_label, val, pkey) in zip(metric_cols, sub_scores):
        delta = None
        if prev and prev.get(pkey) is not None:
            diff = val - prev[pkey]
            delta = diff if diff != 0 else None
        col.metric(label=f"{label}\n{wt_label}", value=val, delta=delta)

    bar_html = ""
    for _, _, val, _ in sub_scores:
        c, _, _ = score_color(val)
        bar_html += (
            f'<div class="score-bar-track" style="flex:1;margin:0 3px;">'
            f'<div class="score-bar-fill" style="background:{c};width:{val}%;"></div></div>'
        )
    st.markdown(
        f'<div style="display:flex;gap:0;margin-top:-0.6rem;margin-bottom:0.4rem;">{bar_html}</div>',
        unsafe_allow_html=True,
    )

    if r.get("quant_detail"):
        st.markdown(f'<p class="meta-text" style="margin-top:0.2rem;">Quantification: {r["quant_detail"]}</p>', unsafe_allow_html=True)

    if prev:
        total_diff = score - prev.get("ats_score", score)
        if total_diff != 0:
            d_color = "#27ae60" if total_diff > 0 else "#e74c3c"
            d_sign  = "+" if total_diff > 0 else ""
            st.markdown(
                f'<p style="font-size:0.85rem;color:{d_color};font-weight:600;">'
                f'vs previous run: {d_sign}{total_diff} pts overall</p>',
                unsafe_allow_html=True,
            )

    st.write("")

    # Keywords
    col_matched, col_missing = st.columns(2)
    with col_matched:
        st.markdown(
            f'<div class="card"><div class="section-label">✅ Matched Keywords '
            f'<span class="text-ok">({len(r["matched_keywords"])})</span></div>',
            unsafe_allow_html=True,
        )
        if r["matched_keywords"]:
            chip_list(r["matched_keywords"], "green")
        else:
            st.markdown('<p class="meta-text">None detected.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_missing:
        st.markdown(
            f'<div class="card"><div class="section-label">❌ Missing Keywords '
            f'<span class="text-error">({len(r["missing_keywords"])})</span></div>',
            unsafe_allow_html=True,
        )
        if r["missing_keywords"]:
            chip_list(r["missing_keywords"], "red")
        else:
            st.markdown('<p class="meta-text text-ok" style="font-weight:500;">Full coverage — none missing!</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # KeyBERT
    kb_resume  = r.get("keybert_resume_kws", [])
    kb_jd      = r.get("keybert_jd_kws", [])
    kb_overlap = r.get("keybert_overlap", [])
    if kb_resume:
        st.write("")
        st.markdown(
            '<div class="section-label" style="margin-bottom:0.4rem;">'
            '🔬 KeyBERT Keyword Extraction'
            '<span class="text-muted" style="font-size:0.72rem;font-weight:400;margin-left:8px;">'
            'BERT embeddings + MMR · zero LLM</span></div>',
            unsafe_allow_html=True,
        )
        kb_cols = st.columns(2) if kb_jd else [st.container()]
        with kb_cols[0]:
            st.markdown(f'<div class="card"><div class="section-label">📄 Resume Keywords ({len(kb_resume)})</div>', unsafe_allow_html=True)
            chip_list(kb_resume, "green")
            st.markdown("</div>", unsafe_allow_html=True)
        if kb_jd:
            with kb_cols[1]:
                st.markdown(f'<div class="card"><div class="section-label">📋 JD Keywords ({len(kb_jd)})</div>', unsafe_allow_html=True)
                for kw in kb_jd:
                    color = "chip-green" if kw in kb_overlap else "chip-red"
                    st.markdown(f'<span class="chip {color}">{kw}</span>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            if kb_overlap:
                st.caption(f"✅ {len(kb_overlap)} of {len(kb_jd)} JD keywords found in your resume by KeyBERT")
            else:
                st.caption("❌ No JD keywords matched in resume by KeyBERT")

    # Strong / Weak areas
    col_strong, col_weak = st.columns(2)
    with col_strong:
        st.markdown('<div class="card"><div class="section-label">💪 Strong Areas</div>', unsafe_allow_html=True)
        for item in r["strong_areas"]:
            st.markdown(f'<p class="card-item">▸ {item}</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_weak:
        st.markdown('<div class="card"><div class="section-label">⚠️ Weak Areas / Gaps</div>', unsafe_allow_html=True)
        for item in r["weak_areas"]:
            st.markdown(f'<p class="card-item card-item-weak">▸ {item}</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Skill gap radar
    st.write("")
    section_heading("📊 Skill Gap Radar", margin_bottom="0.5rem")
    if not r.get("used_jd"):
        st.info("Paste a Job Description above and re-run ATS analysis to see your personalised skill gap radar.", icon="💡")
    else:
        sg = st.session_state.skill_gap_result
        if sg and sg.get("categories"):
            import plotly.graph_objects as go
            cats    = sg["categories"]
            r_scores = sg["resume_scores"]
            j_scores = sg["jd_scores"]
            cats_loop = cats + [cats[0]]
            r_loop    = r_scores + [r_scores[0]]
            j_loop    = j_scores + [j_scores[0]]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=j_loop, theta=cats_loop, fill="toself",
                fillcolor="rgba(230,126,34,0.35)", line=dict(color="#e67e22", width=3), name="JD Required"))
            fig.add_trace(go.Scatterpolar(r=r_loop, theta=cats_loop, fill="toself",
                fillcolor="rgba(26,188,156,0.45)", line=dict(color="#1abc9c", width=3), name="Your Resume"))
            fig.update_layout(
                polar=dict(bgcolor="rgba(248,249,252,1)",
                    radialaxis=dict(visible=True, range=[0,10], tickvals=[2,4,6,8,10],
                        tickfont=dict(size=10, color="#555"), gridcolor="#d8dce8", linecolor="#d8dce8"),
                    angularaxis=dict(tickfont=dict(size=12, color="#1a1a2e"), linecolor="#d8dce8", gridcolor="#d8dce8")),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5, font=dict(size=13)),
                margin=dict(l=50, r=50, t=70, b=30),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=440,
            )
            st.plotly_chart(fig, width='stretch')
            st.caption("Teal = your resume · Orange = JD requirement · Scale 0–10 · Gap between shapes = areas to improve")
        else:
            err = sg.get("error") if sg else None
            if err:
                st.warning(f"Skill gap error: {err}", icon="⚠️")
            else:
                st.info("Skill gap data unavailable — re-run ATS with a job description.", icon="💡")
