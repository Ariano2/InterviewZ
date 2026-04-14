"""tabs/interview.py — Interview Prep & Upskill Hub tab."""
import streamlit as st

from agents.interview_prep import generate_qna
from agents.upskill import recommend_skills, generate_learning_plan, yt_search_url
from ui.components import require_resume, section_heading


def render(target_role: str) -> None:
    if not require_resume("Upload a resume above to use Interview Prep."):
        return

    st.markdown(
        '<div class="tab-intro">'
        '<div class="section-label">🎓 Interview Prep &amp; Upskill Hub</div>'
        '<p>Practice with resume-based MCQs and concept Q&amp;As, then build a personalised '
        'learning plan for any skill you want to add to your arsenal.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    prep_tab, upskill_tab = st.tabs(["🧠  Mock Interview", "🚀  Upskill"])

    # ── MOCK INTERVIEW ────────────────────────────────────────────────────────
    with prep_tab:
        st.markdown('<p class="body-text">Resume-specific questions — your projects, your tech choices, your trade-offs. ~34 questions across Easy / Medium / Hard.</p>', unsafe_allow_html=True)

        if st.button("▶  Generate Interview Questions", key="run_interview"):
            if not st.session_state.groq_client:
                st.error("Please provide a Groq API key first.")
            else:
                with st.spinner("🧠  Reading your resume and generating questions…"):
                    st.session_state.interview_qna = generate_qna(
                        st.session_state.resume_text, target_role,
                        st.session_state.groq_client, model=st.session_state.groq_model,
                    )

        qna = st.session_state.interview_qna
        if not isinstance(qna, dict):
            st.session_state.interview_qna = None
            qna = None

        if qna and (qna.get("easy") or qna.get("medium") or qna.get("hard")):
            easy_qs   = qna.get("easy",   [])
            medium_qs = qna.get("medium", [])
            hard_qs   = qna.get("hard",   [])
            total = len(easy_qs) + len(medium_qs) + len(hard_qs)
            st.markdown(
                f'<p class="meta-text" style="margin:0.4rem 0 0.8rem;">'
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
                            f'<p style="font-size:0.92rem;color:#1a1a2e;line-height:1.75;margin:0 0 4px 0;">{answer}</p>',
                            unsafe_allow_html=True,
                        )
                        if example:
                            lang = (
                                "python" if ("def " in example or "import " in example)
                                else "javascript" if ("=>" in example or "const " in example or "function" in example)
                                else "text"
                            )
                            st.code(example, language=lang)

            with diff_easy:  _render_qna_list(easy_qs,   "E")
            with diff_med:   _render_qna_list(medium_qs, "M")
            with diff_hard:  _render_qna_list(hard_qs,   "H")

    # ── UPSKILL ───────────────────────────────────────────────────────────────
    with upskill_tab:
        st.markdown(
            '<p class="body-text">Enter a skill you want to learn, or let us recommend based on your '
            'resume gaps. Get a 4-week roadmap + curated YouTube resources.</p>',
            unsafe_allow_html=True,
        )

        col_skill_in, col_skill_btn = st.columns([3, 1])
        with col_skill_in:
            custom_skill = st.text_input(
                "Skill to learn",
                placeholder="e.g. Docker, System Design, TypeScript, GraphQL…",
                label_visibility="collapsed",
                key="upskill_custom_input",
            )
        with col_skill_btn:
            run_custom = st.button("▶  Get Plan", key="run_upskill_custom", width='stretch')

        if run_custom and custom_skill.strip():
            if not st.session_state.groq_client:
                st.error("Please provide a Groq API key first.")
            else:
                st.session_state.upskill_selected_skill = custom_skill.strip()
                with st.spinner(f"📅  Building learning plan for **{custom_skill.strip()}**…"):
                    st.session_state.upskill_plan = generate_learning_plan(
                        custom_skill.strip(), target_role,
                        st.session_state.groq_client, model=st.session_state.groq_model,
                    )

        st.divider()

        section_heading("✨ AI-Recommended Skills", margin_bottom="0.5rem")
        st.caption("Based on your resume gaps and target role — click any skill to generate its learning plan.")

        if st.button("🔍  Get Skill Recommendations", key="run_skill_recs"):
            if not st.session_state.groq_client:
                st.error("Please provide a Groq API key first.")
            else:
                missing_kw = (st.session_state.ats_result or {}).get("missing_keywords", [])
                with st.spinner("✨  Analysing skill gaps…"):
                    st.session_state.upskill_recommended = recommend_skills(
                        st.session_state.resume_text, target_role, missing_kw,
                        st.session_state.groq_client, model=st.session_state.groq_model,
                    )

        recs = st.session_state.upskill_recommended
        if recs:
            priority_class = {"High": "priority-high", "Medium": "priority-medium", "Low": "priority-low"}
            rec_cols = st.columns(min(len(recs), 3))
            for idx, rec in enumerate(recs):
                skill_name = rec.get("skill", "")
                priority   = rec.get("priority", "Medium")
                reason     = rec.get("reason", "")
                p_cls      = priority_class.get(priority, "priority-medium")
                with rec_cols[idx % 3]:
                    st.markdown(
                        f'<div class="card" style="padding:14px 16px;margin-bottom:0.5rem;">'
                        f'<div style="font-weight:700;font-size:0.95rem;color:#1a1a2e;">{skill_name}</div>'
                        f'<span class="priority-badge {p_cls}">{priority} Priority</span>'
                        f'<p class="meta-text" style="margin:6px 0 0;line-height:1.5;">{reason}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(f"📅 Plan for {skill_name}", key=f"plan_{idx}", width='stretch'):
                        if not st.session_state.groq_client:
                            st.error("Please provide a Groq API key first.")
                        else:
                            st.session_state.upskill_selected_skill = skill_name
                            with st.spinner(f"📅  Building plan for **{skill_name}**…"):
                                st.session_state.upskill_plan = generate_learning_plan(
                                    skill_name, target_role,
                                    st.session_state.groq_client, model=st.session_state.groq_model,
                                )

        # Learning Plan
        plan = st.session_state.upskill_plan
        sel  = st.session_state.upskill_selected_skill
        if not plan or not sel:
            return

        st.write("")
        section_heading(f"📅 4-Week Plan — {sel}", margin_bottom="0.6rem")

        overview = plan.get("overview", "")
        if overview:
            st.markdown(
                f'<div class="card-blue"><p class="body-text text-primary" style="margin:0;">{overview}</p></div>',
                unsafe_allow_html=True,
            )

        weeks = plan.get("weeks", [])
        if weeks:
            week_cols   = st.columns(len(weeks))
            week_colors = ["#3452c7", "#27ae60", "#e67e22", "#8e44ad"]
            for wi, week in enumerate(weeks):
                wnum   = week.get("week", wi + 1)
                goal   = week.get("goal", "")
                topics = week.get("topics", [])
                wcolor = week_colors[wi % len(week_colors)]
                topics_html = "".join(f'<li>{t}</li>' for t in topics)
                with week_cols[wi]:
                    st.markdown(
                        f'<div class="week-card">'
                        f'<div class="week-card-header" style="color:{wcolor};">Week {wnum}</div>'
                        f'<div class="week-card-goal">{goal}</div>'
                        f'<ul>{topics_html}</ul>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        resources = plan.get("resources", [])
        if not resources:
            return

        st.write("")
        section_heading("🎬 Resources &amp; YouTube Playlists", margin_bottom="0.6rem")
        type_icon  = {"Video": "▶️", "Course": "🎓", "Docs": "📖", "Practice": "💻"}
        type_color = {"Video": "#e74c3c", "Course": "#3452c7", "Docs": "#27ae60", "Practice": "#8e44ad"}

        for res in resources:
            title  = res.get("title", "Resource")
            rtype  = res.get("type", "Video")
            query  = res.get("search_query", title)
            url    = yt_search_url(query)
            icon   = type_icon.get(rtype, "🔗")
            color  = type_color.get(rtype, "#4a4e6a")
            st.markdown(
                f'<div class="resource-row">'
                f'<span style="font-size:1.1rem;">{icon}</span>'
                f'<div style="flex:1;">'
                f'<span class="resource-title">{title}</span>'
                f'<span class="priority-badge" style="color:{color};background:{color}18;margin-left:8px;">{rtype}</span>'
                f'</div>'
                f'<a href="{url}" target="_blank" class="resource-link">Search on YouTube →</a>'
                f'</div>',
                unsafe_allow_html=True,
            )
