"""tabs/jd_tailor.py — JD Tailor + Cover Letter tab."""
import streamlit as st

from agents.jd_tailor import tailor_resume, generate_cover_letter
from agents.resume_structurer import structure_resume
from agents.resume_builder import build_resume_pdf
from ui.components import require_resume, section_heading, chip_list, bullet_diff


def render(target_role: str) -> None:
    if not require_resume("Upload a resume above to tailor it to a job description."):
        return

    st.markdown(
        '<div class="tab-intro">'
        '<div class="section-label">JD Tailor + Cover Letter Generator</div>'
        '<p>Paste a job description — the AI injects missing keywords into your existing bullets '
        'with minimum edits, then generates a tailored cover letter and a downloadable PDF.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

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
            with st.spinner("🎯  Injecting missing JD keywords…"):
                tailor_result = tailor_resume(
                    st.session_state.resume_text, st.session_state.jd_text,
                    target_role, st.session_state.groq_client,
                    model=st.session_state.groq_model,
                )
                st.session_state.jd_tailor_result = tailor_result

            with st.spinner("✉️  Writing cover letter…"):
                cl = generate_cover_letter(
                    st.session_state.resume_text, st.session_state.jd_text,
                    target_role, st.session_state.groq_client,
                    model=st.session_state.groq_model,
                )
                st.session_state.cover_letter = cl

            with st.spinner("📄  Building tailored PDF resume…"):
                try:
                    structure = st.session_state.resume_structure or structure_resume(
                        st.session_state.resume_text,
                        st.session_state.groq_client,
                        model=st.session_state.groq_model,
                    )
                    st.session_state.resume_structure = structure
                    st.session_state.tailored_pdf_bytes = build_resume_pdf(
                        structure, tailor_result.get("rewrites") or [],
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
                    st.session_state.tailored_pdf_bytes = None

    tr = st.session_state.jd_tailor_result
    cl = st.session_state.cover_letter
    if not tr and not cl:
        return

    st.write("")
    col_dl1, col_dl2 = st.columns(2)
    if st.session_state.tailored_pdf_bytes:
        with col_dl1:
            st.download_button(
                label="⬇  Download Tailored Resume (.pdf)",
                data=st.session_state.tailored_pdf_bytes,
                file_name="tailored_resume.pdf",
                mime="application/pdf",
                width='stretch',
            )
    if cl and not cl.startswith("Error"):
        with col_dl2:
            st.download_button(
                label="⬇  Download Cover Letter (.txt)",
                data=cl.encode("utf-8"),
                file_name="cover_letter.txt",
                mime="text/plain",
                width='stretch',
            )

    st.write("")

    if cl:
        section_heading("✉️ Cover Letter", margin_bottom="0.4rem")
        with st.container(border=True):
            st.write(cl)

    st.write("")

    if tr:
        rewrites = tr.get("rewrites") or []
        added    = tr.get("added_keywords") or []

        section_heading(f"🔑 Keywords Injected ({len(added)})", margin_bottom="0.4rem")
        chip_list(added, "green")

        st.write("")

        if rewrites:
            section_heading(f"✍️ Bullet Changes ({len(rewrites)})", margin_bottom="0.4rem")
            for rw in rewrites:
                original = (rw.get("original") or "").strip()
                improved = (rw.get("improved") or "").strip()
                kw       = (rw.get("keyword_added") or "").strip()
                if not original:
                    continue
                bullet_diff(original, improved, f"Keyword added: {kw}" if kw else "")
