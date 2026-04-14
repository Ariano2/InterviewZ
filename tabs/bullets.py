"""tabs/bullets.py — Bullet Rewriter + PDF Resume Builder tab."""
import streamlit as st

from agents.bullet_rewriter import rewrite_bullets
from agents.resume_structurer import structure_resume
from agents.resume_builder import build_resume_pdf
from ui.components import require_resume, section_heading, chip_list, bullet_diff, bullet_removal


def render(target_role: str) -> None:
    if not require_resume("Upload a resume above to rewrite bullet points."):
        return

    st.markdown(
        '<div class="tab-intro">'
        '<div class="section-label">AI Bullet Rewriter + PDF Resume Generator</div>'
        '<p>Rewrites your weakest bullets with strong action verbs &amp; metrics, '
        'then rebuilds your entire resume as a clean, ATS-friendly PDF — ready to download.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ATS keyword context banner
    _ats_missing = (
        (st.session_state.ats_result or {}).get("missing_keywords") or []
        if st.session_state.ats_result and st.session_state.ats_result.get("used_jd")
        else []
    )
    if _ats_missing:
        _kw_chips = "".join(f'<span class="chip chip-green">{k}</span>' for k in _ats_missing[:15])
        st.markdown(
            f'<div class="alert-green">'
            f'<strong>ATS mode active</strong> — rewriter will target these {len(_ats_missing)} '
            f'missing keywords from your last ATS scan:'
            f'<div class="chips" style="margin-top:6px;">{_kw_chips}</div>'
            f'<span class="meta-text">Run ATS Score again after downloading the rewritten PDF to see your improved score.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if st.button("▶  Rewrite Bullets & Build PDF Resume", key="run_bullets"):
        if not st.session_state.groq_client:
            st.error("Please provide a Groq API key first.")
        else:
            with st.spinner("✍️  Rewriting weak bullets…"):
                pairs = rewrite_bullets(
                    st.session_state.resume_text, target_role,
                    st.session_state.groq_client,
                    model=st.session_state.groq_model,
                    jd_keywords=_ats_missing or None,
                )
                st.session_state.bullets_result = pairs

            with st.spinner("🗂️  Parsing resume structure…"):
                structure = structure_resume(
                    st.session_state.resume_text,
                    st.session_state.groq_client,
                    model=st.session_state.groq_model,
                )
                st.session_state.resume_structure = structure

            with st.spinner("📄  Building your PDF resume…"):
                try:
                    st.session_state.resume_pdf_bytes = build_resume_pdf(structure, pairs)
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
                    st.session_state.resume_pdf_bytes = None

    if st.session_state.resume_pdf_bytes:
        st.success("✅ PDF resume built successfully!", icon="📄")
        st.download_button(
            label="⬇  Download Rewritten Resume (.pdf)",
            data=st.session_state.resume_pdf_bytes,
            file_name="rewritten_resume.pdf",
            mime="application/pdf",
            width='stretch',
        )
        st.write("")

    pairs = st.session_state.bullets_result
    if not pairs:
        return

    rewrites = [p for p in pairs if p.get("action") != "remove"]
    removals = [p for p in pairs if p.get("action") == "remove"]

    summary_parts = []
    if rewrites: summary_parts.append(f"{len(rewrites)} rewritten")
    if removals: summary_parts.append(f"{len(removals)} removed")
    section_heading(" · ".join(summary_parts))

    for p in rewrites:
        original = (p.get("original") or "").strip()
        improved = (p.get("improved") or "").strip()
        why      = (p.get("why")      or "").strip()
        if original:
            bullet_diff(original, improved, why)

    if removals:
        st.markdown("**Removed — redundant / zero-value**")
    for p in removals:
        original = (p.get("original") or "").strip()
        why      = (p.get("why")      or "").strip()
        if original:
            bullet_removal(original, why)
