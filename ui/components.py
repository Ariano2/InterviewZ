"""
ui/components.py
Shared UI helpers — call from tab files to avoid repeating HTML patterns.
All functions call st.markdown directly; none return raw strings.
"""
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Guard helpers
# ─────────────────────────────────────────────────────────────────────────────

def require_resume(msg: str = "Upload a resume above to use this feature.") -> bool:
    """Shows info and returns False when no resume is loaded."""
    if not st.session_state.resume_text:
        st.info(msg, icon="👆")
        return False
    return True


def require_groq() -> bool:
    """Shows error and returns False when no Groq API key is set."""
    if not st.session_state.groq_client:
        st.error("Please provide a Groq API key first.")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Layout components
# ─────────────────────────────────────────────────────────────────────────────

def tab_intro(title: str, body: str) -> None:
    """Blue gradient intro banner used at the top of each feature tab."""
    st.markdown(
        f'<div class="tab-intro">'
        f'<div class="section-label">{title}</div>'
        f'<p>{body}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_heading(text: str, margin_bottom: str = "0.6rem") -> None:
    """Section-label with configurable bottom margin."""
    st.markdown(
        f'<div class="section-label" style="margin-bottom:{margin_bottom};">{text}</div>',
        unsafe_allow_html=True,
    )


def chip_list(keywords: list, color: str = "green") -> None:
    """Renders a row of .chip spans inside a .chips flex wrapper."""
    if not keywords:
        return
    chips = "".join(f'<span class="chip chip-{color}">{k}</span>' for k in keywords)
    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)


def alert_green(html_body: str) -> None:
    st.markdown(f'<div class="alert-green">{html_body}</div>', unsafe_allow_html=True)


def alert_blue(html_body: str) -> None:
    st.markdown(f'<div class="alert-blue">{html_body}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Score utilities
# ─────────────────────────────────────────────────────────────────────────────

def score_color(score: int) -> tuple[str, str, str]:
    """Returns (hex_color, bg_color, band_label) based on score 0-100."""
    if score >= 70:
        return "#27ae60", "#e8f8ef", "Strong match"
    if score >= 45:
        return "#e67e22", "#fef3e8", "Moderate match"
    return "#e74c3c", "#fef0f0", "Weak match"


def score_bar(value: int, color: str) -> None:
    """Thin horizontal progress bar using CSS classes."""
    st.markdown(
        f'<div class="score-bar-track">'
        f'<div class="score-bar-fill" style="width:{max(2,value)}%;background:{color};"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bullet diff cards
# ─────────────────────────────────────────────────────────────────────────────

def bullet_diff(original: str, improved: str, why: str = "") -> None:
    """Before/after bullet card using st.container(border=True)."""
    with st.container(border=True):
        st.markdown("**BEFORE**")
        st.write(original)
        st.markdown("**AFTER**")
        st.write(improved)
        if why:
            st.caption(why)


def bullet_removal(original: str, why: str = "") -> None:
    """Strikethrough card for removed bullets."""
    with st.container(border=True):
        st.write(f"~~{original}~~")
        if why:
            st.caption(f"Removed: {why}")


# ─────────────────────────────────────────────────────────────────────────────
# Job card
# ─────────────────────────────────────────────────────────────────────────────

def job_card(job: dict) -> None:
    """Full job listing card with match-score badge and progress bar."""
    score = job.get("match_score", 0)
    if score >= 65:
        badge_bg, badge_color, bar_color = "#e8f5e9", "#1a7a45", "#27ae60"
    elif score >= 45:
        badge_bg, badge_color, bar_color = "#fff8e1", "#a05800", "#f39c12"
    else:
        badge_bg, badge_color, bar_color = "#fdecea", "#922b21", "#e74c3c"

    snippet = (job.get("snippet") or "")[:200]
    ellipsis = "…" if len(job.get("snippet") or "") > 200 else ""

    st.markdown(
        f'<div class="job-card">'
        f'  <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;">'
        f'    <div style="flex:1;min-width:0;">'
        f'      <p class="job-card-title">{job.get("title","N/A")}</p>'
        f'      <p class="job-card-meta">'
        f'        {job.get("company","N/A")} &nbsp;·&nbsp; {job.get("location","N/A")}'
        f'        &nbsp;·&nbsp; {job.get("type","N/A")}'
        f'        &nbsp;·&nbsp; <span class="text-muted">{job.get("source","")}</span>'
        f'        &nbsp;·&nbsp; <span class="text-muted">Posted {job.get("posted","")}</span>'
        f'      </p>'
        f'      <p class="job-card-snippet">{snippet}{ellipsis}</p>'
        f'    </div>'
        f'    <div class="match-badge" style="background:{badge_bg};">'
        f'      <p class="match-score" style="color:{badge_color};">{score}%</p>'
        f'      <p class="match-label" style="color:{badge_color};">match</p>'
        f'    </div>'
        f'  </div>'
        f'  <div class="score-bar-track" style="margin-top:10px;">'
        f'    <div class="score-bar-fill" style="background:{bar_color};width:{max(4,score)}%;"></div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )
