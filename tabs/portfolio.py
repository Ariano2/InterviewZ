"""tabs/portfolio.py — Portfolio Generator + GitHub Pages publisher tab."""
import streamlit as st

from agents.portfolio_generator import generate_portfolio
from agents.github_publisher import (
    request_device_code, poll_for_token, get_github_username, publish_portfolio,
)
from agents.resume_structurer import structure_resume
from ui.components import require_resume, section_heading


def render(target_role: str, github_client_id: str = "", github_client_secret: str = "") -> None:
    if not require_resume("Upload a resume above to generate your portfolio."):
        return

    st.markdown(
        '<div class="tab-intro">'
        '<div class="section-label">🌐 One-Click Portfolio Generator</div>'
        '<p>Generates a fully animated personal website from your resume — then publishes it '
        'to GitHub Pages automatically. Live URL in under 60 seconds.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Template picker ───────────────────────────────────────────────────────
    section_heading("① Choose a Template", margin_bottom="0.5rem")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        luminary_selected = st.button(
            "☀️  Luminary — Clean & Minimal", width='stretch', key="tmpl_luminary",
            type="primary" if st.session_state.get("_tmpl") == "luminary" else "secondary",
        )
        st.caption("White background · Gradient accents · Dot-grid hero · Card layout")
    with col_t2:
        noir_selected = st.button(
            "🌑  Noir — Dark Pro", width='stretch', key="tmpl_noir",
            type="primary" if st.session_state.get("_tmpl") == "noir" else "secondary",
        )
        st.caption("Dark bg · Cyan neon glow · Animated orbs · Terminal about card")

    if luminary_selected:
        st.session_state["_tmpl"] = "luminary"; st.rerun()
    if noir_selected:
        st.session_state["_tmpl"] = "noir"; st.rerun()

    chosen_template = st.session_state.get("_tmpl", "luminary")
    st.markdown(
        f'<p class="label-text text-primary">Selected: '
        f'<strong>{"Luminary" if chosen_template == "luminary" else "Noir"}</strong></p>',
        unsafe_allow_html=True,
    )

    # ── Repo name ─────────────────────────────────────────────────────────────
    st.write("")
    section_heading("② Repo & Site Name", margin_bottom="0.5rem")
    repo_name = st.text_input(
        "GitHub repo name", value="portfolio",
        placeholder="e.g. portfolio  →  username.github.io/portfolio",
        label_visibility="collapsed",
    )
    if repo_name == f"{st.session_state.github_username}.github.io":
        st.warning("This will overwrite your main GitHub Pages site. Rename unless that's intentional.", icon="⚠️")

    # ── GitHub OAuth ──────────────────────────────────────────────────────────
    st.write("")
    section_heading("③ Connect GitHub", margin_bottom="0.5rem")

    if st.session_state.github_token:
        st.success(f"✓ Connected as **{st.session_state.github_username}**", icon="✅")
        if st.button("Disconnect", key="gh_disconnect"):
            for _k in ("github_token", "github_username", "github_device_code", "github_user_code"):
                st.session_state[_k] = ""
            st.rerun()

    elif st.session_state.github_user_code:
        st.markdown(
            f'<div class="gh-auth-card">'
            f'<p><strong>Step 1</strong> — Open '
            f'<a href="https://github.com/login/device" target="_blank" style="color:#27ae60;">'
            f'github.com/login/device</a> in a new tab</p>'
            f'<p><strong>Step 2</strong> — Enter this code:</p>'
            f'<p class="code">{st.session_state.github_user_code}</p>'
            f'<p class="hint">Then click the button below once you\'ve approved.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("✅  I've approved — complete connection", key="gh_poll"):
            with st.spinner("🔐 Checking GitHub authorisation…"):
                _token, _err = poll_for_token(
                    github_client_id, github_client_secret,
                    st.session_state.github_device_code, interval=2, timeout=60,
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
        if github_client_id:
            if st.button("🐙  Connect GitHub", key="gh_connect"):
                with st.spinner("Requesting device code from GitHub…"):
                    _dc_data, _dc_err = request_device_code(github_client_id)
                if _dc_err:
                    st.error(f"GitHub error: {_dc_err}")
                else:
                    st.session_state.github_device_code = _dc_data["device_code"]
                    st.session_state.github_user_code    = _dc_data["user_code"]
                    st.rerun()
            st.caption("Grants repo-only scope. We create one repo on your behalf — nothing else.")
        else:
            st.warning("Add `GITHUB_CLIENT_ID` and `GITHUB_CLIENT_SECRET` to your `.env` to enable GitHub publishing.", icon="⚙️")

    # ── Generate + Publish ────────────────────────────────────────────────────
    st.write("")
    section_heading("④ Generate &amp; Publish", margin_bottom="0.5rem")
    can_publish = bool(st.session_state.github_token and st.session_state.groq_client)

    if st.button("🚀  Generate Portfolio & Publish to GitHub Pages", key="run_portfolio",
                 disabled=not can_publish, width='stretch'):
        with st.spinner("🗂️  Parsing resume structure…"):
            structure = st.session_state.resume_structure or structure_resume(
                st.session_state.resume_text, st.session_state.groq_client,
                model=st.session_state.groq_model,
            )
            st.session_state.resume_structure = structure

        with st.spinner("✨  Generating portfolio content…"):
            files, dummy_sections = generate_portfolio(
                structure, target_role, chosen_template,
                st.session_state.groq_client, model=st.session_state.groq_model,
            )
            st.session_state.portfolio_files          = files
            st.session_state.portfolio_dummy_sections = dummy_sections

        with st.spinner("🐙  Publishing to GitHub Pages…"):
            pages_url, pub_err = publish_portfolio(
                token=st.session_state.github_token,
                files=files,
                repo_name=repo_name.strip() or "portfolio",
                description=f"Portfolio of {(st.session_state.resume_structure or {}).get('name','') or 'Developer'} — built with PrepSense AI",
            )
            st.session_state.portfolio_pages_url = pages_url
            if pub_err and not pages_url:
                st.error(f"Publishing failed: {pub_err}")

    if not can_publish:
        if st.session_state.groq_client:
            st.caption("Connect GitHub above to enable publishing.")
        else:
            st.caption("Provide Groq API key and connect GitHub to publish.")

    # ── Results ───────────────────────────────────────────────────────────────
    pages_url      = st.session_state.portfolio_pages_url
    dummy_sections = st.session_state.portfolio_dummy_sections
    files          = st.session_state.portfolio_files

    if pages_url:
        st.write("")
        st.success("🎉 Portfolio published!", icon="🌐")
        st.markdown(
            f'<a href="{pages_url}" target="_blank" class="live-btn">🔗  Visit Live Site →</a>',
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
        section_heading("⬇ Download Files", margin_bottom="0.4rem")
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        with dl_col1:
            st.download_button("⬇ index.html", data=files["index.html"], file_name="index.html", mime="text/html", width='stretch')
        with dl_col2:
            st.download_button("⬇ style.css",  data=files["style.css"],  file_name="style.css",  mime="text/css",  width='stretch')
        with dl_col3:
            st.download_button("⬇ script.js",  data=files["script.js"],  file_name="script.js",  mime="text/javascript", width='stretch')
