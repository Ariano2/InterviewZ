"""tabs/raw_text.py — Raw parsed resume text tab."""
import streamlit as st
from ui.components import require_resume


def render() -> None:
    if not require_resume("No resume loaded yet. Upload one above."):
        return

    text = st.session_state.resume_text
    st.markdown(
        f'<div class="section-label">Parsed Text · {len(text.split())} words · {len(text)} chars</div>',
        unsafe_allow_html=True,
    )
    preview = text[:6000] + ("\n…[truncated — first 6,000 chars shown]" if len(text) > 6000 else "")
    st.markdown(
        f'<div class="card">'
        f'<pre style="font-family:\'JetBrains Mono\',monospace;font-size:0.85rem;color:#4a4e6a;'
        f'white-space:pre-wrap;line-height:1.75;max-height:520px;overflow-y:auto;margin:0;">{preview}</pre>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.download_button(
        label="⬇ Download parsed text (.txt)",
        data=text,
        file_name="parsed_resume.txt",
        mime="text/plain",
    )
