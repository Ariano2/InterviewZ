"""tabs/job_match.py — Semantic Job Matching tab."""
import streamlit as st

from agents.job_search import search_jobs
from agents.job_matcher import match_jobs_to_resume
from ui.components import job_card


def render(target_role: str, rapidapi_key: str = "") -> None:
    if not st.session_state.resume_text:
        st.info("Upload a resume above to enable semantic job matching.", icon="👆")
        return
    if not rapidapi_key:
        st.warning(
            "Add `RAPIDAPI_KEY` to your `.env` file to enable live job search. "
            "Get a free key at rapidapi.com → subscribe to JSearch.",
            icon="🔑",
        )
        return

    st.markdown(
        '<div class="tab-intro">'
        '<div class="section-label">🔍 Semantic Job Matching</div>'
        '<p>Fetches live jobs from LinkedIn · Indeed · Glassdoor and ranks them by '
        '<strong>semantic similarity</strong> to your resume using BGE-small embeddings.<br>'
        '<span style="font-size:0.85rem;">Match % = cosine similarity between your resume vector and each job description vector.</span>'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Search form
    _jm_col1, _jm_col2, _jm_col3 = st.columns([3, 2, 1])
    with _jm_col1:
        _jm_query = st.text_input(
            "Role / Keywords",
            value=st.session_state.job_match_query or target_role,
            placeholder="e.g. Machine Learning Engineer, Backend SDE",
            key="jm_query_input",
        )
    with _jm_col2:
        _jm_location = st.selectbox(
            "Location",
            ["India", "Bangalore", "Mumbai", "Delhi NCR", "Hyderabad",
             "Pune", "Chennai", "Noida", "Gurgaon", "Remote"],
            key="jm_location",
        )
    with _jm_col3:
        _jm_type = st.selectbox("Type", ["FULLTIME", "INTERN", "PARTTIME", "CONTRACTOR"], key="jm_emp_type")

    _jm_n = st.slider("Number of jobs to fetch", 5, 10, 7, key="jm_n_jobs")

    if st.button("🔍  Find & Match Jobs", type="primary", use_container_width=True, key="jm_search_btn"):
        if not _jm_query.strip():
            st.warning("Enter a role or keywords to search.")
        else:
            st.session_state.job_match_query = _jm_query.strip()
            with st.spinner(f"Fetching jobs for **{_jm_query}** in {_jm_location}…"):
                try:
                    _raw_jobs = search_jobs(
                        query=_jm_query.strip(), location=_jm_location,
                        employment_type=_jm_type, num_results=_jm_n,
                        rapidapi_key=rapidapi_key,
                    )
                except Exception as _e:
                    _err_str = str(_e)
                    if "timed out" in _err_str.lower() or "timeout" in _err_str.lower():
                        st.error("Request timed out — RapidAPI was slow. Try again in a few seconds.", icon="⏱️")
                    else:
                        st.error(f"Job search failed: {_e}")
                    _raw_jobs = []

            if _raw_jobs:
                with st.spinner("Computing semantic match scores…"):
                    try:
                        _matched = match_jobs_to_resume(
                            st.session_state.resume_text, _raw_jobs,
                            resume_embedding=st.session_state.resume_embedding,
                        )
                        st.session_state.job_match_results = _matched
                    except Exception as _e:
                        st.error(f"Matching failed: {_e}")
                        st.session_state.job_match_results = []
            else:
                st.session_state.job_match_results = []
                st.warning("No jobs found. Try broader keywords or a different location.")

    # Results
    _results = st.session_state.job_match_results
    if not _results:
        return

    st.markdown(
        f'<p class="meta-text" style="margin:0.5rem 0 1rem;">'
        f'Showing {len(_results)} jobs · sorted by semantic match to your resume</p>',
        unsafe_allow_html=True,
    )

    for _job in _results:
        job_card(_job)

        _btn1, _btn2, _btn3 = st.columns([2, 2, 3])
        with _btn1:
            if st.button("📊 Analyze ATS", key=f"jm_ats_{_job.get('apply_link','')[:40]}", use_container_width=True):
                st.session_state.jd_text = _job.get("description") or _job.get("snippet") or ""
                st.toast("JD loaded — open the ATS Score tab", icon="📊")
        with _btn2:
            if st.button("✍️ Tailor Resume", key=f"jm_tailor_{_job.get('apply_link','')[:40]}", use_container_width=True):
                st.session_state.jd_text = _job.get("description") or _job.get("snippet") or ""
                st.toast("JD loaded — open the JD Tailor tab", icon="✍️")
        with _btn3:
            _apply = _job.get("apply_link", "")
            if _apply:
                st.link_button("🔗 Apply Now", url=_apply, use_container_width=True)
