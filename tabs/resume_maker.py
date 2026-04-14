"""tabs/resume_maker.py — Make My Resume builder tab."""
import time
import streamlit as st
import streamlit.components.v1 as components

from agents.resume_maker import enhance_bullets, generate_summary as generate_maker_summary, render_resume_html
from agents.resume_builder import build_resume_pdf


def render(target_role: str) -> None:
    _mk_form, _mk_prev = st.columns([11, 9], gap="large")
    d = st.session_state.maker_data

    # ── LEFT: Form ─────────────────────────────────────────────────────────────
    with _mk_form:
        st.markdown(
            '<p style="font-family:Inter,sans-serif;font-weight:700;font-size:1.1rem;'
            'color:#1a1a2e;margin-bottom:4px;">🛠️ Build Your Resume</p>',
            unsafe_allow_html=True,
        )
        if not st.session_state.groq_client:
            st.info("Add your Groq API key in the sidebar to unlock AI bullet & summary generation.", icon="🔑")

        _tab_p, _tab_e, _tab_x, _tab_pr, _tab_sk = st.tabs(
            ["👤 Personal", "🎓 Education", "💼 Experience", "🚀 Projects", "🧠 Skills"]
        )

        # ── Personal ───────────────────────────────────────────────────────────
        with _tab_p:
            _c1, _c2 = st.columns(2)
            with _c1:
                d["name"]     = st.text_input("Full Name",    value=d["name"],     key="mk2_name",  placeholder="Arjun Sharma")
                d["email"]    = st.text_input("Email",         value=d["email"],    key="mk2_email", placeholder="arjun@email.com")
                d["phone"]    = st.text_input("Phone",         value=d["phone"],    key="mk2_phone", placeholder="+91-9876543210")
            with _c2:
                d["location"] = st.text_input("Location",     value=d["location"], key="mk2_loc",   placeholder="Bengaluru, India")
                d["linkedin"] = st.text_input("LinkedIn URL",  value=d["linkedin"], key="mk2_li",    placeholder="linkedin.com/in/arjun")
                d["github"]   = st.text_input("GitHub URL",   value=d["github"],   key="mk2_gh",    placeholder="github.com/arjun")

        # ── Education ──────────────────────────────────────────────────────────
        with _tab_e:
            st.caption("Add as many entries as you need.")
            for _ei, edu in enumerate(d["education"]):
                _eid = edu.get("_id", str(_ei))
                _edu_label = edu.get("degree") or edu.get("institution") or f"Entry {_ei + 1}"
                with st.expander(_edu_label, expanded=(_ei == len(d["education"]) - 1)):
                    _ec1, _ec2 = st.columns(2)
                    with _ec1:
                        edu["degree"]      = st.text_input("Degree / Programme", value=edu.get("degree",""),      key=f"mk2_ed_deg_{_eid}",  placeholder="B.Tech Computer Science")
                        edu["institution"] = st.text_input("Institution",         value=edu.get("institution",""), key=f"mk2_ed_inst_{_eid}", placeholder="VIT University")
                    with _ec2:
                        edu["dates"]    = st.text_input("Dates",          value=edu.get("dates",""),    key=f"mk2_ed_dt_{_eid}",  placeholder="2022 – 2026")
                        edu["gpa"]      = st.text_input("GPA (optional)", value=edu.get("gpa",""),      key=f"mk2_ed_gpa_{_eid}", placeholder="8.5 / 10")
                    edu["location"]     = st.text_input("Location",       value=edu.get("location",""), key=f"mk2_ed_loc_{_eid}", placeholder="Vellore, Tamil Nadu")
                    edu["achievements"] = st.text_area(
                        "Achievements / Coursework (one per line)", value=edu.get("achievements",""),
                        key=f"mk2_ed_ach_{_eid}", placeholder="Ranked 3rd in department", height=75,
                    )
                    if st.button("🗑 Remove", key=f"mk2_ed_rm_{_eid}"):
                        d["education"].pop(_ei); st.rerun()
            if st.button("➕ Add Education", key="mk2_add_edu"):
                d["education"].append({
                    "_id": f"edu_{int(time.time()*1000) % 9999999}",
                    "degree": "", "institution": "", "location": "", "dates": "", "gpa": "", "achievements": "",
                })
                st.rerun()

        # ── Experience ─────────────────────────────────────────────────────────
        with _tab_x:
            st.caption("Describe what you did in 1-2 lines → AI expands into strong bullets.")
            for _xi, exp in enumerate(d["experience"]):
                _xid = exp.get("_id", str(_xi))
                _exp_label = (f"{exp.get('title','')} @ {exp.get('company','')}" if exp.get("title") else f"Entry {_xi + 1}").strip(" @")
                with st.expander(_exp_label, expanded=(_xi == len(d["experience"]) - 1)):
                    _xc1, _xc2 = st.columns(2)
                    with _xc1:
                        exp["title"]   = st.text_input("Job Title", value=exp.get("title",""),   key=f"mk2_xp_t_{_xid}", placeholder="Backend Engineer Intern")
                        exp["company"] = st.text_input("Company",   value=exp.get("company",""), key=f"mk2_xp_c_{_xid}", placeholder="Razorpay")
                    with _xc2:
                        exp["dates"]    = st.text_input("Dates",    value=exp.get("dates",""),    key=f"mk2_xp_d_{_xid}", placeholder="May 2024 – Aug 2024")
                        exp["location"] = st.text_input("Location", value=exp.get("location",""), key=f"mk2_xp_l_{_xid}", placeholder="Remote")
                    exp["_desc"] = st.text_area(
                        "✏️ What did you do? (AI will expand into bullets)", value=exp.get("_desc",""),
                        key=f"mk2_xp_desc_{_xid}",
                        placeholder="e.g. Built REST APIs for order management in Node.js + MongoDB",
                        height=70,
                    )
                    _xbtn_col, _xrm_col = st.columns([3, 1])
                    with _xbtn_col:
                        if st.button("✨ Generate Bullets with AI", key=f"mk2_xp_enh_{_xid}", type="primary"):
                            if not st.session_state.groq_client:
                                st.error("Groq API key required.")
                            elif not exp.get("_desc","").strip():
                                st.warning("Add a brief description first.")
                            elif not exp.get("title","").strip():
                                st.warning("Add a job title first.")
                            else:
                                try:
                                    with st.spinner("Generating bullets…"):
                                        exp["bullets"] = enhance_bullets(
                                            role=exp["title"], company_or_project=exp.get("company",""),
                                            description=exp["_desc"], client=st.session_state.groq_client,
                                            model=st.session_state.groq_model,
                                        )
                                    exp["_desc"] = ""
                                    st.rerun()
                                except Exception as _err:
                                    st.error(str(_err))
                    with _xrm_col:
                        if st.button("🗑 Remove", key=f"mk2_xp_rm_{_xid}"):
                            d["experience"].pop(_xi); st.rerun()
                    if exp.get("bullets"):
                        _btext = st.text_area(
                            "Generated bullets — edit freely (one per line)",
                            value="\n".join(exp["bullets"]), key=f"mk2_xp_bedit_{_xid}", height=100,
                        )
                        exp["bullets"] = [b.strip() for b in _btext.split("\n") if b.strip()]
            if st.button("➕ Add Experience", key="mk2_add_exp"):
                d["experience"].append({
                    "_id": f"exp_{int(time.time()*1000) % 9999999}",
                    "title": "", "company": "", "location": "", "dates": "", "bullets": [], "_desc": "",
                })
                st.rerun()

        # ── Projects ───────────────────────────────────────────────────────────
        with _tab_pr:
            st.caption("Describe what you built in 1-2 lines → AI expands into strong bullets.")
            for _pi, proj in enumerate(d["projects"]):
                _pid = proj.get("_id", str(_pi))
                _proj_label = proj.get("name") or f"Entry {_pi + 1}"
                with st.expander(_proj_label, expanded=(_pi == len(d["projects"]) - 1)):
                    _pc1, _pc2 = st.columns(2)
                    with _pc1:
                        proj["name"] = st.text_input("Project Name", value=proj.get("name",""), key=f"mk2_pr_n_{_pid}", placeholder="PrepSense AI")
                        proj["tech"] = st.text_input("Tech Stack",   value=proj.get("tech",""), key=f"mk2_pr_t_{_pid}", placeholder="Python, Streamlit, Groq")
                    with _pc2:
                        proj["dates"] = st.text_input("Date",        value=proj.get("dates",""), key=f"mk2_pr_d_{_pid}", placeholder="Jan 2025")
                        proj["link"]  = st.text_input("GitHub Link", value=proj.get("link",""),  key=f"mk2_pr_l_{_pid}", placeholder="github.com/you/project")
                    proj["_desc"] = st.text_area(
                        "✏️ What did you build? (AI will expand into bullets)", value=proj.get("_desc",""),
                        key=f"mk2_pr_desc_{_pid}",
                        placeholder="e.g. Real-time chat app using Socket.io + React, 200 concurrent users",
                        height=70,
                    )
                    _pbtn_col, _prm_col = st.columns([3, 1])
                    with _pbtn_col:
                        if st.button("✨ Generate Bullets with AI", key=f"mk2_pr_enh_{_pid}", type="primary"):
                            if not st.session_state.groq_client:
                                st.error("Groq API key required.")
                            elif not proj.get("_desc","").strip():
                                st.warning("Add a brief description first.")
                            elif not proj.get("name","").strip():
                                st.warning("Add a project name first.")
                            else:
                                try:
                                    with st.spinner("Generating bullets…"):
                                        proj["bullets"] = enhance_bullets(
                                            role=proj["name"], company_or_project=proj.get("tech",""),
                                            description=proj["_desc"], client=st.session_state.groq_client,
                                            model=st.session_state.groq_model,
                                        )
                                    proj["_desc"] = ""
                                    st.rerun()
                                except Exception as _err:
                                    st.error(str(_err))
                    with _prm_col:
                        if st.button("🗑 Remove", key=f"mk2_pr_rm_{_pid}"):
                            d["projects"].pop(_pi); st.rerun()
                    if proj.get("bullets"):
                        _pb_text = st.text_area(
                            "Generated bullets — edit freely (one per line)",
                            value="\n".join(proj["bullets"]), key=f"mk2_pr_bedit_{_pid}", height=100,
                        )
                        proj["bullets"] = [b.strip() for b in _pb_text.split("\n") if b.strip()]
            if st.button("➕ Add Project", key="mk2_add_proj"):
                d["projects"].append({
                    "_id": f"proj_{int(time.time()*1000) % 9999999}",
                    "name": "", "tech": "", "dates": "", "link": "", "bullets": [], "_desc": "",
                })
                st.rerun()

        # ── Skills ─────────────────────────────────────────────────────────────
        with _tab_sk:
            if "_mk2_pending_summary" in st.session_state:
                st.session_state["mk2_summary"] = st.session_state.pop("_mk2_pending_summary")
                d["summary"] = st.session_state["mk2_summary"]

            sk = d["skills"]
            _sk1, _sk2 = st.columns(2)
            with _sk1:
                sk["languages"]  = st.text_input("Languages",              value=sk["languages"],  key="mk2_sk_l", placeholder="Python, Java, C++")
                sk["frameworks"] = st.text_input("Frameworks & Libraries",  value=sk["frameworks"], key="mk2_sk_f", placeholder="React, Django, FastAPI")
            with _sk2:
                sk["tools"] = st.text_input("Tools & Platforms", value=sk["tools"],  key="mk2_sk_t", placeholder="Git, Docker, AWS")
                sk["other"] = st.text_input("Other",              value=sk["other"],  key="mk2_sk_o", placeholder="REST APIs, Agile, System Design")

            _ca1, _ca2 = st.columns(2)
            with _ca1:
                d["certifications"] = st.text_area("Certifications (one per line)", value=d["certifications"],
                    key="mk2_certs", placeholder="AWS Certified Cloud Practitioner (2024)", height=90)
            with _ca2:
                d["achievements"] = st.text_area("Achievements (one per line)", value=d["achievements"],
                    key="mk2_ach", placeholder="Winner — Smart India Hackathon 2024", height=90)

            st.divider()
            d["summary"] = st.text_area(
                "Professional Summary", value=d["summary"], key="mk2_summary",
                placeholder="Write manually or generate with AI below…", height=80,
            )
            if st.button("✨ Generate Summary with AI", key="mk2_gen_sum", use_container_width=True):
                if not st.session_state.groq_client:
                    st.error("Groq API key required.")
                else:
                    _exp_titles = [e["title"] for e in d["experience"] if e.get("title")]
                    _all_skills = []
                    for _sk_val in [sk["languages"], sk["frameworks"], sk["tools"]]:
                        _all_skills += [s.strip() for s in _sk_val.split(",") if s.strip()]
                    try:
                        with st.spinner("Writing summary…"):
                            _gen_sum = generate_maker_summary(
                                name=d["name"], target_role=target_role,
                                experience_titles=_exp_titles, skills_flat=_all_skills,
                                client=st.session_state.groq_client, model=st.session_state.groq_model,
                            )
                        d["summary"] = _gen_sum
                        st.session_state["_mk2_pending_summary"] = _gen_sum
                        st.rerun()
                    except Exception as _err:
                        st.error(str(_err))

        st.markdown("<br/>", unsafe_allow_html=True)

        if st.button("🏗️  Export Resume PDF", key="mk2_build", type="primary", use_container_width=True):
            def _csv(s):  return [x.strip() for x in (s or "").split(",") if x.strip()]
            def _lines(s): return [x.strip() for x in (s or "").split("\n") if x.strip()]
            _pdf_data = {
                "name": d["name"], "email": d["email"], "phone": d["phone"],
                "linkedin": d["linkedin"], "github": d["github"], "location": d["location"],
                "summary": d["summary"],
                "education": [
                    {"degree": edu.get("degree",""), "institution": edu.get("institution",""),
                     "location": edu.get("location",""), "dates": edu.get("dates",""),
                     "gpa": edu.get("gpa",""),
                     "bullets": [a.strip() for a in (edu.get("achievements") or "").split("\n") if a.strip()]}
                    for edu in d["education"] if edu.get("degree") or edu.get("institution")
                ],
                "experience": [{k: v for k, v in exp.items() if k != "_desc"} for exp in d["experience"] if exp.get("title")],
                "projects":   [{k: v for k, v in proj.items() if k != "_desc"} for proj in d["projects"] if proj.get("name")],
                "skills": {
                    "languages":  _csv(d["skills"]["languages"]),
                    "frameworks": _csv(d["skills"]["frameworks"]),
                    "tools":      _csv(d["skills"]["tools"]),
                    "other":      _csv(d["skills"]["other"]),
                },
                "certifications": _lines(d["certifications"]),
                "achievements":   _lines(d["achievements"]),
            }
            try:
                with st.spinner("Building PDF…"):
                    st.session_state.maker_pdf_bytes = build_resume_pdf(_pdf_data)
                st.success("PDF ready — download from the preview panel →", icon="✅")
            except Exception as _err:
                st.error(f"PDF build failed: {_err}")

    # ── RIGHT: Live Preview ─────────────────────────────────────────────────────
    with _mk_prev:
        st.markdown(
            '<p style="font-family:Inter,sans-serif;font-weight:700;font-size:1.1rem;'
            'color:#1a1a2e;margin-bottom:6px;">Live Preview</p>',
            unsafe_allow_html=True,
        )
        components.html(render_resume_html(d), height=750, scrolling=True)
        if st.session_state.maker_pdf_bytes:
            _dl_name = (d.get("name", "resume") or "resume").replace(" ", "_")
            st.download_button(
                label="⬇ Download Resume PDF",
                data=st.session_state.maker_pdf_bytes,
                file_name=f"{_dl_name}_resume.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
