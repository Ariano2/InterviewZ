"""tabs/chat.py — Agentic Resume Chat tab."""
import streamlit as st

from agents.chat_agent import chat_with_resume


def render(target_role: str, rapidapi_key: str = "") -> None:
    _job_search_active = bool(rapidapi_key)
    _job_badge = (
        '🔍 <strong>Live job search active</strong> (LinkedIn · Indeed · Glassdoor) — '
        'try <em>"Find ML engineer jobs in Bangalore"</em>'
        if _job_search_active
        else 'Add <code>RAPIDAPI_KEY</code> to .env to enable live job search'
    )
    st.markdown(
        f'<div class="tab-intro">'
        f'<div class="section-label">Agentic Resume Chat</div>'
        f'<p>Every answer is grounded in your resume via RAG. The chat can also '
        f'<strong>run tools on your behalf</strong> — just ask naturally.<br>'
        f'<strong>"What\'s my ATS score?"</strong> · <strong>"Rewrite my bullets"</strong> · '
        f'<strong>"Tailor my resume for [paste JD]"</strong><br>'
        f'<span style="font-size:0.85rem;">{_job_badge}</span></p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Chat history display
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="bubble-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown(f"🤖 {msg['content']}")
                chunks = msg.get("chunks") or []
                if chunks:
                    avg_hybrid = round(
                        sum(c.get("hybrid_score", c.get("cosine_sim", 0.0)) for c in chunks) / len(chunks), 3
                    )
                    avg_color = "#27ae60" if avg_hybrid >= 0.55 else "#e67e22" if avg_hybrid >= 0.35 else "#e74c3c"
                    with st.expander(
                        f"🔍 Retrieved context — {len(chunks)} chunks · avg hybrid score {avg_hybrid:.3f}",
                        expanded=False,
                    ):
                        st.caption("Resume chunks fed to the LLM. Hybrid = BM25 lexical + BGE dense, fused by weighted score normalisation.")
                        for chunk in chunks:
                            hybrid = chunk.get("hybrid_score", chunk.get("cosine_sim", 0.0))
                            dense  = chunk.get("dense_score",  chunk.get("cosine_sim", 0.0))
                            bm25   = chunk.get("bm25_score",   0.0)
                            cidx   = chunk["chunk_index"]
                            text   = chunk["text"]
                            bar_color = "#27ae60" if hybrid >= 0.55 else "#e67e22" if hybrid >= 0.35 else "#e74c3c"
                            st.markdown(
                                f'<div class="chunk-card">'
                                f'<div class="chunk-scores">'
                                f'<span class="meta-text" style="font-weight:700;">Chunk #{cidx}</span>'
                                f'<span style="font-size:0.78rem;font-weight:800;color:{bar_color};">hybrid {hybrid:.3f}</span>'
                                f'<span class="text-primary" style="font-size:0.72rem;">dense {dense:.3f}</span>'
                                f'<span style="font-size:0.72rem;color:#e67e22;">bm25 {bm25:.3f}</span>'
                                f'<div style="flex:1;background:#e0e4f0;border-radius:4px;height:5px;">'
                                f'<div style="background:{bar_color};width:{int(hybrid*100)}%;height:5px;border-radius:4px;"></div>'
                                f'</div></div>'
                                f'<p class="chunk-text">{text[:300]}{"…" if len(text)>300 else ""}</p>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

    # Chat input
    if user_msg := st.chat_input("Ask anything about your resume or interview prep…"):
        if not st.session_state.groq_client:
            st.error("Please provide a Groq API key first.")
        elif not st.session_state.resume_indexed:
            st.warning("Upload & index a resume first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            with st.spinner("Thinking… (may run a tool if needed)"):
                reply, updated_summary, chunks = chat_with_resume(
                    user_message=user_msg,
                    chat_history=st.session_state.chat_history[:-1],
                    groq_client=st.session_state.groq_client,
                    resume_text=st.session_state.resume_text or "",
                    target_role=target_role,
                    session_summary=st.session_state.session_summary,
                    rapidapi_key=rapidapi_key,
                    corpus=st.session_state.vectorstore,
                    access_token=st.session_state.supabase_access_token,
                    model=st.session_state.groq_model,
                    resume_embedding=st.session_state.resume_embedding,
                )
            st.session_state.last_retrieved = [
                {
                    "chunk_index":  c.get("chunk_index", -1),
                    "dense_score":  c.get("dense_score",  0.0),
                    "bm25_score":   c.get("bm25_score",   0.0),
                    "hybrid_score": c.get("hybrid_score", 0.0),
                }
                for c in chunks
            ]
            st.session_state.session_summary = updated_summary
            st.session_state.chat_history.append({"role": "assistant", "content": reply, "chunks": chunks})
            st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑 Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.session_summary = ""
            st.rerun()

    # PCA embedding visualisation
    pca_coords   = st.session_state.pca_coords
    pca_variance = st.session_state.pca_variance
    if not pca_coords:
        return

    st.write("")
    with st.expander(
        f"📊 Embedding Space — PCA of {len(pca_coords)} resume chunks"
        f"  (PC1 {pca_variance[0]*100:.1f}% · PC2 {pca_variance[1]*100:.1f}% variance)",
        expanded=False,
    ):
        import plotly.graph_objects as go
        _retrieved_indices = set()
        for _msg in reversed(st.session_state.chat_history):
            if _msg["role"] == "assistant" and _msg.get("chunks"):
                _retrieved_indices = {c["chunk_index"] for c in _msg["chunks"]}
                break

        _base      = [c for c in pca_coords if c["chunk_index"] not in _retrieved_indices]
        _highlight = [c for c in pca_coords if c["chunk_index"] in _retrieved_indices]

        fig = go.Figure()
        if _base:
            fig.add_trace(go.Scatter(
                x=[c["x"] for c in _base], y=[c["y"] for c in _base],
                mode="markers+text",
                marker=dict(size=14, color="#c0c8e8", line=dict(color="#8890a4", width=1.5)),
                text=[f"#{c['chunk_index']}" for c in _base],
                textposition="top center", textfont=dict(size=10, color="#8890a4"),
                hovertext=[f"Chunk #{c['chunk_index']}<br>{c['text']}" for c in _base],
                hoverinfo="text", name="Resume chunks",
            ))
        if _highlight:
            fig.add_trace(go.Scatter(
                x=[c["x"] for c in _highlight], y=[c["y"] for c in _highlight],
                mode="markers+text",
                marker=dict(size=18, color="#e67e22", symbol="star", line=dict(color="#c0392b", width=2)),
                text=[f"#{c['chunk_index']}" for c in _highlight],
                textposition="top center", textfont=dict(size=11, color="#c0392b", family="Inter, sans-serif"),
                hovertext=[f"★ Retrieved — Chunk #{c['chunk_index']}<br>{c['text']}" for c in _highlight],
                hoverinfo="text", name="Retrieved for last query",
            ))
        fig.update_layout(
            height=400, margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="#f8f9fc", plot_bgcolor="#f8f9fc",
            xaxis=dict(title=f"PC1 ({pca_variance[0]*100:.1f}%)", showgrid=True, gridcolor="#e0e4f0", zeroline=False),
            yaxis=dict(title=f"PC2 ({pca_variance[1]*100:.1f}%)", showgrid=True, gridcolor="#e0e4f0", zeroline=False),
            legend=dict(orientation="h", y=-0.15, font=dict(size=11)), showlegend=True,
        )
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "Each point is a resume chunk projected to 2D via PCA on BGE-small embeddings. "
            "Nearby points are semantically similar. ★ = chunks retrieved for the last query."
        )
