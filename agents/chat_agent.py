"""
agents/chat_agent.py
Multi-turn RAG chat agent with agentic tool calling via Groq.

Available tools the model can autonomously call:
  search_jobs          — live job search (LinkedIn / Indeed / Glassdoor)
  run_ats_analysis     — full ATS score + keyword gap on the user's resume
  rewrite_resume_bullets — rewrite weak bullets with action verbs + metrics
  tailor_resume_to_jd  — inject missing JD keywords into resume bullets

The model decides when to call each tool based on user intent.
Results are formatted as text and fed back to the model for a
natural-language summary — the chat remains conversational throughout.
"""

import json
import re
from groq import Groq
from rag.retriever import retrieve_with_scores
from agents.job_search import search_jobs, format_jobs_for_llm

GROQ_MODEL = "openai/gpt-oss-120b"
MAX_HISTORY_TURNS = 6


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas — passed to Groq on every chat call
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": (
                "Search for live job openings on LinkedIn, Indeed, and Glassdoor. "
                "Call ONLY when the user explicitly asks to find, search, or browse "
                "job listings, openings, or positions to apply to."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Job title or role to search, e.g. 'Machine Learning Engineer'",
                    },
                    "location": {
                        "type": "string",
                        "description": "City or country, e.g. 'Bangalore', 'India'",
                    },
                    "employment_type": {
                        "type": "string",
                        "enum": ["FULLTIME", "PARTTIME", "INTERN", "CONTRACTOR"],
                        "description": "Employment type. Default: FULLTIME",
                    },
                },
                "required": ["query", "location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_ats_analysis",
            "description": (
                "Runs a full ATS analysis on the user's resume — returns an overall score (0-100), "
                "matched and missing keywords, sub-scores, and specific feedback. "
                "Call when user asks: what's my ATS score, how does my resume perform, "
                "keyword analysis, what am I missing, resume rating, resume feedback."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": (
                            "Job description to score against. Extract verbatim from the conversation "
                            "if the user pasted one. Leave as empty string for generic scoring."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rewrite_resume_bullets",
            "description": (
                "Rewrites weak resume bullets with strong action verbs, quantified impact, and "
                "ATS-friendly language. Call when user asks to improve their resume, "
                "rewrite bullets, fix weak points, or make their resume stronger."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tailor_resume_to_jd",
            "description": (
                "Tailors the resume to a specific job description by identifying missing keywords "
                "and injecting them naturally into existing bullets. Call when user provides a JD "
                "and asks to tailor, customize, or optimize their resume for that specific role."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": "The full job description text. Must come from the user's message.",
                    },
                },
                "required": ["jd_text"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool result formatters — convert structured output → LLM-readable text
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_ats(result: dict) -> str:
    score    = result.get("ats_score", 0)
    matched  = result.get("matched_keywords") or []
    missing  = result.get("missing_keywords") or []
    strong   = result.get("strong_areas") or []
    weak     = result.get("weak_areas") or []
    summary  = result.get("summary", "")
    kw_pct   = result.get("keyword_score", 0)
    av_score = result.get("action_verb_score", 0)
    q_score  = result.get("quantification_score", 0)
    s_score  = result.get("section_score", 0)
    f_score  = result.get("formatting_score", 0)

    lines = [
        "=== ATS ANALYSIS COMPLETE ===",
        f"Overall Score: {score}/100",
        "",
        "Sub-scores:",
        f"  Keyword Match      : {kw_pct}%   (50% weight)",
        f"  Action Verbs       : {av_score}/100 (20% weight)",
        f"  Quantification     : {q_score}%   (15% weight)",
        f"  Section Complete   : {s_score}%   (10% weight)",
        f"  Formatting         : {f_score}/100 (5% weight)",
        "",
        f"Matched keywords ({len(matched)}): {', '.join(matched[:12])}",
        f"Missing keywords ({len(missing)}): {', '.join(missing[:12])}",
        "",
        f"Strong areas: {' | '.join(strong[:3])}",
        f"Weak areas  : {' | '.join(weak[:3])}",
        "",
        f"Summary: {summary}",
        "=== END ATS ANALYSIS ===",
        "",
        "INSTRUCTION: Tell the user their score, the 2-3 most impactful missing keywords, "
        "and the single highest-leverage action to improve the score. Be direct and specific. "
        "Mention they can see the full breakdown in the ATS Score tab.",
    ]
    return "\n".join(lines)


def _fmt_bullets(pairs: list) -> str:
    rewrites = [p for p in pairs if p.get("action") == "rewrite"]
    removals = [p for p in pairs if p.get("action") == "remove"]

    lines = [
        "=== BULLET REWRITE COMPLETE ===",
        f"{len(rewrites)} bullets rewritten | {len(removals)} bullets removed",
        "",
    ]
    for i, p in enumerate(rewrites[:5], 1):
        lines += [
            f"Change {i}:",
            f"  BEFORE : {(p.get('original') or '')[:120]}",
            f"  AFTER  : {(p.get('improved') or '')[:140]}",
            f"  WHY    : {(p.get('why') or '')[:120]}",
            "",
        ]
    if len(rewrites) > 5:
        lines.append(f"...and {len(rewrites) - 5} more rewrites not shown here.")

    lines += [
        "=== END BULLET REWRITES ===",
        "",
        "INSTRUCTION: Summarise the key improvement patterns (e.g. 'weak verbs → strong action verbs', "
        "'added metrics'). Show the 2 best before/after examples. Tell the user to visit the "
        "Bullet Rewriter tab to download the full rebuilt PDF resume.",
    ]
    return "\n".join(lines)


def _fmt_jobs_with_scores(jobs: list, query: str, location: str) -> str:
    if not jobs:
        return f"[Job search for '{query}' in {location} returned no results. Advise the user to broaden their search or check back later.]"

    lines = [
        f"=== LIVE JOB LISTINGS WITH RESUME MATCH — '{query}' in {location} ===",
    ]
    for i, j in enumerate(jobs, 1):
        score = j.get("match_score")
        score_str = f"  | Resume match: {score}%" if score is not None else ""
        lines.append(
            f"\n{i}. [{j.get('match_score', '?')}% match] {j['title']} — {j['company']}\n"
            f"   Location: {j['location']} | Type: {j['type']} | Posted: {j['posted']}\n"
            f"   Apply: {j['apply_link']}\n"
            f"   About: {j['snippet']}..."
        )
    lines += [
        "\n=== END JOB LISTINGS ===",
        "Present these as a clean table (Match % | Title | Company | Location | Apply Link), "
        "sorted by match score. Highlight the top 2 matches and briefly explain WHY they match "
        "well based on the resume context. Suggest the user visit the Job Match tab for the full "
        "interactive view with ATS and tailoring shortcuts.",
    ]
    return "\n".join(lines)


def _fmt_tailor(result: dict) -> str:
    rewrites = result.get("rewrites") or []
    keywords = result.get("added_keywords") or []

    lines = [
        "=== JD TAILORING COMPLETE ===",
        f"Keywords injected: {', '.join(keywords)}",
        f"Bullets changed  : {len(rewrites)}",
        "",
    ]
    for i, rw in enumerate(rewrites[:4], 1):
        lines += [
            f"Change {i} — added '{rw.get('keyword_added', '')}':",
            f"  BEFORE : {(rw.get('original') or '')[:120]}",
            f"  AFTER  : {(rw.get('improved') or '')[:140]}",
            "",
        ]
    lines += [
        "=== END TAILORING ===",
        "",
        "INSTRUCTION: Tell the user which keywords were injected and how many bullets changed. "
        "Give 1-2 specific examples. Remind them to go to the JD Tailor tab to download the "
        "tailored PDF resume with all changes applied.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool executor — runs the real Python function, returns formatted string
# ─────────────────────────────────────────────────────────────────────────────

def _execute_tool(
    name: str,
    args: dict,
    resume_text: str,
    target_role: str,
    client: Groq,
    rapidapi_key: str,
    model: str,
    resume_embedding=None,
) -> str:
    try:
        if name == "search_jobs":
            jobs = search_jobs(
                query=args.get("query", ""),
                location=args.get("location", "India"),
                employment_type=args.get("employment_type", "FULLTIME"),
                num_results=5,
                rapidapi_key=rapidapi_key,
            )
            # Semantically rank jobs against the resume if resume text is available
            if resume_text.strip() and jobs:
                from agents.job_matcher import match_jobs_to_resume
                jobs = match_jobs_to_resume(resume_text, jobs, resume_embedding=resume_embedding)
            return _fmt_jobs_with_scores(jobs, args.get("query", ""), args.get("location", "India"))

        if name == "run_ats_analysis":
            from agents.ats_analyzer import analyze_ats
            jd_text = args.get("jd_text") or ""
            result = analyze_ats(
                resume_text=resume_text,
                target_role=target_role,
                client=client,
                jd_text=jd_text,
                model=model,
                resume_embedding=resume_embedding,
            )
            return _fmt_ats(result)

        if name == "rewrite_resume_bullets":
            from agents.bullet_rewriter import rewrite_bullets
            pairs = rewrite_bullets(
                resume_text=resume_text,
                target_role=target_role,
                client=client,
                model=model,
            )
            return _fmt_bullets(pairs)

        if name == "tailor_resume_to_jd":
            from agents.jd_tailor import tailor_resume
            jd_text = args.get("jd_text", "")
            if not jd_text.strip():
                return "[Tailoring failed: no job description was provided. Ask the user to paste the JD.]"
            result = tailor_resume(
                resume_text=resume_text,
                jd_text=jd_text,
                target_role=target_role,
                client=client,
                model=model,
            )
            return _fmt_tailor(result)

        return f"[Unknown tool: {name}]"

    except Exception as e:
        return f"[Tool '{name}' failed: {e}. Let the user know and suggest they use the dedicated tab instead.]"


# ─────────────────────────────────────────────────────────────────────────────
# Markdown post-processor
# ─────────────────────────────────────────────────────────────────────────────

def _clean_markdown(text: str) -> str:
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2022", "-")

    def _strip_long_bold(m: re.Match) -> str:
        inner = m.group(1)
        return inner if len(inner) > 40 else f"**{inner}**"
    text = re.sub(r"\*\*(.+?)\*\*", _strip_long_bold, text, flags=re.DOTALL)

    def _strip_long_italic(m: re.Match) -> str:
        inner = m.group(1)
        return inner if len(inner) > 60 else f"*{inner}*"
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", _strip_long_italic, text, flags=re.DOTALL)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^[\u2022\u25b8\u25e6\u2726\u27a4]\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"(?<!\n)\n(#{1,4} )", r"\n\n\1", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Core LLM calls
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq(messages: list, client: Groq, model: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.65,
            max_tokens=2200,
            top_p=0.92,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Error connecting to Groq: {e}. Check your API key and internet connection."


def _call_groq_with_tools(
    messages: list,
    client: Groq,
    resume_text: str,
    target_role: str,
    rapidapi_key: str,
    model: str,
    resume_embedding=None,
) -> str:
    """
    First pass: LLM decides whether to call a tool.
    If yes: execute the tool, inject result, do a second pass.
    If no: return the first response directly.
    Handles one tool call per turn (most common case).
    """
    # Only include job search tool if rapidapi key exists
    tools = TOOLS if rapidapi_key else [t for t in TOOLS if t["function"]["name"] != "search_jobs"]

    resp1 = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.65,
        max_tokens=2200,
        top_p=0.92,
    )

    choice = resp1.choices[0]
    assistant_msg = choice.message

    # No tool call — return reply directly
    if not assistant_msg.tool_calls:
        return (assistant_msg.content or "").strip()

    # ── Tool was called ────────────────────────────────────────────────────────
    tc   = assistant_msg.tool_calls[0]
    name = tc.function.name
    args = json.loads(tc.function.arguments or "{}")

    tool_result = _execute_tool(
        name=name,
        args=args,
        resume_text=resume_text,
        target_role=target_role,
        client=client,
        rapidapi_key=rapidapi_key,
        model=model,
        resume_embedding=resume_embedding,
    )

    # Append assistant tool-call message + tool result
    messages.append({
        "role": "assistant",
        "content": assistant_msg.content or "",
        "tool_calls": [{
            "id": tc.id,
            "type": "function",
            "function": {"name": name, "arguments": tc.function.arguments},
        }],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": tool_result,
    })

    # Second pass — LLM synthesises a natural-language response from tool output
    resp2 = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.65,
        max_tokens=2200,
        top_p=0.92,
    )
    return (resp2.choices[0].message.content or "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Session summary compression
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_compress_summary(
    existing_summary: str,
    chat_history: list,
    latest_reply: str,
    client: Groq,
    model: str,
) -> str:
    if len(chat_history) < MAX_HISTORY_TURNS * 2:
        return existing_summary

    turns_to_compress = chat_history[:-(MAX_HISTORY_TURNS * 2)]
    if not turns_to_compress:
        return existing_summary

    turns_text = "\n".join(
        f"{t['role'].upper()}: {t['content']}" for t in turns_to_compress
    )
    prompt = (
        "Summarise the key facts, decisions, and action items from these conversation turns "
        "in 120 words or less. Be specific: names, skills, companies, goals mentioned.\n\n"
        f"EXISTING SUMMARY: {existing_summary}\n\n"
        f"TURNS TO COMPRESS:\n{turns_text}\n\n"
        f"LATEST REPLY: {latest_reply}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return existing_summary


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def chat_with_resume(
    user_message: str,
    chat_history: list[dict],
    groq_client: Groq,
    resume_text: str = "",
    target_role: str = "",
    session_summary: str = "",
    rapidapi_key: str = "",
    corpus: list[dict] | None = None,
    access_token: str = "",
    model: str = GROQ_MODEL,
    resume_embedding=None,
) -> tuple[str, str, list]:
    """
    RAG-grounded agentic chat.

    The model can autonomously call:
      - search_jobs          (needs rapidapi_key)
      - run_ats_analysis     (needs resume_text)
      - rewrite_resume_bullets (needs resume_text)
      - tailor_resume_to_jd  (needs resume_text + JD from user)

    Returns (assistant_reply, updated_session_summary, retrieved_chunks).
    retrieved_chunks: list[dict] with text, chunk_index, dense/bm25/hybrid scores.
    """
    # 1. RAG retrieval — hybrid BM25 + dense (Supabase pgvector)
    retrieved_chunks = retrieve_with_scores(
        user_message, k=4, corpus=corpus, access_token=access_token
    )
    context = "\n\n---\n\n".join(c["text"] for c in retrieved_chunks) if retrieved_chunks else ""
    context_block = (
        context if context
        else "No resume indexed yet. Ask the user to upload their resume first."
    )

    # 2. System prompt
    summary_block = (
        f"\n=== PRIOR CONVERSATION SUMMARY ===\n{session_summary}\n=== END SUMMARY ==="
        if session_summary.strip() else ""
    )

    has_tools = bool(resume_text.strip())
    tool_note = (
        "\nAGENTIC TOOLS — you can call these when appropriate:\n"
        "- run_ats_analysis: when user asks for ATS score, keyword gaps, resume rating\n"
        "- rewrite_resume_bullets: when user asks to improve or rewrite their bullets\n"
        "- tailor_resume_to_jd: when user provides a JD and wants resume tailored\n"
        "- search_jobs: when user wants to find/browse live job listings\n"
        "Only call a tool when the user clearly wants that action. For advice or questions, just answer."
        if has_tools else ""
    )

    system_prompt = f"""You are a practical, no-nonsense Indian career & interview mentor.
You help people from any field - software, law, medicine, finance, marketing, CA, design, civil services, teaching, core engineering, sales, HR, etc.

You give realistic, up-to-date (2025-2026) advice tailored to the Indian job market: off-campus hiring, referrals, Naukri/LinkedIn hacks, tier-1 vs tier-2/3 companies, metro vs tier-2 cities, startup vs MNC vs govt/psu, etc.
{summary_block}
=== USER RESUME / CAREER CONTEXT (ground truth - always reference concrete facts from here) ===
{context_block}
=== END CONTEXT ===

Target / Aspired Role: {target_role if target_role.strip() else "not specified - infer from context or ask"}
{tool_note}

FORMATTING RULES:
- Plain hyphen - for all ranges and separators. No em-dashes or unicode punctuation.
- Bold **only short key phrases** (under 4 words). Never bold full sentences.
- Bullet lists: use plain "- " prefix only.
- Max 2 blank lines between sections.
- Tables: simple pipe tables with header separator.
- Headings: use ## or ### only when response has 3+ distinct sections.
- Salary: realistic 2025-2026 India numbers. Never inflate.

CONTENT RULES:
- Never repeat or rephrase the user's question. Start directly with value.
- ONLY state resume facts explicitly written in the context above. Do NOT invent metrics or tools.
- For general career advice, draw on general knowledge but separate it from resume facts.
- Keep answers 200-500 words unless a deep dive is asked.
- End with 1-2 natural follow-up questions.

NEVER use filler openers like "Great question!", "Sure!", "Absolutely!"."""

    # 3. Build messages
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for turn in chat_history[-(MAX_HISTORY_TURNS * 2):]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    # 4. LLM call — with tools if resume is loaded
    if has_tools:
        raw_reply = _call_groq_with_tools(
            messages, groq_client, resume_text, target_role, rapidapi_key, model=model,
            resume_embedding=resume_embedding,
        )
    else:
        raw_reply = _call_groq(messages, groq_client, model=model)

    reply = _clean_markdown(raw_reply)

    # 5. Compress session summary if history is long
    updated_summary = _maybe_compress_summary(
        session_summary, chat_history, reply, groq_client, model=model
    )

    return reply, updated_summary, retrieved_chunks
