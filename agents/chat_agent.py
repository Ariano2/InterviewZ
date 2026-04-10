"""
agents/chat_agent.py
Multi-turn RAG chat agent with agentic job search via Groq tool calling.
The model autonomously decides when to call search_jobs based on user intent.
"""

import json
import re
from groq import Groq
from rag.retriever import retrieve_context
from agents.job_search import search_jobs, format_jobs_for_llm

GROQ_MODEL = "openai/gpt-oss-120b"

# Keep last N turns to manage token usage
MAX_HISTORY_TURNS = 6

# ── Tool schema — passed to Groq on every chat call ───────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": (
                "Search for live job openings on LinkedIn, Indeed, and Glassdoor. "
                "Call this ONLY when the user explicitly asks to find, search, or browse "
                "job listings, openings, or positions to apply to."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Job title or role to search, e.g. 'Machine Learning Engineer', 'Data Analyst'",
                    },
                    "location": {
                        "type": "string",
                        "description": "City or country, e.g. 'Bangalore', 'Mumbai', 'India'. Default: India",
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
    }
]


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
# Main chat function
# ─────────────────────────────────────────────────────────────────────────────

def chat_with_resume(
    user_message: str,
    chat_history: list[dict],
    groq_client: Groq,
    target_role: str = "",
    session_summary: str = "",
    rapidapi_key: str = "",
) -> tuple[str, str]:
    """
    RAG-grounded, agentic chat with optional live job search.
    Uses Groq tool calling — the model decides autonomously when to search.

    Returns (assistant_reply, updated_session_summary).
    """
    # 1. Retrieve relevant resume chunks
    context = retrieve_context(user_message, k=6)
    context_block = (
        context if context
        else "No resume indexed yet. Ask the user to upload their resume first."
    )

    # 2. System prompt
    summary_block = (
        f"\n=== PRIOR CONVERSATION SUMMARY ===\n{session_summary}\n=== END SUMMARY ==="
        if session_summary.strip() else ""
    )

    system_prompt = f"""You are a practical, no-nonsense Indian career & interview mentor.
You help people from any field - software, law, medicine, finance, marketing, CA, design, civil services, teaching, core engineering, sales, HR, etc.

You give realistic, up-to-date (2025-2026) advice tailored to the Indian job market: off-campus hiring, referrals, Naukri/LinkedIn hacks, tier-1 vs tier-2/3 companies, metro vs tier-2 cities, startup vs MNC vs govt/psu, etc.
{summary_block}
=== USER RESUME / CAREER CONTEXT (ground truth - always reference concrete facts from here) ===
{context_block}
=== END CONTEXT ===

Target / Aspired Role (if provided): {target_role if target_role.strip() else "not specified - infer from context or ask"}

TOOL USE:
- You have access to a search_jobs tool that fetches live listings from LinkedIn, Indeed, and Glassdoor.
- Call it when the user asks to find/search/browse job openings or internships.
- Do NOT call it for general advice, resume feedback, salary questions, or interview prep.

FORMATTING RULES - follow exactly:
- Plain hyphen - for all ranges and separators. No em-dashes or unicode punctuation.
- Bold **only short key phrases** (under 4 words). Never bold full sentences.
- Bullet lists: use plain "- " prefix only.
- Max 2 blank lines between sections.
- Tables: simple pipe tables with header separator. Left-align all cells.
- Headings: use ## or ### only when response has 3+ distinct sections.
- Salary: realistic 2025-2026 India numbers. Service MNCs freshers: Rs 3.5-6 LPA. Product/GCC strong profiles: Rs 8-15 LPA. Never inflate.

CONTENT RULES:
- Never repeat or rephrase the user's question. Start directly with value.
- Always tie advice to resume context when possible.
- Keep answers 200-500 words unless a deep dive is asked.
- End with 1-2 natural follow-up questions.

NEVER use filler openers like "Great question!", "Sure!", "Absolutely!". Just deliver value."""

    # 3. Build messages
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for turn in chat_history[-(MAX_HISTORY_TURNS * 2):]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    # 4. First LLM call — with tools if key is available, plain otherwise
    if rapidapi_key:
        raw_reply = _call_groq_with_tools(messages, groq_client, rapidapi_key)
    else:
        raw_reply = _call_groq(messages, groq_client)

    reply = _clean_markdown(raw_reply)

    # 5. Compress session summary if history is long
    updated_summary = _maybe_compress_summary(
        session_summary, chat_history, reply, groq_client
    )

    return reply, updated_summary


# ─────────────────────────────────────────────────────────────────────────────
# Tool-calling flow
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq_with_tools(messages: list[dict], client: Groq, rapidapi_key: str) -> str:
    """
    First pass: LLM decides whether to call search_jobs.
    If it does: execute the tool, inject results, do a second pass.
    If not: return the first response directly.
    """
    resp1 = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.65,
        max_tokens=2200,
        top_p=0.92,
    )

    choice = resp1.choices[0]
    assistant_msg = choice.message

    # No tool call — return the reply as-is
    if not assistant_msg.tool_calls:
        return (assistant_msg.content or "").strip()

    # ── Tool was called ────────────────────────────────────────────────────
    tc = assistant_msg.tool_calls[0]
    args = json.loads(tc.function.arguments)

    try:
        jobs = search_jobs(
            query=args.get("query", ""),
            location=args.get("location", "India"),
            employment_type=args.get("employment_type", "FULLTIME"),
            num_results=5,
            rapidapi_key=rapidapi_key,
        )
        tool_result = format_jobs_for_llm(jobs, args.get("query", ""), args.get("location", "India"))
    except Exception as e:
        tool_result = f"Job search failed ({e}). Tell the user and suggest Naukri / LinkedIn directly."

    # Append assistant tool-call message + tool result
    messages.append({
        "role": "assistant",
        "content": assistant_msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
        ],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": tool_result,
    })

    # Second pass — synthesise final reply with job data
    resp2 = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.65,
        max_tokens=2200,
        top_p=0.92,
    )
    return (resp2.choices[0].message.content or "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq(messages: list[dict], client: Groq) -> str:
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.65,
            max_tokens=2200,
            top_p=0.92,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error connecting to Groq: {e}. Check your API key and internet connection."


def _maybe_compress_summary(
    existing_summary: str,
    chat_history: list[dict],
    latest_reply: str,
    client: Groq,
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
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return existing_summary
