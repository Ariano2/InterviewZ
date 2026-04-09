"""
agents/chat_agent.py
Multi-turn RAG chat agent for Indian careers & resume types.
Uses Groq (openai/gpt-oss-120b)
"""

import re
from groq import Groq
from rag.retriever import retrieve_context

GROQ_MODEL = "openai/gpt-oss-120b"

# Keep last N turns to manage token usage
MAX_HISTORY_TURNS = 6


# ─────────────────────────────────────────────────────────────────────────────────
# Markdown post-processor
# ─────────────────────────────────────────────────────────────────────────────────

def _clean_markdown(text: str) -> str:
    """
    Normalise LLM markdown output so it renders cleanly in Streamlit's
    st.markdown / HTML bubble display. All fixes are pure string transforms -
    no external deps, no regex that touches content meaning.
    """

    # 1. Unicode punctuation -> plain ASCII
    text = text.replace("\u2014", "-").replace("\u2013", "-")   # em/en dash
    text = text.replace("\u2018", "'").replace("\u2019", "'")   # curly single quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')   # curly double quotes
    text = text.replace("\u00a0", " ")                          # non-breaking space
    text = text.replace("\u2022", "-")                          # bullet dot -> hyphen

    # 2. Strip bold (**...**) when the bolded span is longer than 40 chars
    #    (catches LLMs bolding entire sentences - keep only short key-phrase bolding)
    def _strip_long_bold(m: re.Match) -> str:
        inner = m.group(1)
        return inner if len(inner) > 40 else f"**{inner}**"
    text = re.sub(r"\*\*(.+?)\*\*", _strip_long_bold, text, flags=re.DOTALL)

    # 3. Remove italic wrapping around long spans (threshold 60 chars)
    def _strip_long_italic(m: re.Match) -> str:
        inner = m.group(1)
        return inner if len(inner) > 60 else f"*{inner}*"
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", _strip_long_italic, text, flags=re.DOTALL)

    # 4. Collapse 3+ consecutive blank lines -> max 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Normalise exotic bullet characters -> plain hyphen-space
    text = re.sub(r"^[\u2022\u25b8\u25e6\u2726\u27a4]\s+", "- ", text, flags=re.MULTILINE)

    # 6. Ensure a blank line before every markdown heading so Streamlit renders it correctly
    text = re.sub(r"(?<!\n)\n(#{1,4} )", r"\n\n\1", text)

    # 7. Remove trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # 8. Trim leading/trailing blank lines
    text = text.strip()

    return text


# ─────────────────────────────────────────────────────────────────────────────────
# Main chat function
# ─────────────────────────────────────────────────────────────────────────────────

def chat_with_resume(
    user_message: str,
    chat_history: list[dict],
    groq_client: Groq,
    target_role: str = "",
    session_summary: str = "",
) -> tuple[str, str]:
    """
    RAG-grounded chat response for any career / resume type (Indian job market focus).

    chat_history: list of {"role": "user"|"assistant", "content": str}
    Returns (assistant_reply, updated_session_summary).
    """
    # 1. Retrieve relevant resume chunks
    context = retrieve_context(user_message, k=6)

    context_block = (
        context if context
        else "No resume or career document indexed yet. Ask the user to upload their resume/CV/portfolio first."
    )

    # 2. Inject prior summary into system prompt if it exists
    summary_block = (
        f"\n=== PRIOR CONVERSATION SUMMARY ===\n{session_summary}\n=== END SUMMARY ==="
        if session_summary.strip() else ""
    )

    # 3. System prompt
    system_prompt = f"""You are a practical, no-nonsense Indian career & interview mentor.
You help people from any field - software, law, medicine, finance, marketing, CA, design, civil services, teaching, core engineering, sales, HR, etc.

You give realistic, up-to-date (2025-2026) advice tailored to the Indian job market: off-campus hiring, referrals, Naukri/LinkedIn hacks, tier-1 vs tier-2/3 companies, metro vs tier-2 cities, startup vs MNC vs govt/psu, etc.
{summary_block}
=== USER RESUME / CAREER CONTEXT (ground truth - always reference concrete facts from here) ===
{context_block}
=== END CONTEXT ===

Target / Aspired Role (if provided): {target_role if target_role.strip() else "not specified - infer from context or ask"}

FORMATTING RULES - follow exactly, no exceptions:
- Plain hyphen - for all ranges and separators. No em-dashes, en-dashes, or unicode punctuation ever.
- Bold **only short key phrases** (under 4 words). Never bold full sentences or entire list items.
- Bullet lists: use plain "- " prefix only. No dots, arrows, stars, or unicode symbols.
- Max 2 blank lines between sections. No excessive whitespace.
- Tables: simple pipe tables with header separator row. Left-align all cells. No extra padding.
- Headings: use ## or ### only when response has 3+ distinct sections. Otherwise use bold labels inline.
- Salary: realistic 2025-2026 India numbers. Service MNCs freshers: Rs 3.5-6 LPA. Product/GCC strong profiles: Rs 8-15 LPA. Exceptional top-tier: Rs 15-25 LPA. Never inflate for average profiles.

CONTENT RULES:
- Never repeat or rephrase the user's question. Start directly with value.
- Always tie advice to resume context when possible (specific projects, companies, degrees, scores).
- If info is missing from context, say so briefly then give best realistic general advice.
- Keep answers 200-500 words unless a deep dive is explicitly asked.
- End with 1-2 natural follow-up questions to keep momentum.

NEVER use filler openers like "Great question!", "Sure!", "Absolutely!". Just deliver value."""

    # 4. Build messages
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    for turn in chat_history[-(MAX_HISTORY_TURNS * 2):]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": user_message})

    # 5. Call Groq then clean the markdown before returning
    raw_reply = _call_groq(messages, groq_client)
    reply = _clean_markdown(raw_reply)

    # 6. Update session summary
    updated_summary = _maybe_compress_summary(
        session_summary, chat_history, reply, groq_client
    )

    return reply, updated_summary


# ─────────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────────

def _maybe_compress_summary(
    existing_summary: str,
    chat_history: list[dict],
    latest_reply: str,
    client: Groq,
) -> str:
    """
    Compress oldest turns into a short summary once history grows beyond
    MAX_HISTORY_TURNS, so context is not lost when turns scroll out of window.
    """
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
        "in 120 words or less. Be specific: names, skills, companies, goals mentioned. "
        "This will be injected as memory into future turns.\n\n"
        f"EXISTING SUMMARY (if any): {existing_summary}\n\n"
        f"NEW TURNS TO COMPRESS:\n{turns_text}\n\n"
        f"LATEST ASSISTANT REPLY: {latest_reply}"
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
        return f"Error connecting to Groq: {str(e)}. Please check your API key and internet connection."