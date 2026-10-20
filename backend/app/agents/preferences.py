"""Descriptive user preferences, learned from follow-up questions.

Replaces the old scheme (a bare expertise level plus a two-in-a-row refine
counter) with a small set of *topic-scoped, descriptive, dated* notes, e.g.:

    "Detail level: prefers short, plain-language answers (noted 2026-10-18)."
    "Focus area: mainly asks about recourse / retention options (noted 2026-10-18)."

Each note has a topic (a short slug). A new note on a topic a user already
has a note for REPLACES the old one — the newer statement wins, which is how
the system handles a user who contradicts an earlier preference ("actually,
give me more detail now"). Notes are rendered to a short text block that is
appended to the generator prompt, so the LLM sees them on every subsequent
explanation and follow-up, in this session and later ones.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any

from ..llm.client import BaseLLM

MAX_NOTES = 8  # keep the rendered block short enough to stay cheap in every prompt

# Topics we recognise and a short description of what belongs in each, used both
# for the rule-based fallback and as the taxonomy offered to the LLM extractor.
TOPICS = {
    "detail_level": "how much detail / how technical the wording should be",
    "focus_area": "which kind of information this person asks about most (e.g. recourse, specific factors)",
    "tone": "tone or framing preferences (e.g. wants reassurance, wants blunt numbers)",
    "format": "formatting preferences (e.g. wants bullet points, wants a short summary first)",
    "recurring_question": "a specific thing this person repeatedly wants addressed",
}


def today() -> str:
    return dt.date.today().isoformat()


# ---------------------------------------------------------------- rule-based fallback (no LLM / LLM unavailable)
_SIMPLER = re.compile(r"\b(too (?:complex|technical|detailed|much)|simpler|simplify|plain(?: language)?|in (?:plain|simple) (?:terms|words)|eli5|dumb it down)\b", re.I)
_DETAILED = re.compile(r"\b(more detail|more technical|in depth|in-depth|go deeper|be more specific|show (?:me )?the numbers|exact numbers|precise(?:ly)?)\b", re.I)
_SHORT = re.compile(r"\b(shorter|too long|briefly|be brief|keep it short|tl;?dr)\b", re.I)
_LONG = re.compile(r"\b(longer|more thorough|don'?t skip|explain fully|elaborate)\b", re.I)
_BULLETS = re.compile(r"\b(bullet points?|as a list|list format|numbered list)\b", re.I)
_RECOURSE = re.compile(r"\b(recourse|retention|what (?:can|could|would) (?:we|i|help)|lower (?:the|their) risk|reduce (?:the|their) risk)\b", re.I)
_REASSURE = re.compile(r"\b(reassur|less alarming|don'?t want it to sound|sounds (?:harsh|scary|blunt))\b", re.I)
_BLUNT = re.compile(r"\b(just the facts|no sugar.?coating|be direct|be blunt|straight(?:forward)? answer)\b", re.I)


def rule_based_note(query: str) -> dict[str, str] | None:
    """A templated, descriptive preference note from keyword cues. Used when no LLM is configured."""
    q = query or ""
    if _SIMPLER.search(q) or _SHORT.search(q):
        kind = "short, plain-language" if _SIMPLER.search(q) else "shorter"
        return {"topic": "detail_level", "text": f"Detail level: prefers {kind} answers."}
    if _DETAILED.search(q) or _LONG.search(q):
        kind = "more technical, number-heavy" if _DETAILED.search(q) else "longer, more thorough"
        return {"topic": "detail_level", "text": f"Detail level: prefers {kind} answers."}
    if _BULLETS.search(q):
        return {"topic": "format", "text": "Format: prefers answers broken into bullet points where possible."}
    if _REASSURE.search(q):
        return {"topic": "tone", "text": "Tone: prefers a reassuring, less alarming framing of risk."}
    if _BLUNT.search(q):
        return {"topic": "tone", "text": "Tone: prefers blunt, direct wording with no softening."}
    if _RECOURSE.search(q):
        return {"topic": "focus_area", "text": "Focus area: mainly interested in recourse and retention options, not just the risk factors."}
    return None


def llm_note(query: str, explanation_text: str, llm: BaseLLM) -> dict[str, str] | None:
    """Ask the LLM to turn one follow-up turn into a single short, descriptive preference note, if any is implied."""
    system = (
        "You read one follow-up question an HR manager asked about an attrition-risk explanation, and decide whether "
        "it reveals a lasting PREFERENCE about how future explanations should be written for this person (not a fact "
        "about the employee). Topics: " + "; ".join(f"{k} ({v})" for k, v in TOPICS.items()) + ". "
        "If the question is just a one-off factual question with no preference signal, return null. "
        'Otherwise return JSON: {"topic": "<one of the topic slugs>", "text": "<one short, descriptive sentence, '
        "starting with the topic name capitalised, e.g. 'Detail level: prefers concise, plain-language answers.'>\"}. "
        'If there is no preference signal, return {"topic": null, "text": null}.'
    )
    user = f"Question: {query!r}\nExplanation it followed: {explanation_text[:600]!r}"
    try:
        out = llm.json("preference_extract", system, user, {"query": query})
    except Exception:
        return None
    topic, text = out.get("topic"), out.get("text")
    if not topic or not text or topic not in TOPICS:
        return None
    return {"topic": topic, "text": str(text)[:200]}


def extract_preference(query: str, explanation_text: str, llm: BaseLLM | None) -> dict[str, str] | None:
    """One preference note (topic + descriptive text) from a follow-up turn, or None if it implies nothing durable."""
    if llm is not None:
        note = llm_note(query, explanation_text, llm)
        if note is not None:
            return note
    return rule_based_note(query)


# ---------------------------------------------------------------- merge + render
def merge_preferences(notes: list[dict[str, Any]], new: dict[str, str], source_trace_id: str = "") -> list[dict[str, Any]]:
    """Add `new`, replacing any existing note on the same topic (the newer statement always wins)."""
    notes = [n for n in notes if n.get("topic") != new["topic"]]
    notes.append({"topic": new["topic"], "text": new["text"], "updated_at": today(), "source_trace_id": source_trace_id})
    # keep the most recently updated MAX_NOTES notes so the rendered block stays small
    notes.sort(key=lambda n: n.get("updated_at", ""), reverse=True)
    return notes[:MAX_NOTES]


def render_preferences(notes: list[dict[str, Any]]) -> str:
    """Formatted bullet block for the generator prompt and for the UI. Empty string if there are none."""
    if not notes:
        return ""
    by_recency = sorted(notes, key=lambda n: n.get("updated_at", ""), reverse=True)
    lines = [f"- {n['text'].rstrip('.')} (noted {n.get('updated_at', 'earlier')})." for n in by_recency]
    return "\n".join(lines)
