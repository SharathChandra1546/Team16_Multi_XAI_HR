"""LLM critic agent: scores clarity, level fit and relevance (1-5). It never judges facts.

Facts belong to the deterministic verifier. This agent's low score can block an output (quality_low) only while
the `llm_critic` switch is on, and it can never approve an output the verifier or the policy rules rejected,
because it runs after them.
"""
from __future__ import annotations

import json

from ..domains.base import DomainLanguage
from ..llm.client import BaseLLM

DIMENSIONS = ("clarity", "level_fit", "relevance")
MIN_SCORE = 3


def critic_prompt(language: DomainLanguage) -> str:
    return (
        f"You review explanations written for {language.reader} about {language.model_description}. "
        "Score each from 1 to 5: clarity, level_fit (suits the stated reader level), relevance (answers the question). "
        "Do not judge facts. "
        'Return JSON {"clarity": n, "level_fit": n, "relevance": n, "feedback": "one sentence"}.'
    )


def llm_quality(llm: BaseLLM, level: str, question: str, text: str,
                language: DomainLanguage | None = None) -> tuple[dict[str, float], str, str | None]:
    """Returns (scores, feedback, error). Scores are empty when the LLM failed (the failure never blocks)."""
    user = json.dumps({"level": level, "question": question, "explanation": text})
    try:
        out = llm.json("critic", critic_prompt(language or DomainLanguage()), user, {"level": level, "text": text})
        scores = {k: float(out.get(k, 0)) for k in DIMENSIONS}
        return scores, str(out.get("feedback", ""))[:300], None
    except Exception as e:  # critic LLM failure must not block a verified explanation
        return {}, f"LLM critic unavailable: {type(e).__name__}", type(e).__name__
