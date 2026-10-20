"""Reviewer agent: a second LLM, ideally from a different model family than the generator, that reads the
explanation and the evidence and says whether it is faithful and follows the rules.

It is ADVISORY: its verdict is recorded and returned, but it cannot change the outcome. The deterministic verifier
is the gate. The reviewer exists for two reasons: (1) to compare an LLM checker against the deterministic checker
(Phase 3: verifier vs LLM reviewer vs human labels), and (2) as the "LLM critic loop" baseline (Phase 2), where it
is the only checker.

The reviewer's prompt is written from the domain's language and the evidence only; it never sees the verifier's
rules or patterns, so its judgement is independent of them.
"""
from __future__ import annotations

import json
import re

from ..config import settings
from ..domains.base import DomainAgent, DomainLanguage
from ..llm.client import BaseLLM, get_llm
from ..schemas import EvidencePackage, ExplanationPlan, GeneratedExplanation

# provider or model-name fragment -> model family
_FAMILIES = [
    (r"gemini|gemma|google", "google"),
    (r"gpt|o1|o3|o4|openai", "openai"),
    (r"llama|meta", "meta"),
    (r"claude|anthropic", "anthropic"),
    (r"mistral|mixtral", "mistral"),
    (r"qwen", "qwen"),
    (r"deepseek", "deepseek"),
    (r"mock|deterministic", "mock"),
]


def family(llm_id: str) -> str:
    """Model family of an llm id like 'groq:llama-3.3-70b-versatile' (judged by the model name first)."""
    provider, _, model = llm_id.partition(":")
    for text in (model, provider):
        for pat, fam in _FAMILIES:
            if re.search(pat, text or "", re.I):
                return fam
    return provider or "unknown"


def get_reviewer() -> BaseLLM:
    provider = settings.reviewer_provider or settings.judge_provider
    model = settings.reviewer_model if settings.reviewer_provider else settings.judge_model
    return get_llm(provider, model)


def review_prompt(language: DomainLanguage) -> str:
    return (
        f"You check whether an explanation of {language.model_description}'s estimate, written for {language.reader}, "
        "is faithful to the evidence it was given and appropriate. Problems include: a factor, value, direction, "
        "ranking or number that does not match the evidence; describing a factor as a cause; stating the outcome as "
        f"certain; recommending {language.decision_kind}; mentioning or guessing {language.protected_description}; "
        f"judging {language.subject_possessive} character; showing raw codes or column names. "
        'Return JSON {"faithful": true|false, "issues": ["short description", ...]}.'
    )


def review(llm: BaseLLM, pkg: EvidencePackage, plan: ExplanationPlan, gen: GeneratedExplanation,
           domain: DomainAgent, generator_id: str = "") -> dict:
    from .generator import evidence_view

    user = json.dumps({"evidence": evidence_view(pkg, plan), "explanation": gen.explanation}, ensure_ascii=False)
    fam, gen_fam = family(llm.id), family(generator_id) if generator_id else ""
    out = {"model": llm.id, "family": fam, "same_family_as_generator": bool(gen_fam) and fam == gen_fam}
    try:
        res = llm.json("review", review_prompt(domain.language), user, {"text": gen.explanation})
    except Exception as e:
        return out | {"faithful": None, "issues": [], "error": type(e).__name__}
    issues = [str(i)[:200] for i in (res.get("issues") or [])][:10]
    faithful = res.get("faithful")
    return out | {"faithful": faithful if isinstance(faithful, bool) else None, "issues": issues, "error": None}
