"""Detector-independent fault injection: an LLM rewrites a correct explanation so it contains one fault.

Independence from the detector is the point. The fault generator receives:
- the fault's plain-language description from app.faults.TAXONOMY (written without reference to any rule),
- the correct explanation and its claims,
- the evidence the explanation was written from,
- the reader level.

It never receives the verifier's or critic's patterns, violation codes, or rule lists, and it is asked to make the
fault read naturally. Detection rates on these rewrites therefore measure the checker on errors it was not built
around, unlike the template faults (app.faults.inject), whose wording overlaps the patterns.

Use a different model family from the generator when possible (FAULT_PROVIDER / FAULT_MODEL), so the faults are
not just the generator's own habits.

Each returned case records the changed sentence the LLM reports, so a sample can be checked by hand that the
intended fault is really present (Phase 3 reports that manual check).
"""
from __future__ import annotations

import json
from typing import Any

from .agents.generator import evidence_view
from .config import settings
from .domains.base import DomainAgent
from .faults import TAXONOMY
from .llm.client import BaseLLM, get_llm
from .schemas import Claim, EvidencePackage, ExplanationPlan, GeneratedExplanation


def get_fault_llm() -> BaseLLM:
    provider = settings.fault_provider or settings.judge_provider
    model = settings.fault_model if settings.fault_provider else settings.judge_model
    return get_llm(provider, model)


def fault_prompt(domain: DomainAgent, fault: str) -> str:
    lang = domain.language
    return (
        "You help test an explanation-checking system by creating realistic mistakes. You are given a correct "
        f"explanation of {lang.model_description}'s estimate, written for {lang.reader}, together with the evidence "
        "it was written from and its list of claims (each claim is one sentence of the explanation plus the evidence "
        "ids it relies on).\n"
        f"Rewrite the explanation so it contains exactly ONE mistake of this kind: {TAXONOMY[fault]['description']}\n"
        "Make the mistake read naturally, as a careless writer might make it; do not mark or announce it. Change as "
        "little as possible. Keep every other sentence identical. Update the claims list so it matches your new text "
        "(change the text of the claim you altered; add a claim if you added a statement about a factor).\n"
        'Return JSON {"explanation": "...", "claims": [...same shape as given...], '
        '"changed_sentence": "the sentence containing the mistake"}.'
    )


def llm_inject(fault: str, gen: GeneratedExplanation, pkg: EvidencePackage, plan: ExplanationPlan,
               domain: DomainAgent, llm: BaseLLM) -> tuple[GeneratedExplanation | None, dict[str, Any]]:
    """Returns (faulty explanation or None, record). None when the LLM failed or returned the text unchanged."""
    user = json.dumps({"level": plan.level, "evidence": evidence_view(pkg, plan), "explanation": gen.explanation,
                       "claims": [c.model_dump() for c in gen.claims]}, ensure_ascii=False)
    payload = {"fault": fault, "package": pkg.model_dump(), "plan": plan.model_dump(), "domain": domain.name,
               "explanation": gen.model_dump()}
    record: dict[str, Any] = {"fault": fault, "generator": llm.id, "changed_sentence": "", "error": None}
    try:
        out = llm.json("fault", fault_prompt(domain, fault), user, payload)
    except Exception as e:
        record["error"] = type(e).__name__
        return None, record
    text = str(out.get("explanation") or "")
    record["changed_sentence"] = str(out.get("changed_sentence") or out.get("changed") or "")[:400]
    claims = []
    for c in out.get("claims") or []:
        if isinstance(c, dict):
            try:
                claims.append(Claim(**{k: v for k, v in c.items() if k in Claim.model_fields}))
            except Exception:
                claims.append(Claim(text=str(c.get("text", "")), kind="attribution", evidence_ids=[]))
    same_text = " ".join(text.split()) == " ".join(gen.explanation.split())
    same_claims = [c.model_dump() for c in claims] == [c.model_dump() for c in gen.claims]
    if not text.strip() or (same_text and same_claims):  # some faults (a dropped citation) change only the claims
        record["error"] = "unchanged"
        return None, record
    return GeneratedExplanation(explanation=text, claims=claims, source="llm"), record
