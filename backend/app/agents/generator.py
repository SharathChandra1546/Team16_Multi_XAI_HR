"""LLM explanation generator: evidence-grounded, structured output (explanation + claims)."""
from __future__ import annotations

import json

from ..domains.base import DomainAgent
from ..llm.client import BaseLLM
from ..llm.prompts import render_generator
from ..schemas import Claim, EvidencePackage, ExplanationPlan, GeneratedExplanation, Violation


def _ordered(items, plan: ExplanationPlan):
    """Items in the planner's discussion order (if it set one), else in the package's rank order."""
    if not plan.order:
        return list(items)
    pos = {eid: i for i, eid in enumerate(plan.order)}
    return sorted(items, key=lambda it: (pos.get(it.id, len(pos)), getattr(it, "rank", 0)))


def evidence_view(pkg: EvidencePackage, plan: ExplanationPlan) -> dict:
    allowed = set(plan.allowed_ids)
    pr = pkg.prediction
    return {
        "P": {"risk_percent": round(pr.probability * 100, 1), "probability": pr.probability, "band": pr.band,
              "range_percent": [round(pr.interval[0] * 100, 1), round(pr.interval[1] * 100, 1)],
              "review_threshold_percent": round(pr.threshold * 100), "label": pr.label},
        "factors": [
            {"id": it.id, "feature": it.feature, "display_name": it.display_name, "value_label": it.value_label,
             "direction": it.direction, "magnitude": it.magnitude, "rank": it.rank,
             "contribution_percentage_points": round(it.attribution * 100, 1), "glossary_id": it.glossary_id}
            for it in _ordered(pkg.visible, plan) if it.id in allowed
        ],
        "recourse_options": [
            {"id": r.id, "description": r.description, "plain_description": r.short_description,
             "risk_before_percent": round(r.probability_before * 100), "risk_after_percent": round(r.probability_after * 100),
             "probability_after": r.probability_after, "band_after": r.band_after,
             "changes": [c.model_dump() for c in r.changes]}
            for r in pkg.recourse if r.id in allowed
        ],
        "no_recourse_note": pkg.recourse_note,
        "glossary": [g.model_dump(exclude={"source"}) for g in pkg.glossary if g.id in allowed],
        "average_estimate_percent": round(pkg.base_value * 100, 1),
        "withheld_note": next((lim for lim in pkg.limitations if "protected" in lim), None),
    }


def _parse(out: dict) -> GeneratedExplanation:
    claims = []
    for c in out.get("claims") or []:
        if not isinstance(c, dict):
            continue
        try:
            claims.append(Claim(**{k: v for k, v in c.items() if k in Claim.model_fields}))
        except Exception:
            claims.append(Claim(text=str(c.get("text", "")), kind="attribution", evidence_ids=[]))
    return GeneratedExplanation(explanation=str(out.get("explanation") or ""), claims=claims, source="llm")


def generate(
    pkg: EvidencePackage, plan: ExplanationPlan, domain: DomainAgent, llm: BaseLLM, prompt_template: str,
    guidance: list[str], question: str, previous: list[Violation] | None = None, attempt: int = 1,
    repair_instructions: list[str] | None = None, preferences_text: str = "",
) -> GeneratedExplanation:
    system = render_generator(prompt_template, guidance, plan.level, plan.max_words, plan.style,
                              plan.allow_column_names, plan.extra_instructions, preferences_text, domain.language)
    user_payload = {
        "question": question,
        "intent": plan.intent,
        "required_ids": plan.required_ids,
        "evidence": evidence_view(pkg, plan),
    }
    if repair_instructions:
        # structured, self-correcting repair instructions (see app/agents/repair.py) replace a raw violation dump
        user_payload["fix_these_problems_from_previous_attempt"] = repair_instructions
    elif previous:
        user_payload["fix_these_problems_from_previous_attempt"] = [
            {"code": v.code, "message": v.message, "span": v.span} for v in previous
        ]
    user = json.dumps(user_payload, ensure_ascii=False)
    payload = {"package": pkg.model_dump(), "plan": plan.model_dump(), "domain": domain.name, "attempt": attempt}
    try:
        out = llm.json("generate", system, user, payload)
    except Exception:
        # an LLM/API failure counts as a failed attempt; the graph retries and then falls back
        return GeneratedExplanation(explanation="", claims=[], source="llm")
    return _parse(out)
