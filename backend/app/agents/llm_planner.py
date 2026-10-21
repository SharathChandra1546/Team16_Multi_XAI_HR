"""LLM planner (audience agent): adapts the deterministic plan to the question and the reader, within limits.

It may only:
- choose which of the plan's optional factors to focus on, and in what order (a subset of allowed_ids);
- add up to MAX_NOTES short style notes for the generator.

It may never:
- add an evidence id the deterministic plan did not allow (withheld, out-of-level or unknown ids are rejected);
- drop a required id (the risk estimate and the required factor/option are always kept);
- add notes that would loosen a rule: a note is rejected if it mentions ignoring or overriding instructions,
  causes, decisions, certainty or blame (generic list below), or if it matches any of the domain's own policy
  rules (decision wording, protected attributes, blame, prediction-as-fact).

`constrain()` enforces this deterministically, so the planner's output is checked before it is used. Every
rejected item is recorded in the trace.
"""
from __future__ import annotations

import json
import re

from ..domains.base import DomainAgent
from ..llm.client import BaseLLM
from ..schemas import EvidencePackage, ExplanationPlan

MAX_NOTES = 3
MAX_NOTE_CHARS = 160
_LOOSENING = re.compile(
    r"\b(?:ignore|disregard|override|bypass|skip|omit (?:the )?disclosure|rules?|caus\w*|lead(?:s)? to|recommend\w*|"
    r"decision|decide|definitely|certain(?:ly)?|guarantee\w*|protected|personal characteristics?|blame|fault)\b",
    re.I,
)


def _loosens(note: str, domain: DomainAgent | None) -> bool:
    if _LOOSENING.search(note):
        return True
    if domain is not None:
        for rule in domain.policy.critic_rules:
            if any(p.search(note) for p in rule.compiled):
                return True
    return False


def planner_prompt(domain: DomainAgent) -> str:
    lang = domain.language
    return (
        f"You plan an explanation of {lang.model_description}'s estimate for {lang.reader}. You do not write the "
        "explanation. Given the question, the reader level and the candidate factors (already ranked by the model), "
        "choose which candidate factors best answer the question, in the order they should be discussed, and give "
        f"at most {MAX_NOTES} short notes on emphasis or wording for this reader. Only use ids from the candidates. "
        'Return JSON {"focus_ids": ["E1", ...], "notes": ["...", ...]}.'
    )


def advise(pkg: EvidencePackage, plan: ExplanationPlan, domain: DomainAgent, llm: BaseLLM, question: str,
           preferences_text: str = "") -> dict:
    """Ask the LLM planner for advice. Returns the raw advice (unchecked) or {"error": ...}."""
    allowed = set(plan.allowed_ids)
    candidates = [{"id": it.id, "factor": it.display_name, "direction": it.direction, "rank": it.rank,
                   "magnitude": it.magnitude} for it in pkg.visible if it.id in allowed]
    candidates += [{"id": r.id, "option": r.short_description or r.description} for r in pkg.recourse if r.id in allowed]
    user = json.dumps({"question": question, "level": plan.level, "intent": plan.intent,
                       "required_ids": plan.required_ids, "candidates": candidates,
                       "reader_preferences": preferences_text})
    try:
        return llm.json("plan", planner_prompt(domain), user, {"plan": plan.model_dump(), "candidates": candidates})
    except Exception as e:
        return {"error": type(e).__name__}


def constrain(plan: ExplanationPlan, advice: dict, domain: DomainAgent | None = None) -> tuple[ExplanationPlan, dict]:
    """Apply the advice within the limits above. Returns (new plan, record of what was accepted and rejected)."""
    record = {"accepted_ids": [], "rejected_ids": [], "accepted_notes": [], "rejected_notes": [], "error": advice.get("error")}
    if advice.get("error"):
        return plan, record
    allowed = list(plan.allowed_ids)
    optional = [i for i in allowed if i != "P" and not i.startswith("G-") and i not in plan.required_ids]
    focus: list[str] = []
    for i in advice.get("focus_ids") or []:
        i = str(i)
        if i in optional and i not in focus:
            focus.append(i)
        elif i in plan.required_ids or i == "P":
            continue  # always kept anyway
        else:
            record["rejected_ids"].append(i)
    notes = []
    for n in (advice.get("notes") or [])[: MAX_NOTES * 2]:
        n = " ".join(str(n).split())[:MAX_NOTE_CHARS]
        if not n:
            continue
        if _loosens(n, domain) or len(notes) >= MAX_NOTES:
            record["rejected_notes"].append(n)
        else:
            notes.append(n)
    if not focus and not notes:
        return plan, record
    kept_optional = focus or optional  # no usable focus: keep the deterministic scope
    glossary = [i for i in allowed if i.startswith("G-")]
    required_rest = [i for i in plan.required_ids if i != "P"]
    new_allowed = ["P", *required_rest, *[i for i in kept_optional if i not in required_rest], *glossary]
    order = [*required_rest, *[i for i in kept_optional if i not in required_rest]]
    record["accepted_ids"] = focus
    record["accepted_notes"] = notes
    new = plan.model_copy(update={
        "allowed_ids": new_allowed,
        "order": order,
        "extra_instructions": [*plan.extra_instructions, *[f"Audience planner: {n}" for n in notes]],
    })
    return new, record
