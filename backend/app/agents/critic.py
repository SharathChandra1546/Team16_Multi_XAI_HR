"""Critic: domain wording rules (from DomainPolicy), level fit, disclosure, and optional LLM quality scores.

Facts are never judged here; the deterministic verifier owns fidelity.
"""
from __future__ import annotations

import json
import re

from ..domains.base import DomainAgent
from ..llm.client import BaseLLM
from ..schemas import CriticResult, EvidencePackage, ExplanationPlan, GeneratedExplanation, Violation
from .textutil import flesch_reading_ease, norm, sentences, word_count

READABILITY_MIN = {"novice": 50.0, "intermediate": 30.0, "expert": 0.0}
WORD_SLACK = 1.2


def rule_violations(text: str, domain: DomainAgent, pkg: EvidencePackage | None = None) -> list[Violation]:
    out: list[Violation] = []
    allowed_protected: set[str] = set()
    if pkg is not None:
        for it in pkg.visible:
            if it.protected:  # expert view of the audit model: naming the cited attribute is allowed
                spec = domain.spec(it.feature)
                phrases = [spec.display, *spec.aliases, *spec.labels.values()] if spec else []
                allowed_protected |= {a.lower() for a in phrases}
                allowed_protected |= {w.lower() for a in phrases for w in a.split()}
    for rule in domain.policy.critic_rules:
        for sent in sentences(text):
            for pat in rule.compiled:
                for m in pat.finditer(sent):
                    span = m.group(0)
                    if rule.negatable and _negated(sent, m.start()):
                        continue
                    if rule.code == domain.policy.protected_text_rule and span.lower() in allowed_protected:
                        continue
                    if not any(v.code == rule.code and v.span == span for v in out):
                        out.append(Violation(code=rule.code, severity="critical", message=rule.message, span=span))
    return out


def _negated(sentence: str, start: int) -> bool:
    from .textutil import negated

    return negated(sentence, start)


def check_disclosure(final_text: str, domain: DomainAgent) -> Violation | None:
    if norm(domain.policy.disclosure) not in norm(final_text):
        return Violation(code="disclosure_missing", severity="critical",
                         message="The mandatory risk-estimate disclosure is missing.")
    return None


def critique(
    pkg: EvidencePackage, plan: ExplanationPlan, gen: GeneratedExplanation, domain: DomainAgent,
    llm: BaseLLM | None = None, question: str = "",
) -> CriticResult:
    text = gen.explanation or ""
    v = rule_violations(text, domain, pkg)
    low = text.lower()
    for term in plan.banned_terms:
        if re.search(rf"\b{re.escape(term)}\b", low):
            v.append(Violation(code="terminology", severity="critical", span=term,
                               message=f"'{term}' is not suitable for the {plan.level} level."))
    wc = word_count(text)
    if wc > plan.max_words * WORD_SLACK:
        v.append(Violation(code="word_budget", severity="critical",
                           message=f"{wc} words; the {plan.level} budget is {plan.max_words}."))
    fre = flesch_reading_ease(text)
    if fre < READABILITY_MIN[plan.level]:
        v.append(Violation(code="readability", severity="minor",
                           message=f"Reading ease {fre} is below {READABILITY_MIN[plan.level]} for {plan.level}."))
    scores: dict[str, float] = {}
    feedback = ""
    if llm is not None and gen.source == "llm":
        system = (
            "You review explanations written for an HR manager about an attrition model. Score each from 1 to 5: "
            "clarity, level_fit (suits the stated reader level), relevance (answers the question). Do not judge facts. "
            'Return JSON {"clarity": n, "level_fit": n, "relevance": n, "feedback": "one sentence"}.'
        )
        user = json.dumps({"level": plan.level, "question": question, "explanation": text})
        try:
            out = llm.json("critic", system, user, {"level": plan.level, "text": text})
            scores = {k: float(out.get(k, 0)) for k in ("clarity", "level_fit", "relevance")}
            feedback = str(out.get("feedback", ""))[:300]
            if min(scores.values()) < 3:
                v.append(Violation(code="quality_low", severity="critical", message=f"Quality scores {scores}. {feedback}"))
        except Exception as e:  # critic LLM failure must not block a verified explanation
            feedback = f"LLM critic unavailable: {type(e).__name__}"
    approved = not any(x.severity == "critical" for x in v)
    return CriticResult(approved=approved, violations=v, scores=scores, readability=fre, word_count=wc, feedback=feedback)
