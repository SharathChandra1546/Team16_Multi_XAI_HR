"""Critic: domain wording rules (from DomainPolicy), level fit, disclosure, and optional LLM quality scores.

Facts are never judged here; the deterministic verifier owns fidelity.
"""
from __future__ import annotations

import re

from ..domains.base import DomainAgent
from ..llm.client import BaseLLM
from ..schemas import CriticResult, EvidencePackage, ExplanationPlan, GeneratedExplanation, Violation
from .llm_critic import MIN_SCORE, llm_quality
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


# Figures a reader without "include_numbers" may not see: contributions ("percentage points", "pp", signed
# numbers like +3.2) and bare decimals such as a raw probability (0.32). Percentages (the risk estimate, recourse
# before/after) are allowed at every level; wrong ones are the verifier's job (unsupported_number), not this rule's.
_LEVEL_NUMBERS = re.compile(
    r"\bpercentage points?\b|\bpp\b|(?<![\w.])[+\u2212-]\d+(?:\.\d+)?(?![\w%]|\.\d)|(?<![\w.+\u2212-])\d+\.\d+(?![\w%]|\.\d)",
    re.I,
)


def level_violations(text: str, plan: ExplanationPlan, value_labels: list[str] | None = None) -> tuple[list[Violation], float, int]:
    """Checkable constraints of the reader level: terminology, word budget, numbers, readability.

    `value_labels` are spans exempt from the numbers check: the factors' own readable values (e.g. "23.5") and
    fixed system sentences (limitations, the no-recourse note), which are the same at every level.
    """
    v: list[Violation] = []
    low = text.lower()
    for term in plan.banned_terms:
        if re.search(rf"\b{re.escape(term)}\b", low):
            v.append(Violation(code="terminology", severity="critical", span=term,
                               message=f"'{term}' is not suitable for the {plan.level} level."))
    wc = word_count(text)
    if wc > plan.max_words * WORD_SLACK:
        v.append(Violation(code="word_budget", severity="critical",
                           message=f"{wc} words; the {plan.level} budget is {plan.max_words}."))
    if not plan.include_numbers:
        scrubbed = text
        for lab in sorted({lab for lab in (value_labels or []) if lab}, key=len, reverse=True):
            scrubbed = scrubbed.replace(lab, " ")
        for m in _LEVEL_NUMBERS.finditer(scrubbed):
            v.append(Violation(code="level_numbers", severity="critical", span=m.group(0),
                               message=f"Detailed figures are not shown at the {plan.level} level; give the risk "
                                       "percentage and directions only."))
    fre = flesch_reading_ease(text)
    floor = plan.min_reading_ease if plan.min_reading_ease else READABILITY_MIN.get(plan.level, 0.0)
    if fre < floor:
        v.append(Violation(code="readability", severity="minor",
                           message=f"Reading ease {fre} is below {floor} for {plan.level}."))
    return v, fre, wc


def critique(
    pkg: EvidencePackage, plan: ExplanationPlan, gen: GeneratedExplanation, domain: DomainAgent,
    llm: BaseLLM | None = None, question: str = "",
) -> CriticResult:
    text = gen.explanation or ""
    v = rule_violations(text, domain, pkg)
    lv, fre, wc = level_violations(text, plan, [it.value_label for it in pkg.items] +
                                   [c.to_label for r in pkg.recourse for c in r.changes] +
                                   [c.from_label for r in pkg.recourse for c in r.changes] +
                                   [pkg.recourse_note, *pkg.limitations])
    v += lv
    scores: dict[str, float] = {}
    feedback = ""
    if llm is not None and gen.source == "llm":
        scores, feedback, err = llm_quality(llm, plan.level, question, text, domain.language)
        if scores and min(scores.values()) < MIN_SCORE:
            v.append(Violation(code="quality_low", severity="critical", message=f"Quality scores {scores}. {feedback}"))
    approved = not any(x.severity == "critical" for x in v)
    return CriticResult(approved=approved, violations=v, scores=scores, readability=fre, word_count=wc, feedback=feedback)
