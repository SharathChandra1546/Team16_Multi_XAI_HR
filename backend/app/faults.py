"""Fault injection for explanations. Each fault type maps to exactly one expected violation code.

Used by the evaluation scripts (detection rates) and by the mock LLM (to exercise retries).
"""
from __future__ import annotations

import random
import re

from .domains.base import DomainAgent
from .schemas import Claim, EvidencePackage, ExplanationPlan, GeneratedExplanation

FAULTS: dict[str, str] = {
    # fault type -> expected violation code
    "direction_flip": "direction_mismatch",
    "value_swap": "value_mismatch",
    "fake_feature": "unknown_evidence",
    "wrong_top": "rank_mismatch",
    "unsupported_number": "unsupported_number",
    "missing_citation": "missing_citation",
    "causal_claim": "causal_wording",
    "raw_code": "raw_code_leak",
    "decision_wording": "decision_wording",
    "protected_inference": "protected_attribute_inference",
    "blame_wording": "blame_wording",
    "prediction_as_fact": "prediction_as_fact",
}

APPEND = {
    "unsupported_number": "This pattern appeared in 83.7% of similar records.",
    "causal_claim": "Working conditions like these cause attrition.",
    "decision_wording": "They should be dismissed.",
    "protected_inference": "As a younger employee, they may be more mobile.",
    "blame_wording": "The employee seems lazy.",
    "prediction_as_fact": "The employee will leave.",
}

_SWAPS = [("raised", "lowered"), ("pushes the estimate up", "pulls the estimate down"), ("+", "-")]


def _copy(gen: GeneratedExplanation) -> GeneratedExplanation:
    return GeneratedExplanation(explanation=gen.explanation, claims=[c.model_copy() for c in gen.claims], source=gen.source)


def _replace_claim_text(g: GeneratedExplanation, c: Claim, new_text: str) -> None:
    g.explanation = g.explanation.replace(c.text, new_text, 1)
    c.text = new_text


def _attr_claims(g: GeneratedExplanation, pkg: EvidencePackage, non_top: bool = False) -> list[Claim]:
    out = []
    for c in g.claims:
        if c.kind != "attribution" or not c.evidence_ids:
            continue
        it = pkg.item(c.evidence_ids[0])
        if it and (not non_top or it.rank > 1):
            out.append(c)
    return out


def inject(fault: str, gen: GeneratedExplanation, pkg: EvidencePackage, plan: ExplanationPlan, domain: DomainAgent,
           rng: random.Random | None = None) -> GeneratedExplanation | None:
    """Return a faulty copy, or None if the fault does not apply to this explanation."""
    rng = rng or random.Random(0)
    g = _copy(gen)
    if fault in APPEND:
        g.explanation = g.explanation.rstrip() + " " + APPEND[fault]
        return g
    if fault == "raw_code":
        for c in _attr_claims(g, pkg):
            it = pkg.item(c.evidence_ids[0])
            spec = domain.spec(it.feature)
            if spec and spec.categorical and f"({it.value_label})" in c.text:
                raw = spec.options[int(it.value)]
                _replace_claim_text(g, c, c.text.replace(f"({it.value_label})", f"({spec.name}={raw})"))
                return g
        g.explanation = g.explanation.rstrip() + " Profile code: Travel_Rarely."
        return g
    if fault == "fake_feature":
        text = "Bonus eligibility (not eligible) raised the estimate slightly."
        g.explanation = g.explanation.rstrip() + " " + text
        g.claims.append(Claim(text=text, kind="attribution", evidence_ids=["E99"], feature="BonusEligibility",
                              direction="increases", value="not eligible", magnitude="weak"))
        return g
    if fault == "direction_flip":
        cands = _attr_claims(g, pkg)
        if not cands:
            return None
        c = rng.choice(cands)
        new = c.text
        for a, b in _SWAPS:
            if a in new:
                new = new.replace(a, b, 1)
                break
            if b in new:
                new = new.replace(b, a, 1)
                break
        else:
            new = new.replace("lowered", "raised", 1)
        c.direction = "decreases" if c.direction == "increases" else "increases"
        _replace_claim_text(g, c, new)
        return g
    if fault == "value_swap":
        for c in rng.sample(_attr_claims(g, pkg), k=len(_attr_claims(g, pkg))):
            it = pkg.item(c.evidence_ids[0])
            spec = domain.spec(it.feature)
            if spec and spec.categorical:
                others = [lab for code, lab in spec.labels.items() if code != int(it.value)]
                new_label = rng.choice(others)
            else:
                new_val = round(float(it.value) * 1.37 + 3)
                new_label = spec.readable(new_val) if spec else str(new_val)
            if f"({it.value_label})" not in c.text:
                continue
            c.value = new_label
            _replace_claim_text(g, c, c.text.replace(f"({it.value_label})", f"({new_label})", 1))
            return g
        return None
    if fault == "wrong_top":
        cands = _attr_claims(g, pkg, non_top=True)
        if not cands:
            return None
        c = rng.choice(cands)
        c.is_top = True
        _replace_claim_text(g, c, c.text.rstrip(".") + ", the main factor.")
        return g
    if fault == "missing_citation":
        required = set(plan.required_ids)
        cands = [c for c in _attr_claims(g, pkg, non_top=True) if c.evidence_ids[0] not in required]
        if not cands:
            return None
        rng.choice(cands).evidence_ids = []
        return g
    raise ValueError(f"unknown fault {fault}")


def strip_disclosure(text: str, disclosure: str) -> str:
    return re.sub(re.escape(disclosure), "", text).strip()
