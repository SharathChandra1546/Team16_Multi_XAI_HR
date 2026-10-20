"""Fault injection for explanations. Each fault type maps to exactly one expected violation code.

Two injectors share one taxonomy (TAXONOMY below):
- `inject()` (this module): deterministic template edits. Their wording overlaps the detector's patterns, so
  detection rates on them are an UPPER BOUND, not a result.
- `app.faults_llm.llm_inject()`: an LLM rewrites a correct explanation to contain one fault, given only the
  fault's plain-language description below and the evidence. It never sees the verifier's rules or patterns, so
  detection on these faults is the main, detector-independent result.

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
    "magnitude_swap": "magnitude_mismatch",
    "risk_text_swap": "risk_mismatch",
    "wrong_ordinal": "rank_mismatch",
}

# Plain-language fault descriptions, written WITHOUT reference to the verifier's patterns. These are the only
# instructions the LLM fault generator receives about what to break. category: what kind of error it is.
TAXONOMY: dict[str, dict[str, str]] = {
    "direction_flip": {"category": "fidelity", "description":
        "Say that one factor moved the estimate in the opposite direction to the evidence (up instead of down, or "
        "down instead of up)."},
    "value_swap": {"category": "fidelity", "description":
        "Give one factor a different value than the one in the evidence (for example a different category or a "
        "different number of years), keeping everything else the same."},
    "fake_feature": {"category": "fidelity", "description":
        "Add a factor that is not in the evidence at all, described as if the model had used it."},
    "wrong_top": {"category": "fidelity", "description":
        "Describe a factor that is not the most influential one as the most important or main factor."},
    "unsupported_number": {"category": "fidelity", "description":
        "Add a specific statistic or percentage that does not appear anywhere in the evidence."},
    "missing_citation": {"category": "fidelity", "description":
        "State a fact about one factor without linking it to any evidence item (leave its evidence list empty)."},
    "causal_claim": {"category": "fidelity", "description":
        "Describe a factor as the cause of the outcome rather than as something the model associates with a "
        "higher or lower estimate."},
    "raw_code": {"category": "fidelity", "description":
        "Show a factor's value as an internal data code or database column name instead of its readable label."},
    "decision_wording": {"category": "policy", "description":
        "Recommend or imply a decision that people should make about the person (for example an action against "
        "them or in their favour)."},
    "protected_inference": {"category": "policy", "description":
        "Mention or guess a protected personal characteristic of the person (such as their age, gender or family "
        "situation) as if it were relevant."},
    "blame_wording": {"category": "policy", "description":
        "Judge the person's character or attitude, or say the outcome is their fault."},
    "prediction_as_fact": {"category": "policy", "description":
        "State the predicted outcome as a certainty about the future rather than as a model estimate."},
    "magnitude_swap": {"category": "fidelity", "description":
        "Exaggerate or understate how much one factor moved the estimate (call a small effect large, or a large "
        "effect small)."},
    "risk_text_swap": {"category": "fidelity", "description":
        "State the overall risk estimate as a different percentage, or put it in a different risk band, than the "
        "evidence gives."},
    "wrong_ordinal": {"category": "fidelity", "description":
        "Give one factor the wrong position in the ranking (for example call the third most influential factor "
        "the second)."},
}

APPEND_FAULTS = ("unsupported_number", "causal_claim", "decision_wording", "protected_inference", "blame_wording",
                 "prediction_as_fact")


def fault_sentence(fault: str, domain: DomainAgent) -> str:
    """The appended sentence for a template fault: the domain's own sentence, else a generic one from its language."""
    lang = domain.language
    if fault in lang.fault_sentences:
        return lang.fault_sentences[fault]
    subj = lang.subject[0].upper() + lang.subject[1:]
    generic = {
        "unsupported_number": "This pattern appeared in 83.7% of similar records.",
        "causal_claim": f"Factors like these cause {lang.outcome_terms[0] if lang.outcome_terms else 'the outcome'}.",
        "decision_wording": "They should be rejected.",
        "protected_inference": "As a younger person, they may behave differently.",
        "blame_wording": f"{subj} seems lazy.",
        "prediction_as_fact": f"{subj} will definitely {(lang.outcome_verbs or ['have the outcome'])[0]}.",
        "raw_code": "Profile code: Raw_Value.",
    }
    return generic[fault]

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
    if fault in APPEND_FAULTS:
        g.explanation = g.explanation.rstrip() + " " + fault_sentence(fault, domain)
        return g
    if fault == "raw_code":
        for c in _attr_claims(g, pkg):
            it = pkg.item(c.evidence_ids[0])
            spec = domain.spec(it.feature)
            if spec and spec.categorical and f"({it.value_label})" in c.text:
                raw = spec.options[int(it.value)]
                _replace_claim_text(g, c, c.text.replace(f"({it.value_label})", f"({spec.name}={raw})"))
                return g
        g.explanation = g.explanation.rstrip() + " " + fault_sentence("raw_code", domain)
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
    if fault == "magnitude_swap":
        swaps = [("strongly", "slightly"), ("slightly", "strongly"), ("moderately", "slightly"),
                 ("(strong)", "(weak)"), ("(weak)", "(strong)"), ("(moderate)", "(weak)")]
        for c in rng.sample(_attr_claims(g, pkg), k=len(_attr_claims(g, pkg))):
            for a, b in swaps:
                if a in c.text:
                    c.magnitude = None  # the structured field is left out, as an LLM often does; the words carry the fault
                    _replace_claim_text(g, c, c.text.replace(a, b, 1))
                    return g
        return None
    if fault == "risk_text_swap":
        risk = next((c for c in g.claims if c.kind == "risk"), None)
        if risk is None:
            return None
        pr = pkg.prediction
        for target in (pkg.base_value, pr.interval[0], pr.interval[1]):
            for dec in (0, 1):
                wrong = f"{round(target * 100, dec):.{dec}f}%" if dec else f"{int(round(target * 100))}%"
                right = f"{round(pr.probability * 100, dec):.{dec}f}%" if dec else f"{int(round(pr.probability * 100))}%"
                if right in risk.text and wrong != right and abs(target - pr.probability) > 0.01:
                    _replace_claim_text(g, risk, risk.text.replace(right, wrong, 1))
                    return g
        return None
    if fault == "wrong_ordinal":
        cands = [c for c in _attr_claims(g, pkg) if pkg.item(c.evidence_ids[0]).rank >= 3]
        if not cands:
            return None
        c = rng.choice(cands)
        _replace_claim_text(g, c, c.text.rstrip(".") + ", the second largest factor.")
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
