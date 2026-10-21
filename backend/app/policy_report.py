"""Violation report: every violation code in a category, counted per explanation.

Categories:
- fidelity: the text misstates the evidence (verifier codes)
- policy: the text breaks a domain rule (the domain's critic rules, plus the disclosure)
- level: the text does not fit the reader level (terminology, word budget, numbers, readability)
- quality: the LLM critic scored the text low
- other: anything else

Three views per explanation, so Phase 3 can report "policy violations the generator produced" separately from
"policy violations that reached the user":
- first_attempt: violations in the generator's first draft
- all_attempts: every violation across retries
- shown: violations in the text actually shown (from the final verification and critic, including shadow-mode
  checks when a gate is switched off). With all gates on this is empty for verified and fallback outputs.
"""
from __future__ import annotations

from collections import Counter

from .domains.base import DomainAgent

FIDELITY = {
    "unknown_evidence", "withheld_citation", "missing_citation", "not_allowed_evidence", "feature_mismatch",
    "direction_mismatch", "value_mismatch", "magnitude_mismatch", "rank_mismatch", "risk_mismatch",
    "recourse_mismatch", "claim_not_in_text", "missing_required", "uncovered_feature_mention", "unsupported_number",
    "causal_wording", "raw_code_leak", "empty_explanation",
}
LEVEL = {"terminology", "word_budget", "readability", "level_numbers"}
QUALITY = {"quality_low"}
POLICY_EXTRA = {"disclosure_missing"}
CATEGORIES = ("fidelity", "policy", "level", "quality", "other")


def policy_codes(domain: DomainAgent | None) -> set[str]:
    codes = set(POLICY_EXTRA)
    if domain is not None:
        codes |= {r.code for r in domain.policy.critic_rules}
        codes.add(domain.policy.protected_text_rule)
    return codes


def category(code: str, domain: DomainAgent | None = None) -> str:
    if code in FIDELITY:
        return "fidelity"
    if code in policy_codes(domain):
        return "policy"
    if code in LEVEL:
        return "level"
    if code in QUALITY:
        return "quality"
    return "other"


def _tally(violations: list[dict], domain: DomainAgent | None) -> dict:
    out: dict[str, Counter] = {c: Counter() for c in CATEGORIES}
    for v in violations:
        if v.get("severity") != "critical":
            continue
        out[category(v["code"], domain)][v["code"]] += 1
    return {c: dict(n) for c, n in out.items() if n}


def summarize(history: list[dict], shown: list[dict], domain: DomainAgent | None) -> dict:
    """`history`: graph history entries ({attempt, stage, violations}); `shown`: violations of the shown text."""
    first = [v for h in history if h.get("attempt") == 1 for v in h.get("violations", [])]
    every = [v for h in history for v in h.get("violations", [])]
    shown_t = _tally(shown, domain)
    return {
        "first_attempt": _tally(first, domain),
        "all_attempts": _tally(every, domain),
        "shown": shown_t,
        "shown_policy_violations": sum(shown_t.get("policy", {}).values()),
        "shown_fidelity_violations": sum(shown_t.get("fidelity", {}).values()),
        "attempts": max([h.get("attempt", 0) for h in history], default=0),
    }
