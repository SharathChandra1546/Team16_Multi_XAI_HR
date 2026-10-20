"""Structured, violation-code-specific repair instructions for a retry.

Previously a failed attempt's violations were handed back to the generator as
a raw {code, message, span} dump (see generator.generate's old `previous`
argument). That makes the model guess what to actually change. Here each
violation code maps to a specific, imperative instruction, and where the
violation points at one claim/evidence id we pull the *correct* value out of
the Evidence Package so the instruction is self-correcting rather than just
"this was wrong, try again."
"""
from __future__ import annotations

from ..domains.base import DomainLanguage
from ..schemas import EvidencePackage, Violation

# Generic instruction per code when the violation can't be bound to a specific evidence id
# (keys match verifier.py / critic.py violation codes).
GENERIC: dict[str, str] = {
    "missing_citation": "Every claim must cite at least one evidence id (P, E#, R#, or G-#). Add the id you relied on.",
    "unknown_evidence": "Remove the citation to an id that is not in the evidence, or cite the correct id instead.",
    "withheld_citation": "Do not cite a withheld item. Drop that claim or use a different, visible piece of evidence.",
    "not_allowed_evidence": "Only cite ids listed in allowed_ids for this plan.",
    "feature_mismatch": "The claim names a different factor than the evidence id it cites. Match the feature name to the cited id.",
    "direction_mismatch": "State the direction (increases/decreases) exactly as given in the evidence item's 'direction' field, not your own judgement.",
    "value_mismatch": "Copy the value_label of the cited evidence item exactly. Do not round, paraphrase, or recompute it.",
    "magnitude_mismatch": "Use the evidence item's own 'magnitude' field (strong/moderate/weak); do not estimate it yourself.",
    "rank_mismatch": "Only the item with rank 1 may be called the top/largest/main factor. Check rank before using that language.",
    "risk_mismatch": "The stated risk percentage or band must equal the prediction's probability/band in the evidence, to within half a point.",
    "recourse_mismatch": "The stated before/after risk for a recourse option must equal that option's own probability_before/probability_after.",
    "claim_not_in_text": "Every claim's 'text' must appear in the explanation field word for word. Copy it in, do not summarise it.",
    "missing_required": "Include a claim citing every id in required_ids (the risk estimate, and for recourse the top option).",
    "uncovered_feature_mention": "A sentence names a factor with no matching claim. Either add a claim for it or remove the sentence.",
    "unsupported_number": "Remove this number or replace it with one that appears in the evidence JSON (percentages, contributions, thresholds).",
    "causal_wording": "Rewrite as an association, not a cause: 'the model associates X with a higher/lower estimate', never 'X causes/drives/leads to'.",
    "raw_code_leak": "Replace the raw code/column name with the evidence item's value_label.",
    "decision_wording": "Remove any recommendation or mention of {decision_kind} ({decision_examples}).",
    "protected_attribute_inference": "Remove any reference to {protected_description}. Say '{subject}' or 'they'.",
    "blame_wording": "Remove judgemental language about {subject_possessive} character. Describe factors neutrally.",
    "prediction_as_fact": "Describe the outcome as an estimate ('the model estimates...'), never as a certainty ('will leave').",
    "terminology": "Remove the flagged technical term; it is not allowed at this reader level.",
    "word_budget": "The draft is over the word budget for this level. Cut it down.",
    "readability": "Simplify sentence structure and word choice for this reader level.",
    "quality_low": "The reviewer scored this draft low on clarity, level fit, or relevance. See its feedback and address it directly.",
    "disclosure_missing": "Append the mandatory disclosure sentence verbatim at the end.",
    "empty_explanation": "The explanation field was empty. Write the actual explanation text, not just claims.",
}


class _Defaults(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _evidence_fact(pkg: EvidencePackage, eid: str | None) -> str:
    if not eid:
        return ""
    it = pkg.item(eid)
    if it is not None:
        return f" {eid} is {it.feature} = {it.value_label}, direction {it.direction}, rank {it.rank}."
    opt = pkg.option(eid)
    if opt is not None:
        return f" {eid} goes from {opt.probability_before * 100:.0f}% to {opt.probability_after * 100:.0f}% ({opt.band_after} band)."
    if eid == "P":
        pr = pkg.prediction
        return f" P is {pr.probability * 100:.1f}% ({pr.band} band, range {pr.interval[0] * 100:.0f} to {pr.interval[1] * 100:.0f}%)."
    return ""


def build_repair_instructions(violations: list[Violation], pkg: EvidencePackage,
                              language: DomainLanguage | None = None) -> list[str]:
    """One instruction per distinct critical violation, each self-correcting where an evidence id is known.

    Domain words (subject, decision kind, protected attributes) come from the domain's DomainLanguage.
    """
    lang = vars(language or DomainLanguage())
    seen: set[tuple[str, str | None]] = set()
    out: list[str] = []
    for v in violations:
        if v.severity != "critical":
            continue
        key = (v.code, v.span)
        if key in seen:
            continue
        seen.add(key)
        base = GENERIC.get(v.code, v.message).format_map(_Defaults(lang))
        # span is the evidence id for citation-shaped codes; a fact pulled straight from the package beats prose
        fact = _evidence_fact(pkg, v.span) if v.code in (
            "direction_mismatch", "value_mismatch", "magnitude_mismatch", "rank_mismatch", "unknown_evidence", "withheld_citation",
        ) else ""
        out.append(f"[{v.code}] {base}{fact}")
    return out
