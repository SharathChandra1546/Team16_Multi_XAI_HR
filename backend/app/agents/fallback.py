"""Safe template explanations, correct by construction (every sentence is a claim built from the evidence).

Used when retries are exhausted, when no LLM consent is given, and as the mock LLM's base text.
"""
from __future__ import annotations

from ..domains.base import DomainAgent
from ..schemas import Claim, EvidenceItem, EvidencePackage, ExplanationPlan, GeneratedExplanation

MAG_ADVERB = {"strong": "strongly", "moderate": "moderately", "weak": "slightly"}


def pct(p: float, decimals: int = 0) -> str:
    v = round(p * 100, decimals)
    return f"{v:.{decimals}f}" if decimals else f"{int(round(p * 100))}"


def _risk_claim(pkg: EvidencePackage, level: str, variant: int = 0) -> Claim:
    pr = pkg.prediction
    lo, hi = pr.interval
    if level == "novice":
        text = (f"The model estimates a {pct(pr.probability)}% chance that an employee with this profile leaves, "
                f"which is in the {pr.band} risk band.") if variant == 0 else (
                f"For an employee with this profile, the model puts the chance of leaving at {pct(pr.probability)}%, "
                f"in the {pr.band} risk band.")
    elif level == "intermediate":
        text = (f"The model's risk estimate is {pct(pr.probability)}% ({pr.band} band), with a likely range of "
                f"{pct(lo)}% to {pct(hi)}%.") if variant == 0 else (
                f"The estimated attrition risk is {pct(pr.probability)}%, in the {pr.band} band (likely range "
                f"{pct(lo)}% to {pct(hi)}%).")
    else:
        text = (f"Calibrated attrition risk estimate: {pct(pr.probability, 1)}% ({pr.band} band; bootstrap range "
                f"{pct(lo, 1)}% to {pct(hi, 1)}%; review threshold {pct(pr.threshold)}%).")
    return Claim(text=text, kind="risk", evidence_ids=["P"], probability=pr.probability, band=pr.band)


def _attr_claim(it: EvidenceItem, level: str, variant: int = 0) -> Claim:
    name, label = it.display_name, it.value_label
    lname = name[0].lower() + name[1:]
    up = it.direction == "increases"
    top = it.rank == 1
    if level == "novice":
        verb = "pushes the estimate up" if up else "pulls the estimate down"
        text = (f"The biggest factor is {lname} ({label}), which {verb}." if top
                else f"{name} ({label}) {verb}.")
    elif level == "intermediate":
        verb = "raised the estimate" if up else "lowered the estimate"
        if top:
            text = f"The largest factor was {lname} ({label}), which {verb} {MAG_ADVERB[it.magnitude]}."
        else:
            text = (f"{name} ({label}) {verb} {MAG_ADVERB[it.magnitude]}." if variant == 0
                    else f"{name}, at {label}, {verb} {MAG_ADVERB[it.magnitude]}.")
    else:
        sign = "+" if up else "-"
        pts = f"{abs(it.attribution) * 100:.1f}"
        text = (f"Largest contribution: {name} ({label}), {sign}{pts} percentage points ({it.magnitude})." if top
                else f"{name} ({label}): {sign}{pts} percentage points ({it.magnitude}).")
    return Claim(text=text, kind="attribution", evidence_ids=[it.id], feature=it.feature, direction=it.direction,
                 value=label, magnitude=it.magnitude, is_top=top)


def build_template(pkg: EvidencePackage, plan: ExplanationPlan, domain: DomainAgent, variant: int = 0) -> GeneratedExplanation:
    claims: list[Claim] = [_risk_claim(pkg, plan.level, variant)]
    extra_sentences: list[str] = []
    if plan.intent == "recourse":
        if pkg.recourse:
            opts = [o for o in pkg.recourse if o.id in set(plan.allowed_ids)]
            for i, opt in enumerate(opts, start=1):
                novice = plan.level == "novice"
                verb = "drop" if novice else "fall"
                band = "" if novice else f" ({opt.band_after} band)"
                desc = opt.short_description if novice and opt.short_description else opt.description
                text = (f"Option {i}, {desc}: the model's estimate would {verb} from "
                        f"{pct(opt.probability_before)}% to {pct(opt.probability_after)}%{band}.")
                claims.append(Claim(text=text, kind="recourse", evidence_ids=[opt.id],
                                    feature=opt.changes[0].feature if opt.changes else None,
                                    probability=opt.probability_after, band=opt.band_after))
        else:
            extra_sentences.append(pkg.recourse_note or "No actionable change lowers this estimate.")
        extra_sentences.append("These are model estimates for discussion, not predictions of what the employee will do.")
    else:
        allowed = set(plan.allowed_ids)
        for it in pkg.visible:
            if it.id in allowed:
                claims.append(_attr_claim(it, plan.level, variant))
        if any(i.withheld for i in pkg.items):
            extra_sentences.append(domain.policy.withheld_note)
        if plan.level == "expert":
            extra_sentences.append(f"Contributions are Shapley values relative to an average estimate of "
                                   f"{pct(pkg.base_value, 1)}%; they describe the model, not causes.")
        else:
            extra_sentences.append("These are patterns the model learned from past data, not causes.")
    text = " ".join([c.text for c in claims] + extra_sentences)
    return GeneratedExplanation(explanation=text, claims=claims, source="template")


def build_concept(domain: DomainAgent, gid: str) -> GeneratedExplanation:
    e = domain.glossary[gid]
    text = f"{e.term}: {e.definition}"
    if e.values:
        vals = list(e.values.values())
        text += " Possible values: " + (", ".join(vals[:-1]) + " or " + vals[-1] if len(vals) > 1 else vals[0]) + "."
    return GeneratedExplanation(
        explanation=text, source="glossary",
        claims=[Claim(text=text, kind="glossary", evidence_ids=[gid])],
    )
