"""Deterministic claim-level fidelity verifier (domain-agnostic).

Checks every claim against the Canonical Evidence Package and checks the full
text for unsupported numbers, uncited feature mentions, causal wording and raw
codes. Each fault type maps to exactly one violation code.

Codes (all critical unless noted):
  unknown_evidence, withheld_citation, missing_citation, not_allowed_evidence (minor),
  feature_mismatch, direction_mismatch, value_mismatch, magnitude_mismatch, rank_mismatch,
  risk_mismatch, recourse_mismatch, claim_not_in_text, missing_required,
  uncovered_feature_mention, unsupported_number, causal_wording, raw_code_leak, empty_explanation

Claim checks read both the structured fields an LLM fills in AND the claim's own wording, so a claim whose fields
are right but whose sentence says something else is still caught:
  - direction: up/down cues in the sentence must match the evidence direction
  - magnitude: intensity words ("strongly", "slightly", "(weak)") must match the evidence magnitude
  - rank: "largest/main factor" needs rank 1; "second/third largest" needs that rank; "X more than Y" needs
    |attribution(X)| > |attribution(Y)| when both are cited
  - risk: a band word in a risk claim must match the predicted band, and if the claim states percentages one of
    them must be the predicted risk (not, say, the average or the interval bound)
  - recourse: if the claim states percentages, one must be the option's re-predicted risk
"""
from __future__ import annotations

import re

from ..domains.base import DomainAgent
from ..schemas import Claim, EvidencePackage, ExplanationPlan, GeneratedExplanation, VerificationResult, Violation
from ..metrics.fidelity import claim_fidelity_score
from .textutil import extract_numbers, find_rule_matches, norm, sentences

GENERIC_CAUSAL_PATTERNS = [
    r"\bcaus(?:e|es|ed|ing)\b",
    r"\b(?:lead|leads|leading|led) to\b",
    r"\bresult(?:s|ed|ing)? in\b",
    r"\bwill\s+(?:make|cause|keep|stop|prevent|ensure|lead|force|push)\b",
    r"\bprevent(?:s|ing|ed)?\b",
    r"\bis (?:the|a) (?:reason|cause) (?:why|that|for)\b",
]


def causal_patterns(domain: DomainAgent) -> list[str]:
    """Generic causal patterns plus patterns built from the domain's outcome/subject vocabulary and its extras."""
    lang = domain.language
    outcomes = "|".join(lang.outcome_terms) or "the outcome"
    subjects = "|".join(re.escape(t) for t in lang.subject_terms) or "people"
    pats = list(GENERIC_CAUSAL_PATTERNS)
    pats.append(rf"\b(?:drive|drives|driving|drove)\s+(?:\w+\s+)?(?:{outcomes}|{subjects})\b")
    if lang.outcome_verbs:
        verbs = "|".join(re.escape(v) for v in lang.outcome_verbs)
        pats.append(rf"\bmak(?:e|es|ing)\s+(?:{subjects})\s+(?:{verbs})\b")
        stay = [v for v in lang.outcome_verbs if v in ("stay", "remain", "repay", "recover")] or ["stay", "remain"]
        pats.append(rf"\bensur(?:e|es|ing)\s+(?:that\s+)?(?:they|he|she|{subjects})\s+(?:{'|'.join(stay)}|remain)\b")
    pats.append(rf"\b(?:reduc(?:e|es|ed|ing)|lower(?:s|ed|ing)?|increas(?:e|es|ed|ing)|rais(?:e|es|ed|ing)|cut(?:s|ting)?)\s+"
                rf"(?:{outcomes})\b(?!\s+(?:risk|estimate|probability|score|likelihood))")
    return pats + list(domain.policy.causal_extra)


UP_CUES = re.compile(r"\b(?:rais(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|higher|(?:push|pull|mov|driv|bring|brought|nudg|shift)\w*\s+(?:[\w']+\s+){0,3}?up|add(?:s|ed)? to)\b|(?<![\w.])\+\s?\d", re.I)
DOWN_CUES = re.compile(r"\b(?:lower(?:s|ed|ing)?|reduc(?:e|es|ed|ing)|decreas(?:e|es|ed|ing)|less|(?:push|pull|mov|driv|bring|brought|nudg|shift)\w*\s+(?:[\w']+\s+){0,3}?down|offset(?:s)?)\b|(?<![\w.])[-−]\s?\d", re.I)
TOP_CUES = re.compile(r"\b(?:most important|largest|biggest|main|top|strongest|primary|leading)\s+(?:factor|driver|reason|contribution|contributor)\b", re.I)
SECOND_CUES = re.compile(r"\b(second|third|fourth|fifth)[- ](?:largest|biggest|most important|strongest|main|"
                         r"leading|greatest)\b", re.I)
ORDINAL = {"second": 2, "third": 3, "fourth": 4, "fifth": 5}
MORE_THAN = re.compile(r"\b(?:more|larger|bigger|stronger|greater|heavier)\b(?:\s+\w+){0,3}?\s+than\b", re.I)
LESS_THAN = re.compile(r"\b(?:less|smaller|weaker|lighter|lower)\b(?:\s+\w+){0,3}?\s+than\b", re.I)
MAGNITUDE_CUES = {
    "strong": re.compile(r"\b(?:strongly|sharply|substantially|significantly|greatly|heavily|considerably|dramatically|"
                         r"strong|major|a lot)\b", re.I),
    "moderate": re.compile(r"\b(?:moderately|moderate|somewhat)\b", re.I),
    "weak": re.compile(r"\b(?:slightly|marginally|a little|minor|weak|weakly|barely|modestly)\b", re.I),
}
BAND_WORD = re.compile(r"\b(low|medium|high)(?:\s+risk)?\s+band\b|\(\s*(low|medium|high)\s+band\b", re.I)
PERCENT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s?%")
# words that mark a percentage as something other than the point estimate (a range bound, an average, a threshold)
QUALIFIER = re.compile(r"\b(?:range|between|from|to|and|average|baseline|base|mean|typical|threshold|cut-?off|"
                       r"interval|bootstrap|lower|upper|bound|minimum|maximum|up)\b", re.I)


def _headline_percent(text: str) -> tuple[float, str] | None:
    """The first percentage in the sentence that is not qualified as a range bound, average or threshold."""
    for m in PERCENT.finditer(text):
        before = " ".join(text[:m.start()].split()[-3:])
        after = " ".join(text[m.end():].split()[:1])
        if QUALIFIER.search(before) or QUALIFIER.search(after):
            continue
        return float(m.group(1)), m.group(1)
    return None
# identifiers may hold several underscores ("other_installment_plans=A141" escaped the one-underscore pattern)
KEY_VALUE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]+\s*=\s*[A-Za-z0-9_]+")
UNDERSCORE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")
CAMEL = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")

DIR_NORMAL = {
    "increases": "increases", "increase": "increases", "increased": "increases", "raises": "increases",
    "up": "increases", "higher": "increases", "positive": "increases",
    "decreases": "decreases", "decrease": "decreases", "decreased": "decreases", "lowers": "decreases",
    "down": "decreases", "lower": "decreases", "negative": "decreases",
}


def _close(text_value: float, raw: str, target: float) -> bool:
    """A number written with d decimals matches target rounded to d decimals."""
    dec = len(raw.split(".")[1]) if "." in raw else 0
    return abs(text_value - abs(target)) <= 0.5 * 10 ** (-dec) + 1e-9


def allowed_numbers(pkg: EvidencePackage, domain: DomainAgent) -> list[float]:
    nums: list[float] = []
    pr = pkg.prediction
    for v in (pr.probability, *pr.interval, pr.threshold, pkg.base_value):
        nums.append(v * 100)
    # derived: how far the estimate is from the average ("30 points above the average of 27%")
    nums.append(abs(pr.probability - pkg.base_value) * 100)
    nums += [domain.policy.min_risk_drop * 100, 50.0]
    try:
        rt = domain.runtime(pr.model_variant)
        nums += [rt.low_max * 100, rt.high_min * 100]
    except Exception:
        pass
    for it in pkg.items:
        if it.withheld:
            continue
        nums += [abs(it.attribution) * 100, it.share * 100, abs(float(it.value))]
        nums += [v for v, *_ in extract_numbers(it.value_label)]
    for r in pkg.recourse:
        nums += [r.probability_before * 100, r.probability_after * 100,
                 abs(r.probability_before - r.probability_after) * 100]
        for s in [r.description, *[c.from_label for c in r.changes], *[c.to_label for c in r.changes], *r.assumptions]:
            nums += [v for v, *_ in extract_numbers(s)]
    for s in [*pkg.assumptions, pkg.recourse_note]:
        nums += [v for v, *_ in extract_numbers(s)]
    return nums


def _norm_dir(d) -> str | None:
    if d is None:
        return None
    return DIR_NORMAL.get(str(d).strip().lower())


def _value_matches(claim_value, it, spec=None) -> bool:
    if claim_value is None or claim_value == "":
        return True
    if spec is not None and spec.categorical and not isinstance(claim_value, (int, float)):
        # a category is right only if its label is the evidence label: "below 0 DM" must not pass for "0 to 200 DM"
        # just because both contain the number 0
        return norm(str(claim_value)) == norm(it.value_label)
    if isinstance(claim_value, (int, float)) and not isinstance(claim_value, bool):
        nums = [abs(float(it.value))] + [v for v, *_ in extract_numbers(it.value_label)]
        return any(abs(float(claim_value) - n) <= 0.5 + 1e-9 for n in nums)
    cv = norm(str(claim_value))
    if cv == norm(it.value_label):
        return True
    nums = extract_numbers(str(claim_value))
    if nums:
        targets = [abs(float(it.value))] + [v for v, *_ in extract_numbers(it.value_label)]
        return all(any(_close(v, raw, t) for t in targets) for v, raw, *_ in nums)
    return False


def _feature_matches(claim_feature, it) -> bool:
    if not claim_feature:
        return True
    c = str(claim_feature).strip().lower()
    return c in (it.feature.lower(), it.display_name.lower())


def _covered(sentence: str, claim_texts: list[str]) -> bool:
    s = norm(sentence)
    return any(ct and (ct in s or s in ct) for ct in claim_texts)


def verify(pkg: EvidencePackage, plan: ExplanationPlan, gen: GeneratedExplanation, domain: DomainAgent) -> VerificationResult:
    v: list[Violation] = []
    text = gen.explanation or ""
    if not text.strip():
        return VerificationResult(passed=False, fidelity_score=0.0, fidelity=claim_fidelity_score(0, []).as_dict(), n_claims=0,
                                  violations=[Violation(code="empty_explanation", severity="critical", message="No text.")])
    item_ids = {i.id for i in pkg.items}
    allowed = set(plan.allowed_ids)
    gloss = {g.id for g in pkg.glossary}
    ntext = norm(text)
    bad_claims: set[int] = set()

    def add(code: str, msg: str, idx: int | None = None, span: str | None = None, severity: str = "critical"):
        if any(x.code == code and x.claim_index == idx and x.span == span for x in v):
            return
        v.append(Violation(code=code, severity=severity, message=msg, claim_index=idx, span=span))
        if idx is not None and severity == "critical":
            bad_claims.add(idx)

    cited: set[str] = set()
    for idx, c in enumerate(gen.claims):
        _check_claim(pkg, plan, c, idx, add, item_ids, allowed, gloss, cited, domain)
        if c.text and norm(c.text) not in ntext:
            add("claim_not_in_text", "Claim text does not appear in the explanation.", idx)

    for rid in plan.required_ids:
        if rid not in cited:
            add("missing_required", f"Required evidence {rid} is not cited.", None, rid)

    claim_texts = [norm(c.text) for c in gen.claims if c.text]

    # uncited mentions of model features (protected features are handled by the critic's protected rule)
    aliases = domain.all_aliases()
    sensitive = set(domain.policy.sensitive)
    model_feats = set(domain.model_features(pkg.prediction.model_variant))
    for sent in sentences(text):
        ls = sent.lower()
        mentioned = [f for f, als in aliases.items() if f in model_feats and f not in sensitive
                     and any(re.search(rf"\b{re.escape(a)}\b", ls) for a in als)]
        if mentioned and not _covered(sent, claim_texts):
            add("uncovered_feature_mention", f"Sentence mentions {', '.join(mentioned)} without a claim.", None, sent[:80])

    # unsupported numbers (skip numbers inside claims that already failed: one code per fault)
    nums = allowed_numbers(pkg, domain)
    for val, raw, s, e in extract_numbers(text):
        if any(_close(val, raw, n) for n in nums):
            continue
        if re.search(r"\boption\s*$", text[max(0, s - 8):s], re.I) and val.is_integer() and val <= len(pkg.recourse):
            continue  # option numbering
        # skip a number that sits inside a claim already flagged (one code per fault); locate it by its position,
        # not by searching for its digits (a "4" must not be matched to the "4" inside "124" in another sentence)
        in_bad = any(start <= s < start + len(gen.claims[i].text)
                     for i in bad_claims if gen.claims[i].text
                     for start in _occurrences(text, gen.claims[i].text))
        if not in_bad:
            add("unsupported_number", f"Number {raw} is not in the evidence.", None, raw)

    # causal wording (generic + domain extras)
    pats = [re.compile(p, re.I) for p in causal_patterns(domain)]
    for span in find_rule_matches(text, pats, negatable=True):
        add("causal_wording", "Describes an association as a cause.", None, span)

    # raw codes
    for m in KEY_VALUE.finditer(text):
        add("raw_code_leak", "Raw key=value code in text.", None, m.group(0))
    for m in UNDERSCORE.finditer(text):
        add("raw_code_leak", "Raw dataset value in text.", None, m.group(0))
    if not plan.allow_column_names:
        names = {f.name for f in domain.features}
        for m in CAMEL.finditer(text):
            if m.group(0) in names:
                add("raw_code_leak", "Raw column name in text.", None, m.group(0))
    for f in domain.features:
        if not f.categorical or any(ch.isdigit() for lab in f.labels.values() for ch in lab):
            continue
        for a in [f.display.lower(), *f.aliases]:
            pat = (rf"\b{re.escape(a)}\b(?:\s*(?:of|is|was|=|:|at|rated|score|level)\s*(?:[0-9]|yes|no)\b(?![%.,]\d)"
                   rf"|\s*\(\s*(?:[0-9]|yes|no)\s*\))")
            for m in re.finditer(pat, text, re.I):
                add("raw_code_leak", "Raw code shown instead of a readable value.", None, m.group(0))

    critical = [x for x in v if x.severity == "critical"]
    n = len(gen.claims)
    report = claim_fidelity_score(n, v, [c.text for c in gen.claims], text)
    return VerificationResult(passed=not critical, fidelity_score=report.score, fidelity=report.as_dict(),
                              violations=v, n_claims=n)


def _check_claim(pkg, plan, c: Claim, idx, add, item_ids, allowed, gloss, cited, domain=None) -> None:
    ids = [i for i in (c.evidence_ids or []) if i]
    if not ids:
        add("missing_citation", "Claim cites no evidence.", idx)
        return
    for eid in ids:
        known = eid == "P" or eid in item_ids or pkg.option(eid) is not None or eid in gloss
        if not known:
            add("unknown_evidence", f"Cites unknown evidence {eid}.", idx, eid)
            return
        it = pkg.item(eid)
        if it is not None and it.withheld:
            add("withheld_citation", f"Cites withheld evidence {eid}.", idx, eid)
            return
        if eid not in allowed and eid not in gloss:
            add("not_allowed_evidence", f"{eid} is outside this plan's scope.", idx, eid, severity="minor")
        cited.add(eid)

    if c.kind == "risk" or "P" in ids:
        pr = pkg.prediction
        if c.probability is not None and abs(float(c.probability) - pr.probability) > 0.005 + 1e-9:
            add("risk_mismatch", "Stated risk differs from the prediction.", idx, "P")
        if c.band and str(c.band).lower() != pr.band:
            add("risk_mismatch", "Stated band differs from the prediction.", idx, "P")
        if c.kind == "risk":
            for m in BAND_WORD.finditer(c.text or ""):
                if (m.group(1) or m.group(2)).lower() != pr.band:
                    add("risk_mismatch", f"The sentence names the {(m.group(1) or m.group(2)).lower()} band; the "
                        f"prediction is in the {pr.band} band.", idx, "P")
            head = _headline_percent(c.text or "")
            if head and not _close(head[0], head[1], pr.probability * 100):
                add("risk_mismatch", f"The sentence gives the risk as {head[1]}%; the predicted risk is "
                    f"{pr.probability * 100:.1f}%.", idx, "P")
            return
    if c.kind == "recourse":
        opt = next((pkg.option(i) for i in ids if pkg.option(i)), None)
        if opt is None:
            add("recourse_mismatch", "Recourse claim does not cite an option.", idx)
            return
        if c.probability is not None and abs(float(c.probability) - opt.probability_after) > 0.005 + 1e-9:
            add("recourse_mismatch", "Stated post-change risk differs from the re-predicted value.", idx, opt.id)
        if c.feature and not any(str(c.feature).lower() in (ch.feature.lower(), ch.display_name.lower()) for ch in opt.changes):
            add("recourse_mismatch", "Recourse claim names a factor the option does not change.", idx, opt.id)
        if c.band and str(c.band).lower() != opt.band_after:
            add("recourse_mismatch", "Stated post-change band differs.", idx, opt.id)
        pcts = [(float(m.group(1)), m.group(1)) for m in PERCENT.finditer(c.text or "")]
        if pcts and not any(_close(v, raw, opt.probability_after * 100) for v, raw in pcts):
            add("recourse_mismatch", "The sentence does not state the option's re-predicted risk.", idx, opt.id)
        return
    if c.kind == "glossary":
        return
    # attribution claim
    items = [pkg.item(i) for i in ids if pkg.item(i)]
    if not items:
        add("missing_citation", "Attribution claim cites no attribution item.", idx)
        return
    it = items[0]
    if not _feature_matches(c.feature, it):
        add("feature_mismatch", f"Claim names {c.feature} but cites {it.feature}.", idx)
        return
    d = _norm_dir(c.direction)
    # the factor's own value label can contain a cue word ("less than 1 year", "higher education"): not a direction claim
    cue_text = c.text or ""
    for lab in {it.value_label, getattr(it, "display_name", None)} - {None, ""}:
        cue_text = re.sub(re.escape(str(lab)), " ", cue_text, flags=re.I)
    up, down = bool(UP_CUES.search(cue_text)), bool(DOWN_CUES.search(cue_text))
    text_dir = "increases" if up and not down else "decreases" if down and not up else None
    if (d and d != it.direction) or (text_dir and text_dir != it.direction):
        add("direction_mismatch", f"Direction differs from the evidence ({it.direction}).", idx, it.id)
    if not _value_matches(c.value, it, domain.spec(it.feature) if domain is not None else None):
        add("value_mismatch", f"Value differs from the evidence ({it.value_label}).", idx, it.id)
    if c.magnitude and str(c.magnitude).lower() != it.magnitude:
        add("magnitude_mismatch", f"Magnitude differs from the evidence ({it.magnitude}).", idx, it.id)
    elif len(items) == 1:
        said = {m for m, pat in MAGNITUDE_CUES.items() if pat.search(c.text or "")}
        if said and it.magnitude not in said:
            add("magnitude_mismatch", f"The wording says {'/'.join(sorted(said))}; the evidence magnitude is "
                f"{it.magnitude}.", idx, it.id)
    no_ordinals = SECOND_CUES.sub(" ", c.text or "")  # "second largest factor" is not a top-factor claim
    if (c.is_top or TOP_CUES.search(no_ordinals)) and it.rank != 1:
        add("rank_mismatch", "Claims a top factor that is not ranked first.", idx, it.id)
    m2 = SECOND_CUES.search(c.text or "")
    if m2 and it.rank != ORDINAL[m2.group(1).lower()]:
        add("rank_mismatch", f"Calls a factor {m2.group(1).lower()}-ranked; its rank is {it.rank}.", idx, it.id)
    if len(items) >= 2:
        _check_comparison(c, items[:2], idx, add)


def _first_pos(text: str, it) -> int:
    t = text.lower()
    pos = [t.find(n.lower()) for n in (it.display_name, it.feature) if n and t.find(n.lower()) >= 0]
    return min(pos) if pos else -1


def _check_comparison(c: Claim, pair, idx, add) -> None:
    """'X ... more than Y': the factor named first must have the larger absolute attribution (reverse for 'less')."""
    text = c.text or ""
    more, less = bool(MORE_THAN.search(text)), bool(LESS_THAN.search(text))
    if more == less:
        return
    a, b = pair
    pa, pb = _first_pos(text, a), _first_pos(text, b)
    if pa < 0 or pb < 0:
        return
    first, second = (a, b) if pa < pb else (b, a)
    bigger_first = abs(first.attribution) > abs(second.attribution)
    if (more and not bigger_first) or (less and bigger_first):
        add("rank_mismatch", f"Compares {first.display_name} and {second.display_name} in the wrong order.", idx, first.id)


def _occurrences(text: str, sub: str) -> list[int]:
    out, i = [], text.find(sub)
    while i >= 0:
        out.append(i)
        i = text.find(sub, i + 1)
    return out
