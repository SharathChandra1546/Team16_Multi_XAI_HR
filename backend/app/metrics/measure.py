"""One measurement applied identically to the shown text of every system (this framework and every baseline).

It never looks at a system's own structured claims or its internal verdicts; only at the text a reader sees and
the shared evidence package. Per text it reports:

- agreement: value / sign / rank agreement, top-3 coverage and unsupported factor mentions (app/metrics/agreement.py),
  from the deterministic extractor and, if a judge LLM is given, the LLM extractor. Only for explain answers:
  recourse answers state hypothetical new values and an option order, which are not attribution values or ranks;
  for them only unsupported factor mentions count;
- text checks shared with the verifier: unsupported numbers, causal wording, raw codes;
- policy checks: the domain's wording rules (decision, protected attributes, blame, certainty);
- level checks: banned terms, word budget, level_numbers, reading ease;
- error_free_llm (primary when a judge model is given): the same, with the LLM extractor's agreement;
- error_free: no value/sign error, no explicitly stated wrong rank (mention order only lowers RA), no unsupported mention, no unsupported number, no causal wording, no raw
  code, and no policy violation.

Bias to state in the paper: the text and policy checks are the same deterministic rules this framework's pipeline
gates on, so the pipeline is measured by the rules it was filtered with. The agreement metrics (especially with the
LLM extractor) and the Phase 3 human labels are the measures that do not share that bias.
"""
from __future__ import annotations

import re

from ..agents.critic import level_violations, rule_violations
from ..agents.textutil import extract_numbers, find_rule_matches, flesch_reading_ease, sentences, word_count
from ..agents.verifier import KEY_VALUE, UNDERSCORE, _close, allowed_numbers, causal_patterns
from ..domains.base import DomainAgent
from ..llm.client import BaseLLM
from ..schemas import EvidencePackage, ExplanationPlan
from .agreement import agreement, extract_mentions, llm_extract_mentions


def _strip(text: str, domain: DomainAgent) -> str:
    """Remove the domain's mandatory disclosure so it is not measured as content."""
    return text.replace(domain.policy.disclosure, " ").strip()


def text_checks(text: str, pkg: EvidencePackage, domain: DomainAgent) -> dict:
    nums = allowed_numbers(pkg, domain)
    unsupported = []
    for val, raw, s, e in extract_numbers(text):
        if not any(_close(val, raw, n) for n in nums):
            if re.search(r"\boption\s*$", text[max(0, s - 8):s], re.I) and val.is_integer() and val <= 5:
                continue
            unsupported.append(raw)
    pats = [re.compile(p, re.I) for p in causal_patterns(domain)]
    causal = find_rule_matches(text, pats, negatable=True)
    raw_codes = [m.group(0) for m in KEY_VALUE.finditer(text)] + [m.group(0) for m in UNDERSCORE.finditer(text)]
    return {"unsupported_numbers": unsupported, "causal_wording": causal, "raw_codes": raw_codes}


def measure_text(text: str, pkg: EvidencePackage, plan: ExplanationPlan, domain: DomainAgent,
                 judge: BaseLLM | None = None) -> dict:
    body = _strip(text or "", domain)
    det = agreement(extract_mentions(body, pkg, domain), pkg, domain)
    llm_ag = agreement(llm_extract_mentions(body, pkg, domain, judge), pkg, domain) if judge is not None else None
    if plan.intent == "recourse":
        keep = ("n_mentions", "unsupported_mentions")
        det = {k: (det[k] if k in keep else (0 if k.endswith("_errors") else None)) for k in det}
        llm_ag = ({k: (llm_ag[k] if k in keep else (0 if k.endswith("_errors") else None)) for k in llm_ag}
                  if llm_ag else None)
    tc = text_checks(body, pkg, domain)
    pol = [v.model_dump() for v in rule_violations(body, domain, pkg)]
    lv, fre, wc = level_violations(body, plan, [it.value_label for it in pkg.items] + [pkg.recourse_note, *pkg.limitations])
    fidelity_errors = (det["value_errors"] + det["sign_errors"] + det.get("stated_rank_errors", 0) + len(det["unsupported_mentions"])
                       + len(tc["unsupported_numbers"]) + len(tc["causal_wording"]) + len(tc["raw_codes"]))
    text_errors = len(tc["unsupported_numbers"]) + len(tc["causal_wording"]) + len(tc["raw_codes"])
    fid_llm = (llm_ag["value_errors"] + llm_ag["sign_errors"] + llm_ag.get("stated_rank_errors", 0)
               + len(llm_ag["unsupported_mentions"]) + text_errors) if llm_ag else None
    return {
        # primary when an independent judge model is used: the LLM extractor reads paraphrases the rule-based
        # extractor misses, so the deterministic figure (error_free) is reported as a secondary, stricter measure
        "fidelity_errors_llm": fid_llm,
        "error_free_llm": (fid_llm == 0 and not pol) if llm_ag else None,
        "words": word_count(body), "sentences": len(sentences(body)), "reading_ease": fre,
        "agreement_det": {k: v for k, v in det.items() if k != "mentions"},
        "agreement_llm": ({k: v for k, v in llm_ag.items() if k != "mentions"} if llm_ag else None),
        "text_checks": tc,
        "policy_violations": sorted({v["code"] for v in pol}),
        "n_policy_violations": len(pol),
        "level_violations": sorted({v.code for v in lv if v.severity == "critical"}),
        "fidelity_errors": fidelity_errors,
        "error_free": fidelity_errors == 0 and not pol,
        "empty": not body.strip(),
    }
