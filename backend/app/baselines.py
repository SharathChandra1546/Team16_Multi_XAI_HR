"""Baseline explanation systems, re-implemented under one setup for a controlled comparison.

Every system receives the SAME evidence for a case: the canonical evidence package built by this framework (same
model output, same Shapley attributions, same recourse options). Only what each system does with that evidence
differs. Each returns the text a reader would see; the comparison runner measures that text identically for every
system (app/metrics/measure.py).

Systems (names used by eval/run_comparison.py):

- direct      Bello-style translation: the raw attribution table (column names, values, Shapley values) and the
              prediction go to one LLM call that writes the explanation for the reader level. No checks.
- b1          Musicco et al. B1: one LLM call on the explanation card, one generic narrative, no audience, no revision.
- b2          Musicco et al. B2: B1 plus a role description of the reader in the prompt. Single shot.
- b3          Musicco et al. B3: B1 plus a revision loop. A separate reviewer gives a 1-5 score, an accept/revise verdict
              and a comment; the narrator rewrites until accepted or the budget is spent.
- b4          Musicco et al. B4: B3 plus a rule-based format verifier (numeric attribution values present in the prose,
              no list-like or JSON-like formatting, more than one sentence); rewrite unless BOTH accept.
- explingo    Explingo-style (Zytek et al. 2024): narrator with three few-shot exemplars; grader scores accuracy,
              completeness and fluency (0-4) and conciseness with a deterministic length formula; rewrite while the
              average is below the threshold.
- critic_loop He & Martens style (Critic Design): narrator plus a faithfulness critic LLM that lists unfaithful
              statements; the narrator revises until the critic lists none or the round budget is spent.
- critic_rule He & Martens style (Critic-Rule Design): the same loop, but the critic is rule-based. An LLM extractor
              reads each factor's value, direction and rank from the draft, code compares them with the card and turns
              the errors into revision instructions. This is the closest prior design to this framework's verifier
              (it checks facts, but has no policy rules, level rules, escalation or template fallback).
- musicco     Musicco et al. full pipeline (XstrAI-style): Framer (LLM plan: feature order, tone, cautions,
              instructions) -> Narrator (card + plan + audience) -> Reviewer (score, revise flag, route to framer /
              narrator / both, instructions), at most two revision rounds. Reviewer may be a different model family.

These are re-implementations from the papers' descriptions, not the authors' code. Prompts are written for this
framework's evidence format; budgets follow the papers where stated (Musicco: two revision rounds; He & Martens:
three rounds). The conciseness formula of Explingo is approximated (see EXPLINGO_CONCISENESS). State this in the
paper. Musicco et al. released their code (github.com/sisinflab/XstrAI); running it on the same evidence is a
stronger comparison if it can be adapted.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .agents.generator import evidence_view
from .agents.textutil import word_count
from .domains.base import DomainAgent
from .llm.client import BaseLLM, usage_delta
from .schemas import EvidencePackage, ExplanationPlan

LEVEL_DESCRIPTIONS = {
    "novice": "has no technical or statistical background; use plain everyday words and short sentences",
    "intermediate": "is a practitioner comfortable with percentages but not with machine-learning methods",
    "expert": "is technical; precise quantities and method names are welcome",
}
MUSICCO_ROUNDS = 2
B3_BUDGET = 2
CRITIC_ROUNDS = 3
EXPLINGO_ROUNDS = 2
EXPLINGO_THRESHOLD = 3.5


@dataclass
class SystemResult:
    system: str
    text: str
    rounds: int = 1
    accepted: bool | None = None  # the system's own final verdict, if it has a checker
    usage: dict = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)


def _narrative(out: dict) -> str:
    for k in ("narrative", "explanation", "text"):
        if isinstance(out.get(k), str):
            return out[k]
    return ""


def _card(pkg: EvidencePackage, plan: ExplanationPlan) -> dict:
    return evidence_view(pkg, plan)


def _raw_table(pkg: EvidencePackage, plan: ExplanationPlan) -> dict:
    allowed = set(plan.allowed_ids)
    return {"prediction": {"probability": pkg.prediction.probability, "label": pkg.prediction.label},
            "base_value": pkg.base_value,
            "shap": [{"feature": it.feature, "value": it.value, "shap_value": round(it.attribution, 4)}
                     for it in pkg.visible if it.id in allowed]}


def _call(llm: BaseLLM, task: str, system: str, user: Any, pkg, plan, domain, attempt: int) -> dict:
    payload = {"package": pkg.model_dump(), "plan": plan.model_dump(), "domain": domain.name, "attempt": attempt}
    try:
        return llm.json(task, system, json.dumps(user, ensure_ascii=False, default=str), payload)
    except Exception as e:
        return {"error": type(e).__name__}


def _narrator_system(domain: DomainAgent, audience: str | None) -> str:
    base = (f"You write a short narrative that explains a prediction of {domain.language.model_description} from the "
            "explanation card you are given (prediction, factors with their values, direction and contribution). ")
    if audience:
        base += audience
    return base + ' Return JSON {"narrative": "..."}.'


def _audience(domain: DomainAgent, level: str) -> str:
    return (f"The reader is {domain.language.reader} who {LEVEL_DESCRIPTIONS[level]}. Adapt content and wording to "
            "this reader.")


# ---------------------------------------------------------------- single-shot systems
def run_direct(pkg, plan, domain, llm, **_) -> SystemResult:
    system = (f"Translate this SHAP explanation of a machine-learning prediction into a natural-language explanation "
              f"for a {plan.level} reader. " + 'Return JSON {"narrative": "..."}.')
    out = _call(llm, "narrate", system, {"level": plan.level, **_raw_table(pkg, plan)}, pkg, plan, domain, 1)
    return SystemResult("direct", _narrative(out), trace=[{"step": "narrate", "error": out.get("error")}])


def run_b1(pkg, plan, domain, llm, **_) -> SystemResult:
    out = _call(llm, "narrate", _narrator_system(domain, None), {"card": _card(pkg, plan)}, pkg, plan, domain, 1)
    return SystemResult("b1", _narrative(out), trace=[{"step": "narrate", "error": out.get("error")}])


def run_b2(pkg, plan, domain, llm, **_) -> SystemResult:
    out = _call(llm, "narrate", _narrator_system(domain, _audience(domain, plan.level)), {"card": _card(pkg, plan)},
                pkg, plan, domain, 1)
    return SystemResult("b2", _narrative(out), trace=[{"step": "narrate", "error": out.get("error")}])


# ---------------------------------------------------------------- reviewer loops (Musicco B3, B4)
REVIEWER_SYSTEM = (
    "You review a narrative that explains a model prediction against the explanation card it was written from. "
    "Give a quality score from 1 to 5, a verdict \"accept\" or \"revise\", and a short comment saying what to fix. "
    'Return JSON {"score": n, "verdict": "accept|revise", "comment": "..."}.'
)


def format_verifier(text: str, pkg: EvidencePackage, plan: ExplanationPlan) -> list[str]:
    """B4's rule-based format checks (machine-readable properties only, no fact checking)."""
    problems = []
    if not text.strip():
        return ["empty"]
    if re.search(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text):
        problems.append("list-like formatting")
    if re.search(r"^\s*[\[{]|\"\w+\"\s*:", text):
        problems.append("JSON-like formatting")
    if len(re.findall(r"[.!?](?:\s|$)", text)) < 2:
        problems.append("fewer than two sentences")
    allowed = set(plan.allowed_ids)
    shap_vals = {f"{abs(it.attribution) * 100:.1f}" for it in pkg.visible if it.id in allowed} | \
                {f"{abs(it.attribution):.2f}" for it in pkg.visible if it.id in allowed}
    if not any(v in text for v in shap_vals):
        problems.append("no numeric attribution value in the prose")
    return problems


def _reviewer_loop(name: str, pkg, plan, domain, llm, reviewer, with_format: bool, audience: str | None) -> SystemResult:
    card = _card(pkg, plan)
    sys_n = _narrator_system(domain, audience)
    out = _call(llm, "narrate", sys_n, {"card": card}, pkg, plan, domain, 1)
    text = _narrative(out)
    trace = [{"step": "narrate", "round": 0}]
    accepted = False
    for rnd in range(1, B3_BUDGET + 2):
        rv = _call(reviewer, "review_score", REVIEWER_SYSTEM, {"card": card, "narrative": text}, pkg, plan, domain, rnd)
        ok_rev = str(rv.get("verdict", "")).lower() == "accept"
        fmt = format_verifier(text, pkg, plan) if with_format else []
        trace.append({"step": "review", "round": rnd, "score": rv.get("score"), "verdict": rv.get("verdict"),
                      "format_problems": fmt})
        if ok_rev and not fmt:
            accepted = True
            break
        if rnd > B3_BUDGET:
            break
        feedback = [str(rv.get("comment", ""))] + [f"Format: {p}." for p in fmt]
        out = _call(llm, "narrate", sys_n, {"card": card, "previous": text, "revise": feedback}, pkg, plan, domain, rnd + 1)
        text = _narrative(out) or text
        trace.append({"step": "narrate", "round": rnd})
    return SystemResult(name, text, rounds=sum(1 for t in trace if t["step"] == "narrate"), accepted=accepted, trace=trace)


def run_b3(pkg, plan, domain, llm, reviewer=None, **_) -> SystemResult:
    return _reviewer_loop("b3", pkg, plan, domain, llm, reviewer or llm, with_format=False, audience=None)


def run_b4(pkg, plan, domain, llm, reviewer=None, **_) -> SystemResult:
    return _reviewer_loop("b4", pkg, plan, domain, llm, reviewer or llm, with_format=True, audience=None)


# ---------------------------------------------------------------- Explingo-style
GRADER_SYSTEM = (
    "You grade a narrative explanation of a model prediction against the explanation it was written from. Score from "
    "0 to 4: accuracy (every stated fact matches the explanation), completeness (all factors in the explanation are "
    "covered), fluency (reads naturally). "
    'Return JSON {"accuracy": n, "completeness": n, "fluency": n}.'
)


def explingo_conciseness(text: str, n_factors: int) -> float:
    """EXPLINGO_CONCISENESS (approximation): 4 when the narrative uses at most ~20 words per factor, falling
    linearly to 0 at twice that length."""
    budget = max(20, 20 * n_factors)
    w = word_count(text)
    return 4.0 if w <= budget else max(0.0, 4.0 * (2 - w / budget))


_EXEMPLAR_CACHE: dict[tuple, list[dict]] = {}


def explingo_exemplars(domain: DomainAgent, level: str) -> list[dict]:
    """Three labelled exemplars (input explanation -> narrative), built from verified templates on three fixed
    example cases of the domain, so the narrator sees the target style without seeing the case being explained."""
    key = (domain.name, level)
    if key in _EXEMPLAR_CACHE:
        return _EXEMPLAR_CACHE[key]
    from .agents.fallback import build_template
    from .agents.planner import make_plan
    from .xai.evidence import build_package

    rt = domain.runtime("main")
    ex = []
    for _, row in rt.examples.tail(3).iterrows():
        x = {f: float(row[f]) for f in rt.features}
        p = build_package(domain, x, level, "explain")
        pl = make_plan(p, level, "explain")
        ex.append({"explanation": _raw_table(p, pl), "narrative": build_template(p, pl, domain).explanation})
    _EXEMPLAR_CACHE[key] = ex
    return ex


def run_explingo(pkg, plan, domain, llm, reviewer=None, **_) -> SystemResult:
    reviewer = reviewer or llm
    shots = explingo_exemplars(domain, plan.level)
    system = ("You convert a machine-learning explanation (feature, value, contribution) into a short narrative, in the "
              'style of the examples. Return JSON {"narrative": "..."}.')
    table = _raw_table(pkg, plan)
    n_f = len(table["shap"])
    out = _call(llm, "narrate", system, {"examples": shots, "explanation": table}, pkg, plan, domain, 1)
    text = _narrative(out)
    trace, accepted = [{"step": "narrate", "round": 0}], False
    for rnd in range(1, EXPLINGO_ROUNDS + 2):
        g = _call(reviewer, "grade", GRADER_SYSTEM, {"explanation": table, "narrative": text}, pkg, plan, domain, rnd)
        scores = {k: float(g.get(k, 0) or 0) for k in ("accuracy", "completeness", "fluency")}
        scores["conciseness"] = explingo_conciseness(text, n_f)
        avg = sum(scores.values()) / 4
        trace.append({"step": "grade", "round": rnd, "scores": scores, "average": round(avg, 2)})
        if avg >= EXPLINGO_THRESHOLD:
            accepted = True
            break
        if rnd > EXPLINGO_ROUNDS:
            break
        out = _call(llm, "narrate", system, {"examples": shots, "explanation": table, "previous": text,
                                             "rewrite": "The previous narrative scored low; rewrite it."},
                    pkg, plan, domain, rnd + 1)
        text = _narrative(out) or text
        trace.append({"step": "narrate", "round": rnd})
    return SystemResult("explingo", text, rounds=sum(1 for t in trace if t["step"] == "narrate"), accepted=accepted,
                        trace=trace)


# ---------------------------------------------------------------- He & Martens style critic loop
CRITIC_SYSTEM = (
    "You are a faithfulness critic. Compare the narrative with the explanation card. List every statement in the "
    "narrative that is not faithful to the card (a wrong value, direction, ranking or number, or a factor that is not "
    "in the card) and say how to fix it. "
    'Return JSON {"unfaithful": [{"statement": "...", "fix": "..."}]}.'
)


def run_critic_loop(pkg, plan, domain, llm, reviewer=None, **_) -> SystemResult:
    critic = reviewer or llm
    card = _card(pkg, plan)
    sys_n = _narrator_system(domain, _audience(domain, plan.level))
    out = _call(llm, "narrate", sys_n, {"card": card}, pkg, plan, domain, 1)
    text = _narrative(out)
    trace, accepted = [{"step": "narrate", "round": 0}], False
    for rnd in range(1, CRITIC_ROUNDS + 1):
        c = _call(critic, "critique_faithful", CRITIC_SYSTEM, {"card": card, "narrative": text}, pkg, plan, domain, rnd)
        issues = c.get("unfaithful") or []
        trace.append({"step": "critic", "round": rnd, "issues": len(issues)})
        if not issues:
            accepted = True
            break
        out = _call(llm, "narrate", sys_n, {"card": card, "previous": text, "revise": issues}, pkg, plan, domain, rnd + 1)
        text = _narrative(out) or text
        trace.append({"step": "narrate", "round": rnd})
    return SystemResult("critic_loop", text, rounds=sum(1 for t in trace if t["step"] == "narrate"), accepted=accepted,
                        trace=trace)


def rule_feedback(text: str, pkg: EvidencePackage, domain: DomainAgent, extractor: BaseLLM) -> list[str]:
    """Rule-based critic (He & Martens, Critic-Rule): LLM extraction, then code compares with the card."""
    from .metrics.agreement import _value_ok, llm_extract_mentions

    items = {it.feature: it for it in pkg.visible}
    mentions = llm_extract_mentions(text, pkg, domain, extractor)
    issues: list[str] = []
    for m in mentions:
        it = items.get(m.feature)
        spec = domain.spec(m.feature)
        name = spec.display if spec else m.feature
        if it is None:
            issues.append(f"Remove {name}: it is not in the explanation card.")
            continue
        if m.value and not _value_ok(m.value, it, domain):
            issues.append(f"For {name} the card says {it.value_label}, not {m.value}.")
        if m.direction and m.direction != it.direction:
            issues.append(f"{name} {it.direction} the estimate in the card, not the opposite.")
    by_card = [it.feature for it in sorted((items[m.feature] for m in mentions if m.feature in items), key=lambda i: i.rank)]
    in_text = [m.feature for m in sorted((m for m in mentions if m.feature in items), key=lambda m: m.order)]
    if by_card != in_text:
        issues.append("Present the factors in this order: " + ", ".join(domain.spec(f).display for f in by_card) + ".")
    return issues


def run_critic_rule(pkg, plan, domain, llm, reviewer=None, **_) -> SystemResult:
    extractor = reviewer or llm
    card = _card(pkg, plan)
    sys_n = _narrator_system(domain, _audience(domain, plan.level))
    out = _call(llm, "narrate", sys_n, {"card": card}, pkg, plan, domain, 1)
    text = _narrative(out)
    trace, accepted = [{"step": "narrate", "round": 0}], False
    for rnd in range(1, CRITIC_ROUNDS + 1):
        issues = rule_feedback(text, pkg, domain, extractor)
        trace.append({"step": "rule_critic", "round": rnd, "issues": len(issues)})
        if not issues:
            accepted = True
            break
        out = _call(llm, "narrate", sys_n, {"card": card, "previous": text, "revise": [{"statement": "", "fix": i} for i in issues]},
                    pkg, plan, domain, rnd + 1)
        text = _narrative(out) or text
        trace.append({"step": "narrate", "round": rnd})
    return SystemResult("critic_rule", text, rounds=sum(1 for t in trace if t["step"] == "narrate"), accepted=accepted,
                        trace=trace)


# ---------------------------------------------------------------- Musicco-style full pipeline
FRAMER_SYSTEM = (
    "You plan, but do not write, an explanation of a model prediction for a specific reader. From the explanation "
    "card, decide the order in which to present the factors, the tone, the cautions the narrative must include, and "
    "instructions for the writer. "
    'Return JSON {"feature_order": ["E1", ...], "tone": "...", "cautions": ["..."], "instructions": ["..."]}.'
)
MUSICCO_REVIEWER = (
    "You are the gatekeeper for a narrative explanation. Check it against the explanation card and the plan: card "
    "grounding (values, directions, numbers), fit for the reader, communication safety and epistemic framing. Return a "
    "score from 1 to 5, whether a revision is needed, where to route it (\"narrator\" to rewrite the text, \"framer\" "
    "to re-plan, or \"both\"), and instructions. "
    'Return JSON {"score": n, "revise": true|false, "route": "narrator|framer|both", "instructions": ["..."]}.'
)


def run_musicco(pkg, plan, domain, llm, reviewer=None, **_) -> SystemResult:
    reviewer = reviewer or llm
    card = _card(pkg, plan)
    aud = _audience(domain, plan.level)
    frame = _call(llm, "frame", FRAMER_SYSTEM + " " + aud, {"card": card}, pkg, plan, domain, 1)
    sys_n = _narrator_system(domain, aud)
    out = _call(llm, "narrate", sys_n, {"card": card, "plan": frame}, pkg, plan, domain, 1)
    text = _narrative(out)
    trace, accepted = [{"step": "frame", "round": 0}, {"step": "narrate", "round": 0}], False
    for rnd in range(1, MUSICCO_ROUNDS + 2):
        rv = _call(reviewer, "review_route", MUSICCO_REVIEWER, {"card": card, "plan": frame, "narrative": text},
                   pkg, plan, domain, rnd)
        revise = bool(rv.get("revise"))
        trace.append({"step": "review", "round": rnd, "score": rv.get("score"), "revise": revise, "route": rv.get("route")})
        if not revise:
            accepted = True
            break
        if rnd > MUSICCO_ROUNDS:
            break
        if rv.get("route") in ("framer", "both"):
            frame = _call(llm, "frame", FRAMER_SYSTEM + " " + aud,
                          {"card": card, "previous_plan": frame, "revise": rv.get("instructions")}, pkg, plan, domain, rnd + 1)
            trace.append({"step": "frame", "round": rnd})
        out = _call(llm, "narrate", sys_n, {"card": card, "plan": frame, "previous": text,
                                            "revise": rv.get("instructions")}, pkg, plan, domain, rnd + 1)
        text = _narrative(out) or text
        trace.append({"step": "narrate", "round": rnd})
    return SystemResult("musicco", text, rounds=sum(1 for t in trace if t["step"] == "narrate"), accepted=accepted,
                        trace=trace)


SYSTEMS: dict[str, Callable[..., SystemResult]] = {
    "direct": run_direct, "b1": run_b1, "b2": run_b2, "b3": run_b3, "b4": run_b4, "explingo": run_explingo,
    "critic_loop": run_critic_loop, "critic_rule": run_critic_rule, "musicco": run_musicco,
}


def run_baseline(name: str, pkg: EvidencePackage, plan: ExplanationPlan, domain: DomainAgent, llm: BaseLLM,
                 reviewer: BaseLLM | None = None) -> SystemResult:
    """Run one baseline and record the LLM usage it caused (generator and reviewer counted together)."""
    clients = [llm] + ([reviewer] if reviewer is not None and reviewer is not llm else [])
    before = [c.usage_snapshot() for c in clients]
    res = SYSTEMS[name](pkg, plan, domain, llm, reviewer=reviewer)
    deltas = [usage_delta(b, c.usage_snapshot()) for b, c in zip(before, clients)]
    res.usage = {k: sum(d[k] for d in deltas) for k in ("calls", "cached_calls", "prompt_tokens", "completion_tokens")}
    res.usage["estimated"] = any(d["estimated"] for d in deltas)
    return res
