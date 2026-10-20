"""Router: decision-use guardrail, domain routing and intent detection (domain-agnostic).

Guardrail layer 1 is deterministic (DomainPolicy patterns from every registered
domain). Layer 2 is an optional LLM classifier; a request is refused if either
layer flags it.
"""
from __future__ import annotations

import re

from ..domains.base import DomainLanguage
from ..domains.registry import DomainRegistry
from ..llm.client import BaseLLM
from ..schemas import RoutingResult

_ACTORS = ["we", "i", "management", "the company", "the organi[sz]ation", "the team"]
_LEVERS = ["reducing", "removing", "changing", "increasing", "lowering", "raising", "adding"]


def recourse_patterns(lang: DomainLanguage) -> list[str]:
    """Recourse-question patterns, built from the domain's language (outcome words, actors, levers)."""
    risk_words = "|".join(["risk", "chance", "likelihood", "probability", "estimate", *lang.outcome_terms])
    actors = "|".join(_ACTORS + [re.escape(a) for a in lang.actor_terms])
    levers = "|".join(_LEVERS + [re.escape(t) for t in lang.lever_terms])
    subj = "|".join(["we", "they", "i", re.escape(lang.subject)])
    return [
        rf"\b(?:lower|reduce|decrease|bring down|cut|improve|mitigate|minimi[sz]e)\b.{{0,40}}\b(?:{risk_words})\b",
        rf"\bwhat (?:can|could|would|should|might) (?:{actors}) (?:do|change|try|offer)\b",
        r"\bhow (?:can|could|do|would|might) (?:we|i) (?:keep|retain|help|support)\b",
        r"\bwhat would (?:help|change|make a difference|lower|reduce)\b",
        r"\b(?:recourse|what[- ]if|retention (?:options|levers|actions|ideas)|levers?)\b",
        rf"\b(?:would|could|does|do|will) (?:a |an )?(?:{levers})\b.{{0,60}}\b(?:risk|chance|estimate|help|lower|reduce)\b",
        rf"\bif (?:{subj})\b.{{0,60}}\b(?:risk|chance|estimate)\b",
        r"\b(?:would|could|will)\b.{0,40}\b(?:change|lower|reduce|affect|help)\b.{0,30}\b(?:risk|estimate|band)\b",
    ]


_CONCEPT = [
    r"\bwhat (?:does|do)\b.{0,60}\b(?:mean|stand for|refer to|measure)\b",
    r"\b(?:define|definition of|meaning of|what is meant by|explain the term|how is .{1,40} measured)\b",
]


def instance_words(lang: DomainLanguage) -> re.Pattern:
    poss = re.escape(lang.subject_possessive.split()[-1]) if lang.subject_possessive else "person's"
    return re.compile(rf"\b(?:this|their|his|her|why|flagged|{poss})\b", re.I)


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def normalise(q: str) -> str:
    q = q.lower().replace("’", "'")
    q = re.sub(r"[^a-z0-9'%\- ]+", " ", q)
    return " ".join(q.split())


class Router:
    def __init__(self, registry: DomainRegistry, threshold: float = 0.6):
        self.registry = registry
        self.threshold = threshold

    # ---------------------------------------------------------------- guardrail
    def guard_rules(self, query: str) -> str | None:
        q = normalise(query)
        for d in self.registry.list():
            pol = d.policy
            for p in pol.guard_patterns:
                if re.search(p, q, re.I):
                    return f"decision_request ({d.name}: hard rule)"
            if _any(pol.guard_soft, q) and _any(pol.guard_frames, q):
                return f"decision_request ({d.name}: decision framing)"
        return None

    def guard_prompt(self) -> str:
        """LLM guard instructions, built from every registered domain's language (no domain is hard-coded)."""
        scopes = "; ".join(f"{d.title} ({d.language.risk_name} for {d.language.reader})" for d in self.registry.list())
        decisions = "; ".join(f"{d.language.decision_kind} such as {d.language.decision_examples}" for d in self.registry.list())
        return (
            f"You are a safety classifier for an analytics tool that only estimates and explains model risk estimates: {scopes}. "
            f"Decide whether the user is asking the tool to make, recommend or rank a decision about people ({decisions}, "
            "or choosing who to keep, approve or reject). Questions about what would lower the model's risk estimate are NOT "
            'decision requests. Return JSON: {"decision_request": true|false, "reason": "short reason"}.'
        )

    def guard_llm(self, query: str, llm: BaseLLM) -> str | None:
        try:
            out = llm.json("guard", self.guard_prompt(), f"User request: {query!r}", {"query": query})
        except Exception:
            return None  # layer 1 already ran; LLM failure must not block legitimate use
        return "decision_request (LLM classifier)" if out.get("decision_request") is True else None

    # ---------------------------------------------------------------- routing
    def route(self, query: str, instance: dict | None, domain: str | None, llm: BaseLLM | None = None) -> RoutingResult:
        reason = self.guard_rules(query)
        if not reason and llm is not None:
            reason = self.guard_llm(query, llm)
        if reason:
            return RoutingResult(domain=domain if self.registry.has(domain) else None, confidence=1.0,
                                 intent="out_of_scope", refusal_reason=reason, method="guardrail")

        # domain
        if self.registry.has(domain):
            dom, conf = domain, 1.0
        else:
            q = normalise(query)
            scored = []
            for d in self.registry.list():
                hits = sum(1 for k in d.routing_keywords if re.search(rf"\b{re.escape(k)}\b", q))
                overlap = 0.0
                if instance:
                    names = {f.name for f in d.features}
                    overlap = len(names & set(instance)) / max(1, len(names))
                scored.append((max(min(1.0, 0.4 + 0.2 * hits) if hits else 0.0, overlap), d.name))
            scored.sort(reverse=True)
            conf, dom = scored[0] if scored else (0.0, None)
            if len(scored) > 1 and scored[1][0] == conf and conf > 0:
                conf = min(conf, 0.5)  # tie between domains: ask the user
        if dom is None or conf < self.threshold:
            return RoutingResult(domain=None, confidence=conf, intent="explain", method="rules")

        # intent
        agent = self.registry.get(dom)
        if _any(_CONCEPT, query) or (re.match(r"^\s*what(?: is|'s| are)\b", query, re.I)
                                     and not instance_words(agent.language).search(query) and agent.concept_lookup(query)):
            term = agent.concept_lookup(query)
            if term:
                return RoutingResult(domain=dom, confidence=conf, intent="concept", concept_term=term)
        if _any(recourse_patterns(agent.language), query):
            return RoutingResult(domain=dom, confidence=conf, intent="recourse")
        return RoutingResult(domain=dom, confidence=conf, intent="explain")
