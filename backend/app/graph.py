"""Orchestration graph (LangGraph). Domain-agnostic: no node or edge refers to a specific domain.

route -> (refuse | clarify | concept | prepare)
prepare -> xai -> gate -> (evidence_error | escalate | plan)
plan -> generate -> verify -> (critic | generate | exhausted)
critic -> (finalize | generate | exhausted)
exhausted -> (fallback -> finalize) | withheld (batch mode)

If the `langgraph` package is not installed, a minimal engine with the same
semantics (node returns a partial state update; conditional edges by function)
is used so the system still runs.
"""
from __future__ import annotations

import time
from typing import Any, Callable, TypedDict

from .agents.critic import check_disclosure, critique
from .agents.fallback import build_concept, build_template
from .agents.generator import generate
from .agents.planner import make_plan
from .agents.repair import build_repair_instructions
from .agents.router import Router
from .agents.verifier import verify
from .config import settings
from .domains.registry import DomainRegistry
from .llm.client import BaseLLM
from .schemas import EvidencePackage, ExplanationPlan, GeneratedExplanation
from .xai.evidence import build_package

try:  # pragma: no cover - environment dependent
    from langgraph.graph import END, START, StateGraph

    HAS_LANGGRAPH = True
except Exception:
    HAS_LANGGRAPH = False
    START, END = "__start__", "__end__"


class MiniGraph:
    """Tiny stand-in for langgraph.StateGraph (sequential, dict-merge state updates)."""

    def __init__(self, _schema=None):
        self.nodes: dict[str, Callable] = {}
        self.edges: dict[str, str] = {}
        self.cond: dict[str, tuple[Callable, dict]] = {}

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, a, b):
        self.edges[a] = b

    def add_conditional_edges(self, a, fn, mapping):
        self.cond[a] = (fn, mapping)

    def compile(self):
        return self

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        state = dict(state)
        cur = self.edges[START]
        limit = (config or {}).get("recursion_limit", 60)
        for _ in range(limit):
            if cur == END:
                return state
            state.update(self.nodes[cur](state) or {})
            if cur in self.cond:
                fn, mapping = self.cond[cur]
                cur = mapping[fn(state)]
            else:
                cur = self.edges.get(cur, END)
        raise RuntimeError("graph recursion limit reached")


class GraphState(TypedDict, total=False):
    request: dict  # may carry "preferences_text": the requesting user's rendered preference notes (see agents/preferences.py)
    use_llm: bool
    prompt: dict
    routing: dict
    domain: str
    level: str
    intent: str
    variant: str
    instance: dict
    assumed: list
    package: dict
    plan: dict
    generated: dict
    verification: dict
    critic: dict
    attempts: int
    history: list
    status: str
    message: str
    notices: list
    disclosure: str
    trace: list


def _t(state: dict, node: str, t0: float, **info) -> list:
    return [*state.get("trace", []), {"node": node, "ms": round((time.perf_counter() - t0) * 1000, 1), **info}]


class ExplanationGraph:
    def __init__(self, registry: DomainRegistry, llm: BaseLLM, use_langgraph: bool | None = None):
        self.registry = registry
        self.llm = llm
        self.router = Router(registry, settings.router_confidence_threshold)
        use = HAS_LANGGRAPH if use_langgraph is None else (use_langgraph and HAS_LANGGRAPH)
        self.engine = "langgraph" if use else "minigraph"
        self.app = self._build(StateGraph if use else MiniGraph)

    # ------------------------------------------------------------------ nodes
    def n_route(self, s):
        t0 = time.perf_counter()
        req = s["request"]
        r = self.router.route(req.get("query", ""), req.get("instance"), req.get("domain"),
                              self.llm if s.get("use_llm") else None)
        return {"routing": r.model_dump(), "domain": r.domain, "intent": r.intent,
                "trace": _t(s, "route", t0, intent=r.intent, domain=r.domain, confidence=r.confidence)}

    def c_route(self, s):
        r = s["routing"]
        if r["intent"] == "out_of_scope":
            return "refuse"
        if not r["domain"]:
            return "clarify"
        if r["intent"] == "concept":
            return "concept"
        return "prepare"

    def n_refuse(self, s):
        dom = self.registry.get(s["domain"]) if s.get("domain") else self.registry.list()[0]
        return {"status": "refused", "message": dom.policy.refusal_message, "trace": _t(s, "refuse", time.perf_counter())}

    def n_clarify(self, s):
        names = ", ".join(d.title for d in self.registry.list())
        return {"status": "clarify", "message": f"Which area is this question about? Available: {names}.",
                "trace": _t(s, "clarify", time.perf_counter())}

    def n_concept(self, s):
        d = self.registry.get(s["domain"])
        gen = build_concept(d, s["routing"]["concept_term"])
        return {"status": "concept", "generated": gen.model_dump(), "trace": _t(s, "concept", time.perf_counter())}

    def n_prepare(self, s):
        t0 = time.perf_counter()
        d = self.registry.get(s["domain"])
        req = s["request"]
        variant = req.get("model_variant", "main")
        raw = req.get("instance") or {}
        if not raw:
            return {"status": "error", "message": "Provide the employee's details to explain a risk estimate.",
                    "trace": _t(s, "prepare", t0, error="no instance")}
        inst, errors = d.parse_instance(raw, variant)
        if errors:
            return {"status": "error", "message": "Some fields are invalid: " + "; ".join(errors.values()),
                    "trace": _t(s, "prepare", t0, error="validation")}
        return {"instance": inst, "variant": variant, "assumed": list(req.get("assumed_fields") or []),
                "level": s.get("level") or "intermediate", "trace": _t(s, "prepare", t0)}

    def c_prepare(self, s):
        return "stop" if s.get("status") == "error" else "xai"

    def n_xai(self, s):
        t0 = time.perf_counter()
        d = self.registry.get(s["domain"])
        pkg = build_package(d, s["instance"], s["level"], s["intent"], s["variant"], s.get("assumed"))
        return {"package": pkg.model_dump(), "trace": _t(s, "xai", t0, method=pkg.method, valid=pkg.valid)}

    def c_gate(self, s):
        pkg = s["package"]
        if not pkg["valid"]:
            return "evidence_error"
        pr = pkg["prediction"]
        if pr["band_uncertain"] or pr["out_of_distribution"]:
            return "escalate"
        return "plan"

    def n_evidence_error(self, s):
        failed = [c["name"] for c in s["package"]["checks"] if not c["passed"] and c["severity"] == "critical"]
        return {"status": "error", "message": "Explanation unavailable: the evidence failed validation (" + ", ".join(failed) + ").",
                "trace": _t(s, "evidence_error", time.perf_counter())}

    def n_escalate(self, s):
        d = self.registry.get(s["domain"])
        pr = s["package"]["prediction"]
        if pr["out_of_distribution"]:
            msg = d.policy.escalation_ood.format(reasons="; ".join(pr["ood_reasons"]))
        else:
            rt = d.runtime(s["variant"])
            lo, hi = pr["interval"]
            msg = d.policy.escalation_uncertain.format(low=round(lo * 100), high=round(hi * 100), band_a=rt.band(lo), band_b=rt.band(hi))
        return {"status": "escalated", "message": msg, "notices": [msg], "trace": _t(s, "escalate", time.perf_counter())}

    def n_plan(self, s):
        pkg = EvidencePackage(**s["package"])
        plan = make_plan(pkg, s["level"], s["intent"])
        return {"plan": plan.model_dump(), "attempts": 0, "history": [], "trace": _t(s, "plan", time.perf_counter())}

    def n_generate(self, s):
        t0 = time.perf_counter()
        d = self.registry.get(s["domain"])
        pkg, plan = EvidencePackage(**s["package"]), ExplanationPlan(**s["plan"])
        attempt = s.get("attempts", 0) + 1
        prev = []
        repair = []
        if s.get("history"):
            from .schemas import Violation

            prev = [Violation(**v) for v in s["history"][-1]["violations"] if v["severity"] == "critical"]
            repair = build_repair_instructions(prev, pkg)
        if s.get("use_llm"):
            p = s.get("prompt") or {}
            gen = generate(pkg, plan, d, self.llm, p.get("template", ""), p.get("guidance", []),
                           s["request"].get("query", ""), prev, attempt,
                           repair_instructions=repair, preferences_text=s["request"].get("preferences_text", ""))
        else:
            gen = build_template(pkg, plan, d, variant=attempt % 2)
        return {"generated": gen.model_dump(), "attempts": attempt,
                "trace": _t(s, "generate", t0, attempt=attempt, source=gen.source, repairs=len(repair))}

    def n_verify(self, s):
        t0 = time.perf_counter()
        d = self.registry.get(s["domain"])
        res = verify(EvidencePackage(**s["package"]), ExplanationPlan(**s["plan"]), GeneratedExplanation(**s["generated"]), d)
        hist = [*s.get("history", []), {"attempt": s["attempts"], "stage": "verifier",
                                         "violations": [v.model_dump() for v in res.violations], "passed": res.passed}]
        return {"verification": res.model_dump(), "history": hist,
                "trace": _t(s, "verify", t0, passed=res.passed, codes=sorted({v.code for v in res.violations}))}

    def c_after_check(self, key):
        def fn(s):
            ok = s[key]["passed"] if key == "verification" else s[key]["approved"]
            if ok:
                return "next"
            return "retry" if s["attempts"] < settings.max_attempts else "exhausted"
        return fn

    def n_critic(self, s):
        t0 = time.perf_counter()
        d = self.registry.get(s["domain"])
        res = critique(EvidencePackage(**s["package"]), ExplanationPlan(**s["plan"]), GeneratedExplanation(**s["generated"]),
                       d, self.llm if s.get("use_llm") else None, s["request"].get("query", ""))
        hist = [*s.get("history", []), {"attempt": s["attempts"], "stage": "critic",
                                         "violations": [v.model_dump() for v in res.violations], "passed": res.approved}]
        return {"critic": res.model_dump(), "history": hist,
                "trace": _t(s, "critic", t0, approved=res.approved, codes=sorted({v.code for v in res.violations}))}

    def c_exhausted(self, s):
        return "withheld" if s["request"].get("batch_mode") else "fallback"

    def n_exhausted(self, s):
        return {"trace": _t(s, "retries_exhausted", time.perf_counter())}

    def n_withheld(self, s):
        return {"status": "withheld", "generated": None,
                "message": "Explanation withheld: it did not pass verification after the maximum number of attempts.",
                "trace": _t(s, "withheld", time.perf_counter())}

    def n_fallback(self, s):
        t0 = time.perf_counter()
        d = self.registry.get(s["domain"])
        pkg, plan = EvidencePackage(**s["package"]), ExplanationPlan(**s["plan"])
        gen = build_template(pkg, plan, d)
        ver = verify(pkg, plan, gen, d)
        cri = critique(pkg, plan, gen, d)
        status = "fallback" if ver.passed and cri.approved else "error"
        return {"generated": gen.model_dump(), "verification": ver.model_dump(), "critic": cri.model_dump(), "status": status,
                "trace": _t(s, "fallback", t0, passed=ver.passed and cri.approved)}

    def n_finalize(self, s):
        d = self.registry.get(s["domain"])
        notices = list(s.get("notices") or [])
        status = s.get("status") or "verified"
        if status == "concept":
            return {"trace": _t(s, "finalize", time.perf_counter())}
        pkg = s["package"]
        gen = s["generated"]
        final_text = (gen["explanation"].rstrip() + " " + d.policy.disclosure).strip()
        missing = check_disclosure(final_text, d)
        if missing:  # cannot happen with the fixed template, but the critic enforces it
            return {"status": "error", "message": missing.message, "trace": _t(s, "finalize", time.perf_counter())}
        if pkg["prediction"]["band"] == "high":
            notices.append(d.policy.high_risk_notice)
        if pkg.get("assumptions"):
            notices.append("Not provided, so typical values from the training data were assumed: "
                           + "; ".join(pkg["assumptions"]) + ".")
        if gen.get("source") == "template" and not s.get("use_llm") and status != "fallback":
            notices.append("Written from verified templates because the language model was not used for this request.")
        if status == "fallback":
            notices.append("The language model's drafts did not pass verification, so a verified template is shown.")
        return {"status": status, "disclosure": d.policy.disclosure, "notices": notices,
                "trace": _t(s, "finalize", time.perf_counter())}

    # ------------------------------------------------------------------ wiring
    def _build(self, G):
        # Node names get a suffix: LangGraph rejects node names that equal state keys (e.g. "plan", "critic").
        N = lambda name: END if name == END else f"{name}_node"  # noqa: E731
        g = G(GraphState)
        for name in ["route", "refuse", "clarify", "concept", "prepare", "xai", "evidence_error", "escalate", "plan",
                     "generate", "verify", "critic", "exhausted", "withheld", "fallback", "finalize"]:
            g.add_node(N(name), getattr(self, f"n_{name}"))

        def cond(src, fn, mapping):
            g.add_conditional_edges(N(src), fn, {k: N(v) for k, v in mapping.items()})

        g.add_edge(START, N("route"))
        cond("route", self.c_route, {"refuse": "refuse", "clarify": "clarify", "concept": "concept", "prepare": "prepare"})
        g.add_edge(N("refuse"), END)
        g.add_edge(N("clarify"), END)
        g.add_edge(N("concept"), N("finalize"))
        cond("prepare", self.c_prepare, {"stop": END, "xai": "xai"})
        cond("xai", self.c_gate, {"evidence_error": "evidence_error", "escalate": "escalate", "plan": "plan"})
        g.add_edge(N("evidence_error"), END)
        g.add_edge(N("escalate"), END)
        g.add_edge(N("plan"), N("generate"))
        g.add_edge(N("generate"), N("verify"))
        cond("verify", self.c_after_check("verification"), {"next": "critic", "retry": "generate", "exhausted": "exhausted"})
        cond("critic", self.c_after_check("critic"), {"next": "finalize", "retry": "generate", "exhausted": "exhausted"})
        cond("exhausted", self.c_exhausted, {"withheld": "withheld", "fallback": "fallback"})
        g.add_edge(N("withheld"), END)
        g.add_edge(N("fallback"), N("finalize"))
        g.add_edge(N("finalize"), END)
        return g.compile()

    def run(self, request: dict, level: str, use_llm: bool, prompt: dict) -> dict:
        state = {"request": request, "level": level, "use_llm": use_llm, "prompt": prompt, "trace": [], "notices": []}
        return self.app.invoke(state, {"recursion_limit": 60})
