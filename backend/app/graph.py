"""Orchestration graph (LangGraph). Domain-agnostic: no node or edge refers to a specific domain.

route -> (refuse | clarify | concept | prepare)
prepare -> xai -> gate -> (evidence_error | escalate | plan)
plan -> [llm planner, optional] -> generate -> verify -> (critic | generate | exhausted)
critic -> (review | generate | exhausted)
review [advisory reviewer, optional] -> finalize
exhausted -> (fallback -> finalize) | withheld (batch mode or fallback switched off)

Every component can be switched off through PipelineConfig (app/pipeline.py). A switched-off CHECK still runs in
shadow mode: its result is recorded but does not gate the output, so ablations can report what reached the user.
LLM agents (planner, critic score, reviewer) can advise but never turn a verifier rejection into an approval: the
verifier and the policy rules run before them and their verdicts decide the branch.

Each trace entry names the agent that ran (app/agents/registry.py), whether it is deterministic or an LLM, and
for LLM agents the model id.

If the `langgraph` package is not installed, a minimal engine with the same
semantics (node returns a partial state update; conditional edges by function)
is used so the system still runs.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypedDict

from . import policy_report
from .agents import llm_planner, reviewer
from .agents.critic import check_disclosure, critique
from .agents.fallback import build_concept, build_template
from .agents.generator import generate
from .agents.planner import make_plan
from .agents.registry import meta
from .agents.repair import build_repair_instructions
from .agents.router import Router
from .agents.verifier import verify
from .config import settings
from .domains.registry import DomainRegistry
from .escalation import escalation_info
from .llm.client import BaseLLM
from .pipeline import PipelineConfig
from .provenance import build_provenance
from .schemas import EvidencePackage, ExplanationPlan, GeneratedExplanation, Violation
from .xai.evidence import build_package

log = logging.getLogger("mxai.graph")

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
    request: dict  # may carry "preferences_text" (profile agent) and "forced_intent" (follow-up turns)
    use_llm: bool
    prompt: dict
    pipeline: dict
    routing: dict
    domain: str
    level: str
    intent: str
    variant: str
    instance: dict
    assumed: list
    package: dict
    plan: dict
    plan_advice: dict
    generated: dict
    verification: dict
    critic: dict
    review: dict
    attempts: int
    history: list
    status: str
    message: str
    notices: list
    disclosure: str
    policy_report: dict
    provenance: list
    trace: list


def _t(state: dict, node: str, t0: float, **info) -> list:
    entry = {"node": node, **meta(node), "ms": round((time.perf_counter() - t0) * 1000, 1), **info}
    return [*state.get("trace", []), entry]


def _cfg(s: dict) -> PipelineConfig:
    return PipelineConfig.from_dict(s.get("pipeline")) if s.get("pipeline") else PipelineConfig()


def _critical(violations: list[dict]) -> list[dict]:
    return [v for v in violations if v.get("severity") == "critical"]


class ExplanationGraph:
    def __init__(self, registry: DomainRegistry, llm: BaseLLM, use_langgraph: bool | None = None,
                 reviewer_llm: BaseLLM | None = None):
        self.registry = registry
        self.llm = llm
        self._reviewer = reviewer_llm
        self.router = Router(registry, settings.router_confidence_threshold)
        use = HAS_LANGGRAPH if use_langgraph is None else (use_langgraph and HAS_LANGGRAPH)
        self.engine = "langgraph" if use else "minigraph"
        self.app = self._build(StateGraph if use else MiniGraph)

    @property
    def reviewer_llm(self) -> BaseLLM:
        if self._reviewer is None:
            self._reviewer = reviewer.get_reviewer()
        return self._reviewer

    # ------------------------------------------------------------------ nodes
    def n_route(self, s):
        t0 = time.perf_counter()
        req = s["request"]
        r = self.router.route(req.get("query", ""), req.get("instance"), req.get("domain"),
                              self.llm if s.get("use_llm") else None)
        intent = r.intent
        forced = req.get("forced_intent")
        if forced in ("explain", "recourse") and intent != "out_of_scope" and r.domain:
            intent = forced  # follow-up turn: the caller already classified the question
        return {"routing": r.model_dump() | {"intent": intent}, "domain": r.domain, "intent": intent,
                "trace": _t(s, "route", t0, intent=intent, domain=r.domain, confidence=r.confidence,
                            method=r.method, model=self.llm.id if s.get("use_llm") else None)}

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
            return {"status": "error", "message": d.language.missing_instance_message,
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
        cfg = _cfg(s)
        d = self.registry.get(s["domain"])
        pkg = build_package(d, s["instance"], s["level"], s["intent"], s["variant"], s.get("assumed"))
        rt = d.runtime(s["variant"])
        esc = escalation_info(rt, s["instance"], pkg.prediction.model_dump(), cfg.escalation_mode,
                              cfg.escalation_threshold, enabled=cfg.escalation)
        pkg.prediction.escalation = esc
        warn = {c.name: c.detail for c in pkg.checks if c.severity == "warning" and not c.passed}
        if warn:
            log.warning("evidence warnings for package %s: %s", pkg.package_id, warn)
        return {"package": pkg.model_dump(),
                "trace": _t(s, "xai", t0, method=pkg.method, valid=pkg.valid, escalation_score=esc["score"],
                            would_escalate=esc["would_escalate"], warnings=sorted(warn))}

    def c_gate(self, s):
        pkg = s["package"]
        if not pkg["valid"]:
            return "evidence_error"
        if (pkg["prediction"].get("escalation") or {}).get("escalate"):
            return "escalate"
        return "plan"

    def n_evidence_error(self, s):
        failed = [c["name"] for c in s["package"]["checks"] if not c["passed"] and c["severity"] == "critical"]
        return {"status": "error", "message": "Explanation unavailable: the evidence failed validation (" + ", ".join(failed) + ").",
                "trace": _t(s, "evidence_error", time.perf_counter())}

    def n_escalate(self, s):
        d = self.registry.get(s["domain"])
        pr = s["package"]["prediction"]
        esc = pr.get("escalation") or {}
        if pr["out_of_distribution"]:
            msg = d.policy.escalation_ood.format(reasons="; ".join(pr["ood_reasons"]))
        elif pr["band_uncertain"]:
            rt = d.runtime(s["variant"])
            lo, hi = pr["interval"]
            msg = d.policy.escalation_uncertain.format(low=round(lo * 100), high=round(hi * 100), band_a=rt.band(lo), band_b=rt.band(hi))
        else:  # score mode: escalated by the score alone
            msg = d.policy.escalation_ood.format(reasons=f"escalation score {esc.get('score', 0):.2f} is above the "
                                                         f"review threshold {esc.get('threshold')}")
        return {"status": "escalated", "message": msg, "notices": [msg],
                "trace": _t(s, "escalate", time.perf_counter(), score=esc.get("score"), mode=esc.get("mode"))}

    def n_plan(self, s):
        t0 = time.perf_counter()
        cfg = _cfg(s)
        d = self.registry.get(s["domain"])
        pkg = EvidencePackage(**s["package"])
        plan = make_plan(pkg, s["level"], s["intent"])
        trace = _t(s, "plan", t0, allowed=len(plan.allowed_ids))
        advice_record: dict = {}
        if cfg.llm_planner and s.get("use_llm"):
            t1 = time.perf_counter()
            advice = llm_planner.advise(pkg, plan, d, self.llm, s["request"].get("query", ""),
                                        s["request"].get("preferences_text", ""))
            plan, advice_record = llm_planner.constrain(plan, advice, d)
            trace = [*trace, {"node": "llm_plan", **meta("llm_plan"), "model": self.llm.id,
                              "ms": round((time.perf_counter() - t1) * 1000, 1),
                              "accepted_ids": advice_record["accepted_ids"], "rejected_ids": advice_record["rejected_ids"],
                              "notes": len(advice_record["accepted_notes"]), "rejected_notes": len(advice_record["rejected_notes"]),
                              "error": advice_record.get("error")}]
        return {"plan": plan.model_dump(), "plan_advice": advice_record, "attempts": 0, "history": [], "trace": trace}

    def n_generate(self, s):
        t0 = time.perf_counter()
        cfg = _cfg(s)
        d = self.registry.get(s["domain"])
        pkg, plan = EvidencePackage(**s["package"]), ExplanationPlan(**s["plan"])
        attempt = s.get("attempts", 0) + 1
        prev: list[Violation] = []
        repair: list[str] = []
        if s.get("history") and cfg.repair:
            # feedback from every gate that blocked the previous attempt
            last_attempt = s["history"][-1]["attempt"]
            blocking = [v for h in s["history"] if h["attempt"] == last_attempt for v in h.get("blocking", [])]
            prev = [Violation(**v) for v in blocking]
            repair = build_repair_instructions(prev, pkg, d.language)
        if s.get("use_llm"):
            p = s.get("prompt") or {}
            gen = generate(pkg, plan, d, self.llm, p.get("template", ""), p.get("guidance", []),
                           s["request"].get("query", ""), prev if cfg.repair else None, attempt,
                           repair_instructions=repair, preferences_text=s["request"].get("preferences_text", ""))
        else:
            gen = build_template(pkg, plan, d, variant=attempt % 2)
        return {"generated": gen.model_dump(), "attempts": attempt,
                "trace": _t(s, "generate", t0, attempt=attempt, source=gen.source, repairs=len(repair),
                            model=self.llm.id if s.get("use_llm") else "template")}

    def n_verify(self, s):
        t0 = time.perf_counter()
        cfg = _cfg(s)
        d = self.registry.get(s["domain"])
        res = verify(EvidencePackage(**s["package"]), ExplanationPlan(**s["plan"]), GeneratedExplanation(**s["generated"]), d)
        res.gated = cfg.verifier
        vs = [v.model_dump() for v in res.violations]
        hist = [*s.get("history", []), {"attempt": s["attempts"], "stage": "verifier", "gated": cfg.verifier,
                                         "violations": vs, "blocking": _critical(vs) if cfg.verifier else [],
                                         "passed": res.passed}]
        return {"verification": res.model_dump(), "history": hist,
                "trace": _t(s, "verify", t0, passed=res.passed, gated=cfg.verifier, cfs=res.fidelity_score,
                            codes=sorted({v.code for v in res.violations}))}

    def _max_attempts(self, s) -> int:
        return _cfg(s).max_attempts or settings.max_attempts

    def c_after_verify(self, s):
        v = s["verification"]
        if v["passed"] or not v.get("gated", True):
            return "next"
        return "retry" if s["attempts"] < self._max_attempts(s) else "exhausted"

    def _critic_gate(self, s, cri: dict, d) -> bool:
        """Approval counting only the categories whose switches are on."""
        return not any(self._blocks(s, v, d) for v in cri["violations"])

    def n_critic(self, s):
        t0 = time.perf_counter()
        cfg = _cfg(s)
        d = self.registry.get(s["domain"])
        use_llm_critic = s.get("use_llm") and cfg.llm_critic
        res = critique(EvidencePackage(**s["package"]), ExplanationPlan(**s["plan"]), GeneratedExplanation(**s["generated"]),
                       d, self.llm if use_llm_critic else None, s["request"].get("query", ""))
        cri = res.model_dump()
        cri["gated_approved"] = self._critic_gate(s, cri, d)
        gated = cfg.policy_rules or cfg.level_rules or cfg.llm_critic
        hist = [*s.get("history", []), {"attempt": s["attempts"], "stage": "critic", "gated": gated,
                                         "violations": cri["violations"],
                                         "blocking": [v for v in cri["violations"] if self._blocks(s, v, d)],
                                         "passed": cri["gated_approved"]}]
        return {"critic": cri, "history": hist,
                "trace": _t(s, "critic", t0, approved=cri["gated_approved"], all_rules_approved=res.approved,
                            codes=sorted({v.code for v in res.violations}), scores=res.scores or None,
                            model=self.llm.id if use_llm_critic else None)}

    def _blocks(self, s, v: dict, d) -> bool:
        if v.get("severity") != "critical":
            return False
        cfg = _cfg(s)
        cat = policy_report.category(v["code"], d)
        return {"policy": cfg.policy_rules, "level": cfg.level_rules, "quality": cfg.llm_critic}.get(cat, True)

    def c_after_critic(self, s):
        if s["critic"]["gated_approved"]:
            return "next"
        return "retry" if s["attempts"] < self._max_attempts(s) else "exhausted"

    def n_review(self, s):
        cfg = _cfg(s)
        if not (cfg.reviewer and s.get("use_llm")):
            return {}
        t0 = time.perf_counter()
        d = self.registry.get(s["domain"])
        rv = reviewer.review(self.reviewer_llm, EvidencePackage(**s["package"]), ExplanationPlan(**s["plan"]),
                             GeneratedExplanation(**s["generated"]), d, generator_id=self.llm.id)
        return {"review": rv, "trace": _t(s, "review", t0, model=rv["model"], faithful=rv["faithful"],
                                          issues=len(rv["issues"]), same_family=rv["same_family_as_generator"])}

    def c_exhausted(self, s):
        return "withheld" if s["request"].get("batch_mode") or not _cfg(s).fallback else "fallback"

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
        status = "fallback" if ver.passed and cri.approved else "error"  # templates are always fully gated
        cri_d = cri.model_dump() | {"gated_approved": cri.approved}
        return {"generated": gen.model_dump(), "verification": ver.model_dump(), "critic": cri_d, "status": status,
                "trace": _t(s, "fallback", t0, passed=ver.passed and cri.approved)}

    def n_finalize(self, s):
        d = self.registry.get(s["domain"])
        notices = list(s.get("notices") or [])
        status = s.get("status") or "verified"
        if status == "concept":
            return {"trace": _t(s, "finalize", time.perf_counter())}
        pkg = s["package"]
        gen = s["generated"]
        # The disclosure is attached here, by construction, to every explanation that is shown. check_disclosure
        # is a guard against a future code change that drops it, not an independent test of generated text.
        final_text = (gen["explanation"].rstrip() + " " + d.policy.disclosure).strip()
        missing = check_disclosure(final_text, d)
        if missing:
            return {"status": "error", "message": missing.message, "trace": _t(s, "finalize", time.perf_counter())}
        # what the shown text still violates (non-empty only when a gate ran in shadow mode)
        shown = _critical((s.get("verification") or {}).get("violations", []) + (s.get("critic") or {}).get("violations", []))
        if status == "verified" and shown:
            status = "unverified"
            notices.append("Ablation run: a checking component was switched off and the shown text did not pass it.")
        if pkg["prediction"]["band"] == "high":
            notices.append(d.policy.high_risk_notice)
        if pkg.get("assumptions"):
            notices.append("Not provided, so typical values from the training data were assumed: "
                           + "; ".join(pkg["assumptions"]) + ".")
        if gen.get("source") == "template" and not s.get("use_llm") and status != "fallback":
            notices.append("Written from verified templates because the language model was not used for this request.")
        if status == "fallback":
            notices.append("The language model's drafts did not pass verification, so a verified template is shown.")
        report = policy_report.summarize(s.get("history") or [], shown, d)
        prov = build_provenance(gen, pkg, d, plan=s.get("plan"))
        return {"status": status, "disclosure": d.policy.disclosure, "notices": notices, "policy_report": report,
                "provenance": prov,
                "trace": _t(s, "finalize", time.perf_counter(), shown_violations=len(shown))}

    # ------------------------------------------------------------------ wiring
    def _build(self, G):
        # Node names get a suffix: LangGraph rejects node names that equal state keys (e.g. "plan", "critic").
        N = lambda name: END if name == END else f"{name}_node"  # noqa: E731
        g = G(GraphState)
        for name in ["route", "refuse", "clarify", "concept", "prepare", "xai", "evidence_error", "escalate", "plan",
                     "generate", "verify", "critic", "review", "exhausted", "withheld", "fallback", "finalize"]:
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
        cond("verify", self.c_after_verify, {"next": "critic", "retry": "generate", "exhausted": "exhausted"})
        cond("critic", self.c_after_critic, {"next": "review", "retry": "generate", "exhausted": "exhausted"})
        g.add_edge(N("review"), N("finalize"))
        cond("exhausted", self.c_exhausted, {"withheld": "withheld", "fallback": "fallback"})
        g.add_edge(N("withheld"), END)
        g.add_edge(N("fallback"), N("finalize"))
        g.add_edge(N("finalize"), END)
        return g.compile()

    def run(self, request: dict, level: str, use_llm: bool, prompt: dict,
            pipeline: PipelineConfig | dict | None = None) -> dict:
        cfg = pipeline if isinstance(pipeline, PipelineConfig) else PipelineConfig.from_dict(pipeline)
        state = {"request": request, "level": level, "use_llm": use_llm, "prompt": prompt, "trace": [], "notices": [],
                 "pipeline": cfg.as_dict()}
        return self.app.invoke(state, {"recursion_limit": 80})
