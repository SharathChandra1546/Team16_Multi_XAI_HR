"""Phase 1: pipeline switches (shadow mode), LLM agents cannot override gates, escalation score, agent trace."""
import os
import random
import tempfile

from app.agents.fallback import build_template
from app.escalation import escalation_info, uncertainty_component
from app.faults import inject
from app.llm.client import BaseLLM
from app.pipeline import ABLATIONS, PipelineConfig, ablation
from app.schemas import EvidencePackage, ExplanationPlan, ExplanationRequest
from app.service import Service
from app.store import Store

NO_ESC = {"escalation": False}  # keep cases explainable so the generation loop always runs


def _raw(hr, x):
    return {f: hr.readable(f, v) if hr.spec(f).categorical else v for f, v in x.items()}


class ScriptedLLM(BaseLLM):
    """Generator that injects a fault on chosen attempts; critic and reviewer always approve.

    Used to show that LLM agents (critic score 5/5, reviewer 'faithful') can never turn a verifier rejection into
    an approved output.
    """

    name, model = "scripted", "fault-on-attempts"

    def __init__(self, fault: str, bad_attempts: set[int] | None = None, review_faithful: bool = True):
        self.fault, self.bad_attempts, self.review_faithful = fault, bad_attempts, review_faithful
        self.registry = None

    def json(self, task, system, user, payload=None):
        payload = payload or {}
        if task == "generate":
            pkg, plan = EvidencePackage(**payload["package"]), ExplanationPlan(**payload["plan"])
            d = self.registry.get(payload["domain"])
            gen = build_template(pkg, plan, d, variant=payload["attempt"] % 2)
            if self.bad_attempts is None or payload["attempt"] in self.bad_attempts:
                bad = inject(self.fault, gen, pkg, plan, d, random.Random(payload["attempt"]))
                gen = bad or gen
            return {"explanation": gen.explanation, "claims": [c.model_dump() for c in gen.claims]}
        if task == "critic":
            return {"clarity": 5, "level_fit": 5, "relevance": 5, "feedback": "excellent"}
        if task == "review":
            return {"faithful": self.review_faithful, "issues": [] if self.review_faithful else ["made-up issue"]}
        if task == "guard":
            return {"decision_request": False}
        if task == "plan":
            return {"focus_ids": [c["id"] for c in payload.get("candidates", [])], "notes": []}
        return {}


def _svc(registry, llm):
    llm.registry = registry
    return Service(registry=registry, store=Store(os.path.join(tempfile.mkdtemp(), "t.sqlite3")), llm=llm,
                   reviewer_llm=llm, pipeline=PipelineConfig(**NO_ESC))


def _req(hr, x, q=None):
    return ExplanationRequest(query=q or hr.language.default_question, instance=_raw(hr, x), level="intermediate",
                              domain="hr")


def test_named_ablations():
    assert set(ABLATIONS) >= {"full", "no_verifier", "no_critic", "no_repair", "no_escalation", "no_fallback"}
    assert ablation("no_verifier").disabled() == ["verifier"]
    assert PipelineConfig().disabled() == []
    assert PipelineConfig.from_dict({"verifier": False, "bogus": 1}).verifier is False


def test_verifier_gate_retries_then_verifies(registry, hr, instances):
    svc = _svc(registry, ScriptedLLM("direction_flip", bad_attempts={1}))
    r = svc.explain(_req(hr, instances[0]), PipelineConfig(**NO_ESC))
    assert r.status == "verified" and r.attempts == 2, (r.status, r.attempts)
    assert "direction_mismatch" in r.policy_report["first_attempt"]["fidelity"]
    assert r.policy_report["shown"] == {} and r.verification.fidelity_score == 1.0


def test_verifier_off_is_shadow_mode(registry, hr, instances):
    svc = _svc(registry, ScriptedLLM("direction_flip", bad_attempts={1}))
    r = svc.explain(_req(hr, instances[0]), PipelineConfig(verifier=False, **NO_ESC))
    assert r.status == "unverified" and r.attempts == 1
    assert r.verification.gated is False and not r.verification.passed
    assert r.policy_report["shown_fidelity_violations"] >= 1


def test_policy_rules_off_lets_policy_violation_through(registry, hr, instances):
    svc = _svc(registry, ScriptedLLM("decision_wording", bad_attempts={1}))
    on = svc.explain(_req(hr, instances[0]), PipelineConfig(**NO_ESC))
    assert on.status == "verified" and on.policy_report["first_attempt"]["policy"] == {"decision_wording": 1}
    off = svc.explain(_req(hr, instances[0]), PipelineConfig(policy_rules=False, **NO_ESC))
    assert off.status == "unverified" and off.policy_report["shown_policy_violations"] == 1


def test_llm_agents_cannot_override_the_verifier(registry, hr, instances):
    """Generator always wrong; LLM critic scores 5/5 and reviewer says faithful: the output is still not verified."""
    svc = _svc(registry, ScriptedLLM("value_swap", bad_attempts=None))
    cfg = PipelineConfig(reviewer=True, llm_planner=True, **NO_ESC)
    for x in instances[:3]:
        r = svc.explain(_req(hr, x), cfg)
        assert r.status == "fallback", r.status  # verified template, never the faulty LLM text
        assert r.explanation.source == "template" and r.verification.passed
        assert "value_mismatch" in r.policy_report["all_attempts"]["fidelity"]


def test_no_fallback_withholds(registry, hr, instances):
    svc = _svc(registry, ScriptedLLM("value_swap", bad_attempts=None))
    r = svc.explain(_req(hr, instances[0]), PipelineConfig(fallback=False, **NO_ESC))
    assert r.status == "withheld" and r.explanation is None


def test_no_repair_sends_no_instructions(registry, hr, instances):
    svc = _svc(registry, ScriptedLLM("direction_flip", bad_attempts={1}))
    on = svc.explain(_req(hr, instances[0]), PipelineConfig(**NO_ESC))
    off = svc.explain(_req(hr, instances[0]), PipelineConfig(repair=False, **NO_ESC))
    gen_on = [t for t in on.trace if t["node"] == "generate"]
    gen_off = [t for t in off.trace if t["node"] == "generate"]
    assert gen_on[1]["repairs"] >= 1 and gen_off[1]["repairs"] == 0


def test_reviewer_is_advisory(registry, hr, instances):
    svc = _svc(registry, ScriptedLLM("direction_flip", bad_attempts=set(), review_faithful=False))
    r = svc.explain(_req(hr, instances[0]), PipelineConfig(reviewer=True, **NO_ESC))
    assert r.status == "verified" and r.review["faithful"] is False and r.review["issues"]
    assert any(t["node"] == "review" and t["kind"] == "llm" and t["authority"] == "advise" for t in r.trace)


def test_trace_names_agents(service, hr, instances):
    r = service.explain(_req(hr, instances[1]), PipelineConfig(**NO_ESC))
    nodes = {t["node"]: t for t in r.trace}
    assert nodes["profile"]["kind"] == "deterministic"
    assert nodes["verify"]["kind"] == "deterministic" and nodes["verify"]["authority"] == "gate"
    assert nodes["generate"]["kind"] == "llm" and nodes["generate"]["model"]
    assert r.pipeline["verifier"] is True and r.level_source == "explicit"


def test_llm_planner_runs_and_is_logged(service, hr, instances):
    r = service.explain(_req(hr, instances[1]), PipelineConfig(llm_planner=True, **NO_ESC))
    entry = next(t for t in r.trace if t["node"] == "llm_plan")
    assert entry["kind"] == "llm" and entry["rejected_ids"] == []
    assert r.status in ("verified", "fallback")


# ---------------------------------------------------------------- escalation
def test_uncertainty_component():
    assert uncertainty_component(0.2, 0.15, 0.25, "low", 0.3, 0.6) == 0.0
    assert abs(uncertainty_component(0.25, 0.2, 0.4, "low", 0.3, 0.6) - 0.5) < 1e-9
    assert uncertainty_component(0.5, 0.5, 0.5, "medium", 0.3, 0.6) == 0.0


def test_escalation_rule_mode_matches_original_rule(hr, instances):
    rt = hr.runtime("main")
    for x in instances[:20]:
        pred = rt.predict(x)
        info = escalation_info(rt, x, pred)
        assert info["would_escalate"] == bool(pred["band_uncertain"] or pred["ood_reasons"])
        assert 0.0 <= info["score"] <= 1.0
        assert (info["uncertainty"] > 0) == pred["band_uncertain"]
        never = escalation_info(rt, x, pred, mode="score", threshold=1.01)
        assert never["would_escalate"] is False


def test_escalation_score_logged_and_switch_off_explains(service, hr, instances):
    rt = hr.runtime("main")
    x = next((x for x in instances if rt.predict(x)["band_uncertain"]), None)
    assert x is not None, "no uncertain case in the examples"
    r = service.explain(_req(hr, x), PipelineConfig())
    assert r.status == "escalated" and r.prediction.escalation["escalate"]
    r2 = service.explain(_req(hr, x), PipelineConfig(escalation=False))
    assert r2.status in ("verified", "fallback")
    assert r2.prediction.escalation["would_escalate"] and not r2.prediction.escalation["escalate"]
    r3 = service.explain(_req(hr, x), PipelineConfig(escalation_mode="score", escalation_threshold=1.01))
    assert r3.status in ("verified", "fallback") and r3.prediction.escalation["mode"] == "score"
