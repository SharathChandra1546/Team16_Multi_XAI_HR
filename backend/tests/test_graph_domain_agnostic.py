"""The same graph and verification code route and serve a second, unrelated domain with no changes."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from app.domains.base import Action, Actionability, DomainAgent, DomainPolicy, FeatureSpec, TextRule
from app.domains.registry import DomainRegistry
from app.ml.runtime import ModelRuntime
from app.ml.training import TrainConfig, train_and_validate
from app.schemas import ExplanationRequest, GlossaryEntry

APP = Path(__file__).resolve().parents[1] / "app"
GENERIC = ["graph.py", "recourse.py", "agents/verifier.py", "agents/critic.py", "agents/router.py", "agents/planner.py",
           "agents/generator.py", "agents/fallback.py", "xai/evidence.py", "xai/validator.py", "service.py"]


def test_no_domain_specific_code_in_generic_modules():
    for f in GENERIC:
        src = (APP / f).read_text()
        assert not re.search(r"['\"]hr['\"]|HrAgent|OverTime|Attrition", src), f


class ToyDomain(DomainAgent):
    name = "toy"
    title = "Toy loan default"
    routing_keywords = ["loan", "default", "borrower", "credit"]
    positive_label, negative_label = "likely to default", "likely to repay"
    features = [
        FeatureSpec("income", "Income", "Yearly income.", "numeric", "T-INCOME", min=0, max=1000, aliases=["income"]),
        FeatureSpec("debt", "Debt ratio", "Debt to income ratio.", "numeric", "T-DEBT", min=0, max=100, step=0.1, aliases=["debt ratio"]),
        FeatureSpec("late", "Late payments", "Late payments last year.", "numeric", "T-LATE", min=0, max=50, aliases=["late payments"]),
        FeatureSpec("owner", "Home owner", "Owns a home.", "binary", "T-OWNER", options={0: "No", 1: "Yes"},
                    labels={0: "does not own a home", 1: "owns a home"}, aliases=["home owner"]),
    ]
    glossary = {f.glossary_id: GlossaryEntry(id=f.glossary_id, term=f.display, definition=f.description) for f in features}
    policy = DomainPolicy(
        sensitive=[], actionability={"debt": Actionability("actionable"), "late": Actionability("immutable"),
                                     "income": Actionability("immutable"), "owner": Actionability("immutable")},
        disclosure="This is a model-based estimate, not a decision; a person must review it.",
        high_risk_notice="High estimate: review required.", escalation_uncertain="Uncertain {low} {high} {band_a} {band_b}",
        escalation_ood="Unusual ({reasons})", refusal_message="Decision requests are refused.",
        guard_patterns=[r"\bdeny (?:the )?loan\b"], critic_rules=[TextRule("decision_wording", [r"\bdeny\b"], "decision")],
    )

    def __init__(self):
        rng = np.random.default_rng(1)
        n = 600
        X = pd.DataFrame({"income": rng.integers(20, 200, n).astype(float), "debt": rng.uniform(0, 60, n).round(1),
                          "late": rng.integers(0, 6, n).astype(float), "owner": rng.integers(0, 2, n).astype(float)})
        logit = -2 + 0.05 * X.debt + 0.5 * X.late - 0.01 * X.income - 0.5 * X.owner
        y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
        art, man = train_and_validate(X, y, TrainConfig(features=list(X.columns), nominal=[], candidates=["logreg"]),
                                      log=lambda *_: None, fast=True)
        art["version"] = "toy-1"
        self.rt = ModelRuntime(art, man)

    def runtime(self, variant="main"):
        return self.rt

    def recourse_actions(self, x, variant="main"):
        return [Action(f"debt_{k}", f"lowering the debt ratio to {max(0, x['debt'] - k):g}", {"debt": float(max(0, x["debt"] - k))},
                       ["debt"]) for k in (10, 20)]


def test_same_graph_serves_two_domains(hr):
    from app.service import Service
    from app.store import Store
    import tempfile, os

    reg = DomainRegistry()
    reg.register(hr)
    toy = ToyDomain()
    reg.register(toy)
    svc = Service(registry=reg, store=Store(os.path.join(tempfile.mkdtemp(), "t.sqlite3")))
    rt = toy.runtime()
    ex = rt.examples.iloc[0]
    inst = {f: float(ex[f]) for f in rt.features}
    r = svc.explain(ExplanationRequest(query="Why is this borrower likely to default on the loan?", instance=inst, level="intermediate"))
    assert r.domain == "toy" and r.status in ("verified", "escalated"), (r.status, r.message)
    if r.status == "verified":
        assert r.verification.passed and r.disclosure == toy.policy.disclosure
    r2 = svc.explain(ExplanationRequest(query="What would lower the default risk for this loan borrower?", instance=inst, domain="toy"))
    assert r2.intent == "recourse"
    r3 = svc.explain(ExplanationRequest(query="Should we deny the loan?", instance=inst))
    assert r3.status == "refused"
    hr_rt = hr.runtime("main")
    hx = {f: (hr.readable(f, hr_rt.examples.iloc[0][f]) if hr.spec(f).categorical else float(hr_rt.examples.iloc[0][f]))
          for f in hr_rt.features}
    r4 = svc.explain(ExplanationRequest(query="Why is this employee flagged?", instance=hx))
    assert r4.domain == "hr"


def test_langgraph_engine_if_installed(hr, instances):
    """Runs the real LangGraph engine when the package is installed (it is in requirements.txt)."""
    import pytest
    from app.graph import HAS_LANGGRAPH, ExplanationGraph
    from app.domains.registry import DomainRegistry
    from app.llm.client import get_llm
    from app.llm import mock as mock_llm

    if not HAS_LANGGRAPH:
        pytest.skip("langgraph not installed")
    reg = DomainRegistry()
    reg.register(hr)
    mock_llm.DOMAIN_RESOLVER = reg.get
    g = ExplanationGraph(reg, get_llm("mock"), use_langgraph=True)
    assert g.engine == "langgraph"
    raw = {f: hr.readable(f, v) if hr.spec(f).categorical else v for f, v in instances[0].items()}
    for q in ["Why is this employee flagged?", "What would lower their risk?", "Should we fire them?", "What does attrition mean?"]:
        s = g.run({"query": q, "instance": raw, "domain": None}, "intermediate", True, {"template": "", "guidance": []})
        assert s.get("status") in ("verified", "escalated", "refused", "concept"), (q, s.get("status"), s.get("message"))


def test_library_explainers_if_installed(hr, instances):
    """With shap and lime installed, the library explainers are used and still pass the evidence validator."""
    import pytest
    from app.xai.lime_local import HAS_LIME
    from app.xai.shapley import HAS_SHAP
    from app.xai.evidence import build_package

    if not (HAS_SHAP and HAS_LIME):
        pytest.skip("shap/lime not installed")
    pkg = build_package(hr, instances[0], "expert", "explain")
    assert "shap.PermutationExplainer" in pkg.method and "lime.LimeTabularExplainer" in pkg.method, pkg.method
    assert pkg.valid, [c for c in pkg.checks if not c.passed]
