"""Cross-domain suite: the SAME assertions on every registered domain (HR, finance, healthcare).

Each test runs once per domain in KNOWN_DOMAINS, so adding a domain adds it to this suite automatically.
"""
import json
import random
from pathlib import Path

import pytest

from app.agents.critic import critique
from app.agents.fallback import build_template
from app.agents.planner import make_plan
from app.agents.router import Router
from app.agents.verifier import verify
from app.domains.base import validate_domain
from app.domains.registry import KNOWN_DOMAINS, default_registry
from app.faults import FAULTS, inject
from app.pipeline import PipelineConfig
from app.recourse import find_recourse
from app.schemas import ExplanationRequest
from app.xai.evidence import build_package

DOMAINS = list(KNOWN_DOMAINS)
DATA = Path(__file__).resolve().parents[1] / "eval" / "data"
_REG = default_registry()


def _dom(name):
    return _REG.get(name)


def _xs(d, n=6, variant="main"):
    rt = d.runtime(variant)
    return [{f: float(r[f]) for f in rt.features} for _, r in rt.examples.head(n).iterrows()]


def _raw(d, x):
    return {f: d.readable(f, v) if d.spec(f).categorical else v for f, v in x.items()}


def _codes(pkg, plan, gen, d):
    return {v.code for v in verify(pkg, plan, gen, d).violations + critique(pkg, plan, gen, d).violations
            if v.severity == "critical"}


@pytest.mark.parametrize("name", DOMAINS)
def test_registered_and_contract(name):
    assert name in _REG.names()
    assert validate_domain(_dom(name)) == []


@pytest.mark.parametrize("name", DOMAINS)
def test_templates_clean_and_faults_detected_exactly(name):
    d = _dom(name)
    rng = random.Random(0)
    applied = {f: 0 for f in FAULTS}
    for x in _xs(d, 4):
        for intent in ("explain", "recourse"):
            for level in ("novice", "intermediate", "expert"):
                pkg = build_package(d, x, level, intent)
                plan = make_plan(pkg, level, intent)
                for variant in (0, 1):
                    g = build_template(pkg, plan, d, variant)
                    assert _codes(pkg, plan, g, d) == set(), (name, g.explanation)
                clean = build_template(pkg, plan, d)
                for fault, code in FAULTS.items():
                    bad = inject(fault, clean, pkg, plan, d, rng)
                    if bad is None:
                        continue
                    applied[fault] += 1
                    assert _codes(pkg, plan, bad, d) == {code}, (name, fault, bad.explanation)
    missing = [f for f, n in applied.items() if n == 0 and f != "magnitude_swap"]
    assert not missing, (name, missing)


@pytest.mark.parametrize("name", DOMAINS)
def test_no_raw_codes_and_disclosure(name):
    d = _dom(name)
    from app.service import Service
    from app.store import Store
    import os
    import tempfile

    svc = Service(registry=_REG, store=Store(os.path.join(tempfile.mkdtemp(), "t.sqlite3")))
    x = _xs(d, 3)[1]
    r = svc.explain(ExplanationRequest(query=d.language.default_question, instance=_raw(d, x), level="intermediate",
                                       domain=name), PipelineConfig(escalation=False))
    assert r.status in ("verified", "fallback"), (name, r.status, r.message)
    assert r.disclosure == d.policy.disclosure and r.domain == name
    import re

    text = r.explanation.explanation
    codes = {str(v) for f in d.features if f.categorical for v in f.options.values()
             if re.fullmatch(r"[A-Za-z]+\d+", str(v))}  # dataset codes such as A11 (pure numbers are not codes)
    assert not [c for c in codes if re.search(rf"\b{c}\b", text)], (name, text)


@pytest.mark.parametrize("name", DOMAINS)
def test_protected_attributes_withheld_below_expert(name):
    d = _dom(name)
    if not d.policy.sensitive:
        pytest.skip("no protected attributes in this domain's model")
    x = _xs(d, 2, "audit")[0]
    for level in ("novice", "intermediate"):
        pkg = build_package(d, x, level, "explain", variant="audit")
        assert all(it.withheld for it in pkg.items if it.feature in d.policy.sensitive)
    pkg = build_package(d, x, "expert", "explain", variant="audit")
    assert any(not it.withheld for it in pkg.items if it.feature in d.policy.sensitive)
    assert not set(d.policy.sensitive) & set(d.runtime("main").features), "main model must not use protected attributes"


@pytest.mark.parametrize("name", DOMAINS)
def test_recourse_only_changes_actionable_and_rechecks(name):
    d = _dom(name)
    n = 0
    for x in _xs(d, 12):
        opts, note = find_recourse(d, x)
        for o in opts:
            n += 1
            assert o.rechecked and o.probability_after < o.probability_before
            for ch in o.changes:
                st = d.policy.status(ch.feature)
                assert st in ("actionable", "partial") or (st == "consequence" and ch.consequence), (name, ch)
    assert n > 0, name


@pytest.mark.parametrize("name", DOMAINS)
def test_guardrail_sets(name):
    r = Router(_REG)
    files = {"hr": ["guardrail_cases.json"]}.get(name, [f"guardrail_cases_{name}.json"])
    for fn in files:
        d = json.loads((DATA / fn).read_text())
        assert all(r.guard_rules(q) for q in d["refuse"]), [q for q in d["refuse"] if not r.guard_rules(q)]
        assert not any(r.guard_rules(q) for q in d["allow"]), [q for q in d["allow"] if r.guard_rules(q)]


def test_heldout_guardrail_sets_have_no_false_refusals():
    """Held-out sets are reported, not tuned against: refusal recall is a result, but legitimate questions must pass."""
    r = Router(_REG)
    for p in DATA.glob("guardrail_heldout_*.json"):
        d = json.loads(p.read_text())
        assert len(d["refuse"]) >= 40 and len(d["allow"]) >= 25, p.name
        wrongly = [q for q in d["allow"] if r.guard_rules(q)]
        assert not wrongly, (p.name, wrongly)


@pytest.mark.parametrize("name", DOMAINS)
def test_routing_by_instance(name):
    d = _dom(name)
    x = _xs(d, 1)[0]
    res = Router(_REG).route("Why is this case flagged?", _raw(d, x), None)
    assert res.domain == name, (name, res)


def test_mixed_domain_routing_never_misroutes():
    """Questions without a domain or case details: never sent to the wrong domain; vague ones ask to clarify."""
    r = Router(_REG)
    cases = [c for c in json.loads((DATA / "routing_cases.json").read_text())["cases"]
             if c["domain"] is None or _REG.has(c["domain"])]
    wrong = [(c["query"], c["domain"], r.route(c["query"], None, None).domain) for c in cases]
    wrong = [w for w in wrong if w[2] is not None and w[2] != w[1]]
    assert not wrong, wrong
    vague = [c["query"] for c in cases if c["domain"] is None]
    assert all(r.route(q, None, None).domain is None for q in vague)
