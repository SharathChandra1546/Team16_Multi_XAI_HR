"""Verifier: templates pass; each injected fault is detected with exactly its expected code."""
import random

from app.agents.critic import critique
from app.agents.fallback import build_template
from app.agents.planner import make_plan
from app.agents.verifier import verify
from app.faults import FAULTS, inject
from app.xai.evidence import build_package

LEVELS = ["novice", "intermediate", "expert"]


def _cases(hr, instances, n=6):
    for x in instances[:n]:
        for intent in ("explain", "recourse"):
            for level in LEVELS:
                pkg = build_package(hr, x, level, intent)
                yield pkg, make_plan(pkg, level, intent)


def _codes(pkg, plan, gen, hr):
    v = verify(pkg, plan, gen, hr)
    c = critique(pkg, plan, gen, hr)
    return {x.code for x in v.violations + c.violations if x.severity == "critical"}


def test_templates_are_clean(hr, instances):
    for pkg, plan in _cases(hr, instances):
        for variant in (0, 1):
            gen = build_template(pkg, plan, hr, variant)
            assert _codes(pkg, plan, gen, hr) == set(), gen.explanation


def test_each_fault_detected_with_one_code(hr, instances):
    rng = random.Random(0)
    applied = {f: 0 for f in FAULTS}
    for pkg, plan in _cases(hr, instances):
        clean = build_template(pkg, plan, hr)
        for fault, code in FAULTS.items():
            bad = inject(fault, clean, pkg, plan, hr, rng)
            if bad is None:
                continue
            applied[fault] += 1
            codes = _codes(pkg, plan, bad, hr)
            assert codes == {code}, f"{fault}: expected {{{code}}}, got {codes}\n{bad.explanation}"
    assert all(n > 0 for n in applied.values()), applied


def test_claim_must_cite_and_match(hr, package_and_plan):
    pkg, plan = package_and_plan
    gen = build_template(pkg, plan, hr)
    gen.claims[1].evidence_ids = ["E1", "ZZ"] if gen.claims[1].evidence_ids != ["E1"] else ["ZZ"]
    assert not verify(pkg, plan, gen, hr).passed


def test_text_not_backed_by_claim_fails(hr, package_and_plan):
    pkg, plan = package_and_plan
    gen = build_template(pkg, plan, hr)
    top = pkg.visible[0]
    gen.explanation += f" {top.display_name} is also worth noting."
    codes = {v.code for v in verify(pkg, plan, gen, hr).violations}
    assert "uncovered_feature_mention" in codes


def test_causal_wording_rule(hr, package_and_plan):
    pkg, plan = package_and_plan
    gen = build_template(pkg, plan, hr)
    for bad in ["Overtime causes attrition.", "Low pay leads to resignations.", "Removing overtime will prevent departures."]:
        g = gen.model_copy(deep=True)
        g.explanation += " " + bad
        assert "causal_wording" in {v.code for v in verify(pkg, plan, g, hr).violations}, bad
    ok = gen.model_copy(deep=True)
    ok.explanation += " This does not mean overtime causes attrition."
    assert "causal_wording" not in {v.code for v in verify(pkg, plan, ok, hr).violations}


def test_raw_code_rule(hr, package_and_plan):
    pkg, plan = package_and_plan
    gen = build_template(pkg, plan, hr)
    bad = ["Overtime: yes.", "Job satisfaction of 1 was noted.", "Overtime (Yes) matters.", "OverTime=Yes was recorded.",
           "Business travel is Travel_Frequently."]
    good = ["Overtime (no overtime) matters less.", "Job satisfaction (High) lowered the estimate.", "Job level (level 2) is shown."]
    for s in bad:
        g = gen.model_copy(deep=True)
        g.explanation += " " + s
        assert "raw_code_leak" in {v.code for v in verify(pkg, plan, g, hr).violations}, s
    for s in good:
        g = gen.model_copy(deep=True)
        g.explanation += " " + s
        assert "raw_code_leak" not in {v.code for v in verify(pkg, plan, g, hr).violations}, s
