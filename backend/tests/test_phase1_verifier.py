"""Phase 1: verifier hardening (text-level magnitude, rank, risk checks), CFS v2, and level rules."""
import random

from app.agents.critic import level_violations
from app.agents.fallback import build_template
from app.agents.planner import make_plan
from app.agents.verifier import verify
from app.faults import FAULTS, inject
from app.metrics.fidelity import claim_fidelity_score
from app.schemas import Claim, Violation
from app.xai.evidence import build_package

LEVELS = ["novice", "intermediate", "expert"]


def _pkg(hr, x, level="intermediate", intent="explain"):
    pkg = build_package(hr, x, level, intent)
    return pkg, make_plan(pkg, level, intent)


def _codes(res):
    return {v.code for v in res.violations if v.severity == "critical"}


# ---------------------------------------------------------------- text-level claim checks
def test_magnitude_words_must_match_evidence(hr, instances):
    pkg, plan = _pkg(hr, instances[0])
    gen = build_template(pkg, plan, hr)
    for c in gen.claims:
        if c.kind != "attribution":
            continue
        it = pkg.item(c.evidence_ids[0])
        wrong = {"strong": "slightly", "moderate": "slightly", "weak": "strongly"}[it.magnitude]
        right = {"strong": "strongly", "moderate": "moderately", "weak": "slightly"}[it.magnitude]
        g = gen.model_copy(deep=True)
        cc = g.claims[gen.claims.index(c)]
        new = cc.text.replace(right, wrong)
        g.explanation = g.explanation.replace(cc.text, new)
        cc.text, cc.magnitude = new, None
        assert "magnitude_mismatch" in _codes(verify(pkg, plan, g, hr)), new
        return
    raise AssertionError("no attribution claim")


def test_ordinal_rank_words(hr, instances):
    for x in instances[:10]:
        pkg, plan = _pkg(hr, x)
        gen = build_template(pkg, plan, hr)
        by_rank = {pkg.item(c.evidence_ids[0]).rank: i for i, c in enumerate(gen.claims) if c.kind == "attribution"}
        if 2 not in by_rank or 3 not in by_rank:
            continue
        for rank, expect_bad in ((2, False), (3, True)):
            g = gen.model_copy(deep=True)
            c = g.claims[by_rank[rank]]
            new = c.text.rstrip(".") + ", the second largest factor."
            g.explanation = g.explanation.replace(c.text, new)
            c.text = new
            assert ("rank_mismatch" in _codes(verify(pkg, plan, g, hr))) == expect_bad, new
        return
    raise AssertionError("no case with ranks 2 and 3")


def test_comparison_between_two_cited_factors(hr, instances):
    pkg, plan = _pkg(hr, instances[0])
    gen = build_template(pkg, plan, hr)
    a, b = pkg.visible[0], pkg.visible[1]  # |a| >= |b|
    for text, bad in [(f"{a.display_name} moved the estimate more than {b.display_name.lower()} did.", False),
                      (f"{b.display_name} moved the estimate more than {a.display_name.lower()} did.", True),
                      (f"{b.display_name} moved the estimate less than {a.display_name.lower()} did.", False)]:
        g = gen.model_copy(deep=True)
        g.explanation += " " + text
        g.claims.append(Claim(text=text, kind="attribution", evidence_ids=[a.id, b.id]))
        assert ("rank_mismatch" in _codes(verify(pkg, plan, g, hr))) == bad, text


def test_risk_sentence_band_and_percentage(hr, instances):
    pkg, plan = _pkg(hr, instances[0])
    gen = build_template(pkg, plan, hr)
    risk = gen.claims[0]
    pr = pkg.prediction
    other = next(b for b in ("low", "medium", "high") if b != pr.band)
    g = gen.model_copy(deep=True)
    new = risk.text.replace(f"({pr.band} band)", f"({other} band)")
    g.explanation = g.explanation.replace(risk.text, new)
    g.claims[0].text = new
    assert "risk_mismatch" in _codes(verify(pkg, plan, g, hr))
    # a percentage qualified as an average or a range bound is not the headline figure
    ok = (f"Compared with the {round(pkg.base_value * 100)}% average, the model's risk estimate is "
          f"{round(pr.probability * 100)}% ({pr.band} band).")
    g = gen.model_copy(deep=True)
    g.explanation = g.explanation.replace(risk.text, ok)
    g.claims[0].text = ok
    assert "risk_mismatch" not in _codes(verify(pkg, plan, g, hr)), ok


def test_new_template_faults_apply_and_are_detected(hr, instances):
    rng = random.Random(1)
    seen = {f: 0 for f in ("magnitude_swap", "risk_text_swap", "wrong_ordinal")}
    for x in instances[:8]:
        for level in LEVELS:
            pkg, plan = _pkg(hr, x, level)
            clean = build_template(pkg, plan, hr)
            for f in seen:
                bad = inject(f, clean, pkg, plan, hr, rng)
                if bad is None:
                    continue
                seen[f] += 1
                assert FAULTS[f] in _codes(verify(pkg, plan, bad, hr)), (f, bad.explanation)
    assert all(seen.values()), seen


# ---------------------------------------------------------------- CFS v2
def test_cfs_counts_text_level_violations(hr, instances):
    pkg, plan = _pkg(hr, instances[0])
    gen = build_template(pkg, plan, hr)
    clean = verify(pkg, plan, gen, hr)
    assert clean.passed and clean.fidelity_score == 1.0 and clean.fidelity["n_units"] == len(gen.claims)
    g = gen.model_copy(deep=True)
    g.explanation += " This pattern appeared in 83.7% of similar records. Conditions like these cause attrition."
    res = verify(pkg, plan, g, hr)
    assert not res.passed
    assert res.fidelity["n_unbound_sentences"] == 2
    assert res.fidelity_score == round(len(g.claims) / (len(g.claims) + 2), 4) < 1.0


def test_cfs_binds_span_to_claim_and_counts_missing_required(hr, instances):
    pkg, plan = _pkg(hr, instances[0])
    gen = build_template(pkg, plan, hr)
    g = gen.model_copy(deep=True)
    c = g.claims[1]
    new = c.text.rstrip(".") + ", which causes attrition."
    g.explanation = g.explanation.replace(c.text, new)
    c.text = new
    res = verify(pkg, plan, g, hr)
    assert res.fidelity["n_unbound_sentences"] == 0 and res.fidelity["n_clean"] == len(g.claims) - 1
    g2 = gen.model_copy(deep=True)
    g2.claims = [cl for cl in g2.claims if cl.kind != "risk"]
    g2.explanation = " ".join(cl.text for cl in g2.claims)
    res2 = verify(pkg, plan, g2, hr)
    assert res2.fidelity["n_missing_required"] >= 1 and res2.fidelity_score < 1.0


def test_cfs_is_one_exactly_when_verifier_passes(hr, instances):
    rng = random.Random(0)
    for x in instances[:4]:
        for level in LEVELS:
            for intent in ("explain", "recourse"):
                pkg, plan = _pkg(hr, x, level, intent)
                clean = build_template(pkg, plan, hr)
                for g in [clean] + [inject(f, clean, pkg, plan, hr, rng) for f in FAULTS]:
                    if g is None:
                        continue
                    res = verify(pkg, plan, g, hr)
                    assert (res.fidelity_score == 1.0) == res.passed, (res.fidelity, _codes(res))


def test_cfs_edge_cases():
    assert claim_fidelity_score(0, []).score == 0.0
    v = [Violation(code="unsupported_number", severity="critical", message="", span="83.7")]
    r = claim_fidelity_score(2, v)  # no texts given: conservative, one unbound unit
    assert r.n_unbound_sentences == 1 and r.score == round(2 / 3, 4)
    minor = [Violation(code="not_allowed_evidence", severity="minor", message="", claim_index=0)]
    r = claim_fidelity_score(2, minor)
    assert r.score == 1.0 and r.weighted_score == 0.75


# ---------------------------------------------------------------- level rules
def test_level_numbers_rule(hr, instances):
    pkg, plan_nov = _pkg(hr, instances[0], "novice")
    _, plan_int = _pkg(hr, instances[0], "intermediate")
    for text in ["Overtime added 3.2 percentage points.", "Overtime: +3.2.", "The probability is 0.32."]:
        assert "level_numbers" in {v.code for v in level_violations(text, plan_nov)[0]}, text
        assert "level_numbers" not in {v.code for v in level_violations(text, plan_int)[0]}, text
    for text in ["The model estimates a 32% chance.", "The estimate would drop from 45% to 30%."]:
        assert "level_numbers" not in {v.code for v in level_violations(text, plan_nov)[0]}, text
    assert "level_numbers" not in {v.code for v in level_violations("Debt ratio (23.5) raised it.", plan_nov, ["23.5"])[0]}


def test_readability_floor_comes_from_the_plan(hr, instances):
    pkg = build_package(hr, instances[0], "novice", "explain")
    assert make_plan(pkg, "novice", "explain").min_reading_ease == 50.0
    assert make_plan(pkg, "expert", "explain").min_reading_ease == 0.0
