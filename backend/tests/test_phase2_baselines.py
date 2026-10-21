"""Phase 2: baselines, uniform measurement, agreement metrics, usage accounting, comparison runner."""
import os
import random

from app.agents.fallback import build_template
from app.agents.planner import make_plan
from app.baselines import SYSTEMS, format_verifier, run_baseline
from app.faults import inject
from app.llm.client import BaseLLM, get_llm
from app.metrics.agreement import agreement, extract_mentions
from app.metrics.measure import measure_text
from app.xai.evidence import build_package


def _case(hr, x, level="intermediate", intent="explain"):
    pkg = build_package(hr, x, level, intent)
    return pkg, make_plan(pkg, level, intent)


def test_verified_templates_measure_clean_with_full_agreement(hr, instances):
    for x in instances[:4]:
        for level in ("novice", "intermediate", "expert"):
            for intent in ("explain", "recourse"):
                pkg, plan = _case(hr, x, level, intent)
                m = measure_text(build_template(pkg, plan, hr).explanation, pkg, plan, hr)
                assert m["error_free"], (level, intent, m["text_checks"], m["policy_violations"], m["agreement_det"])
                if intent == "explain":
                    a = m["agreement_det"]
                    assert a["sign_agreement"] == 1.0 and a["rank_agreement"] == 1.0, a
                    assert a["value_agreement"] in (1.0, None) and a["top3_coverage"] == 1.0


def test_measurement_catches_faults(hr, instances):
    pkg, plan = _case(hr, instances[0])
    clean = build_template(pkg, plan, hr)
    rng = random.Random(0)
    for fault, field in [("direction_flip", "sign_errors"), ("value_swap", "value_errors")]:
        bad = inject(fault, clean, pkg, plan, hr, rng)
        m = measure_text(bad.explanation, pkg, plan, hr)
        assert not m["error_free"] and m["agreement_det"][field] >= 1, (fault, m["agreement_det"])
    for fault in ("decision_wording", "protected_inference", "causal_claim", "unsupported_number"):
        bad = inject(fault, clean, pkg, plan, hr, rng)
        assert not measure_text(bad.explanation, pkg, plan, hr)["error_free"], fault


def test_unsupported_mention_of_withheld_factor(hr, audit_instances):
    hr_rt = hr.runtime("main")
    x = {f: float(v) for f, v in hr_rt.examples.iloc[0][hr_rt.features].items()}
    pkg, plan = _case(hr, x)
    text = build_template(pkg, plan, hr).explanation + " Marital status also played a part."
    a = agreement(extract_mentions(text, pkg, hr), pkg, hr)
    assert "MaritalStatus" in a["unsupported_mentions"]


def test_every_baseline_runs_and_counts_usage(registry, hr, instances):
    from app.llm import mock

    mock.DOMAIN_RESOLVER = registry.get
    llm = get_llm("mock")
    pkg, plan = _case(hr, instances[0])
    for name in SYSTEMS:
        r = run_baseline(name, pkg, plan, hr, llm)
        assert r.text.strip(), name
        assert r.usage["calls"] >= 1 and r.usage["prompt_tokens"] > 0 and r.usage["estimated"], (name, r.usage)
    assert run_baseline("b3", pkg, plan, hr, llm).usage["calls"] == 2  # narrate + one accepting review


def test_b4_format_verifier():
    from app.schemas import ExplanationPlan

    plan = ExplanationPlan(level="expert", intent="explain", allowed_ids=[], required_ids=[], max_words=200,
                           include_numbers=True, allow_column_names=True, banned_terms=[], style="")

    class Pkg:
        visible = []

    assert "list-like formatting" in format_verifier("- Overtime raised it.\n- Pay lowered it.", Pkg(), plan)
    assert "JSON-like formatting" in format_verifier('{"a": 1}. Second sentence.', Pkg(), plan)
    assert "fewer than two sentences" in format_verifier("Only one sentence.", Pkg(), plan)
    assert format_verifier("", Pkg(), plan) == ["empty"]


class FailingReviewer(BaseLLM):
    name, model = "strict", "reject-all"

    def json(self, task, system, user, payload=None):
        self.record_usage(task, 10, 5)
        return {"score": 1, "verdict": "revise", "comment": "no", "revise": True, "route": "both", "instructions": ["x"],
                "unfaithful": [{"statement": "s", "fix": "f"}], "accuracy": 0, "completeness": 0, "fluency": 0}


def test_loops_respect_their_budgets(registry, hr, instances):
    from app.llm import mock

    mock.DOMAIN_RESOLVER = registry.get
    llm, strict = get_llm("mock"), FailingReviewer()
    pkg, plan = _case(hr, instances[0])
    assert run_baseline("b3", pkg, plan, hr, llm, strict).rounds == 3  # first draft + 2 revisions
    assert run_baseline("musicco", pkg, plan, hr, llm, strict).rounds == 3  # first draft + 2 revision rounds
    assert run_baseline("critic_loop", pkg, plan, hr, llm, strict).rounds == 4  # first draft + 3 rounds
    r = run_baseline("explingo", pkg, plan, hr, llm, strict)
    assert r.rounds == 3 and r.accepted is False
    assert r.usage["calls"] == 3 + 3  # narrator calls + grader calls, both counted


def test_comparison_runner_end_to_end():
    import subprocess
    import sys
    from pathlib import Path

    backend = Path(__file__).resolve().parents[1]
    env = dict(os.environ, LLM_PROVIDER="mock", MOCK_FAULT_RATE="0.5")
    out = subprocess.run([sys.executable, "eval/run_comparison.py", "--systems", "ours:full,ours:no_verifier,b2,musicco",
                          "--n", "2", "--levels", "novice", "--intents", "explain"],
                         cwd=backend, env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "| ours:full |" in out.stdout and "Coverage-risk" in out.stdout and "Do not report them" in out.stdout


def test_rule_critic_flags_wrong_value_sign_and_unsupported_factor(registry, hr, instances):
    """He & Martens' rule-based critic: code compares the extracted claims with the card and writes the fix."""
    from app.baselines import rule_feedback
    from app.metrics.agreement import Mention
    import app.metrics.agreement as ag

    pkg, plan = _case(hr, instances[0])
    top = pkg.visible[0]
    flipped = "decreases" if top.direction == "increases" else "increases"

    class Stub:
        name = "stub"

        def json(self, *a, **k):
            return {}

    real = ag.llm_extract_mentions
    ag.llm_extract_mentions = lambda *a, **k: [
        Mention(top.feature, True, value="999999", direction=flipped, stated_rank=None, order=1),
        Mention("NotAFeature", False, order=2)]
    try:
        issues = rule_feedback("x", pkg, hr, Stub())
    finally:
        ag.llm_extract_mentions = real
    assert any("not" in i and top.value_label in i for i in issues)
    assert any(top.direction in i for i in issues)
    assert any(i.startswith("Remove") for i in issues)


def test_rule_critic_runs_and_respects_budget(registry, hr, instances):
    from app.llm import mock

    mock.DOMAIN_RESOLVER = registry.get
    llm = get_llm("mock")
    pkg, plan = _case(hr, instances[0])
    r = run_baseline("critic_rule", pkg, plan, hr, llm, llm)
    assert r.text.strip() and 1 <= r.rounds <= 4


# ---------------------------------------------------------------- regressions found when running on real data
def test_risk_text_swap_matches_whole_numbers(hr, instances):
    """Real HR data had an estimate of 1.3% with a range ending in "2.1%". The injector searched for "1%" as text,
    matched inside "2.1%" and wrote "2.15%", a change no checker could flag."""
    from app.agents.verifier import verify

    pkg, plan = _case(hr, instances[0], "expert")  # the expert text prints the bootstrap range
    pr = pkg.prediction.model_copy(update={"probability": 0.0133, "interval": (0.0051, 0.0215), "band": "low"})
    pkg = pkg.model_copy(update={"prediction": pr})
    clean = build_template(pkg, plan, hr)
    bad = inject("risk_text_swap", clean, pkg, plan, hr, random.Random(0))
    assert bad is not None and bad.explanation != clean.explanation
    assert "risk_mismatch" in {v.code for v in verify(pkg, plan, bad, hr).violations}


def test_extractor_reads_value_labels_that_contain_a_factor_name(registry, hr, instances):
    """"Overtime (works overtime)" must yield the value; and a label that contains ANOTHER factor's name
    ("Credit history (existing credits repaid duly so far)") must not be read as a mention of that factor."""
    pkg, _ = _case(hr, instances[0])
    ms = extract_mentions("Overtime (works overtime) lowered the estimate moderately.", pkg, hr)
    ot = next(m for m in ms if m.feature == "OverTime")
    assert ot.value == "works overtime" and ot.direction == "decreases"

    fin = registry.get("finance")
    rt = fin.runtime("main")
    x = {f: float(rt.examples.iloc[0][f]) for f in rt.features}
    fpkg = build_package(fin, x, "expert", "explain")
    ms = extract_mentions("Credit history (existing credits repaid duly so far) raised the estimate, +2.1 percentage points.",
                          fpkg, fin)
    assert [m.feature for m in ms] == ["credit_history"]
    assert ms[0].value == "existing credits repaid duly so far" and ms[0].direction == "increases"


def test_value_label_cue_words_are_not_direction_claims(registry):
    """Finance label "less than 1 year" contains the down-cue "less"; "Employment length (less than 1 year) moves the
    estimate up." was rejected as direction_mismatch once the rewrite removed the up-cue ("pushes ... up")."""
    from app.agents.verifier import verify

    fin = registry.get("finance")
    rt = fin.runtime("main")
    for i in range(len(rt.examples.head(40))):
        x = {f: float(rt.examples.iloc[i][f]) for f in rt.features}
        pkg = build_package(fin, x, "novice", "explain")
        plan = make_plan(pkg, "novice", "explain")
        gen = build_template(pkg, plan, fin)
        hit = [c for c in gen.claims if c.kind == "attribution" and c.direction == "increases" and "pushes the estimate up" in c.text
               and any(w in c.text.lower() for w in ("less than", "higher", "lower"))]
        if hit:
            g = gen.model_copy(deep=True)
            g.explanation = g.explanation.replace("pushes the estimate up", "moves the estimate up")
            for c in g.claims:
                c.text = c.text.replace("pushes the estimate up", "moves the estimate up")
            assert "direction_mismatch" not in {v.code for v in verify(pkg, plan, g, fin).violations}
            return
    import pytest
    pytest.skip("no finance example with a cue word in a value label among the first 40")


def test_risk_text_swap_never_edits_the_review_threshold(registry):
    from app.agents.verifier import verify

    fin = registry.get("finance")
    rt = fin.runtime("main")
    x = {f: float(rt.examples.iloc[0][f]) for f in rt.features}
    pkg = build_package(fin, x, "expert", "explain")
    pr = pkg.prediction.model_copy(update={"probability": 0.197, "interval": (0.088, 0.331), "threshold": 0.20, "band": "low"})
    pkg = pkg.model_copy(update={"prediction": pr})
    plan = make_plan(pkg, "expert", "explain")
    clean = build_template(pkg, plan, fin)
    bad = inject("risk_text_swap", clean, pkg, plan, fin, random.Random(0))
    if bad is not None:
        assert "risk_mismatch" in {v.code for v in verify(pkg, plan, bad, fin).violations}
        assert "review threshold 20%" in bad.explanation or "threshold" not in bad.explanation


def test_raw_code_with_several_underscores_is_caught(registry):
    """Real finance run: "(other_installment_plans=A141)" passed, because both raw-code patterns allowed one underscore."""
    from app.agents.verifier import verify

    fin = registry.get("finance")
    rt = fin.runtime("main")
    x = {f: float(rt.examples.iloc[0][f]) for f in rt.features}
    pkg = build_package(fin, x, "intermediate", "explain")
    plan = make_plan(pkg, "intermediate", "explain")
    g = build_template(pkg, plan, fin)
    g.explanation += " Plans were other_installment_plans=A141."
    assert "raw_code_leak" in {v.code for v in verify(pkg, plan, g, fin).violations}
