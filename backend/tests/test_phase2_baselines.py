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
