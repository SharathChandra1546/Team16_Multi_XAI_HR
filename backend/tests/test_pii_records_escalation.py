"""PII scrubbing, audit/feedback records, escalation and consent."""
from app.pii import domain_vocab, scrub
from app.schemas import ExplanationRequest, FeedbackRequest


def test_pii_masks_names_ids_contacts(registry):
    v = domain_vocab(registry)
    cases = {
        "Why is Priya Sharma flagged?": "[NAME]",
        "Explain employee 10234": "[ID]",
        "Mail priya@acme.com": "[EMAIL]",
        "Call +91 98765 43210 about this": "[PHONE]",
        "What would lower risk for Mr. Rao?": "[NAME]",
    }
    for q, token in cases.items():
        out, n = scrub(q, v)
        assert token in out and n >= 1, (q, out)
    keep = "Why is the Sales Executive in Research & Development flagged?"
    assert scrub(keep, v)[0] == keep


def _raw(hr, x):
    return {f: hr.readable(f, v) if hr.spec(f).categorical else v for f, v in x.items()}


def test_trace_and_feedback_records(service, hr, instances):
    r = service.explain(ExplanationRequest(query="Why is Priya flagged?", instance=_raw(hr, instances[1]), user_id="u-1"))
    rec = service.store.get_trace(r.trace_id)
    assert rec is not None and rec.verdict and rec.prompt_version
    assert "Priya" not in rec.query and rec.pii_masked >= 1
    out = service.feedback(FeedbackRequest(trace_id=r.trace_id, rating=4, understood=True, trust=3, comment="ok"))
    assert out["ok"]
    fb = [f for f in service.store.feedback() if f.trace_id == r.trace_id]
    assert fb and fb[0].rating == 4 and fb[0].prompt_version == rec.prompt_version


def test_honeypot_rejected(service, hr, instances):
    import pytest
    from app.service import SpamError

    with pytest.raises(SpamError):
        service.explain(ExplanationRequest(query="Why?", instance=_raw(hr, instances[0]), website="http://spam"))


def test_escalation_replaces_explanation(service, hr, instances):
    seen = {"escalated": 0, "verified": 0}
    for x in instances:
        r = service.explain(ExplanationRequest(query="Why is this employee flagged?", instance=_raw(hr, x)))
        if r.prediction and (r.prediction.band_uncertain or r.prediction.out_of_distribution):
            assert r.status == "escalated" and r.explanation is None and "human review" in r.message
            seen["escalated"] += 1
        elif r.status == "verified":
            seen["verified"] += 1
    assert seen["verified"] > 0


def test_ood_input_escalates(service, hr, instances):
    x = dict(instances[0])
    x["MonthlyIncome"] = 99000.0  # valid for the form, far outside the training range
    r = service.explain(ExplanationRequest(query="Why is this employee flagged?", instance=_raw(hr, x)))
    assert r.status == "escalated" and "unlike the data" in r.message


def test_cloud_mode_requires_consent():
    from app.config import settings
    from app.service import Service

    class FakeCloud:
        name, model = "gemini", "x"
        id = "gemini:x"

        def json(self, *a, **k):
            raise AssertionError("must not be called without consent")

    old = settings.llm_provider
    settings.llm_provider = "gemini"
    try:
        svc = Service(llm=FakeCloud())
        assert svc._use_llm(False) is False and svc._use_llm(True) is True
    finally:
        settings.llm_provider = old


def test_intake_orders_by_influence_and_flags_defaults(service, hr):
    from app.schemas import IntakeRequest

    rt = hr.runtime("main")
    first = service.intake(IntakeRequest(instance={}))
    assert first["next"][0]["name"] == rt.global_importance[0]["feature"]
    done = service.intake(IntakeRequest(instance={"Department": "Sales", "YearsAtCompany": 1}, finish=True))
    assumed = {a["feature"] for a in done["assumed"]}
    assert "Department" not in assumed and "JobRole" in assumed
    inst = done["instance_codes"]
    pairs = {(int(d), int(r)) for d, r in rt.domain_stats["role_department_pairs"]}
    assert (int(inst["Department"]), int(inst["JobRole"])) in pairs
    assert inst["YearsInCurrentRole"] <= inst["YearsAtCompany"] <= inst["TotalWorkingYears"]
