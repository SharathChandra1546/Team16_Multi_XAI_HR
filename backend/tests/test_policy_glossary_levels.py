"""DomainPolicy, glossary, readable values, disclosure and level differences."""
import re

from app.agents.fallback import build_template
from app.agents.planner import LEVELS, make_plan
from app.agents.textutil import word_count
from app.schemas import ExplanationRequest
from app.xai.evidence import build_package


def test_main_model_excludes_protected(hr):
    feats = set(hr.model_features("main"))
    assert not feats & set(hr.policy.sensitive)
    assert set(hr.policy.sensitive) <= set(hr.model_features("audit"))


def test_protected_withheld_below_expert(hr, audit_instances):
    for x in audit_instances[:8]:
        for level in ("novice", "intermediate"):
            pkg = build_package(hr, x, level, "explain", variant="audit")
            for it in pkg.items:
                if it.feature in hr.policy.sensitive:
                    assert it.withheld and it.rank == 0
            plan = make_plan(pkg, level, "explain")
            text = build_template(pkg, plan, hr).explanation.lower()
            for f in hr.policy.sensitive:
                spec = hr.spec(f)
                assert not re.search(rf"\b{spec.display.lower()}\b", text)
        pkg = build_package(hr, x, "expert", "explain", variant="audit")
        assert not any(it.withheld for it in pkg.items)


def test_every_feature_has_glossary_entry(hr):
    for f in hr.features:
        assert f.glossary_id in hr.glossary, f.name


def test_no_raw_codes_in_text(hr, instances):
    names = {f.name for f in hr.features}
    raws = {str(v) for f in hr.features for v in f.options.values() if "_" in str(v)}
    for x in instances[:10]:
        for level in ("novice", "intermediate", "expert"):
            for intent in ("explain", "recourse"):
                pkg = build_package(hr, x, level, intent)
                text = build_template(pkg, make_plan(pkg, level, intent), hr).explanation
                assert "=" not in text
                assert not any(r in text for r in raws)
                if level != "expert":
                    assert not any(re.search(rf"\b{n}\b", text) for n in names if re.search(r"[a-z][A-Z]", n))


def test_levels_differ(hr, instances):
    assert LEVELS["novice"]["max_words"] < LEVELS["intermediate"]["max_words"] < LEVELS["expert"]["max_words"]
    counts = {lvl: [] for lvl in LEVELS}
    for x in instances[:10]:
        for lvl in LEVELS:
            pkg = build_package(hr, x, lvl, "explain")
            text = build_template(pkg, make_plan(pkg, lvl, "explain"), hr).explanation
            counts[lvl].append(word_count(text))
            assert "probability" not in text.lower() or lvl != "novice"
    mean = {k: sum(v) / len(v) for k, v in counts.items()}
    assert mean["novice"] < mean["intermediate"] < mean["expert"], mean


def test_disclosure_on_every_explanation(service, instances, hr):
    for x in instances[:6]:
        raw = {f: hr.readable(f, v) if hr.spec(f).categorical else v for f, v in x.items()}
        r = service.explain(ExplanationRequest(query="Why is this employee flagged?", instance=raw, level="novice"))
        if r.status in ("verified", "fallback"):
            assert r.disclosure == hr.policy.disclosure
            assert "risk estimate, not a decision" in r.disclosure and "review" in r.disclosure
            if r.prediction.band == "high":
                assert hr.policy.high_risk_notice in r.notices


def test_disclosure_check_catches_missing(hr):
    from app.agents.critic import check_disclosure

    assert check_disclosure("Some text without it.", hr) is not None
    assert check_disclosure("Text. " + hr.policy.disclosure, hr) is None
