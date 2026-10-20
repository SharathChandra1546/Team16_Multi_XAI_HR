"""Actionable-only recourse: every option is re-predicted, lowers risk, and respects constraints."""
from app.recourse import find_recourse, recheck_option

IMMUTABLE = ["YearsAtCompany", "TotalWorkingYears", "NumCompaniesWorked", "Education", "EducationField", "Department",
             "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "PerformanceRating", "PercentSalaryHike"]


def test_every_option_lowers_risk_and_respects_policy(hr, instances):
    rt = hr.runtime("main")
    ds = rt.domain_stats
    n_options = 0
    for x in instances:
        options, note = find_recourse(hr, x)
        assert options or note
        for o in options:
            n_options += 1
            assert recheck_option(hr, x, o), o
            assert o.probability_after <= o.probability_before - hr.policy.min_risk_drop or \
                rt.band_rank(o.band_after) < rt.band_rank(o.band_before)
            changed = {c.feature for c in o.changes}
            assert not changed & set(IMMUTABLE), changed
            for c in o.changes:
                st = hr.policy.status(c.feature)
                assert st in ("actionable", "partial") or (st == "consequence" and c.consequence), c
            if "MonthlyIncome" in changed and "promotion" not in o.actions:
                new = next(a for a in o.actions if a.startswith("salary"))
                pct = int(new.split("_")[1])
                assert pct <= 20
    assert n_options > 0


def test_salary_stays_in_level_band(hr, instances):
    ds = hr.runtime("main").domain_stats
    for x in instances:
        for a in hr.recourse_actions(x):
            if "MonthlyIncome" in a.changes:
                level = int(a.changes.get("JobLevel", x["JobLevel"]))
                band = ds["income_by_level"].get(level) or ds["income_by_level"].get(str(level))
                assert band["min"] <= a.changes["MonthlyIncome"] <= band["max"]
                if a.id.startswith("salary"):
                    assert a.changes["MonthlyIncome"] <= x["MonthlyIncome"] * 1.2 + 1


def test_distance_is_partial_with_assumption(hr, instances):
    assert hr.policy.status("DistanceFromHome") == "partial"
    for x in instances:
        for a in hr.recourse_actions(x):
            if "DistanceFromHome" in a.changes:
                assert a.assumptions and "remote-work" in a.assumptions[0]
