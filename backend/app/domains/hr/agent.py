"""HrAgent: the HR attrition domain (HR-manager perspective)."""
from __future__ import annotations

import json
from collections import Counter
from functools import cached_property

from ...config import MODELS_DIR
from ...ml.runtime import ModelRuntime
from ..base import Action, DomainAgent
from .policy import POLICY
from .spec import FEATURES, GLOSSARY

SALARY_STEPS = [0.05, 0.10, 0.15, 0.20]  # bounded realistic band: at most +20% and within the job level's range
SPECS = {f.name: f for f in FEATURES}


class HrAgent(DomainAgent):
    name = "hr"
    title = "Employee attrition risk"
    description = "Estimates and explains employee attrition risk for HR managers. Risk estimates only; people decide."
    routing_keywords = [
        "employee", "attrition", "leave", "leaving", "quit", "resign", "turnover", "retention", "retain", "flight risk",
        "overtime", "salary", "income", "promotion", "job", "role", "manager", "hr", "staff", "workforce",
        "satisfaction", "tenure", "training", "work-life", "department", "risk", "estimate", "flagged", "factors",
        "travel", "pay", "hybrid", "remote", "band", "involvement", "balance", "promoting", "stock", "commute",
    ]
    positive_label = "likely to leave"
    negative_label = "likely to stay"
    features = FEATURES
    glossary = GLOSSARY
    policy = POLICY

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self._runtimes: dict[str, ModelRuntime] = {}

    @cached_property
    def manifest(self) -> dict:
        p = self.models_dir / "hr_manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def runtime(self, variant: str = "main") -> ModelRuntime:
        if variant not in self._runtimes:
            path = self.models_dir / f"hr_{variant}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Model not trained: {path}. Run scripts/train_hr.py first.")
            self._runtimes[variant] = ModelRuntime.load(path, self.manifest.get(variant, {}), variant)
        return self._runtimes[variant]

    @property
    def data_source(self) -> str:
        return self.manifest.get("dataset", {}).get("source", "unknown")

    # ---------------------------------------------------------------- recourse levers
    def recourse_actions(self, x: dict[str, float], variant: str = "main") -> list[Action]:
        rt = self.runtime(variant)
        ds = rt.domain_stats
        acts: list[Action] = []
        if x["OverTime"] == 1:
            acts.append(Action("overtime_off", "removing regular overtime", {"OverTime": 0.0}, ["OverTime"], cost=2, short="no regular overtime"))
        if x["BusinessTravel"] == 2:
            acts.append(Action("travel_rarely", "reducing business travel to rare trips", {"BusinessTravel": 1.0},
                               ["BusinessTravel"], cost=2, short="less business travel"))
        if x["BusinessTravel"] >= 1:
            acts.append(Action("travel_none", "removing business travel", {"BusinessTravel": 0.0}, ["BusinessTravel"], cost=3, short="no business travel"))

        level = int(x["JobLevel"])
        band = ds["income_by_level"].get(str(level)) or ds["income_by_level"].get(level)
        income = x["MonthlyIncome"]
        for k in SALARY_STEPS:
            new = round(min(income * (1 + k), band["max"]))
            if new > income:
                acts.append(Action(f"salary_{int(k * 100)}",
                                   f"a {int(k * 100)}% raise in monthly income (to {new:,})" if new == round(income * (1 + k))
                                   else f"a raise in monthly income to {new:,}, the top of the range for {SPECS['JobLevel'].readable(level)}",
                                   {"MonthlyIncome": float(new)}, ["MonthlyIncome"], cost=1.5 + 10 * k,
                                   short="higher pay"))
        if x["StockOptionLevel"] < 3:
            nl = x["StockOptionLevel"] + 1
            acts.append(Action("stock_up", f"raising the stock option level to {SPECS['StockOptionLevel'].readable(nl)}",
                               {"StockOptionLevel": nl}, ["StockOptionLevel"], cost=2.5, short="more stock options"))
        tmax = max(6.0, ds.get("training_max", 6.0))
        for extra in (1, 2):
            if x["TrainingTimesLastYear"] + extra <= tmax:
                n = int(x["TrainingTimesLastYear"] + extra)
                acts.append(Action(f"training_{extra}", f"adding {extra} training session{'s' if extra > 1 else ''} (to {n} a year)",
                                   {"TrainingTimesLastYear": float(n)}, ["TrainingTimesLastYear"], cost=0.5 + 0.5 * extra,
                                   short="more training"))
        if level < 5:
            nb = ds["income_by_level"].get(str(level + 1)) or ds["income_by_level"].get(level + 1)
            if nb:
                new_income = float(round(min(max(income, nb["p25"]), nb["max"])))
                new_income = max(new_income, nb["min"])
                acts.append(Action(
                    "promotion",
                    f"a promotion to {SPECS['JobLevel'].readable(level + 1)} (years since last promotion reset to 0, "
                    f"monthly income {new_income:,.0f})",
                    {"JobLevel": float(level + 1), "YearsSinceLastPromotion": 0.0, "MonthlyIncome": new_income},
                    ["JobLevel"], consequences=["YearsSinceLastPromotion", "MonthlyIncome"], cost=4, short="a promotion",
                ))
        dept = int(x["Department"])
        for d, r in ds["role_department_pairs"]:
            if int(d) == dept and int(r) != int(x["JobRole"]) and int(r) != 3:  # same department; not to Manager
                acts.append(Action(
                    f"role_{int(r)}", f"a move to the {SPECS['JobRole'].readable(r)} role (years in current role reset to 0)",
                    {"JobRole": float(r), "YearsInCurrentRole": 0.0}, ["JobRole"], consequences=["YearsInCurrentRole"], cost=4, short="a role change",
                ))
        dist = x["DistanceFromHome"]
        if dist > 2:
            hybrid = float(max(1, round(dist / 2)))
            acts.append(Action("distance_hybrid", f"hybrid work, modelled as halving the distance from home (to {int(hybrid)})",
                               {"DistanceFromHome": hybrid}, ["DistanceFromHome"], cost=3, short="hybrid work",
                               assumptions=["Hybrid work is modelled as halving the distance from home; the dataset has no "
                                            "remote-work column."]))
            low = float(rt.ood["ranges"]["DistanceFromHome"][0])
            if low < hybrid:
                acts.append(Action("distance_remote", f"remote work, modelled as the minimum distance in the data ({int(low)})",
                                   {"DistanceFromHome": low}, ["DistanceFromHome"], cost=3.5, short="remote work",
                                   assumptions=["Remote work is modelled as the minimum distance in the training data; the "
                                                "dataset has no remote-work column."]))
        return acts

    def consistent_defaults(self, filled: dict[str, float], assumed: set[str], variant: str = "main") -> dict[str, float]:
        out = dict(filled)
        rt = self.runtime(variant)
        pairs = {(int(d), int(r)) for d, r in rt.domain_stats["role_department_pairs"]}
        if ("JobRole" in assumed or "Department" in assumed) and (int(out["Department"]), int(out["JobRole"])) not in pairs:
            if "JobRole" in assumed:  # most common role in the given department
                rows = list(rt.background[["Department", "JobRole"]].itertuples(index=False)) + \
                    list(rt.examples[["Department", "JobRole"]].itertuples(index=False))
                counts = Counter(int(r) for d, r in rows if int(d) == int(out["Department"]))
                options = [r for d, r in pairs if d == int(out["Department"])]
                out["JobRole"] = float(max(options, key=lambda r: counts.get(r, 0)))
            else:
                out["Department"] = float(next(d for d, r in sorted(pairs) if r == int(out["JobRole"])))
        # time fields: role, promotion and manager years <= years at company <= total working years
        if "TotalWorkingYears" in assumed:
            out["TotalWorkingYears"] = max(out["TotalWorkingYears"], out["YearsAtCompany"])
        if "YearsAtCompany" in assumed:
            out["YearsAtCompany"] = min(out["YearsAtCompany"], out["TotalWorkingYears"])
        for f in ("YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"):
            if f in assumed:
                out[f] = min(out[f], out["YearsAtCompany"])
        return out
