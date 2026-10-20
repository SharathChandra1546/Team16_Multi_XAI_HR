"""FinanceAgent: creditworthiness (German Credit), credit-analyst perspective."""
from __future__ import annotations

import json
from functools import cached_property

from ...config import MODELS_DIR
from ...ml.runtime import ModelRuntime
from ..base import Action, DomainAgent
from .policy import POLICY
from .spec import FEATURES, GLOSSARY

SPECS = {f.name: f for f in FEATURES}


class FinanceAgent(DomainAgent):
    name = "finance"
    title = "Credit risk"
    description = "Estimates and explains credit risk for credit analysts. Risk estimates only; people decide."
    routing_keywords = [
        "loan", "credit", "applicant", "borrower", "lend", "lending", "default", "repay", "mortgage", "bank",
        "creditworthiness", "installment", "savings", "checking", "guarantor", "debt", "amount", "duration", "interest",
    ]
    positive_label = "bad credit risk"
    negative_label = "good credit risk"
    features = FEATURES
    glossary = GLOSSARY
    policy = POLICY

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self._runtimes: dict[str, ModelRuntime] = {}

    @cached_property
    def manifest(self) -> dict:
        p = self.models_dir / "finance_manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def runtime(self, variant: str = "main") -> ModelRuntime:
        if variant not in self._runtimes:
            path = self.models_dir / f"finance_{variant}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Model not trained: {path}. Run scripts/train_domain.py --domain finance.")
            self._runtimes[variant] = ModelRuntime.load(path, self.manifest.get(variant, {}), variant)
        return self._runtimes[variant]

    @property
    def data_source(self) -> str:
        return self.manifest.get("dataset", {}).get("source", "unknown")

    # ---------------------------------------------------------------- recourse levers
    def recourse_actions(self, x: dict[str, float], variant: str = "main") -> list[Action]:
        ds = self.runtime(variant).domain_stats
        acts: list[Action] = []
        amount, dur, inst = x["credit_amount"], x["duration"], x["installment_rate"]
        a_min = ds.get("amount_min", 250.0)
        for k in (0.10, 0.25):
            new = float(round(amount * (1 - k)))
            if new >= a_min:
                changes = {"credit_amount": new}
                conseq = []
                if k >= 0.25 and inst > 1:
                    changes["installment_rate"] = inst - 1
                    conseq = ["installment_rate"]
                acts.append(Action(f"amount_{int(k * 100)}", f"requesting {int(k * 100)}% less credit "
                                   f"({SPECS['credit_amount'].readable(new)})", changes, ["credit_amount"],
                                   consequences=conseq, cost=1 + 4 * k, short="a smaller loan",
                                   assumptions=["A smaller amount is modelled as lowering the installment rate by one "
                                                "level when the amount falls by a quarter."] if conseq else []))
        d_min = ds.get("duration_min", 4.0)
        for m in (6, 12):
            new = dur - m
            if new >= max(d_min, 4):
                changes = {"duration": float(new)}
                conseq = []
                if inst < 4:
                    changes["installment_rate"] = inst + 1
                    conseq = ["installment_rate"]
                acts.append(Action(f"duration_{m}", f"shortening the loan by {m} months (to {int(new)} months)", changes,
                                   ["duration"], consequences=conseq, cost=1.5 + m / 12, short="a shorter loan",
                                   assumptions=["A shorter term means larger instalments, modelled as raising the "
                                                "installment rate by one level."] if conseq else []))
        if int(x["other_debtors"]) == 0:
            acts.append(Action("guarantor", "adding a guarantor", {"other_debtors": 2.0}, ["other_debtors"], cost=3,
                               short="a guarantor"))
            acts.append(Action("coapplicant", "adding a co-applicant", {"other_debtors": 1.0}, ["other_debtors"],
                               cost=3, short="a co-applicant"))
        if int(x["savings"]) in (0, 1, 2):
            nxt = float(int(x["savings"]) + 1)
            acts.append(Action("savings_up", f"savings growing to {SPECS['savings'].readable(nxt)}", {"savings": nxt},
                               ["savings"], cost=3.5, short="more savings",
                               assumptions=["Savings growth takes time; this shows the estimate after it."]))
        if int(x["checking_status"]) == 0:
            acts.append(Action("checking_up", "a checking balance of 0 to 200 DM", {"checking_status": 1.0},
                               ["checking_status"], cost=3.5, short="a positive checking balance",
                               assumptions=["A recovered checking balance takes time; this shows the estimate after it."]))
        return acts
