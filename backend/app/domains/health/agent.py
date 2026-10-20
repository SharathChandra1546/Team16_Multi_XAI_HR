"""HealthAgent: diabetes risk (Pima Indians Diabetes data), clinician perspective."""
from __future__ import annotations

import json
from functools import cached_property

from ...config import MODELS_DIR
from ...ml.runtime import ModelRuntime
from ..base import Action, DomainAgent
from .policy import POLICY
from .spec import FEATURES, GLOSSARY

SPECS = {f.name: f for f in FEATURES}
_NOT_ADVICE = "Shows the model's estimate after the change; it is not treatment advice and needs clinical judgement."


class HealthAgent(DomainAgent):
    name = "health"
    title = "Diabetes risk"
    description = "Estimates and explains diabetes risk for clinicians. Risk estimates only; clinicians decide."
    routing_keywords = [
        "patient", "diabetes", "diabetic", "glucose", "bmi", "insulin", "blood pressure", "clinician", "clinical",
        "pregnancies", "pedigree", "medical", "health", "skin", "measurement",
    ]
    positive_label = "diabetes by the study's criteria"
    negative_label = "no diabetes by the study's criteria"
    features = FEATURES
    glossary = GLOSSARY
    policy = POLICY

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self._runtimes: dict[str, ModelRuntime] = {}

    @cached_property
    def manifest(self) -> dict:
        p = self.models_dir / "health_manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def runtime(self, variant: str = "main") -> ModelRuntime:
        if variant not in self._runtimes:
            path = self.models_dir / f"health_{variant}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Model not trained: {path}. Run scripts/train_domain.py --domain health.")
            self._runtimes[variant] = ModelRuntime.load(path, self.manifest.get(variant, {}), variant)
        return self._runtimes[variant]

    @property
    def data_source(self) -> str:
        return self.manifest.get("dataset", {}).get("source", "unknown")

    # ---------------------------------------------------------------- modelled changes (recourse)
    def recourse_actions(self, x: dict[str, float], variant: str = "main") -> list[Action]:
        ds = self.runtime(variant).domain_stats
        acts: list[Action] = []
        bmi_min = ds.get("bmi_min", 18.5)
        for d in (2.0, 4.0, 6.0):
            new = round(x["BMI"] - d, 1)
            if new >= bmi_min:
                acts.append(Action(f"bmi_{int(d)}", f"a BMI {d:g} points lower (to {SPECS['BMI'].readable(new)})",
                                   {"BMI": new}, ["BMI"], cost=2 + d / 2, short="a lower BMI",
                                   assumptions=[_NOT_ADVICE]))
        bp_min = ds.get("bp_min", 50.0)
        for d in (5.0, 10.0):
            new = float(x["BloodPressure"] - d)
            if new >= bp_min:
                acts.append(Action(f"bp_{int(d)}", f"a diastolic blood pressure {d:g} mm Hg lower (to "
                                   f"{SPECS['BloodPressure'].readable(new)})", {"BloodPressure": new},
                                   ["BloodPressure"], cost=2 + d / 10, short="lower blood pressure",
                                   assumptions=[_NOT_ADVICE]))
        return acts
