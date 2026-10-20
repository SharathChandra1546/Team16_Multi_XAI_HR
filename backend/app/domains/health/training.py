"""Training spec for the healthcare domain (used by scripts/train_domain.py)."""
from __future__ import annotations

import pandas as pd

from ...ml.domain_training import DomainTraining
from .data import REAL_PATH, STANDIN_PATH, generate_standin, load_dataset
from .spec import FEATURES, PROTECTED


def age_band(a: float) -> str:
    return "21-29" if a < 30 else "30-39" if a < 40 else "40-49" if a < 50 else "50+"


def protected_groups(X: pd.DataFrame) -> dict[str, pd.Series]:
    """Age stays in the model as a clinical factor; the audit reports error and flag rates by age band."""
    return {"age_band": X["Age"].map(age_band)}


def domain_stats(X: pd.DataFrame) -> dict:
    """Bounds for modelled changes, from the TRAINING split only."""
    return {"bmi_min": float(max(18.5, X.BMI.quantile(0.02))), "bp_min": float(max(50.0, X.BloodPressure.quantile(0.02)))}


TRAINING = DomainTraining(
    name="health",
    features=[f.name for f in FEATURES],
    nominal=[],
    protected=list(PROTECTED),
    real_path=REAL_PATH,
    standin_path=STANDIN_PATH,
    load=load_dataset,
    generate_standin=generate_standin,
    write_standin=lambda df, p: df.to_csv(p, index=False),
    protected_groups=protected_groups,
    domain_stats=domain_stats,
    cost_fn=5.0,
    cost_fp=1.0,
    cost_note=("Missing a person with diabetes (false negative) is weighted 5 times a false alarm: an undetected case "
               "risks untreated hyperglycaemia and complications, while a false alarm leads to a confirmatory test "
               "(e.g. HbA1c or a repeat glucose test) that is cheap and low-risk. This ratio is a modelling choice; "
               "it should be reviewed with a clinician and varied in a sensitivity analysis."),
    outcome_label="Observed diabetes rate",
    aggregate_columns=[],
)
