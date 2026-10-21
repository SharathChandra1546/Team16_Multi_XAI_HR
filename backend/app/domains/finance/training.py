"""Training spec for the finance domain (used by scripts/train_domain.py)."""
from __future__ import annotations

import pandas as pd

from ...ml.domain_training import DomainTraining
from .data import STANDIN_PATH, UCI_PATH, generate_standin, load_dataset
from .spec import FEATURES, PROTECTED

SPEC = {f.name: f for f in FEATURES}


def age_band(a: float) -> str:
    return "under 25" if a < 25 else "25-34" if a < 35 else "35-49" if a < 50 else "50+"


def protected_groups(X: pd.DataFrame) -> dict[str, pd.Series]:
    sex = X["personal_status_sex"].map(lambda c: "female" if int(c) in (1, 4) else "male")
    return {"age_band": X["age"].map(age_band), "sex": sex,
            "foreign_worker": X["foreign_worker"].map(lambda c: SPEC["foreign_worker"].labels[int(c)])}


def domain_stats(X: pd.DataFrame) -> dict:
    """Recourse bounds from the TRAINING split only."""
    return {"amount_min": float(X.credit_amount.min()), "duration_min": float(X.duration.min()),
            "duration_max": float(X.duration.max())}


TRAINING = DomainTraining(
    name="finance",
    features=[f.name for f in FEATURES],
    nominal=[f.name for f in FEATURES if f.kind == "nominal"],
    protected=list(PROTECTED),
    real_path=UCI_PATH,
    standin_path=STANDIN_PATH,
    load=load_dataset,
    generate_standin=generate_standin,
    write_standin=lambda df, p: df.to_csv(p, index=False),
    protected_groups=protected_groups,
    domain_stats=domain_stats,
    cost_fn=5.0,
    cost_fp=1.0,
    cost_note=("The dataset's documented cost matrix: classifying a bad credit risk as good costs 5, classifying a "
               "good risk as bad costs 1."),
    outcome_label="Observed bad-risk rate",
    aggregate_columns=["purpose", "checking_status", "savings", "credit_history"],
)
