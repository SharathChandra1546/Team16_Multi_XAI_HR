"""What a domain package supplies so the generic trainer (scripts/train_domain.py) can build its models.

A domain package exposes `TRAINING: DomainTraining` in `app/domains/<name>/training.py`. The trainer then:
loads the real dataset (or the clearly labelled synthetic stand-in), trains a main model without the protected
attributes and, if there are any, an audit model with them, computes recourse statistics on the training split only,
runs the fairness audit at each model's own threshold, and writes `<name>_main.joblib`, `<name>_audit.joblib`,
`<name>_manifest.json` and a reliability plot. Training, calibration, thresholds, uncertainty and the OOD detector
are the shared code in app/ml/training.py, so every domain is validated the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd


@dataclass
class DomainTraining:
    name: str
    features: list[str]  # all features in model space (main model drops `protected`)
    nominal: list[str]  # one-hot encoded features
    protected: list[str]  # excluded from the main model; used by the audit model and the fairness audit
    real_path: Path
    standin_path: Path
    load: Callable[[Path], tuple[pd.DataFrame, pd.Series, dict]]  # -> (X in model space, y, info with "source")
    generate_standin: Callable[[], pd.DataFrame]  # raw-format rows; written to standin_path
    write_standin: Callable[[pd.DataFrame, Path], None]
    protected_groups: Callable[[pd.DataFrame], dict[str, pd.Series]]  # group labels for the fairness audit
    domain_stats: Callable[[pd.DataFrame], dict] = lambda X: {}  # recourse bounds; given the TRAINING split only
    cost_fn: float = 5.0
    cost_fp: float = 1.0
    cost_note: str = ""
    outcome_label: str = "Observed rate"  # y-axis label of the reliability plot
    aggregate_columns: list[str] = field(default_factory=list)  # features summarised in the aggregate view
    data_note: str = ""
