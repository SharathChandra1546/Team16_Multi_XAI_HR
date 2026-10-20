"""Loading and validating the IBM HR Attrition CSV, plus a clearly labelled synthetic stand-in for tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...config import DATA_DIR
from .spec import DROPPED, FEATURES, TARGET

IBM_FILENAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
IBM_PATH = DATA_DIR / IBM_FILENAME
STANDIN_PATH = DATA_DIR / "hr_standin_TEST_ONLY.csv"
KAGGLE_URL = "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset"
EXPECTED_ROWS = 1470
EXPECTED_LEAVERS = 237

SPECS = {f.name: f for f in FEATURES}
REQUIRED_COLUMNS = [TARGET, *SPECS.keys()]


class DataError(ValueError):
    pass


def to_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataset values -> model space (numeric codes). Raises DataError on unknown categories."""
    out = pd.DataFrame(index=df.index)
    for name, spec in SPECS.items():
        col = df[name]
        if spec.categorical:
            lookup = {str(v).strip().lower(): c for c, v in spec.options.items()}
            mapped = col.astype(str).str.strip().str.lower().map(lookup)
            bad = col[mapped.isna()].unique().tolist()
            if bad:
                raise DataError(f"Column {name} has unknown values: {bad[:5]}")
            out[name] = mapped.astype(float)
        else:
            vals = pd.to_numeric(col, errors="coerce")
            if vals.isna().any():
                raise DataError(f"Column {name} has non-numeric or missing values")
            out[name] = vals.astype(float)
    return out


def load_dataset(path: Path | str) -> tuple[pd.DataFrame, pd.Series, dict]:
    path = Path(path)
    if not path.exists():
        raise DataError(
            f"Dataset not found at {path}. Download it from {KAGGLE_URL} and place {IBM_FILENAME} in {DATA_DIR}."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Missing columns: {missing}")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise DataError("Dataset has missing values in required columns")
    y_raw = df[TARGET].astype(str).str.strip().str.lower()
    if not set(y_raw.unique()) <= {"yes", "no"}:
        raise DataError("Attrition must be Yes/No")
    y = (y_raw == "yes").astype(int).rename("y")
    X = to_codes(df)
    if "EmployeeNumber" in df.columns:
        X.index = df["EmployeeNumber"].astype(str)
        y.index = X.index
    info = {
        "path": str(path),
        "rows": int(len(df)),
        "leavers": int(y.sum()),
        "attrition_rate": round(float(y.mean()), 4),
        "dropped_columns": {c: r for c, r in DROPPED.items() if c in df.columns},
        "warnings": [],
    }
    standin = path.name == STANDIN_PATH.name
    info["source"] = "synthetic_standin" if standin else "ibm_kaggle"
    if not standin and (info["rows"] != EXPECTED_ROWS or info["leavers"] != EXPECTED_LEAVERS):
        info["warnings"].append(
            f"Expected {EXPECTED_ROWS} rows and {EXPECTED_LEAVERS} leavers for the IBM dataset; "
            f"got {info['rows']} and {info['leavers']}. Check the file."
        )
    if standin:
        info["warnings"].append("SYNTHETIC STAND-IN DATA FOR SOFTWARE TESTS ONLY. Do not report these numbers.")
    return X, y, info


def generate_standin(n: int = 1470, seed: int = 7, target_rate: float = 0.16) -> pd.DataFrame:
    """Synthetic rows with the IBM schema, for software tests only (not a substitute for the dataset)."""
    rng = np.random.default_rng(seed)
    dept = rng.choice([0, 1, 2], size=n, p=[0.05, 0.65, 0.30])
    roles_by_dept = {0: [1, 3], 1: [6, 2, 4, 0, 5, 3], 2: [7, 8, 3]}
    role_p = {0: [0.9, 0.1], 1: [0.30, 0.27, 0.15, 0.14, 0.08, 0.06], 2: [0.72, 0.18, 0.10]}
    role = np.array([rng.choice(roles_by_dept[d], p=role_p[d]) for d in dept])
    level = np.clip(rng.choice([1, 2, 3, 4, 5], size=n, p=[0.37, 0.36, 0.15, 0.07, 0.05]) + (role == 3) * 2, 1, 5)
    income_rng = {1: (1000, 5000), 2: (2000, 10000), 3: (4000, 14000), 4: (9000, 18000), 5: (17000, 20000)}
    income = np.array([rng.integers(*income_rng[l]) for l in level])
    age = np.clip(rng.normal(37, 9, n).round(), 18, 60)
    total_years = np.clip((age - 18) * rng.uniform(0.3, 1.0, n), 0, 40).round()
    years_company = np.clip(total_years * rng.uniform(0.1, 1.0, n), 0, 40).round()
    years_role = np.clip(years_company * rng.uniform(0.2, 1.0, n), 0, 18).round()
    years_promo = np.clip(years_company * rng.uniform(0.0, 0.7, n), 0, 15).round()
    years_mgr = np.clip(years_company * rng.uniform(0.2, 1.0, n), 0, 17).round()
    marital = rng.choice([0, 1, 2], size=n, p=[0.22, 0.46, 0.32])
    stock = np.where(marital == 2, 0, rng.choice([0, 1, 2, 3], size=n, p=[0.2, 0.5, 0.2, 0.1]))
    overtime = rng.random(n) < 0.28
    travel = rng.choice([0, 1, 2], size=n, p=[0.10, 0.71, 0.19])
    dist = np.clip(rng.gamma(1.6, 6, n).round() + 1, 1, 29)
    jobsat, envsat, relsat, jobinv, wlb = (rng.integers(1, 5, n) for _ in range(5))
    perf = np.where(rng.random(n) < 0.85, 3, 4)
    hike = np.where(perf == 4, rng.integers(20, 26, n), rng.integers(11, 20, n))
    ncomp = rng.integers(0, 10, n)
    train = rng.integers(0, 7, n)
    edu = rng.integers(1, 6, n)
    field = rng.choice(6, size=n)
    gender = rng.random(n) < 0.6
    logit = (
        1.5 * overtime - 0.4 * (jobsat - 2.5) - 0.35 * (envsat - 2.5) - 0.3 * (wlb - 2.5) - 0.3 * (jobinv - 2.5)
        + 0.6 * (travel == 2) + 0.03 * (dist - 9) - 0.00012 * (income - 6500) + 0.6 * (stock == 0)
        - 0.05 * (years_company - 7) + 0.08 * (years_promo - 2) + 0.12 * (ncomp - 2.7) - 0.04 * (age - 37)
        + 0.4 * (marital == 2) + 0.5 * (level == 1) - 0.08 * (train - 2.8) + rng.normal(0, 0.5, n)
    )
    lo, hi = -10.0, 10.0
    for _ in range(60):  # intercept so the attrition rate matches target_rate
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(logit + mid)))).mean()
        lo, hi = (mid, hi) if rate < target_rate else (lo, mid)
    p = 1 / (1 + np.exp(-(logit + lo)))
    y = rng.random(n) < p

    def lab(name, codes):
        return [SPECS[name].options[int(c)] for c in codes]

    df = pd.DataFrame({
        "Age": age.astype(int), "Attrition": np.where(y, "Yes", "No"),
        "BusinessTravel": lab("BusinessTravel", travel), "DailyRate": rng.integers(100, 1500, n),
        "Department": lab("Department", dept), "DistanceFromHome": dist.astype(int), "Education": edu,
        "EducationField": lab("EducationField", field), "EmployeeCount": 1, "EmployeeNumber": np.arange(1, n + 1),
        "EnvironmentSatisfaction": envsat, "Gender": np.where(gender, "Male", "Female"),
        "HourlyRate": rng.integers(30, 101, n), "JobInvolvement": jobinv, "JobLevel": level,
        "JobRole": lab("JobRole", role), "JobSatisfaction": jobsat, "MaritalStatus": lab("MaritalStatus", marital),
        "MonthlyIncome": income, "MonthlyRate": rng.integers(2000, 27000, n), "NumCompaniesWorked": ncomp,
        "Over18": "Y", "OverTime": np.where(overtime, "Yes", "No"), "PercentSalaryHike": hike,
        "PerformanceRating": perf, "RelationshipSatisfaction": relsat, "StandardHours": 80,
        "StockOptionLevel": stock, "TotalWorkingYears": total_years.astype(int), "TrainingTimesLastYear": train,
        "WorkLifeBalance": wlb, "YearsAtCompany": years_company.astype(int),
        "YearsInCurrentRole": years_role.astype(int), "YearsSinceLastPromotion": years_promo.astype(int),
        "YearsWithCurrManager": years_mgr.astype(int),
    })
    return df
