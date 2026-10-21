"""Loading the Pima Indians Diabetes data, plus a clearly labelled synthetic stand-in for tests.

Put `diabetes.csv` in backend/data/ (Kaggle: https://www.kaggle.com/datasets/mathchi/diabetes-data-set, the copy
Musicco et al. 2026 used; the same 768 rows are in the UCI/NIDDK release). Expected columns: Pregnancies, Glucose,
BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome.

Missing values: the dataset records missing Glucose, BloodPressure, SkinThickness, Insulin and BMI as 0, which is not
a possible measurement. They are replaced by the median of the observed values of that column over the whole file
(Insulin is missing in about half the rows, SkinThickness in about 30%). Using all rows for the medians leaks a
little information from the test split into the imputed values; report this, or impute inside the training split if
you change the pipeline. The number of imputed values per column is recorded in the manifest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...config import DATA_DIR
from .spec import FEATURES, TARGET

FILENAME = "diabetes.csv"
REAL_PATH = DATA_DIR / FILENAME
STANDIN_PATH = DATA_DIR / "health_standin_TEST_ONLY.csv"
KAGGLE_URL = "https://www.kaggle.com/datasets/mathchi/diabetes-data-set"
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
EXPECTED_ROWS, EXPECTED_POS = 768, 268
NAMES = [f.name for f in FEATURES]


class DataError(ValueError):
    pass


def load_dataset(path: Path | str) -> tuple[pd.DataFrame, pd.Series, dict]:
    path = Path(path)
    if not path.exists():
        raise DataError(f"Dataset not found at {path}. Download {FILENAME} from {KAGGLE_URL} and place it in {DATA_DIR}.")
    df = pd.read_csv(path)
    missing = [c for c in [*NAMES, TARGET] if c not in df.columns]
    if missing:
        raise DataError(f"Missing columns: {missing}")
    X = df[NAMES].apply(pd.to_numeric, errors="coerce").astype(float)
    if X.isna().any().any():
        raise DataError("Non-numeric values in feature columns")
    imputed = {}
    for c in ZERO_IS_MISSING:
        zero = X[c] <= 0
        imputed[c] = int(zero.sum())
        if zero.any():
            med = float(X.loc[~zero, c].median())
            X.loc[zero, c] = round(med, 1) if c == "BMI" else round(med)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    if not set(y.dropna().unique()) <= {0, 1} or y.isna().any():
        raise DataError("Outcome must be 0/1")
    y = y.astype(int).rename("y")
    standin = path.name == STANDIN_PATH.name
    info = {"path": str(path), "rows": int(len(df)), "positives": int(y.sum()), "positive_rate": round(float(y.mean()), 4),
            "imputed_zero_values": imputed, "imputation": "median of observed values over all rows",
            "source": "synthetic_standin" if standin else "pima_niddk", "warnings": []}
    if standin:
        info["warnings"].append("SYNTHETIC STAND-IN DATA FOR SOFTWARE TESTS ONLY. Do not report these numbers.")
    elif (info["rows"], info["positives"]) != (EXPECTED_ROWS, EXPECTED_POS):
        info["warnings"].append(f"Expected {EXPECTED_ROWS} rows and {EXPECTED_POS} positives; got {info['rows']} and "
                                f"{info['positives']}. Check the file.")
    return X, y, info


def generate_standin(n: int = 768, seed: int = 5, target_rate: float = 0.35) -> pd.DataFrame:
    """Synthetic rows with the Pima schema (including zeros for missing values), for software tests only."""
    rng = np.random.default_rng(seed)
    age = np.clip(21 + rng.gamma(2.0, 6.5, n).round(), 21, 81)
    preg = np.clip(rng.poisson(np.clip((age - 20) / 6, 0.5, 6)), 0, 15)
    bmi = np.clip(rng.normal(32, 6.5, n), 18.2, 60).round(1)
    glucose = np.clip(rng.normal(115 + 0.4 * (bmi - 32) + 0.3 * (age - 33), 28, n), 56, 199).round()
    bp = np.clip(rng.normal(70 + 0.3 * (age - 33), 11, n), 38, 122).round()
    skin = np.clip(rng.normal(20 + 0.8 * (bmi - 32), 8, n), 7, 63).round()
    insulin = np.clip(rng.gamma(2.2, 60, n) + 0.8 * (glucose - 110), 14, 846).round()
    dpf = np.clip(rng.gamma(2.0, 0.24, n), 0.078, 2.42).round(3)
    logit = (0.035 * (glucose - 120) + 0.08 * (bmi - 32) + 0.9 * (dpf - 0.47) + 0.03 * (age - 33) + 0.08 * (preg - 3.8)
             + 0.004 * (bp - 70) + rng.normal(0, 0.4, n))
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (1 / (1 + np.exp(-(logit + mid)))).mean() < target_rate else (lo, mid)
    y = rng.random(n) < 1 / (1 + np.exp(-(logit + lo)))
    df = pd.DataFrame({"Pregnancies": preg.astype(int), "Glucose": glucose.astype(int), "BloodPressure": bp.astype(int),
                       "SkinThickness": skin.astype(int), "Insulin": insulin.astype(int), "BMI": bmi,
                       "DiabetesPedigreeFunction": dpf, "Age": age.astype(int), TARGET: y.astype(int)})
    # missing measurements recorded as 0, as in the real data
    for c, share in (("Insulin", 0.48), ("SkinThickness", 0.30), ("BloodPressure", 0.05), ("BMI", 0.015), ("Glucose", 0.007)):
        df.loc[rng.random(n) < share, c] = 0
    return df
