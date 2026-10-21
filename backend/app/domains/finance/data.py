"""Loading the Statlog German Credit data, plus a clearly labelled synthetic stand-in for tests.

Accepted files (put in backend/data/):
- german.data: the original UCI file (space-separated, no header, 20 attributes then the class 1 = good, 2 = bad).
  Download: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data (CC BY 4.0).
- any CSV with a header that uses this package's feature names and a `class` column (1/2 or good/bad), with
  categorical values given as the dataset's codes (A11, ...) or as this package's readable labels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...config import DATA_DIR
from .spec import FEATURES, TARGET

UCI_FILENAME = "german.data"
UCI_PATH = DATA_DIR / UCI_FILENAME
STANDIN_PATH = DATA_DIR / "finance_standin_TEST_ONLY.csv"
UCI_URL = "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data"
UCI_ORDER = ["checking_status", "duration", "credit_history", "purpose", "credit_amount", "savings", "employment",
             "installment_rate", "personal_status_sex", "other_debtors", "residence_since", "property", "age",
             "other_installment_plans", "housing", "existing_credits", "job", "num_dependents", "telephone",
             "foreign_worker"]
EXPECTED_ROWS, EXPECTED_BAD = 1000, 300
SPECS = {f.name: f for f in FEATURES}


class DataError(ValueError):
    pass


def to_codes(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for name, spec in SPECS.items():
        col = df[name]
        if spec.categorical:
            lookup = {str(v).strip().lower(): c for c, v in spec.options.items()}
            lookup |= {str(v).strip().lower(): c for c, v in spec.labels.items()}
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


def read_raw(path: Path) -> pd.DataFrame:
    if path.name == UCI_FILENAME or path.suffix == ".data":
        df = pd.read_csv(path, sep=r"\s+", header=None)
        if df.shape[1] != 21:
            raise DataError(f"{path.name} should have 21 columns (20 attributes and the class); found {df.shape[1]}")
        df.columns = [*UCI_ORDER, TARGET]
        return df
    df = pd.read_csv(path)
    missing = [c for c in [*UCI_ORDER, TARGET] if c not in df.columns]
    if missing:
        raise DataError(f"Missing columns: {missing}")
    return df


def load_dataset(path: Path | str) -> tuple[pd.DataFrame, pd.Series, dict]:
    path = Path(path)
    if not path.exists():
        raise DataError(f"Dataset not found at {path}. Download {UCI_FILENAME} from {UCI_URL} and place it in {DATA_DIR}.")
    df = read_raw(path)
    y_raw = df[TARGET].astype(str).str.strip().str.lower()
    if not set(y_raw.unique()) <= {"1", "2", "good", "bad"}:
        raise DataError("class must be 1/2 or good/bad")
    y = y_raw.isin(["2", "bad"]).astype(int).rename("y")  # positive class = bad credit risk
    X = to_codes(df)
    standin = path.name == STANDIN_PATH.name
    info = {"path": str(path), "rows": int(len(df)), "positives": int(y.sum()), "positive_rate": round(float(y.mean()), 4),
            "source": "synthetic_standin" if standin else "uci_statlog", "warnings": []}
    if standin:
        info["warnings"].append("SYNTHETIC STAND-IN DATA FOR SOFTWARE TESTS ONLY. Do not report these numbers.")
    elif (info["rows"], info["positives"]) != (EXPECTED_ROWS, EXPECTED_BAD):
        info["warnings"].append(f"Expected {EXPECTED_ROWS} rows and {EXPECTED_BAD} bad risks; got {info['rows']} and "
                                f"{info['positives']}. Check the file.")
    return X, y, info


def generate_standin(n: int = 1000, seed: int = 11, target_rate: float = 0.30) -> pd.DataFrame:
    """Synthetic rows with the German Credit schema, for software tests only (not a substitute for the dataset)."""
    rng = np.random.default_rng(seed)
    checking = rng.choice(4, n, p=[0.27, 0.27, 0.06, 0.40])
    history = rng.choice(5, n, p=[0.04, 0.05, 0.53, 0.09, 0.29])
    purpose = rng.choice(11, n, p=[0.23, 0.10, 0.18, 0.28, 0.01, 0.02, 0.05, 0.01, 0.01, 0.10, 0.01])
    duration = np.clip(rng.gamma(3.0, 7.0, n).round(), 4, 72)
    amount = np.clip(np.exp(rng.normal(7.8, 0.75, n)) * (1 + duration / 60), 250, 18500).round()
    savings = rng.choice(5, n, p=[0.60, 0.10, 0.06, 0.05, 0.19])
    employment = rng.choice(5, n, p=[0.06, 0.17, 0.34, 0.17, 0.26])
    inst = rng.choice([1, 2, 3, 4], n, p=[0.14, 0.23, 0.16, 0.47])
    personal = rng.choice(5, n, p=[0.05, 0.31, 0.54, 0.09, 0.01])
    debtors = rng.choice(3, n, p=[0.91, 0.04, 0.05])
    residence = rng.choice([1, 2, 3, 4], n, p=[0.13, 0.31, 0.15, 0.41])
    prop = rng.choice(4, n, p=[0.28, 0.23, 0.33, 0.16])
    age = np.clip(rng.gamma(6.0, 6.0, n).round() + 19, 19, 75)
    plans = rng.choice(3, n, p=[0.14, 0.05, 0.81])
    housing = rng.choice(3, n, p=[0.18, 0.71, 0.11])
    credits = rng.choice([1, 2, 3, 4], n, p=[0.63, 0.33, 0.03, 0.01])
    job = rng.choice(4, n, p=[0.02, 0.20, 0.63, 0.15])
    deps = rng.choice([1, 2], n, p=[0.85, 0.15])
    phone = rng.choice(2, n, p=[0.60, 0.40])
    foreign = rng.choice(2, n, p=[0.96, 0.04])
    logit = (0.9 * (checking == 0) + 0.5 * (checking == 1) - 0.8 * (checking == 3) + 0.035 * (duration - 20)
             + 0.00008 * (amount - 3200) + 0.9 * (history <= 1) - 0.5 * (history == 4) + 0.4 * (savings == 0)
             - 0.4 * (savings >= 3) + 0.15 * (inst - 2.9) - 0.25 * (employment - 2) - 0.7 * (debtors == 2)
             + 0.3 * (housing == 0) + 0.35 * (plans == 0) - 0.4 * (purpose == 1) + 0.3 * (purpose == 0)
             - 0.012 * (age - 35) + rng.normal(0, 0.5, n))
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (1 / (1 + np.exp(-(logit + mid)))).mean() < target_rate else (lo, mid)
    y = rng.random(n) < 1 / (1 + np.exp(-(logit + lo)))

    def lab(name, codes):
        return [SPECS[name].options[int(c)] for c in codes]

    return pd.DataFrame({
        "checking_status": lab("checking_status", checking), "duration": duration.astype(int),
        "credit_history": lab("credit_history", history), "purpose": lab("purpose", purpose),
        "credit_amount": amount.astype(int), "savings": lab("savings", savings), "employment": lab("employment", employment),
        "installment_rate": inst, "personal_status_sex": lab("personal_status_sex", personal),
        "other_debtors": lab("other_debtors", debtors), "residence_since": residence, "property": lab("property", prop),
        "age": age.astype(int), "other_installment_plans": lab("other_installment_plans", plans),
        "housing": lab("housing", housing), "existing_credits": credits, "job": lab("job", job),
        "num_dependents": deps, "telephone": lab("telephone", phone), "foreign_worker": lab("foreign_worker", foreign),
        TARGET: np.where(y, 2, 1),
    })
