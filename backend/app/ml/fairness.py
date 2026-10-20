"""Fairness audit: error and flag rates by group, disparate-impact ratio, and proxy-feature check."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

FOUR_FIFTHS = 0.8
PROXY_AUC_FLAG = 0.70
MIN_GROUP_N = 30  # groups smaller than this are reported but excluded from the ratio and the TPR/FPR gaps


def group_rates(y: np.ndarray, p: np.ndarray, t: float, groups: pd.Series) -> list[dict]:
    flag = p >= t
    rows = []
    for g in sorted(groups.unique(), key=str):
        mask = (groups == g).values
        yy, ff = y[mask], flag[mask]
        pos, neg = (yy == 1).sum(), (yy == 0).sum()
        tp, fp = int((ff & (yy == 1)).sum()), int((ff & (yy == 0)).sum())
        rows.append({
            "group": str(g),
            "n": int(mask.sum()),
            "base_rate": float(yy.mean()) if mask.any() else None,
            "flag_rate": float(ff.mean()) if mask.any() else None,
            "tpr": tp / pos if pos else None,
            "fpr": fp / neg if neg else None,
            "fnr": 1 - tp / pos if pos else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "small_group": bool(mask.sum() < MIN_GROUP_N),
        })
    return rows


def _proxy_auc(X: pd.DataFrame, target: pd.Series, nominal: list[str], seed: int) -> float:
    enc = LabelEncoder().fit(target.astype(str))
    t = enc.transform(target.astype(str))
    numeric = [c for c in X.columns if c not in nominal]
    pre = ColumnTransformer([("num", StandardScaler(), numeric),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in X.columns if c in nominal])])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pr = cross_val_predict(pipe, X, t, cv=cv, method="predict_proba")
    if len(enc.classes_) == 2:
        return float(roc_auc_score(t, pr[:, 1]))
    return float(roc_auc_score(t, pr, multi_class="ovr", average="macro"))


def proxy_check(X: pd.DataFrame, protected: dict[str, pd.Series], nominal: list[str], seed: int = 0, top: int = 5) -> dict:
    """How well can each protected attribute be predicted from the model's features (all together and one by one)?"""
    out = {}
    for attr, target in protected.items():
        all_auc = _proxy_auc(X, target, nominal, seed)
        singles = []
        for c in X.columns:
            singles.append({"feature": c, "auc": _proxy_auc(X[[c]], target, [c] if c in nominal else [], seed)})
        singles.sort(key=lambda r: -r["auc"])
        out[attr] = {
            "all_features_auc": all_auc,
            "flagged": all_auc >= PROXY_AUC_FLAG,
            "top_single_features": singles[:top],
            "note": f"AUC from 5-fold logistic regression. {PROXY_AUC_FLAG:.2f} or higher is flagged as a strong proxy.",
        }
    return out


def fairness_report(y: np.ndarray, p: np.ndarray, t: float, groups: dict[str, pd.Series]) -> dict:
    """Group rates at the model's own decision threshold `t` (the same threshold that flags cases in use).

    One rule for every attribute and every model: groups with fewer than MIN_GROUP_N rows are listed but left out
    of the disparate-impact ratio and the TPR/FPR gaps, because their rates are too noisy to compare.
    """
    out = {}
    for attr, g in groups.items():
        rows = group_rates(y, p, t, g)
        big = [r for r in rows if not r["small_group"] and r["n"] > 0]
        flag_rates = [(r["group"], r["flag_rate"]) for r in big if r["flag_rate"] is not None]
        dir_, lo, hi = None, (None, None), (None, None)
        if len(flag_rates) >= 2:
            lo = min(flag_rates, key=lambda r: r[1])
            hi = max(flag_rates, key=lambda r: r[1])
            dir_ = lo[1] / hi[1] if hi[1] > 0 else None
        tprs = [r["tpr"] for r in big if r["tpr"] is not None]
        fprs = [r["fpr"] for r in big if r["fpr"] is not None]
        out[attr] = {
            "groups": rows,
            "threshold": t,
            "min_group_n": MIN_GROUP_N,
            "excluded_small_groups": [r["group"] for r in rows if r["small_group"]],
            "disparate_impact_ratio": dir_,
            "dir_groups": {"lowest_flag_rate": lo[0], "highest_flag_rate": hi[0]},
            "passes_four_fifths": None if dir_ is None else dir_ >= FOUR_FIFTHS,
            "tpr_gap": max(tprs) - min(tprs) if len(tprs) >= 2 else None,
            "fpr_gap": max(fprs) - min(fprs) if len(fprs) >= 2 else None,
        }
    return out
