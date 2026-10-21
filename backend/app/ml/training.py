"""Domain-agnostic training, validation and artefact building for tabular risk models.

Produces: a calibrated model, a cost-optimal threshold, risk bands, a bootstrap
ensemble for uncertainty, an out-of-distribution detector, and a manifest with
cross-validated and held-out metrics (with confidence intervals).
"""
from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..xai.shapley import builtin_permutation_shapley


@dataclass
class TrainConfig:
    features: list[str]
    nominal: list[str]  # one-hot encoded; all other features are scaled numerics
    cost_fn: float = 5.0  # cost of missing a leaver
    cost_fp: float = 1.0  # cost of a false flag
    high_band_min: float = 0.5
    test_size: float = 0.2
    cv_splits: int = 5
    cv_repeats: int = 3
    calib_cv: int = 5
    n_bootstrap_ci: int = 1000
    n_ensemble: int = 25
    reliability_bins: int = 8
    candidates: list[str] = field(default_factory=lambda: ["logreg", "hgb"])
    seed: int = 42


def build_model(kind: str, cfg: TrainConfig, calib_cv: int | None = None, seed: int | None = None):
    seed = cfg.seed if seed is None else seed
    numeric = [f for f in cfg.features if f not in cfg.nominal]
    pre = ColumnTransformer(
        [("num", StandardScaler(), numeric),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [f for f in cfg.nominal if f in cfg.features])],
        remainder="drop",
    )
    if kind == "logreg":
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000)
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=3, max_iter=200, l2_regularization=1.0, class_weight="balanced", random_state=seed
        )
    else:
        raise ValueError(f"unknown model kind {kind}")
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    # Platt (sigmoid) calibration: isotonic overfits with ~240 positives.
    return CalibratedClassifierCV(pipe, method="sigmoid", cv=calib_cv or cfg.calib_cv)


def proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def oof_proba(kind: str, cfg: TrainConfig, X: pd.DataFrame, y: pd.Series, seed: int) -> np.ndarray:
    out = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = build_model(kind, cfg).fit(X.iloc[tr], y.iloc[tr])
        out[te] = proba(m, X.iloc[te])
    return out


def cost(y: np.ndarray, flag: np.ndarray, cfg: TrainConfig) -> float:
    fn = int(((y == 1) & (~flag)).sum())
    fp = int(((y == 0) & flag).sum())
    return (cfg.cost_fn * fn + cfg.cost_fp * fp) / len(y)


def choose_threshold(y: np.ndarray, p: np.ndarray, cfg: TrainConfig) -> tuple[float, list[dict]]:
    grid = np.round(np.arange(0.02, 0.91, 0.01), 2)
    curve = [{"t": float(t), "cost": round(cost(y, p >= t, cfg), 5)} for t in grid]
    best = min(curve, key=lambda r: (r["cost"], r["t"]))
    return best["t"], curve


def metrics_at(y: np.ndarray, p: np.ndarray, t: float, cfg: TrainConfig) -> dict:
    flag = p >= t
    tp = int(((y == 1) & flag).sum())
    fp = int(((y == 0) & flag).sum())
    fn = int(((y == 1) & ~flag).sum())
    tn = int(((y == 0) & ~flag).sum())
    rec = tp / (tp + fn) if tp + fn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    return {
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "recall": rec,
        "precision": prec,
        "f1": 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan"),
        "cost_per_employee": cost(y, flag, cfg),
        "cost_flag_nobody": cost(y, np.zeros_like(flag), cfg),
        "cost_flag_everyone": cost(y, np.ones_like(flag), cfg),
        "flag_rate": float(flag.mean()),
        "accuracy": (tp + tn) / len(y),
        "brier": float(brier_score_loss(y, p)),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, t: float, cfg: TrainConfig, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    keys = ["roc_auc", "pr_auc", "recall", "precision", "cost_per_employee", "accuracy"]
    vals: dict[str, list[float]] = {k: [] for k in keys}
    n = len(y)
    for _ in range(cfg.n_bootstrap_ci):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        m = metrics_at(y[idx], p[idx], t, cfg)
        for k in keys:
            if m[k] == m[k]:
                vals[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items() if v}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def reliability(y: np.ndarray, p: np.ndarray, bins: int) -> dict:
    order = np.argsort(p)
    chunks = np.array_split(order, bins)  # equal-count bins (stable with few positives)
    rows, ece = [], 0.0
    for c in chunks:
        if len(c) == 0:
            continue
        k, n = int(y[c].sum()), len(c)
        lo, hi = wilson(k, n)
        mp, obs = float(p[c].mean()), k / n
        ece += n / len(y) * abs(mp - obs)
        rows.append({"mean_predicted": mp, "observed": obs, "ci_low": lo, "ci_high": hi, "n": n})
    return {"bins": rows, "ece": ece, "brier": float(brier_score_loss(y, p))}


def band_of(p: float, low_max: float, high_min: float) -> str:
    if p < low_max:
        return "low"
    if p < high_min:
        return "medium"
    return "high"


def fit_ood(X: pd.DataFrame, seed: int) -> dict:
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(X.values)
    scores = iso.score_samples(X.values)
    return {
        "iso": iso,
        "score_threshold": float(np.percentile(scores, 1)),
        "score_median": float(np.median(scores)),  # used by the escalation score (app/escalation.py)
        "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in X.columns},
    }


def global_importance(model, X_bg: pd.DataFrame, X_eval: pd.DataFrame, n_perm: int = 8, seed: int = 0) -> list[dict]:
    cols = list(X_bg.columns)

    def f(a: np.ndarray) -> np.ndarray:
        return proba(model, pd.DataFrame(a, columns=cols))

    vals = []
    for i in range(len(X_eval)):
        phi, _, _ = builtin_permutation_shapley(f, X_eval.values[i], X_bg.values, n_perm=n_perm, seed=seed + i)
        vals.append(phi)
    arr = np.array(vals)
    out = [
        {"feature": c, "mean_abs": float(np.abs(arr[:, j]).mean()), "mean": float(arr[:, j].mean())}
        for j, c in enumerate(cols)
    ]
    return sorted(out, key=lambda r: -r["mean_abs"])


def _clean(obj):
    """Make floats JSON safe (NaN -> None)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if f != f else round(f, 6)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def train_and_validate(
    X: pd.DataFrame, y: pd.Series, cfg: TrainConfig, log: Callable[[str], None] = print, fast: bool = False
) -> tuple[dict, dict]:
    """Returns (artifact, manifest). X must contain exactly cfg.features in model space."""
    t0 = time.time()
    X = X[cfg.features].astype(float)
    y = y.astype(int)
    if fast:
        cfg.cv_repeats, cfg.n_bootstrap_ci, cfg.n_ensemble = 1, 200, 8
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed)

    # 1) model selection with repeated stratified CV (PR-AUC is the selection metric)
    rskf = RepeatedStratifiedKFold(n_splits=cfg.cv_splits, n_repeats=cfg.cv_repeats, random_state=cfg.seed)
    cv_results = {}
    for kind in cfg.candidates:
        rocs, prs = [], []
        for tr, te in rskf.split(Xtr, ytr):
            m = build_model(kind, cfg).fit(Xtr.iloc[tr], ytr.iloc[tr])
            p = proba(m, Xtr.iloc[te])
            rocs.append(roc_auc_score(ytr.iloc[te], p))
            prs.append(average_precision_score(ytr.iloc[te], p))
        cv_results[kind] = {
            "roc_auc_mean": float(np.mean(rocs)), "roc_auc_sd": float(np.std(rocs, ddof=1)),
            "roc_auc_range95": [float(np.percentile(rocs, 2.5)), float(np.percentile(rocs, 97.5))],
            "pr_auc_mean": float(np.mean(prs)), "pr_auc_sd": float(np.std(prs, ddof=1)),
            "pr_auc_range95": [float(np.percentile(prs, 2.5)), float(np.percentile(prs, 97.5))],
            "folds": len(rocs),
        }
        log(f"  CV {kind}: ROC-AUC {np.mean(rocs):.3f} +/- {np.std(rocs):.3f}, PR-AUC {np.mean(prs):.3f} +/- {np.std(prs):.3f}")
    best = max(cv_results, key=lambda k: cv_results[k]["pr_auc_mean"])
    log(f"  selected model: {best}")

    # 2) threshold from out-of-fold predictions on the training split (never the test split)
    p_oof_tr = oof_proba(best, cfg, Xtr, ytr, seed=cfg.seed)
    t, cost_curve = choose_threshold(ytr.values, p_oof_tr, cfg)
    high_min = max(cfg.high_band_min, t)

    # 3) final model on the training split, evaluated once on the held-out split
    model = build_model(best, cfg).fit(Xtr, ytr)
    p_te = proba(model, Xte)
    test_metrics = metrics_at(yte.values, p_te, t, cfg)
    test_ci = bootstrap_ci(yte.values, p_te, t, cfg, seed=cfg.seed)

    # cost-ratio sensitivity: the same model and folds, with the threshold re-chosen for other FN:FP ratios on the
    # training-split out-of-fold predictions, then evaluated once on the held-out split. Shows how much the flag rate
    # and the headline numbers depend on the assumed cost ratio.
    sensitivity = []
    for ratio in (1.0, 2.0, 3.0, 5.0, 10.0, 20.0):
        cfg_r = dataclasses.replace(cfg, cost_fn=ratio * cfg.cost_fp)
        t_r, _ = choose_threshold(ytr.values, p_oof_tr, cfg_r)
        m_r = metrics_at(yte.values, p_te, t_r, cfg_r)
        sensitivity.append({"fn_to_fp": ratio, "threshold": float(t_r), "flag_rate": m_r["flag_rate"],
                            "recall": m_r["recall"], "precision": m_r["precision"],
                            "cost_per_case": m_r["cost_per_employee"], "cost_flag_nobody": m_r["cost_flag_nobody"],
                            "cost_flag_everyone": m_r["cost_flag_everyone"]})

    # 4) out-of-fold predictions on all rows (reliability curve, fairness, aggregate views)
    p_oof_all = oof_proba(best, cfg, X, y, seed=cfg.seed + 1)

    # 5) bootstrap ensemble for per-case uncertainty
    rng = np.random.default_rng(cfg.seed)
    ensemble = []
    for b in range(cfg.n_ensemble):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        if ytr.iloc[idx].nunique() < 2:
            continue
        ensemble.append(build_model(best, cfg, calib_cv=3, seed=cfg.seed + b).fit(Xtr.iloc[idx], ytr.iloc[idx]))

    # 6) OOD detector + escalation rate on the test split
    ood = fit_ood(Xtr, cfg.seed)
    from ..config import settings  # the same interval percentiles the runtime uses for escalation

    ens_te = np.array([proba(m, Xte) for m in ensemble])
    lo_te = np.percentile(ens_te, settings.interval_low_pct, axis=0)
    hi_te = np.percentile(ens_te, settings.interval_high_pct, axis=0)
    uncertain = np.array([band_of(a, t, high_min) != band_of(b, t, high_min) for a, b in zip(lo_te, hi_te)])
    ood_flag = ood["iso"].score_samples(Xte.values) < ood["score_threshold"]

    # 7) global importance on a sample of held-out rows
    bg = Xtr.sample(n=min(50, len(Xtr)), random_state=cfg.seed)
    imp_rows = Xte.sample(n=min(40 if fast else 120, len(Xte)), random_state=cfg.seed)
    importance = global_importance(model, bg.iloc[:20], imp_rows, n_perm=6 if fast else 8, seed=cfg.seed)

    stats = {
        "medians": {c: float(Xtr[c].median()) for c in cfg.features if c not in cfg.nominal},
        "modes": {c: float(Xtr[c].mode().iloc[0]) for c in cfg.features},
    }
    artifact = {
        "features": list(cfg.features),
        "nominal": list(cfg.nominal),
        "kind": best,
        "model": model,
        "ensemble": ensemble,
        "threshold": float(t),
        "band_edges": {"low_max": float(t), "high_min": float(high_min)},
        "ood": ood,
        "background": bg,
        "stats": stats,
        "global_importance": importance,
        "examples": Xte.assign(_y=yte.values, _p=p_te).head(60),
    }
    manifest = _clean({
        "model_kind": best,
        "features": list(cfg.features),
        "n_train": len(Xtr), "n_test": len(Xte),
        "positives_train": int(ytr.sum()), "positives_test": int(yte.sum()),
        "cost_matrix": {"false_negative": cfg.cost_fn, "false_positive": cfg.cost_fp,
                        "note": "Cost per employee = (cost_fn*FN + cost_fp*FP) / n. Lower is better."},
        "calibration": "Platt (sigmoid), 5-fold CalibratedClassifierCV",
        "class_weighting": "balanced",
        "threshold": t,
        "band_edges": {"low": f"p < {t:.2f}", "medium": f"{t:.2f} <= p < {high_min:.2f}", "high": f"p >= {high_min:.2f}"},
        "cv": cv_results,
        "test": {"metrics": test_metrics, "ci95": test_ci},
        "headline": {
            "roc_auc": [test_metrics["roc_auc"], test_ci.get("roc_auc")],
            "pr_auc": [test_metrics["pr_auc"], test_ci.get("pr_auc")],
            "recall": [test_metrics["recall"], test_ci.get("recall")],
            "cost_per_employee": [test_metrics["cost_per_employee"], test_ci.get("cost_per_employee")],
            "note": "Accuracy is reported only as a secondary metric; the base rate of staying is about 84%.",
        },
        "reliability_test": reliability(yte.values, p_te, cfg.reliability_bins),
        "reliability_oof_all": reliability(y.values, p_oof_all, cfg.reliability_bins),
        "uncertainty": {
            "ensemble_size": len(ensemble),
            "interval": f"{settings.interval_low_pct:g}th-{settings.interval_high_pct:g}th percentile of the bootstrap ensemble",
            "band_uncertain_rate_test": float(uncertain.mean()),
            "ood_rate_test": float(ood_flag.mean()),
            "escalation_rate_test": float((uncertain | ood_flag).mean()),
        },
        "cost_curve": cost_curve,
        "cost_sensitivity": sensitivity,
        "global_importance": importance,
        "train_seconds": round(time.time() - t0, 1),
    })
    artifact["oof_all"] = p_oof_all
    artifact["train_index"] = list(Xtr.index)  # callers pop this to compute training-split-only statistics
    return artifact, manifest
