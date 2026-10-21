"""Train and validate the HR attrition models (main + audit), then write artefacts and the manifest.

  python scripts/train_hr.py            # uses data/WA_Fn-UseC_-HR-Employee-Attrition.csv
  python scripts/train_hr.py --standin  # synthetic stand-in, software tests only
  python scripts/train_hr.py --fast     # fewer CV repeats / bootstraps (quick checks)

Outputs in backend/models/: hr_main.joblib, hr_audit.joblib, hr_manifest.json, reliability_hr.png
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import MODELS_DIR  # noqa: E402
from app.domains.hr.data import IBM_PATH, STANDIN_PATH, generate_standin, load_dataset  # noqa: E402
from app.domains.hr.spec import FEATURES, PROTECTED  # noqa: E402
from app.ml.fairness import fairness_report, proxy_check  # noqa: E402
from app.ml.training import TrainConfig, band_of, train_and_validate  # noqa: E402

NOMINAL_ALL = [f.name for f in FEATURES if f.kind == "nominal"]
SPEC = {f.name: f for f in FEATURES}


def age_band(a: float) -> str:
    if a < 30:
        return "18-29"
    if a < 40:
        return "30-39"
    if a < 50:
        return "40-49"
    return "50+"


def protected_groups(Xall: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "gender": Xall["Gender"].map(lambda c: SPEC["Gender"].labels[int(c)]),
        "age_band": Xall["Age"].map(age_band),
        "marital_status": Xall["MaritalStatus"].map(lambda c: SPEC["MaritalStatus"].labels[int(c)]),
    }


def hr_domain_stats(X: pd.DataFrame) -> dict:
    income = {}
    for lvl, g in X.groupby("JobLevel"):
        income[int(lvl)] = {"min": float(g.MonthlyIncome.min()), "p25": float(g.MonthlyIncome.quantile(0.25)),
                            "max": float(g.MonthlyIncome.max())}
    roles = sorted({(int(d), int(r)) for d, r in zip(X.Department, X.JobRole)})
    return {"income_by_level": income, "role_department_pairs": roles,
            "training_max": float(X.TrainingTimesLastYear.max())}


def aggregate_view(Xall: pd.DataFrame, y: pd.Series, p: np.ndarray, art: dict) -> dict:
    low_max, high_min = art["band_edges"]["low_max"], art["band_edges"]["high_min"]
    df = Xall.assign(_y=y.values, _p=p, _flag=p >= art["threshold"])
    out = {}
    for col in ["Department", "JobRole", "OverTime", "BusinessTravel", "JobLevel"]:
        rows = []
        for code, g in df.groupby(col):
            rows.append({"group": SPEC[col].readable(code), "n": int(len(g)), "observed_rate": float(g._y.mean()),
                         "mean_estimate": float(g._p.mean()), "flag_rate": float(g._flag.mean())})
        out[col] = sorted(rows, key=lambda r: -r["observed_rate"])
    bands = [band_of(v, low_max, high_min) for v in p]
    out["bands"] = {b: int(bands.count(b)) for b in ["low", "medium", "high"]}
    out["n"] = int(len(df))
    out["observed_rate"] = float(df._y.mean())
    out["note"] = "Out-of-fold estimates (5-fold) on all rows. Group summaries, not individual judgements."
    return out


def plot_reliability(manifest_main: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=150)
    for ax, key, title in [(axes[0], "reliability_oof_all", "Out-of-fold, all rows"),
                           (axes[1], "reliability_test", "Held-out test split")]:
        rel = manifest_main[key]
        xs = [b["mean_predicted"] for b in rel["bins"]]
        ys = [b["observed"] for b in rel["bins"]]
        err = [[b["observed"] - b["ci_low"] for b in rel["bins"]], [b["ci_high"] - b["observed"] for b in rel["bins"]]]
        ax.plot([0, 1], [0, 1], linestyle="--", color="#8a9099", linewidth=1, label="Perfect calibration")
        ax.errorbar(xs, ys, yerr=err, fmt="o-", color="#0b6e69", ecolor="#0b6e69", capsize=3, label="Observed (95% Wilson CI)")
        ax.set_xlim(0, max(0.8, max(xs) + 0.05))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability (bin)")
        ax.set_ylabel("Observed attrition rate")
        ax.set_title(f"{title}\nECE {rel['ece']:.3f}, Brier {rel['brier']:.3f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    try:  # compress
        from PIL import Image

        Image.open(path).convert("P", palette=Image.ADAPTIVE, colors=64).save(path, optimize=True)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--standin", action="store_true")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    if args.standin:
        if not STANDIN_PATH.exists():
            STANDIN_PATH.parent.mkdir(parents=True, exist_ok=True)
            generate_standin().to_csv(STANDIN_PATH, index=False)
        path = STANDIN_PATH
    else:
        path = Path(args.data) if args.data else IBM_PATH
    Xall, y, info = load_dataset(path)
    print(f"Data: {info['rows']} rows, {info['leavers']} leavers ({info['attrition_rate']:.1%}), source={info['source']}")
    for w in info["warnings"]:
        print("WARNING:", w)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:10]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    manifest: dict = {"dataset": info | {"sha256_prefix": digest}, "trained_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    main_features = [f.name for f in FEATURES if f.name not in PROTECTED]
    variants = {"main": main_features, "audit": [f.name for f in FEATURES]}
    oof = {}
    for variant, feats in variants.items():
        print(f"\nTraining {variant} model ({len(feats)} features)")
        cfg = TrainConfig(features=feats, nominal=[c for c in NOMINAL_ALL if c in feats])
        art, man = train_and_validate(Xall, y, cfg, fast=args.fast)
        train_idx = art.pop("train_index")
        # recourse bounds (income ranges per level, valid role/department pairs, training max) come from the
        # training split only, so nothing about the held-out test rows leaks into recourse
        art["domain_stats"] = hr_domain_stats(Xall.loc[train_idx])
        art["domain_stats_source"] = "training split"
        man["domain_stats_source"] = "training split"
        art["version"] = f"hr-{variant}-{stamp}-{digest}"
        art["data_source"] = info["source"]
        oof[variant] = art.pop("oof_all")
        if variant == "main":
            art["aggregate"] = aggregate_view(Xall, y, oof[variant], art)
        man["version"] = art["version"]
        manifest[variant] = man
        joblib.dump(art, MODELS_DIR / f"hr_{variant}.joblib", compress=3)
        h = man["headline"]
        print(f"  test ROC-AUC {h['roc_auc'][0]:.3f} {h['roc_auc'][1]}, PR-AUC {h['pr_auc'][0]:.3f} {h['pr_auc'][1]}, "
              f"recall {h['recall'][0]:.3f} {h['recall'][1]}, cost/employee {h['cost_per_employee'][0]:.3f}")

    groups = protected_groups(Xall)
    t_main, t_audit = manifest["main"]["threshold"], manifest["audit"]["threshold"]
    manifest["fairness"] = {
        "main": fairness_report(y.values, oof["main"], t_main, groups),
        "audit": fairness_report(y.values, oof["audit"], t_audit, groups),
        "proxy_check": proxy_check(Xall[main_features], groups,
                                   [c for c in NOMINAL_ALL if c in main_features], seed=0),
        "note": ("Out-of-fold predictions (5-fold) on all rows. The main model does not use age, gender or marital "
                 "status; the audit model does, for comparison only."),
    }
    manifest["aggregate"] = joblib.load(MODELS_DIR / "hr_main.joblib")["aggregate"]
    plot_reliability(manifest["main"], MODELS_DIR / "reliability_hr.png")
    (MODELS_DIR / "hr_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote models to {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
