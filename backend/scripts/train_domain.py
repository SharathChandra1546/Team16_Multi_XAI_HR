"""Train and validate any domain's models (main + audit) from its DomainTraining spec, then write artefacts.

  python scripts/train_domain.py --domain finance            # real dataset (see the domain's data.py for the file)
  python scripts/train_domain.py --domain health --standin   # synthetic stand-in, software tests only
  python scripts/train_domain.py --domain finance --fast     # fewer CV repeats / bootstraps (quick checks)

HR keeps its own script (scripts/train_hr.py); this one is for every domain added after it.
Outputs in backend/models/: <name>_main.joblib, <name>_audit.joblib (if the domain has protected attributes),
<name>_manifest.json, reliability_<name>.png
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import MODELS_DIR  # noqa: E402
from app.ml.domain_training import DomainTraining  # noqa: E402
from app.ml.fairness import fairness_report, proxy_check  # noqa: E402
from app.ml.training import TrainConfig, band_of, train_and_validate  # noqa: E402


def load_spec(name: str) -> DomainTraining:
    return importlib.import_module(f"app.domains.{name}.training").TRAINING


def aggregate_view(spec: DomainTraining, Xall, y, p: np.ndarray, art: dict, readable) -> dict:
    low_max, high_min = art["band_edges"]["low_max"], art["band_edges"]["high_min"]
    df = Xall.assign(_y=y.values, _p=p, _flag=p >= art["threshold"])
    out = {}
    for col in spec.aggregate_columns:
        rows = []
        for code, g in df.groupby(col):
            rows.append({"group": readable(col, code), "n": int(len(g)), "observed_rate": float(g._y.mean()),
                         "mean_estimate": float(g._p.mean()), "flag_rate": float(g._flag.mean())})
        out[col] = sorted(rows, key=lambda r: -r["observed_rate"])
    bands = [band_of(v, low_max, high_min) for v in p]
    out["bands"] = {b: int(bands.count(b)) for b in ["low", "medium", "high"]}
    out["n"] = int(len(df))
    out["observed_rate"] = float(df._y.mean())
    out["note"] = "Out-of-fold estimates (5-fold) on all rows. Group summaries, not individual judgements."
    return out


def plot_reliability(manifest_main: dict, path: Path, ylabel: str) -> None:
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
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability (bin)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nECE {rel['ece']:.3f}, Brier {rel['brier']:.3f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--standin", action="store_true")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    spec = load_spec(args.domain)
    from app.domains.registry import agent_class

    agent = agent_class(args.domain)(models_dir=MODELS_DIR)

    if args.standin:
        if not spec.standin_path.exists():
            spec.standin_path.parent.mkdir(parents=True, exist_ok=True)
            spec.write_standin(spec.generate_standin(), spec.standin_path)
        path = spec.standin_path
    else:
        path = Path(args.data) if args.data else spec.real_path
    Xall, y, info = spec.load(path)
    print(f"[{spec.name}] Data: {len(Xall)} rows, {int(y.sum())} positives ({y.mean():.1%}), source={info['source']}")
    for w in info.get("warnings", []):
        print("WARNING:", w)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:10]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    manifest: dict = {"dataset": info | {"sha256_prefix": digest}, "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "cost_rationale": spec.cost_note}
    main_features = [f for f in spec.features if f not in spec.protected]
    variants = {"main": main_features}
    if spec.protected:
        variants["audit"] = list(spec.features)
    oof = {}
    for variant, feats in variants.items():
        print(f"\nTraining {variant} model ({len(feats)} features)")
        cfg = TrainConfig(features=feats, nominal=[c for c in spec.nominal if c in feats],
                          cost_fn=spec.cost_fn, cost_fp=spec.cost_fp)
        art, man = train_and_validate(Xall, y, cfg, fast=args.fast)
        train_idx = art.pop("train_index")
        art["domain_stats"] = spec.domain_stats(Xall.loc[train_idx])
        art["domain_stats_source"] = "training split"
        man["domain_stats_source"] = "training split"
        art["version"] = f"{spec.name}-{variant}-{stamp}-{digest}"
        art["data_source"] = info["source"]
        oof[variant] = art.pop("oof_all")
        if variant == "main":
            art["aggregate"] = aggregate_view(spec, Xall, y, oof[variant], art, agent.readable)
        man["version"] = art["version"]
        manifest[variant] = man
        joblib.dump(art, MODELS_DIR / f"{spec.name}_{variant}.joblib", compress=3)
        h = man["headline"]
        print(f"  test ROC-AUC {h['roc_auc'][0]:.3f} {h['roc_auc'][1]}, PR-AUC {h['pr_auc'][0]:.3f} {h['pr_auc'][1]}, "
              f"recall {h['recall'][0]:.3f} {h['recall'][1]}, cost/case {h['cost_per_employee'][0]:.3f}")

    groups = spec.protected_groups(Xall)
    if groups:
        fair = {v: fairness_report(y.values, oof[v], manifest[v]["threshold"], groups) for v in variants}
        fair["proxy_check"] = proxy_check(Xall[main_features], groups, [c for c in spec.nominal if c in main_features], seed=0)
        fair["note"] = ("Out-of-fold predictions (5-fold) on all rows. The main model does not use the protected "
                        f"attributes ({', '.join(spec.protected) or 'none'}); the audit model does, for comparison only.")
        manifest["fairness"] = fair
    manifest["aggregate"] = joblib.load(MODELS_DIR / f"{spec.name}_main.joblib")["aggregate"]
    plot_reliability(manifest["main"], MODELS_DIR / f"reliability_{spec.name}.png", spec.outcome_label)
    (MODELS_DIR / f"{spec.name}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {spec.name} models to {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
