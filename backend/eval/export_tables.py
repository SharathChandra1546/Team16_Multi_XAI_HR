"""Export thesis tables from the training manifest (model metrics with CIs, calibration, fairness, proxy check).

  python eval/export_tables.py   -> results/model_and_fairness_tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ci(v) -> str:
    if not v or v[1] is None:
        return "n/a"
    point, (lo, hi) = v[0], v[1]
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def pct(v) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def main() -> int:
    m = json.loads((ROOT / "models" / "hr_manifest.json").read_text(encoding="utf-8"))
    ds = m["dataset"]
    out = ["# Model and fairness tables", ""]
    if ds.get("source") == "synthetic_standin":
        out += ["> Synthetic stand-in data. Retrain on the IBM CSV before using these tables.", ""]
    out += [f"Data: {ds['rows']} rows, {ds['leavers']} leavers ({ds['attrition_rate']:.1%}). Source: {ds['source']}.", ""]
    out += ["## Held-out test metrics (95% bootstrap CI)", "",
            "| Model | Kind | ROC-AUC | PR-AUC | Recall | Cost per employee | Threshold |", "|---|---|---|---|---|---|---|"]
    for v in ("main", "audit"):
        h = m[v]["headline"]
        out.append(f"| {v} | {m[v]['model_kind']} | {ci(h['roc_auc'])} | {ci(h['pr_auc'])} | {ci(h['recall'])} | "
                   f"{ci(h['cost_per_employee'])} | {m[v]['threshold']:.2f} |")
    t = m["main"]["test"]["metrics"]
    out += ["", f"Cost matrix: missed leaver = {m['main']['cost_matrix']['false_negative']}, false flag = "
            f"{m['main']['cost_matrix']['false_positive']}. Baselines: flag nobody {t['cost_flag_nobody']:.3f}, flag everyone "
            f"{t['cost_flag_everyone']:.3f}. Accuracy (secondary) {t['accuracy']:.3f}.", ""]
    out += ["## Cross-validation (training split, repeated stratified 5-fold)", "", "| Candidate | ROC-AUC mean +/- sd | PR-AUC mean +/- sd | Folds |",
            "|---|---|---|---|"]
    for k, r in m["main"]["cv"].items():
        out.append(f"| {k} | {r['roc_auc_mean']:.3f} +/- {r['roc_auc_sd']:.3f} | {r['pr_auc_mean']:.3f} +/- {r['pr_auc_sd']:.3f} | {r['folds']} |")
    rel = m["main"]["reliability_oof_all"]
    out += ["", f"## Calibration (main model, out-of-fold): ECE {rel['ece']:.3f}, Brier {rel['brier']:.3f}", "",
            "| Mean predicted | Observed | 95% CI | n |", "|---|---|---|---|"]
    out += [f"| {b['mean_predicted']:.3f} | {b['observed']:.3f} | [{b['ci_low']:.3f}, {b['ci_high']:.3f}] | {b['n']} |" for b in rel["bins"]]
    u = m["main"]["uncertainty"]
    out += ["", f"Escalation on the test split: band uncertain {pct(u['band_uncertain_rate_test'])}, out of distribution "
            f"{pct(u['ood_rate_test'])}, total {pct(u['escalation_rate_test'])}.", ""]
    f = m["fairness"]
    for variant in ("main", "audit"):
        out += [f"## Fairness: {variant} model", ""]
        for attr, r in f[variant].items():
            out += [f"### {attr}: disparate-impact ratio {r['disparate_impact_ratio']:.2f} "
                    f"({'passes' if r['passes_four_fifths'] else 'fails'} four-fifths rule)", "",
                    "| Group | n | Base rate | Flag rate | TPR | FPR | Precision |", "|---|---|---|---|---|---|---|"]
            for g in r["groups"]:
                small = " (small)" if g["small_group"] else ""
                out.append(f"| {g['group']}{small} | {g['n']} | {pct(g['base_rate'])} | {pct(g['flag_rate'])} | {pct(g['tpr'])} | "
                           f"{pct(g['fpr'])} | {pct(g['precision'])} |")
            out.append("")
    out += ["## Proxy check (can the main model's features predict protected attributes?)", "",
            "| Attribute | AUC (all features) | Flagged | Strongest single features |", "|---|---|---|---|"]
    for attr, r in f["proxy_check"].items():
        tops = ", ".join(f"{s['feature']} ({s['auc']:.2f})" for s in r["top_single_features"][:3])
        out.append(f"| {attr} | {r['all_features_auc']:.3f} | {'yes' if r['flagged'] else 'no'} | {tops} |")
    out += ["", "Reliability figure: models/reliability_hr.png", ""]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "model_and_fairness_tables.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
