"""Export thesis tables from a domain's training manifest (model metrics with CIs, calibration, fairness, proxy check).

  python eval/export_tables.py                    -> results/model_and_fairness_tables.md (HR)
  python eval/export_tables.py --domain finance   -> results/model_and_fairness_tables_finance.md
  python eval/export_tables.py --all              -> one file per trained domain
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


def export(domain: str) -> str:
    m = json.loads((ROOT / "models" / f"{domain}_manifest.json").read_text(encoding="utf-8"))
    ds = m["dataset"]
    out = [f"# Model and fairness tables: {domain}", ""]
    if ds.get("source") == "synthetic_standin":
        out += ["> Synthetic stand-in data. Retrain on the real dataset before using these tables.", ""]
    pos = ds.get("leavers", ds.get("positives"))
    prate = ds.get("attrition_rate", ds.get("positive_rate"))
    out += [f"Data: {ds['rows']} rows, {pos} positives ({prate:.1%}). Source: {ds['source']}.", ""]
    if m.get("cost_rationale"):
        out += [f"Cost matrix rationale: {m['cost_rationale']}", ""]
    if ds.get("imputed_zero_values"):
        out += [f"Imputed values (zero recorded as missing): {ds['imputed_zero_values']} ({ds.get('imputation')}).", ""]
    variants = [v for v in ("main", "audit") if v in m]
    out += ["## Held-out test metrics (95% bootstrap CI)", "",
            "| Model | Kind | ROC-AUC | PR-AUC | Recall | Cost per case | Threshold |", "|---|---|---|---|---|---|---|"]
    for v in variants:
        h = m[v]["headline"]
        out.append(f"| {v} | {m[v]['model_kind']} | {ci(h['roc_auc'])} | {ci(h['pr_auc'])} | {ci(h['recall'])} | "
                   f"{ci(h['cost_per_employee'])} | {m[v]['threshold']:.2f} |")
    t = m["main"]["test"]["metrics"]
    out += ["", f"Cost matrix: missed positive = {m['main']['cost_matrix']['false_negative']}, false flag = "
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
    f = m.get("fairness", {})
    for variant in [v for v in variants if v in f]:
        out += [f"## Fairness: {variant} model", ""]
        for attr, r in f[variant].items():
            dir_ = r["disparate_impact_ratio"]
            verdict = "n/a" if dir_ is None else ("passes" if r["passes_four_fifths"] else "fails") + " four-fifths rule"
            out += [f"### {attr}: disparate-impact ratio {'n/a' if dir_ is None else format(dir_, '.2f')} ({verdict})", "",
                    "| Group | n | Base rate | Flag rate | TPR | FPR | Precision |", "|---|---|---|---|---|---|---|"]
            for g in r["groups"]:
                small = " (small)" if g["small_group"] else ""
                out.append(f"| {g['group']}{small} | {g['n']} | {pct(g['base_rate'])} | {pct(g['flag_rate'])} | {pct(g['tpr'])} | "
                           f"{pct(g['fpr'])} | {pct(g['precision'])} |")
            out.append("")
    out += ["## Proxy check (can the main model's features predict protected attributes?)", "",
            "| Attribute | AUC (all features) | Flagged | Strongest single features |", "|---|---|---|---|"]
    for attr, r in f.get("proxy_check", {}).items():
        tops = ", ".join(f"{s['feature']} ({s['auc']:.2f})" for s in r["top_single_features"][:3])
        out.append(f"| {attr} | {r['all_features_auc']:.3f} | {'yes' if r['flagged'] else 'no'} | {tops} |")
    out += ["", f"Reliability figure: models/reliability_{domain}.png", ""]
    return "\n".join(out)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="hr")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    domains = sorted(p.name[:-len("_manifest.json")] for p in (ROOT / "models").glob("*_manifest.json")) if a.all else [a.domain]
    (ROOT / "results").mkdir(exist_ok=True)
    for d in domains:
        text = export(d)
        name = "model_and_fairness_tables.md" if d == "hr" else f"model_and_fairness_tables_{d}.md"
        (ROOT / "results" / name).write_text(text, encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
