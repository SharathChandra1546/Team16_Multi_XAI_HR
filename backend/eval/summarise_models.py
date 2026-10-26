"""One table per paper table, straight from the model manifests (no retyping).

  python eval/summarise_models.py

Writes results/real_data_summary.md: Table 1 (held-out performance by domain), Table 2 (cost baselines), Table 3
(fairness per domain, main model) and the escalation rate on the test split. Every number is read from
models/<domain>_manifest.json, so it is whatever the models on disk were trained on; the first line says which source.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.domains.registry import KNOWN_DOMAINS  # noqa: E402


def ci(v, c, d=3):
    return f"{v:.{d}f} [{c[0]:.{d}f}, {c[1]:.{d}f}]" if c else f"{v:.{d}f}"


def main() -> int:
    man = {d: json.loads((ROOT / "models" / f"{d}_manifest.json").read_text(encoding="utf-8")) for d in KNOWN_DOMAINS
           if (ROOT / "models" / f"{d}_manifest.json").exists()}
    out = ["# Real-data model summary", "",
           "Sources: " + "; ".join(f"{d} = {m['dataset']['source']} ({m['dataset']['rows']} rows)" for d, m in man.items()), "",
           "## Table 1. Held-out performance by domain (main model, 95% bootstrap CIs)", "",
           "| Domain | Model | n test (positives) | ROC-AUC | PR-AUC | Recall | Precision | Cost per case | Brier | ECE | Threshold |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d, m in man.items():
        k, t = m["main"], m["main"]["test"]
        me, c = t["metrics"], t["ci95"]
        out.append(f"| {d} | {k['model_kind']} | {k['n_test']} ({k['positives_test']}) | {ci(me['roc_auc'], c['roc_auc'])} | "
                   f"{ci(me['pr_auc'], c['pr_auc'])} | {ci(me['recall'], c['recall'])} | {ci(me['precision'], c['precision'])} | "
                   f"{ci(me['cost_per_employee'], c['cost_per_employee'])} | {me['brier']:.3f} | {k['reliability_oof_all']['ece']:.3f} | {k['threshold']:.2f} |")
    out += ["", "ECE is on out-of-fold predictions over all rows; the other columns are on the held-out split.", "",
            "## Table 2. Cost per case (missed positive : false flag as in the manifest)", "",
            "| Domain | Cost matrix | Flag nobody | Flag everyone | Model | Flag rate of the model |", "|---|---|---|---|---|---|"]
    for d, m in man.items():
        k, me = m["main"], m["main"]["test"]["metrics"]
        cm = k["cost_matrix"]
        out.append(f"| {d} | {cm['false_negative']:g} : {cm['false_positive']:g} | {me['cost_flag_nobody']:.3f} | {me['cost_flag_everyone']:.3f} | "
                   f"{me['cost_per_employee']:.3f} | {me['flag_rate']:.1%} |")
    out += ["", "## Escalation on the test split (rule mode)", "", "| Domain | Uncertain band | Out of distribution | Total |", "|---|---|---|---|"]
    for d, m in man.items():
        u = m["main"]["uncertainty"]
        out.append(f"| {d} | {u['band_uncertain_rate_test']:.1%} | {u['ood_rate_test']:.1%} | {u['escalation_rate_test']:.1%} |")
    out += ["", "## Table 2b. Cost-ratio sensitivity (threshold re-chosen on training-split out-of-fold predictions, evaluated on the held-out split)", "",
            "| Domain | Missed : false flag | Threshold | Flag rate | Recall | Precision | Cost per case | Flag nobody | Flag everyone |", "|---|---|---|---|---|---|---|---|---|"]
    for d, m in man.items():
        for r in m["main"].get("cost_sensitivity", []):
            out.append(f"| {d} | {r['fn_to_fp']:g} : 1 | {r['threshold']:.2f} | {r['flag_rate']:.1%} | {r['recall']:.3f} | {r['precision']:.3f} | "
                       f"{r['cost_per_case']:.3f} | {r['cost_flag_nobody']:.3f} | {r['cost_flag_everyone']:.3f} |")
    out += ["", "## Table 3. Fairness, main model (out-of-fold, all rows)", "",
            "| Domain | Attribute | Disparate-impact ratio | Four-fifths | Lowest / highest flag rate | Proxy AUC (flagged at 0.70 or more) |",
            "|---|---|---|---|---|---|"]
    for d, m in man.items():
        fair = m.get("fairness", {})
        for attr, r in fair.get("main", {}).items():
            if not isinstance(r, dict) or "disparate_impact_ratio" not in r:
                continue
            px = fair.get("proxy_check", {}).get(attr, {})
            dir_ = r["disparate_impact_ratio"]
            lo_hi = r.get("dir_groups", {})
            out.append(f"| {d} | {attr} | {'n/a' if dir_ is None else f'{dir_:.2f}'} | "
                       f"{'n/a' if r.get('passes_four_fifths') is None else ('pass' if r['passes_four_fifths'] else 'fail')} | "
                       f"{lo_hi.get('lowest_flag_rate', '?')} / {lo_hi.get('highest_flag_rate', '?')} | "
                       f"{px.get('all_features_auc', float('nan')):.3f}{' (flagged)' if px.get('flagged') else ''} |")
    out += ["", "Groups smaller than 30 are listed in the per-domain tables and excluded from the ratio. For healthcare, age is "
            "a clinical factor in the model, so compare error rates between age bands (see model_and_fairness_tables_health.md)."]
    (ROOT / "results" / "real_data_summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
