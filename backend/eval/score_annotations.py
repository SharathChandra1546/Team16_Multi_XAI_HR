"""Score completed annotation sheets: inter-annotator agreement, per-system human results, checker vs human.

  python eval/score_annotations.py --dir annotation/<tag> [--adjudicated annotation/<tag>/adjudicated.csv]

Reads sheet_<annotator>.csv files (all completed) and items_private.json. Writes in the same folder:
  scores.json / scores.md   kappa per label, per-system human rates with Wilson 95% intervals, checker-vs-human table
  adjudication.csv          items where the two annotators disagree on a binary label (resolve these, save as
                            adjudicated.csv with the same columns plus a final value, then re-run with --adjudicated)

Labels: faithful, unsupported_claim, policy_violation (0/1); level_fit, clarity (1 to 5); notes (free text).
"human_error" for an item = not faithful, or unsupported_claim, or policy_violation (by consensus).
Quadratic-weighted kappa is used for the 1 to 5 scales, Cohen's kappa for the 0/1 labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.training import wilson  # noqa: E402

BIN = ["faithful", "unsupported_claim", "policy_violation"]
ORD = ["level_fit", "clarity"]


def read_sheet(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            out[row["item_id"]] = row
    return out


def to_int(v, lo, hi):
    try:
        x = int(str(v).strip())
    except ValueError:
        return None
    return x if lo <= x <= hi else None


def cohen(a: list[int], b: list[int], cats: list[int]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)


def weighted_kappa(a: list[int], b: list[int], lo: int, hi: int) -> float | None:
    n, k = len(a), hi - lo + 1
    if n == 0:
        return None
    w = lambda i, j: ((i - j) / (k - 1)) ** 2
    obs = sum(w(x - lo, y - lo) for x, y in zip(a, b)) / n
    ma = [a.count(lo + i) / n for i in range(k)]
    mb = [b.count(lo + i) / n for i in range(k)]
    exp = sum(w(i, j) * ma[i] * mb[j] for i in range(k) for j in range(k))
    return None if exp == 0 else 1 - obs / exp


def rate(k, n):
    lo, hi = wilson(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--adjudicated", default="")
    args = ap.parse_args()
    d = Path(args.dir)
    private = json.loads((d / "items_private.json").read_text(encoding="utf-8"))["items"]
    sheets = {p.stem.removeprefix("sheet_"): read_sheet(p) for p in sorted(d.glob("sheet_*.csv"))}
    if len(sheets) < 2:
        print("Need at least two annotator sheets.")
        return 1

    problems = []
    clean: dict[str, dict[str, dict]] = {}
    for a, rows in sheets.items():
        clean[a] = {}
        for iid, row in rows.items():
            rec, bad = {}, []
            for c in BIN:
                v = to_int(row.get(c), 0, 1)
                bad += [c] if v is None else []
                rec[c] = v
            for c in ORD:
                v = to_int(row.get(c), 1, 5)
                bad += [c] if v is None else []
                rec[c] = v
            if bad:
                problems.append(f"{a} {iid}: missing or invalid {', '.join(bad)}")
            else:
                clean[a][iid] = rec
    if problems:
        print(f"{len(problems)} problems; first 15:")
        print("\n".join(problems[:15]))
        print("Fix the sheets and run again. Only complete rows count.")

    res = {"annotators": list(sheets), "n_items": len(private), "pairs": {}}
    for a, b in combinations(sheets, 2):
        common = sorted(set(clean[a]) & set(clean[b]))
        pair = {"n_common": len(common)}
        for c in BIN:
            xa, xb = [clean[a][i][c] for i in common], [clean[b][i][c] for i in common]
            pair[c] = {"kappa": cohen(xa, xb, [0, 1]), "agreement": sum(x == y for x, y in zip(xa, xb)) / len(common) if common else None}
        for c in ORD:
            xa, xb = [clean[a][i][c] for i in common], [clean[b][i][c] for i in common]
            pair[c] = {"kappa_quadratic": weighted_kappa(xa, xb, 1, 5), "agreement": sum(x == y for x, y in zip(xa, xb)) / len(common) if common else None}
        res["pairs"][f"{a}-{b}"] = pair

    first, second = list(sheets)[:2]
    common = sorted(set(clean[first]) & set(clean[second]))
    final: dict[str, dict] = {}
    disputes = []
    adjud = {}
    if args.adjudicated and Path(args.adjudicated).exists():
        for row in csv.DictReader(Path(args.adjudicated).open(newline="", encoding="utf-8-sig")):
            adjud[row["item_id"]] = row
    for i in common:
        a, b = clean[first][i], clean[second][i]
        rec = {}
        for c in BIN:
            if a[c] == b[c]:
                rec[c] = a[c]
            elif i in adjud and to_int(adjud[i].get(f"final_{c}"), 0, 1) is not None:
                rec[c] = to_int(adjud[i][f"final_{c}"], 0, 1)
            else:
                rec[c] = None
        for c in ORD:
            rec[c] = (a[c] + b[c]) / 2
        if any(rec[c] is None for c in BIN):
            disputes.append(i)
        final[i] = rec
    res["n_adjudication_needed"] = len(disputes)
    if disputes:
        with (d / "adjudication.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            cols = ["item_id"] + [f"{who}_{c}" for c in BIN for who in (first, second)] + [f"final_{c}" for c in BIN]
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for i in disputes:
                row = {"item_id": i} | {f"{who}_{c}": clean[who][i][c] for c in BIN for who in (first, second)}
                w.writerow(row | {f"final_{c}": "" for c in BIN})

    def human_error(r):
        return None if any(r[c] is None for c in BIN) else int(r["faithful"] == 0 or r["unsupported_claim"] == 1 or r["policy_violation"] == 1)

    per = defaultdict(lambda: defaultdict(list))
    tp = defaultdict(lambda: [0, 0, 0, 0])  # checker flags error x human error: [both, checker only, human only, neither]
    for i, r in final.items():
        he = human_error(r)
        p = private[i]
        key = f"{p['domain']}/{p['system']}"
        for k in (key, f"all/{p['system']}"):
            if he is not None:
                per[k]["human_error_free"].append(1 - he)
            per[k]["level_fit"].append(r["level_fit"])
            per[k]["clarity"].append(r["clarity"])
            if he is not None:
                # the independent LLM scorer's verdict when the run had one, else the rule-based checker's
                auto_ok = p["auto"].get("error_free_llm")
                ce = 0 if (p["auto"]["error_free"] if auto_ok is None else auto_ok) else 1
                cell = 0 if (ce and he) else 1 if ce else 2 if he else 3
                tp[k][cell] += 1
    res["systems"] = {}
    for k, v in sorted(per.items()):
        ef = v["human_error_free"]
        res["systems"][k] = {
            "n": len(v["clarity"]), "human_error_free": rate(sum(ef), len(ef)),
            "level_fit_mean": sum(v["level_fit"]) / len(v["level_fit"]), "clarity_mean": sum(v["clarity"]) / len(v["clarity"]),
            "checker_vs_human": dict(zip(["both_flag", "checker_only", "human_only", "neither"], tp[k]))}
    p = res["systems"]
    for k, v in p.items():
        c = v["checker_vs_human"]
        he = c["both_flag"] + c["human_only"]
        cf = c["both_flag"] + c["checker_only"]
        v["checker_recall_of_human_errors"] = rate(c["both_flag"], he)
        v["checker_precision"] = rate(c["both_flag"], cf)
    # Cohen's kappa between the automatic verdict and the (adjudicated) human verdict, all items
    cells = [0, 0, 0, 0]
    for k, v in tp.items():
        if k.startswith("all/"):
            cells = [a + b for a, b in zip(cells, v)]
    both, conly, honly, neither = cells
    n_all = sum(cells)
    if n_all:
        po = (both + neither) / n_all
        pc, ph = (both + conly) / n_all, (both + honly) / n_all
        pe = pc * ph + (1 - pc) * (1 - ph)
        res["checker_vs_human_kappa"] = {"n": n_all, "observed_agreement": po,
                                         "kappa": None if pe == 1 else (po - pe) / (1 - pe), "cells": cells}
    (d / "scores.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    f = lambda x: "n/a" if x is None else f"{x:.2f}"
    ci = lambda r: "n/a" if r["rate"] is None else f"{r['rate']:.1%} [{r['ci95'][0]:.1%}, {r['ci95'][1]:.1%}] ({r['k']}/{r['n']})"
    md = ["# Human evaluation scores", "", f"Annotators: {', '.join(sheets)}. Items: {res['n_items']}. Disputed on a binary label: {len(disputes)}"
          + (" (resolve in adjudication.csv, save as adjudicated.csv and re-run with --adjudicated)." if disputes else "."), "",
          "## Inter-annotator agreement", "", "| Pair | Items | " + " | ".join(BIN + ORD) + " |", "|---|---|" + "---|" * (len(BIN) + len(ORD))]
    for pair, v in res["pairs"].items():
        md.append(f"| {pair} | {v['n_common']} | " + " | ".join([f(v[c]["kappa"]) for c in BIN] + [f(v[c]["kappa_quadratic"]) for c in ORD]) + " |")
    md += ["", "Cohen's kappa for 0/1 labels, quadratic-weighted kappa for 1 to 5 scales. Report with the number of items.", "",
           "## Systems by human consensus", "", "| Domain / system | n | Human error-free | Level fit | Clarity | Checker recall of human errors | Checker precision |", "|---|---|---|---|---|---|---|"]
    for k, v in p.items():
        md.append(f"| {k} | {v['n']} | {ci(v['human_error_free'])} | {v['level_fit_mean']:.2f} | {v['clarity_mean']:.2f} | "
                  f"{ci(v['checker_recall_of_human_errors'])} | {ci(v['checker_precision'])} |")
    md += ["", "The checker is the automatic measurement in app/metrics/measure.py. Recall: of the items humans marked as having an error, "
           "the share the checker also flagged. Precision: of the items the checker flagged, the share humans also marked."]
    kv = res.get("checker_vs_human_kappa")
    if kv:
        md += ["", f"Automatic verdict vs (adjudicated) human verdict, error present: kappa {f(kv['kappa'])} "
               f"(agreement {kv['observed_agreement']:.1%}, n = {kv['n']}; cells both/auto only/human only/neither = {kv['cells']})."]
    (d / "scores.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
