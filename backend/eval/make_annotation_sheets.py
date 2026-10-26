"""Build blind annotation sheets for human evaluation from a comparison run.

  python eval/make_annotation_sheets.py --results results/comparison_<stamp>.json --per-system 24 --annotators A,B

Writes annotation/<tag>/ :
  sheet_<annotator>.csv   one per annotator: the SAME items, each shuffled differently, system names hidden
  items_private.json      item id -> system, domain, case, level, intent, and the automatic measurements
                          (do NOT give this file to annotators; score_annotations.py needs it)
  README.txt              what to send to whom

The annotator sees the evidence the system saw (prediction, visible factors, modelled changes), the reader level,
the question and the explanation, and fills six columns (see docs/ANNOTATION_GUIDE.md). Both annotators label every
item so Cohen's kappa can be computed on all of them.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.domains.registry import default_registry  # noqa: E402
from app.xai.evidence import build_package  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABEL_COLS = ["faithful", "unsupported_claim", "policy_violation", "level_fit", "clarity", "notes"]


def render_evidence(pkg) -> str:
    p = pkg.prediction
    lines = [f"Model estimate: {p.probability:.0%} ({p.band} band), likely range {p.interval[0]:.0%} to {p.interval[1]:.0%}."]
    vis = [i for i in pkg.items if not i.withheld]
    if vis:
        lines.append("Factors shown at this level (largest effect first):")
        for i in sorted(vis, key=lambda i: i.rank):
            lines.append(f"  {i.id} {i.display_name} = {i.value_label}: {i.direction} the estimate, {i.magnitude}, "
                         f"{i.attribution * 100:+.1f} points (rank {i.rank})")
    if pkg.recourse:
        lines.append("Modelled changes (checked by re-running the model):")
        for r in pkg.recourse:
            lines.append(f"  {r.id} {r.description} -> estimate {r.probability_before:.0%} to {r.probability_after:.0%}")
    elif pkg.recourse_note:
        lines.append(f"Modelled changes: none. {pkg.recourse_note}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results/comparison_<stamp>.json from run_comparison.py")
    ap.add_argument("--per-system", type=int, default=24, help="items per system per domain (stratified by level and intent)")
    ap.add_argument("--annotators", default="A,B")
    ap.add_argument("--systems", default="", help="comma list; default every system in the run")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    res = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if res.get("mock_or_standin"):
        print("WARNING: this run used the mock LLM or stand-in data. Sheets built from it are for a dry run only.")
    reg = default_registry()
    rng = random.Random(args.seed)
    wanted = set(args.systems.split(",")) if args.systems else set(res["systems"])
    groups = defaultdict(list)
    for r in res["records"]:
        if r["system"] in wanted and r["text"].strip():
            groups[(r["domain"], r["system"])].append(r)

    # The same (domain, case, level, intent) is used for every system as far as the sample allows, so annotators can
    # compare systems on identical evidence. Choose cases first, then take every system's text for them.
    cells = defaultdict(set)
    for (dom, _), rs in groups.items():
        for r in rs:
            cells[dom].add((r["case"], r["level"], r["intent"]))
    chosen = {}
    for dom, cs in cells.items():
        cs = sorted(cs)
        rng.shuffle(cs)
        by_lvl = defaultdict(list)
        for c in cs:
            by_lvl[(c[1], c[2])].append(c)
        picked, keys, i = [], sorted(by_lvl), 0
        while len(picked) < min(args.per_system, len(cs)) and any(by_lvl.values()):
            k = keys[i % len(keys)]
            i += 1
            if by_lvl[k]:
                picked.append(by_lvl[k].pop())
        chosen[dom] = set(picked)

    items = []
    for (dom, system), rs in sorted(groups.items()):
        for r in rs:
            if (r["case"], r["level"], r["intent"]) in chosen[dom]:
                items.append(r)
    rng.shuffle(items)

    tag = args.tag or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "annotation" / tag
    out.mkdir(parents=True, exist_ok=True)
    private, rows = {}, []
    for n, r in enumerate(items, 1):
        item_id = f"I{n:04d}"
        d = reg.get(r["domain"])
        rt = d.runtime("main")
        row = next(row for idx, row in rt.examples.iterrows() if str(idx) == str(r["case"]))
        x = {f: float(row[f]) for f in rt.features}
        pkg = build_package(d, x, r["level"], r["intent"])
        q = d.language.default_question if r["intent"] == "explain" else "What would lower the risk?"
        rows.append({"item_id": item_id, "domain": r["domain"], "reader": d.language.reader, "level": r["level"],
                     "question": q, "evidence": render_evidence(pkg), "explanation": r["text"].strip()})
        m = r["m"]
        private[item_id] = {"system": r["system"], "domain": r["domain"], "case": r["case"], "level": r["level"],
                            "intent": r["intent"], "auto": {"error_free": m["error_free"], "error_free_llm": m.get("error_free_llm"),
                                                          "fidelity_errors": m["fidelity_errors"],
                                                          "policy_violations": m["n_policy_violations"],
                                                          "level_violations": len(m.get("level_violations") or [])}}
    for a in [x.strip() for x in args.annotators.split(",") if x.strip()]:
        order = list(rows)
        random.Random(f"{args.seed}-{a}").shuffle(order)
        with (out / f"sheet_{a}.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) + LABEL_COLS)
            w.writeheader()
            for row in order:
                w.writerow(row | {c: "" for c in LABEL_COLS})
    (out / "items_private.json").write_text(json.dumps({"source": args.results, "seed": args.seed, "items": private},
                                                       indent=1), encoding="utf-8")
    (out / "README.txt").write_text(
        "Send each annotator ONLY their own sheet_<name>.csv and docs/ANNOTATION_GUIDE.md.\n"
        "Annotators must work alone and not discuss items until both sheets are returned.\n"
        "Keep items_private.json (it names the system behind every item) away from annotators.\n"
        "Then: python eval/score_annotations.py --dir annotation/" + tag + "\n", encoding="utf-8")
    per = defaultdict(int)
    for v in private.values():
        per[(v["domain"], v["system"])] += 1
    print(f"{len(rows)} items in {out}")
    for k, v in sorted(per.items()):
        print(f"  {k[0]:8s} {k[1]:22s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
