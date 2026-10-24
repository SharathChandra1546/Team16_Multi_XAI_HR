"""Re-measure the texts of an earlier comparison run with the current measurement code (no new generations).

    python eval/rescore_comparison.py results/comparison_<stamp>.json [--judge]

Writes results/comparison_<stamp>_rescored.{json,md}. Use it after a measurement fix so every system is scored by
the same, current rules. --judge re-runs the LLM extractor (JUDGE_PROVIDER); identical prompts come from the cache.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "eval"))

import run_comparison as rc  # noqa: E402
from app.agents.planner import make_plan  # noqa: E402
from app.domains.registry import default_registry  # noqa: E402
from app.llm.client import get_judge  # noqa: E402
from app.metrics.measure import measure_text  # noqa: E402
from app.xai.evidence import build_package  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--judge", action="store_true")
    a = ap.parse_args()
    src = Path(a.path)
    res = json.loads(src.read_text(encoding="utf-8"))
    reg = default_registry()
    judge = get_judge() if a.judge else None
    cases: dict[str, dict] = {}
    pk: dict[tuple, tuple] = {}
    for r in res["records"]:
        d = reg.get(r["domain"])
        if r["domain"] not in cases:
            rt = d.runtime("main")
            cases[r["domain"]] = {str(i): {f: float(row[f]) for f in rt.features} for i, row in rt.examples.iterrows()}
        key = (r["domain"], r["case"], r["level"], r["intent"])
        if key not in pk:
            pkg = build_package(d, cases[r["domain"]][str(r["case"])], r["level"], r["intent"])
            pk[key] = (pkg, make_plan(pkg, r["level"], r["intent"]))
        pkg, plan = pk[key]
        r["m"] = measure_text(r["text"], pkg, plan, d, judge)
    res["summary"] = rc.summarize(res["records"])
    res["rescored_from"] = src.name
    if judge is not None:
        res["judge"] = judge.id
    out = src.with_name(src.stem + "_rescored")
    out.with_suffix(".json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    md = rc.to_markdown(res)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
