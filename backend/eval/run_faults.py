"""Fault-injection experiment: detection rate per fault type and false-rejection rate on clean text.

  python eval/run_faults.py --n 40                       # template base texts, deterministic checks
  python eval/run_faults.py --n 40 --base llm            # base texts written by the configured LLM (must pass checks)
  python eval/run_faults.py --n 40 --reviewer            # also run the LLM-reviewer baseline (JUDGE_PROVIDER)
  python eval/run_faults.py --domains hr,finance         # side-by-side once a second domain is registered
  python eval/run_faults.py --n 20 --faults llm          # detector-independent faults written by FAULT_PROVIDER
  python eval/run_faults.py --n 20 --faults both         # template (upper bound) and LLM faults side by side

Rates are reported with Wilson 95% intervals. "Detection" = the expected code was raised; "any rejection" = the
text was rejected for any reason (an LLM-written fault can legitimately trip a different code than expected).

Writes results/faults_<timestamp>.json and .md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agents.critic import critique  # noqa: E402
from app.agents.fallback import build_template  # noqa: E402
from app.agents.generator import evidence_view, generate  # noqa: E402
from app.agents.planner import make_plan  # noqa: E402
from app.agents.verifier import verify  # noqa: E402
from app.config import settings  # noqa: E402
from app.domains.registry import default_registry  # noqa: E402
from app.agents.reviewer import get_reviewer, review  # noqa: E402
from app.faults import FAULTS, TAXONOMY, inject  # noqa: E402
from app.faults_llm import get_fault_llm, llm_inject  # noqa: E402
from app.ml.training import wilson  # noqa: E402
from app.llm import mock as mock_llm  # noqa: E402
from app.llm.client import get_judge, get_llm  # noqa: E402
from app.llm.prompts import DEFAULT_PROMPTS  # noqa: E402
from app.xai.evidence import build_package  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
LEVELS = ["novice", "intermediate", "expert"]

# Meaning-preserving rewrites used to measure false rejections on varied, correct text.
CLEAN_REWRITES = [
    ("raised the estimate", "increased the estimate"),
    ("lowered the estimate", "reduced the estimate"),
    ("The largest factor was", "The most important factor was"),
    ("These are patterns the model learned from past data, not causes.",
     "These are patterns in past data that the model learned; they are not causes."),
    ("pushes the estimate up", "moves the estimate up"),
]
CLEAN_APPENDS = [
    "These estimates support a conversation, not a decision.",
    "A person should review this estimate together with other information.",
]



def codes_for(pkg, plan, gen, d):
    v = verify(pkg, plan, gen, d)
    c = critique(pkg, plan, gen, d)
    return {x.code for x in v.violations + c.violations if x.severity == "critical"}


def reviewer_flags(judge, pkg, plan, gen, d=None) -> bool:
    """True if the advisory reviewer agent (app/agents/reviewer.py) says the text is not faithful."""
    from app.domains.registry import default_registry

    d = d or default_registry().get(pkg.domain)
    return review(judge, pkg, plan, gen, d).get("faithful") is False


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [lo, hi]}


def base_texts(d, x, level, intent, base, llm, prompt):
    pkg = build_package(d, x, level, intent)
    plan = make_plan(pkg, level, intent)
    if base == "llm":
        gen = generate(pkg, plan, d, llm, prompt["template"], prompt["guidance"], d.language.default_question)
        if codes_for(pkg, plan, gen, d):
            return pkg, plan, None  # only correct LLM texts can serve as fault-injection bases
        return pkg, plan, gen
    return pkg, plan, build_template(pkg, plan, d)


def run_domain(d, n, base, use_reviewer, seed, injector="template"):
    rng = random.Random(seed)
    llm = get_llm()
    judge = get_reviewer() if use_reviewer else None
    fault_llm = get_fault_llm() if injector == "llm" else None
    llm_records = []
    prompt = DEFAULT_PROMPTS[0]
    rt = d.runtime("main")
    xs = [{f: float(r[f]) for f in rt.features} for _, r in rt.examples.head(n).iterrows()]
    det = defaultdict(lambda: {"n": 0, "detected": 0, "exact": 0, "rejected": 0, "reviewer_detected": 0,
                               "collateral": defaultdict(int)})
    clean = {"n": 0, "rejected": 0, "reviewer_rejected": 0, "codes": defaultdict(int)}
    skipped_bases = 0
    for x in xs:
        for intent in ("explain", "recourse"):
            for level in LEVELS:
                pkg, plan, gen = base_texts(d, x, level, intent, base, llm, prompt)
                if gen is None:
                    skipped_bases += 1
                    continue
                variants = [gen]
                for a, b in CLEAN_REWRITES:
                    if a in gen.explanation:
                        g = gen.model_copy(deep=True)
                        g.explanation = g.explanation.replace(a, b)
                        for c in g.claims:
                            c.text = c.text.replace(a, b)
                        variants.append(g)
                for s in CLEAN_APPENDS:
                    g = gen.model_copy(deep=True)
                    g.explanation += " " + s
                    variants.append(g)
                for g in variants:
                    cs = codes_for(pkg, plan, g, d)
                    clean["n"] += 1
                    if cs:
                        clean["rejected"] += 1
                        for c in cs:
                            clean["codes"][c] += 1
                    if judge and reviewer_flags(judge, pkg, plan, g):
                        clean["reviewer_rejected"] += 1
                for fault, code in FAULTS.items():
                    if fault_llm is not None:
                        bad, rec = llm_inject(fault, gen, pkg, plan, d, fault_llm)
                        llm_records.append(rec | {"level": level, "intent": intent})
                    else:
                        bad = inject(fault, gen, pkg, plan, d, rng)
                    if bad is None:
                        continue
                    r = det[fault]
                    r["n"] += 1
                    cs = codes_for(pkg, plan, bad, d)
                    r["detected"] += code in cs
                    r["exact"] += cs == {code}
                    r["rejected"] += bool(cs)
                    if fault_llm is not None:
                        llm_records[-1]["codes"] = sorted(cs)
                    for c in cs - {code}:
                        r["collateral"][c] += 1
                    if judge and reviewer_flags(judge, pkg, plan, bad):
                        r["reviewer_detected"] += 1
    per_fault = {}
    for f, r in det.items():
        per_fault[f] = {"expected_code": FAULTS[f], "category": TAXONOMY[f]["category"], "n": r["n"],
                        "detection_rate": r["detected"] / r["n"], "detection": rate(r["detected"], r["n"]),
                        "exact_code_rate": r["exact"] / r["n"], "any_rejection": rate(r["rejected"], r["n"]),
                        "reviewer_detection_rate": (r["reviewer_detected"] / r["n"]) if judge else None,
                        "reviewer_detection": rate(r["reviewer_detected"], r["n"]) if judge else None,
                        "collateral_codes": dict(r["collateral"])}
    tot_n = sum(r["n"] for r in det.values())
    return {
        "domain": d.name, "base": base, "injector": injector, "fault_llm": fault_llm.id if fault_llm else None,
        "n_cases": len(xs), "skipped_bases_not_clean": skipped_bases,
        "per_fault": per_fault,
        "overall": {"detection": rate(sum(r["detected"] for r in det.values()), tot_n),
                    "any_rejection": rate(sum(r["rejected"] for r in det.values()), tot_n),
                    "reviewer_detection": rate(sum(r["reviewer_detected"] for r in det.values()), tot_n) if judge else None},
        "clean": {"n": clean["n"], "false_rejection_rate": clean["rejected"] / max(1, clean["n"]),
                  "false_rejection": rate(clean["rejected"], clean["n"]),
                  "reviewer_false_rejection_rate": (clean["reviewer_rejected"] / max(1, clean["n"])) if judge else None,
                  "codes": dict(clean["codes"])},
        "llm_fault_records": llm_records,
    }


def to_markdown(res: dict) -> str:
    out = [f"# Fault injection results ({res['created']})", "",
           f"LLM: `{res['llm']}`. Reviewer: `{res['judge'] or 'not run'}`. Data source: `{res['data_source']}`.", ""]
    if "mock" in res["llm"] and res["base"] == "llm" or res["data_source"] == "synthetic_standin":
        out += ["> These numbers come from a mock LLM and/or synthetic stand-in data. They test the software only and "
                "must not be reported as results.", ""]
    def ci(r):
        if not r or r["rate"] is None:
            return "n/a"
        return f"{r['rate']:.1%} [{r['ci95'][0]:.1%}, {r['ci95'][1]:.1%}]"

    for dres in res["domains"]:
        label = "LLM-written faults (detector-independent)" if dres["injector"] == "llm" else "template faults (upper bound)"
        out += [f"## Domain: {dres['domain']} (base texts: {dres['base']}; {label})", "",
                "| Fault | Expected code | n | Detection (95% CI) | Any rejection | Exact code | LLM reviewer |",
                "|---|---|---|---|---|---|---|"]
        for f, r in dres["per_fault"].items():
            out.append(f"| {f} | {r['expected_code']} | {r['n']} | {ci(r['detection'])} | {ci(r['any_rejection'])} | "
                       f"{r['exact_code_rate']:.1%} | {ci(r['reviewer_detection'])} |")
        o = dres["overall"]
        out += ["", f"Overall: detection {ci(o['detection'])}, any rejection {ci(o['any_rejection'])}, "
                    f"reviewer {ci(o['reviewer_detection'])}."]
        c = dres["clean"]
        rv = "n/a" if c["reviewer_false_rejection_rate"] is None else f"{c['reviewer_false_rejection_rate']:.1%}"
        out += ["", f"Clean texts: n = {c['n']}, false-rejection rate = {ci(c['false_rejection'])} "
                    f"(LLM reviewer: {rv}). Codes on clean texts: {c['codes'] or 'none'}.", ""]
    if len(res["domains"]) > 1:
        out += ["## Side by side", "", "| Fault | " + " | ".join(f"{d['domain']} ({d['injector']})" for d in res["domains"]) + " |",
                "|---|" + "---|" * len(res["domains"])]
        for f in FAULTS:
            cells = [f"{d['per_fault'][f]['detection_rate']:.1%}" if f in d["per_fault"] else "n/a" for d in res["domains"]]
            out.append(f"| {f} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--base", choices=["template", "llm"], default="template")
    ap.add_argument("--reviewer", action="store_true")
    ap.add_argument("--domains", default="hr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--faults", choices=["template", "llm", "both"], default="template")
    args = ap.parse_args()
    reg = default_registry()
    mock_llm.DOMAIN_RESOLVER = reg.get
    res = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "llm": get_llm().id,
           "judge": get_reviewer().id if args.reviewer else None, "base": args.base,
           "data_source": getattr(reg.get(args.domains.split(",")[0]), "data_source", "unknown"), "domains": []}
    injectors = ["template", "llm"] if args.faults == "both" else [args.faults]
    for name in args.domains.split(","):
        for inj in injectors:
            res["domains"].append(run_domain(reg.get(name), args.n, args.base, args.reviewer, args.seed, inj))
    RESULTS.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"faults_{stamp}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = to_markdown(res)
    (RESULTS / f"faults_{stamp}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
