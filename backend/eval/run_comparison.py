"""Comparison and ablation runner: every system, the same evidence, one measurement.

  python eval/run_comparison.py --preset baselines --n 30              # our pipeline vs all baselines
  python eval/run_comparison.py --preset ablations --n 30              # our pipeline with each component removed
  python eval/run_comparison.py --preset all --domains hr,finance,health --n 30 --judge
  python eval/run_comparison.py --systems ours:full,b2,musicco --levels novice,expert --intents explain,recourse

Systems:
  ours:<config>  this framework through the full graph, with a PipelineConfig from app.pipeline.ABLATIONS
                 (full, no_verifier, no_policy_rules, no_critic, no_repair, no_fallback, with_llm_agents, ...).
                 Escalation runs in shadow mode for all `ours:` systems so every system explains every case; the
                 escalation score is recorded and used for the coverage-risk table instead.
  direct, b1, b2, b3, b4, explingo, critic_loop, musicco   see app/baselines.py

For each case (domain x held-out example x level x intent) every system gets the same evidence package. The text
each system would show is measured with app/metrics/measure.py (deterministic extractor; with --judge also the LLM
extractor). Nothing a system reports about itself is used, including this framework's own claims and verdicts.

Outputs results/comparison_<timestamp>.json (every record + summary) and .md (tables with Wilson 95% intervals):
- error-free rate, fidelity-error rate, policy-violation rate (per code), unsupported factor mentions
- value / sign / rank agreement and top-3 coverage (deterministic and LLM extractor)
- withheld / fallback / unverified rates (ours), LLM calls, tokens and latency per explanation
- coverage-risk: for escalation-score thresholds, the share of cases explained and the error rate among them
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agents.planner import make_plan  # noqa: E402
from app.agents.reviewer import get_reviewer  # noqa: E402
from app.baselines import SYSTEMS, run_baseline  # noqa: E402
from app.domains.registry import default_registry  # noqa: E402
from app.escalation import escalation_info  # noqa: E402
from app.llm import mock as mock_llm  # noqa: E402
from app.llm.client import get_judge, get_llm, usage_delta  # noqa: E402
from app.metrics.measure import measure_text  # noqa: E402
from app.ml.training import wilson  # noqa: E402
from app.pipeline import ABLATIONS, PipelineConfig  # noqa: E402
from app.schemas import ExplanationRequest  # noqa: E402
from app.service import Service  # noqa: E402
from app.store import Store  # noqa: E402
from app.xai.evidence import build_package  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
PRESETS = {
    "baselines": ["ours:full", *SYSTEMS],
    "ablations": [f"ours:{a}" for a in ABLATIONS],
    "all": [f"ours:{a}" for a in ABLATIONS] + list(SYSTEMS),
}
THRESHOLDS = [round(0.1 * i, 1) for i in range(0, 11)] + [1.01]


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [lo, hi]}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def run_ours(svc: Service, config: str, d, x: dict, level: str, intent: str, question: str) -> dict:
    cfg = PipelineConfig(**(ABLATIONS[config] | {"escalation": False}))
    raw = {f: (d.readable(f, v) if d.spec(f).categorical else v) for f, v in x.items()}
    clients = [svc.llm] + ([svc.graph._reviewer] if svc.graph._reviewer is not None and svc.graph._reviewer is not svc.llm else [])
    before = [c.usage_snapshot() for c in clients]
    forced = intent if intent in ("explain", "recourse") else None
    resp = svc.explain(ExplanationRequest(query=question, instance=raw, level=level, domain=d.name, consent_cloud=True),
                       cfg, forced_intent=forced)
    deltas = [usage_delta(b, c.usage_snapshot()) for b, c in zip(before, clients)]
    usage = {k: sum(dd[k] for dd in deltas) for k in ("calls", "cached_calls", "prompt_tokens", "completion_tokens")}
    text = resp.explanation.explanation if resp.explanation else ""
    return {"text": text, "status": resp.status, "attempts": resp.attempts, "usage": usage,
            "accepted": resp.status in ("verified", "fallback")}


def cases_for(d, n: int, offset: int):
    rt = d.runtime("main")
    rows = rt.examples.iloc[offset:offset + n]
    return [(str(idx), {f: float(r[f]) for f in rt.features}) for idx, r in rows.iterrows()]


def summarize(records: list[dict]) -> dict:
    by = defaultdict(list)
    for r in records:
        by[(r["domain"], r["system"])].append(r)
    out = {}
    for (dom, sysname), rs in sorted(by.items()):
        shown = [r for r in rs if not r["m"]["empty"]]
        n = len(rs)
        ef = sum(r["m"]["error_free"] for r in shown)
        fid = sum(r["m"]["fidelity_errors"] > 0 for r in shown)
        pol = sum(r["m"]["n_policy_violations"] > 0 for r in shown)
        codes = Counter(c for r in shown for c in r["m"]["policy_violations"])
        tc = Counter()
        for r in shown:
            for k, v in r["m"]["text_checks"].items():
                tc[k] += bool(v)
        ag = lambda key, kind="agreement_det": _mean([r["m"][kind][key] for r in shown if r["m"].get(kind)])  # noqa: E731
        explain = [r for r in shown if r["intent"] == "explain"]
        status = Counter(r.get("status", "shown" if not r["m"]["empty"] else "empty") for r in rs)
        cov = []
        for t in THRESHOLDS:
            kept = [r for r in rs if r["escalation_score"] < t]
            kept_shown = [r for r in kept if not r["m"]["empty"]]
            err = sum(not r["m"]["error_free"] for r in kept_shown)
            cov.append({"threshold": t, "coverage": round(len(kept) / n, 4) if n else None,
                        "error_rate": round(err / len(kept_shown), 4) if kept_shown else None, "n": len(kept)})
        rule = [r for r in rs if not r["would_escalate_rule"]]
        rule_shown = [r for r in rule if not r["m"]["empty"]]
        out[f"{dom}/{sysname}"] = {
            "domain": dom, "system": sysname, "n_cases": n, "n_shown": len(shown),
            "status": dict(status),
            "error_free": rate(ef, len(shown)),
            "any_fidelity_error": rate(fid, len(shown)),
            "any_policy_violation": rate(pol, len(shown)),
            "policy_codes": dict(codes),
            "text_checks": dict(tc),
            "unsupported_mention_rate": rate(sum(bool(r["m"]["agreement_det"]["unsupported_mentions"]) for r in shown), len(shown)),
            "agreement_det": {"value": _mean([r["m"]["agreement_det"]["value_agreement"] for r in explain]),
                              "sign": _mean([r["m"]["agreement_det"]["sign_agreement"] for r in explain]),
                              "rank": _mean([r["m"]["agreement_det"]["rank_agreement"] for r in explain]),
                              "top3_coverage": _mean([r["m"]["agreement_det"]["top3_coverage"] for r in explain])},
            "agreement_llm": ({"value": _mean([r["m"]["agreement_llm"]["value_agreement"] for r in explain]),
                               "sign": _mean([r["m"]["agreement_llm"]["sign_agreement"] for r in explain]),
                               "rank": _mean([r["m"]["agreement_llm"]["rank_agreement"] for r in explain])}
                              if explain and explain[0]["m"].get("agreement_llm") else None),
            "by_level": {lv: {"words": _mean([r["m"]["words"] for r in shown if r["level"] == lv]),
                              "reading_ease": _mean([r["m"]["reading_ease"] for r in shown if r["level"] == lv]),
                              "level_violation_rate": _mean([float(bool(r["m"]["level_violations"])) for r in shown if r["level"] == lv])}
                         for lv in sorted({r["level"] for r in rs})},
            "cost": {"calls": _mean([r["usage"]["calls"] for r in rs]),
                     "tokens": _mean([r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"] for r in rs]),
                     "latency_ms": _mean([r["latency_ms"] for r in rs])},
            "error_free_outside_rule_escalation": rate(sum(r["m"]["error_free"] for r in rule_shown), len(rule_shown)),
            "coverage_risk": cov,
        }
    return out


def to_markdown(res: dict) -> str:
    def ci(r):
        if not r or r["rate"] is None:
            return "n/a"
        return f"{r['rate']:.1%} [{r['ci95'][0]:.0%}, {r['ci95'][1]:.0%}]"

    def f(x, pct=False):
        return "n/a" if x is None else (f"{x:.1%}" if pct else f"{x:.2f}")

    out = [f"# System comparison ({res['created']})", "",
           f"Generator `{res['llm']}`, reviewer `{res['reviewer']}`, judge (LLM extractor) `{res['judge'] or 'not used'}`. "
           f"Cases: {res['n']} per domain x levels {res['levels']} x intents {res['intents']}.", ""]
    if res["mock_or_standin"]:
        out += ["> Mock LLM and/or synthetic stand-in data: these numbers test the software only. Do not report them.", ""]
    out += ["> Measurement shares the deterministic text and policy rules this framework gates on, which favours it. "
            "Report the LLM-extractor agreement and the Phase 3 human labels alongside these numbers.", ""]
    for dom in res["domains"]:
        rows = [v for v in res["summary"].values() if v["domain"] == dom]
        out += [f"## {dom}", "",
                "| System | n shown | Error-free | Any fidelity error | Any policy violation | VA | SA | RA | Top-3 cov. | Calls | Tokens |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for v in rows:
            a = v["agreement_det"]
            out.append(f"| {v['system']} | {v['n_shown']}/{v['n_cases']} | {ci(v['error_free'])} | {ci(v['any_fidelity_error'])} | "
                       f"{ci(v['any_policy_violation'])} | {f(a['value'])} | {f(a['sign'])} | {f(a['rank'])} | "
                       f"{f(a['top3_coverage'])} | {f(v['cost']['calls'])} | {f(v['cost']['tokens'])} |")
        if any(v["agreement_llm"] for v in rows):
            out += ["", "LLM-extractor agreement:", "", "| System | VA | SA | RA |", "|---|---|---|---|"]
            for v in rows:
                a = v["agreement_llm"] or {}
                out.append(f"| {v['system']} | {f(a.get('value'))} | {f(a.get('sign'))} | {f(a.get('rank'))} |")
        out += ["", "Policy violation codes:", ""]
        for v in rows:
            if v["policy_codes"]:
                out.append(f"- {v['system']}: {v['policy_codes']}")
        out += ["", "Coverage-risk (escalation-score threshold: share explained / error rate among explained):", "",
                "| System | " + " | ".join(str(t) for t in THRESHOLDS) + " |", "|---|" + "---|" * len(THRESHOLDS)]
        for v in rows:
            cells = [f"{c['coverage']:.0%}/{'n/a' if c['error_rate'] is None else format(c['error_rate'], '.0%')}"
                     for c in v["coverage_risk"]]
            out.append(f"| {v['system']} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--systems", default="")
    ap.add_argument("--domains", default="hr")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N held-out examples")
    ap.add_argument("--levels", default="novice,intermediate,expert")
    ap.add_argument("--intents", default="explain")
    ap.add_argument("--judge", action="store_true", help="also measure with the LLM extractor (JUDGE_PROVIDER)")
    args = ap.parse_args()
    systems = [s for s in (args.systems.split(",") if args.systems else PRESETS[args.preset or "baselines"]) if s]
    for s in systems:
        if not (s in SYSTEMS or (s.startswith("ours:") and s[5:] in ABLATIONS)):
            raise SystemExit(f"unknown system {s}")
    reg = default_registry()
    mock_llm.DOMAIN_RESOLVER = reg.get
    llm, reviewer = get_llm(), get_reviewer()
    judge = get_judge() if args.judge else None
    svc = Service(registry=reg, store=Store(os.path.join(tempfile.mkdtemp(), "cmp.sqlite3")), llm=llm,
                  reviewer_llm=reviewer)
    levels, intents = args.levels.split(","), args.intents.split(",")
    records = []
    for dname in args.domains.split(","):
        d = reg.get(dname)
        rt = d.runtime("main")
        for cid, x in cases_for(d, args.n, args.offset):
            esc = escalation_info(rt, x, rt.predict(x))
            for level in levels:
                for intent in intents:
                    pkg = build_package(d, x, level, intent)
                    plan = make_plan(pkg, level, intent)
                    q = d.language.default_question if intent == "explain" else "What would lower the risk?"
                    for s in systems:
                        t0 = time.perf_counter()
                        if s.startswith("ours:"):
                            o = run_ours(svc, s[5:], d, x, level, intent, q)
                        else:
                            r = run_baseline(s, pkg, plan, d, llm, reviewer)
                            o = {"text": r.text, "status": "shown" if r.text.strip() else "empty", "attempts": r.rounds,
                                 "usage": r.usage, "accepted": r.accepted}
                        lat = (time.perf_counter() - t0) * 1000
                        m = measure_text(o["text"], pkg, plan, d, judge)
                        records.append({"domain": dname, "system": s, "case": cid, "level": level, "intent": intent,
                                        "status": o["status"], "attempts": o["attempts"], "accepted": o["accepted"],
                                        "usage": o["usage"], "latency_ms": round(lat, 1),
                                        "escalation_score": esc["score"], "would_escalate_rule": esc["rule_decision"],
                                        "text": o["text"], "m": m})
                print(f"{dname} case {cid} done", flush=True)
    data_sources = {n: getattr(reg.get(n), "data_source", "unknown") for n in args.domains.split(",")}
    res = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "llm": llm.id,
           "reviewer": reviewer.id, "judge": judge.id if judge else None, "systems": systems,
           "domains": args.domains.split(","), "n": args.n, "levels": levels, "intents": intents,
           "data_sources": data_sources,
           "mock_or_standin": llm.name == "mock" or any(v == "synthetic_standin" for v in data_sources.values()),
           "summary": summarize(records), "records": records}
    RESULTS.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"comparison_{stamp}.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    md = to_markdown(res)
    (RESULTS / f"comparison_{stamp}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
