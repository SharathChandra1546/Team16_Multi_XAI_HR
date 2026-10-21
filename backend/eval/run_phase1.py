"""Phase I experiment: how often the raw LLM output is wrong, what the full pipeline delivers, and how levels differ.

  python eval/run_phase1.py --n 30                 # uses LLM_PROVIDER from .env
  python eval/run_phase1.py --n 30 --reviewer      # also compares an LLM reviewer with the deterministic checks

For each held-out employee x level x intent:
  A. raw first attempt: violations found by the verifier + critic (errors that would reach users without checks)
  B. full pipeline: final status (verified / fallback / escalated), attempts, latency
  C. optional LLM reviewer on the raw first attempt: agreement with the deterministic checks
Writes results/phase1_<timestamp>.json and .md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agents.critic import critique  # noqa: E402
from app.agents.generator import generate  # noqa: E402
from app.agents.planner import make_plan  # noqa: E402
from app.agents.textutil import flesch_reading_ease, word_count  # noqa: E402
from app.agents.verifier import verify  # noqa: E402
from app.config import settings  # noqa: E402
from app.agents.reviewer import get_reviewer  # noqa: E402
from app.schemas import ExplanationRequest  # noqa: E402
from app.service import Service  # noqa: E402
from app.xai.evidence import build_package  # noqa: E402
from eval.run_faults import reviewer_flags  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
LEVELS = ["novice", "intermediate", "expert"]
QUESTIONS = {"explain": None, "recourse": "What would lower their risk?"}  # None: the domain's default question


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--reviewer", action="store_true")
    ap.add_argument("--domain", default="hr")
    args = ap.parse_args()
    svc = Service()
    d = svc.registry.get(args.domain)
    rt = d.runtime("main")
    judge = get_reviewer() if args.reviewer else None
    prompt = svc.store.active_prompt("generator")
    xs = [{f: float(r[f]) for f in rt.features} for _, r in rt.examples.head(args.n).iterrows()]

    raw_codes = Counter()
    raw_pass = defaultdict(list)
    status = Counter()
    attempts, latency = [], []
    level_stats = defaultdict(lambda: {"words": [], "reading_ease": [], "terminology_violations": 0})
    agree = Counter()
    n_packages = stability_warn = agreement_warn = 0
    shown_policy = shown_fidelity = 0
    escalation_scores = []
    for x in xs:
        raw_inst = {f: (d.readable(f, v) if d.spec(f).categorical else v) for f, v in x.items()}
        for intent, q in QUESTIONS.items():
            q = q or d.language.default_question
            for level in LEVELS:
                pkg = build_package(d, x, level, intent)
                n_packages += 1
                stability_warn += any(c.name == "stability" and not c.passed for c in pkg.checks)
                agreement_warn += any(c.name == "method_agreement" and not c.passed for c in pkg.checks)
                plan = make_plan(pkg, level, intent)
                if pkg.prediction.band_uncertain or pkg.prediction.out_of_distribution:
                    status["escalated (no generation)"] += 1
                    continue
                gen = generate(pkg, plan, d, svc.llm, prompt["template"], prompt["guidance"], q)
                v = verify(pkg, plan, gen, d)
                c = critique(pkg, plan, gen, d)
                crit = {x2.code for x2 in v.violations + c.violations if x2.severity == "critical"}
                for code in crit:
                    raw_codes[code] += 1
                raw_pass[level].append(not crit)
                if judge:
                    flagged = reviewer_flags(judge, pkg, plan, gen)
                    agree[("checks_fail" if crit else "checks_pass", "reviewer_fail" if flagged else "reviewer_pass")] += 1
                t0 = time.perf_counter()
                resp = svc.explain(ExplanationRequest(query=q, instance=raw_inst, level=level, domain=args.domain,
                                                      consent_cloud=True))
                latency.append((time.perf_counter() - t0) * 1000)
                status[resp.status] += 1
                attempts.append(resp.attempts)
                if resp.policy_report:
                    shown_policy += resp.policy_report.get("shown_policy_violations", 0)
                    shown_fidelity += resp.policy_report.get("shown_fidelity_violations", 0)
                if resp.prediction is not None and resp.prediction.escalation:
                    escalation_scores.append(resp.prediction.escalation["score"])
                if resp.explanation:
                    t = resp.explanation.explanation
                    level_stats[level]["words"].append(word_count(t))
                    level_stats[level]["reading_ease"].append(flesch_reading_ease(t))
                    level_stats[level]["terminology_violations"] += sum(
                        1 for vv in (resp.critic.violations if resp.critic else []) if vv.code == "terminology")
    n_raw = sum(len(v) for v in raw_pass.values())
    res = {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "llm": svc.llm.id, "judge": judge.id if judge else None, "data_source": getattr(d, "data_source", "unknown"),
        "prompt_version": prompt.get("version"), "n_employees": len(xs),
        "raw_first_attempt": {"n": n_raw, "pass_rate": sum(sum(v) for v in raw_pass.values()) / max(1, n_raw),
                              "pass_rate_by_level": {k: sum(v) / len(v) for k, v in raw_pass.items() if v},
                              "violation_counts": dict(raw_codes.most_common())},
        "pipeline": {"status": dict(status), "mean_attempts": statistics.mean(attempts) if attempts else None,
                     "median_latency_ms": statistics.median(latency) if latency else None,
                     "p95_latency_ms": sorted(latency)[int(0.95 * (len(latency) - 1))] if latency else None},
        "levels": {k: {"mean_words": statistics.mean(v["words"]) if v["words"] else None,
                       "mean_reading_ease": statistics.mean(v["reading_ease"]) if v["reading_ease"] else None,
                       "terminology_violations": v["terminology_violations"]} for k, v in level_stats.items()},
        "reviewer_vs_checks": {f"{a}/{b}": n for (a, b), n in agree.items()} if judge else None,
        "evidence_warnings": {"n_packages": n_packages,
                              "stability_warning_rate": stability_warn / max(1, n_packages),
                              "method_agreement_warning_rate": agreement_warn / max(1, n_packages)},
        "shown_violations": {"policy": shown_policy, "fidelity": shown_fidelity},
        "escalation_scores": escalation_scores,
    }
    RESULTS.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"phase1_{stamp}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    warn = ("> Mock LLM and/or synthetic stand-in data: software test only, not a result.\n\n"
            if svc.llm.name == "mock" or res["data_source"] == "synthetic_standin" else "")
    md = [f"# Phase I results ({res['created']})", "", warn + f"LLM `{res['llm']}`, prompt `{res['prompt_version']}`, "
          f"{res['n_employees']} employees x 3 levels x 2 intents.", "",
          "## Raw first attempt (before verification)", "",
          f"Pass rate: {res['raw_first_attempt']['pass_rate']:.1%} of {n_raw}. By level: "
          + ", ".join(f"{k} {v:.1%}" for k, v in res["raw_first_attempt"]["pass_rate_by_level"].items()), "",
          "| Violation code | Count |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in res["raw_first_attempt"]["violation_counts"].items()] or ["| none | 0 |"]
    md += ["", "## Full pipeline", "", f"Status: {res['pipeline']['status']}. Mean attempts: {res['pipeline']['mean_attempts']}. "
           f"Median latency: {res['pipeline']['median_latency_ms']} ms (p95 {res['pipeline']['p95_latency_ms']} ms).", "",
           "## Levels", "", "| Level | Mean words | Mean reading ease | Terminology violations |", "|---|---|---|---|"]
    for k in LEVELS:
        v = res["levels"].get(k)
        if v and v["mean_words"] is not None:
            md.append(f"| {k} | {v['mean_words']:.0f} | {v['mean_reading_ease']:.0f} | {v['terminology_violations']} |")
    ew = res["evidence_warnings"]
    md += ["", "## Evidence warnings", "",
           f"Shapley stability warnings: {ew['stability_warning_rate']:.1%} of {ew['n_packages']} packages; "
           f"Shapley/LIME direction disagreement warnings: {ew['method_agreement_warning_rate']:.1%}. "
           "Raise SHAP_PERMUTATIONS if the stability rate is high.", "",
           f"Violations in shown text: policy {res['shown_violations']['policy']}, "
           f"fidelity {res['shown_violations']['fidelity']} (should be 0 with all gates on)."]
    if judge:
        md += ["", "## LLM reviewer vs deterministic checks", "", f"{res['reviewer_vs_checks']}"]
    text = "\n".join(md) + "\n"
    (RESULTS / f"phase1_{stamp}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
