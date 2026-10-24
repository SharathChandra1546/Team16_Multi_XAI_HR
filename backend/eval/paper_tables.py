"""Build the paper's result tables from all saved Phase 3 runs (merged, de-duplicated, re-measured).

    python eval/paper_tables.py [--judge] [--since 2026-10-24]

Inputs (results/): comparison_*.json for the main comparison (writer qwen) and ablations (writer gemma),
faults_*.json (LLM-written and template faults), models/*_manifest.json.
Every text is re-measured with the current measurement code, so all systems are scored by the same rules;
--judge re-runs the LLM extractor (answers come from the cache when the text was scored before).

Outputs (results/paper/): tables.md (everything, with caveats), one .tex and one .csv per table, numbers.json
(every headline number, for the verification pass), runs.json (which files and models went in).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "eval"))

RESULTS = BACKEND / "results"
OUT = RESULTS / "paper"
MAIN_WRITER = "qwen/qwen3.8-27b"
ABL_WRITER = "gemma-4-26b-a4b-it"
DOMAINS = ["finance", "health"]
SYS_ORDER = ["ours:full", "direct", "b3", "explingo", "critic_rule", "musicco"]
SYS_LABEL = {"ours:full": "Ours (full pipeline)", "direct": "Direct prompting", "b3": "B3 (LLM reviewer loop)",
             "explingo": "Explingo-style", "critic_rule": "Critic-Rule (He \\& Martens)",
             "musicco": "Musicco et al.-style"}
ABL_ORDER = ["ours:full", "ours:no_verifier", "ours:no_critic", "ours:no_repair", "ours:no_policy_rules"]
ABL_LABEL = {"ours:full": "Full pipeline", "ours:no_verifier": "-- claim verifier", "ours:no_critic": "-- critic",
             "ours:no_repair": "-- repair loop", "ours:no_policy_rules": "-- policy rules"}


def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - a) / d, (c + a) / d)


def rate(k, n):
    lo, hi = wilson(k, n)
    if lo is not None:
        lo, hi = max(0.0, lo), min(1.0, hi)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "lo": lo, "hi": hi}


def fmt(r, tex=False):
    if not r or r["n"] == 0:
        return "n/a"
    pct = "\\%" if tex else "%"
    return f"{100 * r['rate']:.0f}{pct} [{100 * r['lo']:.0f}, {100 * r['hi']:.0f}]"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def f2(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


# ------------------------------------------------------------------------------------------------ loading
def load_runs(since: str):
    runs, recs = [], {}
    for f in sorted(RESULTS.glob("comparison_*.json")):
        if "rescored" in f.name:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("created", "") < since or d.get("llm", "").startswith("mock"):
            continue
        writer = d["llm"].split(":", 1)[1]
        if writer not in (MAIN_WRITER, ABL_WRITER):
            continue
        runs.append({"file": f.name, "llm": d["llm"], "reviewer": d.get("reviewer"), "judge": d.get("judge"),
                     "n_records": len(d["records"])})
        for r in d["records"]:
            key = (writer, r["domain"], str(r["case"]), r["level"], r["intent"], r["system"])
            failed = str(r.get("status", "")).startswith("llm_failed")
            old = recs.get(key)
            # keep a successful generation over a failed one; otherwise the latest file wins
            if old is None or (old["_failed"] and not failed) or (old["_failed"] == failed):
                recs[key] = dict(r, _failed=failed, _writer=writer, _file=f.name)
    return runs, list(recs.values())


def remeasure(recs, use_judge: bool):
    from app.agents.planner import make_plan
    from app.domains.registry import default_registry
    from app.llm.client import get_judge
    from app.metrics.measure import measure_text
    from app.xai.evidence import build_package

    reg = default_registry()
    judge = get_judge() if use_judge else None
    cases, pk = {}, {}
    for r in recs:
        if r["_failed"]:
            continue
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
    return judge.id if judge else None


# ------------------------------------------------------------------------------------------------ tables
def system_row(rs):
    shown = [r for r in rs if not r["_failed"] and not r["m"]["empty"]]
    scored = [r for r in shown if r["m"].get("error_free_llm") is not None]
    llm = [r["m"]["agreement_llm"] for r in scored if r["m"].get("agreement_llm")]
    return {
        "n_generated": len(rs), "n_failed": sum(r["_failed"] for r in rs), "n_shown": len(shown),
        "n_empty": sum(1 for r in rs if not r["_failed"] and r["m"]["empty"]),
        "status": dict(Counter(r.get("status") for r in rs if not r["_failed"])),
        "error_free_llm": rate(sum(r["m"]["error_free_llm"] for r in scored), len(scored)),
        "error_free_rule": rate(sum(r["m"]["error_free"] for r in shown), len(shown)),
        "fidelity_error_llm": rate(sum(r["m"]["fidelity_errors_llm"] > 0 for r in scored), len(scored)),
        "policy_violation": rate(sum(r["m"]["n_policy_violations"] > 0 for r in shown), len(shown)),
        "va": mean([a["value_agreement"] for a in llm]), "sa": mean([a["sign_agreement"] for a in llm]),
        "ra": mean([a["rank_agreement"] for a in llm]),
        "top3": mean([a["top3_coverage"] for a in llm]),
        "words": mean([r["m"]["words"] for r in shown]),
        "calls": mean([r["usage"]["calls"] for r in rs if not r["_failed"]]),
        "tokens": mean([r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"] for r in rs if not r["_failed"]]),
        "latency_s": mean([r["latency_ms"] / 1000 for r in rs if not r["_failed"]]),
    }


def error_breakdown(rs):
    c = Counter()
    for r in rs:
        if r["_failed"] or r["m"]["empty"]:
            continue
        m, a = r["m"], (r["m"].get("agreement_llm") or {})
        c["value"] += bool(a.get("value_errors"))
        c["direction"] += bool(a.get("sign_errors"))
        c["stated rank"] += bool(a.get("stated_rank_errors"))
        c["unsupported factor"] += bool(a.get("unsupported_mentions"))
        c["unsupported number"] += bool(m["text_checks"]["unsupported_numbers"])
        c["causal wording"] += bool(m["text_checks"]["causal_wording"])
        c["raw code"] += bool(m["text_checks"]["raw_codes"])
        for code in m["policy_violations"]:
            c[f"policy: {code}"] += 1
    return c


def paired(recs, base_sys, other, writer):
    by = defaultdict(dict)
    for r in recs:
        if r["_writer"] != writer or r["_failed"] or r["m"]["empty"] or r["m"].get("error_free_llm") is None:
            continue
        by[(r["domain"], r["case"], r["level"], r["intent"])][r["system"]] = r["m"]["error_free_llm"]
    b = c = both = neither = 0
    for v in by.values():
        if base_sys in v and other in v:
            x, y = v[base_sys], v[other]
            b += x and not y
            c += y and not x
            both += x and y
            neither += not x and not y
    return {"pairs": b + c + both + neither, "ours_only": b, "other_only": c, "both": both, "neither": neither,
            "p_mcnemar": mcnemar_exact(b, c)}


def write_csv(name, header, rows):
    with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def write_tex(name, caption, label, header, rows, colspec=None):
    colspec = colspec or ("l" + "c" * (len(header) - 1))
    lines = ["% generated by backend/eval/paper_tables.py -- do not edit by hand", "\\begin{table}[t]", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\footnotesize", f"\\begin{{tabular}}{{{colspec}}}",
             "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(x) for x in row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (OUT / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="re-run the LLM extractor (cached answers are free)")
    ap.add_argument("--since", default="2026-10-24")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs, recs = load_runs(a.since)
    judge_id = remeasure(recs, a.judge)
    numbers: dict = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                     "judge": judge_id or "as stored", "runs": runs}
    md = ["# Phase 3 results (reduced real-LLM study): finance and health", "",
          f"Generated {numbers['generated']} by `backend/eval/paper_tables.py` from {len(runs)} run files. "
          "Every text was re-measured with the current measurement code. Do not edit numbers by hand.", ""]

    # ---------------- main comparison
    main_recs = [r for r in recs if r["_writer"] == MAIN_WRITER and r["intent"] == "explain"]
    systems = [s for s in SYS_ORDER if any(r["system"] == s for r in main_recs)]
    rows_tex, rows_csv, rows_md, num_main = [], [], [], {}
    for s in systems:
        rs = [r for r in main_recs if r["system"] == s]
        row = system_row(rs)
        per = {dom: system_row([r for r in rs if r["domain"] == dom]) for dom in DOMAINS}
        num_main[s] = {"pooled": row, **per}
        cells = [fmt(row["error_free_llm"], True), fmt(per["finance"]["error_free_llm"], True),
                 fmt(per["health"]["error_free_llm"], True), fmt(row["error_free_rule"], True),
                 fmt(row["policy_violation"], True), f2(row["va"]), f2(row["sa"]), f2(row["ra"]),
                 f2(row["calls"], 1), f"{row['tokens'] / 1000:.1f}k" if row["tokens"] else "n/a"]
        rows_tex.append([SYS_LABEL[s]] + cells)
        rows_md.append([SYS_LABEL[s].replace("\\&", "&"), row["n_shown"]] + [c.replace("\\%", "%") for c in cells])
        rows_csv.append([s, row["n_shown"], row["error_free_llm"]["k"], row["error_free_llm"]["n"],
                         row["error_free_llm"]["rate"], row["error_free_llm"]["lo"], row["error_free_llm"]["hi"],
                         per["finance"]["error_free_llm"]["rate"], per["health"]["error_free_llm"]["rate"],
                         row["error_free_rule"]["rate"], row["policy_violation"]["rate"], row["va"], row["sa"],
                         row["ra"], row["top3"], row["calls"], row["tokens"], row["latency_s"], row["n_failed"]])
    head = ["System", "Error-free (pooled)", "Finance", "Health", "Error-free (rule-based)", "Policy viol.",
            "VA", "SA", "RA", "Calls", "Tokens"]
    write_tex("table_main_comparison",
              "Explanation correctness by system (writer: Qwen 3.8 27B; scorer: gpt-oss-120b). Error-free: no wrong "
              "value, direction or stated rank, no unsupported factor or number, no causal wording, no raw code and no "
              "policy violation. 95\\% Wilson intervals. VA/SA/RA: value, sign and rank agreement (LLM extractor).",
              "tab:main", head, rows_tex, "lccccccccc" + "c")
    write_csv("table_main_comparison", ["system", "n", "ef_k", "ef_n", "ef_rate", "ef_lo", "ef_hi", "ef_finance",
                                        "ef_health", "ef_rule_based", "policy_violation", "va", "sa", "ra", "top3",
                                        "calls", "tokens", "latency_s", "failed_generations"], rows_csv)
    md += ["## Table 1. Main comparison", "", "Writer for every system: Qwen 3.8 27B (Groq). Each system's own "
           "review/critique steps use the same writer. Scorer: gpt-oss-120b (Groq, a different model family). "
           "Questions: \"why this estimate\" (explain); levels: novice and expert.", ""]
    md += md_table(["System", "n"] + head[1:], rows_md) + [""]

    # paired tests vs ours
    num_pairs = {}
    rows = []
    for s in systems:
        if s == "ours:full":
            continue
        p = paired(main_recs, "ours:full", s, MAIN_WRITER)
        num_pairs[s] = p
        rows.append([SYS_LABEL[s], p["pairs"], p["ours_only"], p["other_only"], p["both"], p["neither"],
                     f"{p['p_mcnemar']:.3g}"])
    head_p = ["Baseline", "Pairs", "Only ours correct", "Only baseline correct", "Both", "Neither", "Exact McNemar p"]
    write_tex("table_paired_tests", "Paired comparison with our pipeline on the same cases and levels (error-free, "
              "LLM scorer). Exact two-sided McNemar test.", "tab:paired", head_p, rows)
    write_csv("table_paired_tests", head_p, rows)
    md += ["### Paired tests (same case and level)", ""] + md_table(head_p, [[c.replace("\\&", "&") if isinstance(c, str) else c for c in r] for r in rows]) + [""]

    # error breakdown
    rows, num_err = [], {}
    kinds = ["value", "direction", "stated rank", "unsupported factor", "unsupported number", "causal wording",
             "raw code"]
    for s in systems:
        rs = [r for r in main_recs if r["system"] == s]
        c = error_breakdown(rs)
        num_err[s] = dict(c)
        n = num_main[s]["pooled"]["n_shown"]
        pol = sum(v for k, v in c.items() if k.startswith("policy"))
        rows.append([SYS_LABEL[s], n] + [c.get(k, 0) for k in kinds] + [pol])
    head_e = ["System", "Texts"] + [k.capitalize() for k in kinds] + ["Policy"]
    write_tex("table_error_types", "Texts with at least one error of each type (main comparison).", "tab:errors",
              head_e, rows)
    write_csv("table_error_types", head_e, rows)
    md += ["### Error types (texts with at least one)", ""] + md_table(head_e, [[c.replace("\\&", "&") if isinstance(c, str) else c for c in r] for r in rows]) + [""]

    # ---------------- ablations
    abl = [r for r in recs if r["_writer"] == ABL_WRITER and r["intent"] == "explain"]
    num_abl = {}
    if abl:
        rows, rows_md = [], []
        for s in [x for x in ABL_ORDER if any(r["system"] == x for r in abl)]:
            row = system_row([r for r in abl if r["system"] == s])
            num_abl[s] = row
            st = row["status"]
            n = max(1, sum(st.values()))
            cells = [row["n_shown"], fmt(row["error_free_llm"], True), fmt(row["fidelity_error_llm"], True),
                     fmt(row["policy_violation"], True), f"{100 * st.get('fallback', 0) / n:.0f}\\%",
                     f"{100 * (st.get('withheld', 0) + st.get('escalated', 0)) / n:.0f}\\%", f2(row["calls"], 1)]
            rows.append([ABL_LABEL[s]] + cells)
            rows_md.append([ABL_LABEL[s]] + [str(c).replace("\\%", "%") for c in cells])
        head_a = ["Variant", "Texts", "Error-free", "Any fidelity error", "Policy viol.", "Template fallback",
                  "Withheld/escalated", "Calls"]
        write_tex("table_ablations", "Ablation of our pipeline (writer: Gemma 4 26B; scorer: gpt-oss-120b), finance and "
                  "health pooled. 95\\% Wilson intervals.", "tab:ablation", head_a, rows)
        write_csv("table_ablations", head_a, rows_md)
        md += ["## Table 2. Ablations", "", "Writer: Gemma 4 26B (Gemini API) for every variant; scorer gpt-oss-120b.", ""]
        md += md_table(head_a, rows_md) + [""]

    # ---------------- faults
    num_faults = {}
    llm_files, tmpl_files = [], []
    for f in sorted(RESULTS.glob("faults_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        doms = d.get("domains", [])
        if any(x.get("injector") == "llm" for x in doms) and d.get("created", "") >= a.since:
            llm_files.append((f, d))
        elif doms and all(x.get("injector") == "template" for x in doms) and len(doms) >= 3 and "llm" in d and d["llm"].startswith("mock"):
            tmpl_files.append((f, d))
    for title, files, key in (("LLM-written faults (gpt-oss-20b; detector-independent)", llm_files, "llm"),
                              ("Template faults (upper bound)", tmpl_files[-1:], "template")):
        if not files:
            continue
        # LLM faults run one (domain, case) chunk per file: merge the chunks, latest file per chunk wins
        chunks = {}
        for f, d in files:
            for x in d["domains"]:
                if x.get("injector") != key:
                    continue
                ck = (x["domain"], d.get("offset", 0), d.get("n"), tuple(d.get("levels") or []), f.name if key == "template" else "")
                chunks[ck] = (f, x)
        agg = defaultdict(lambda: [0, 0])     # blocked: any critical violation (the text is not shown)
        exact = defaultdict(lambda: [0, 0])   # flagged with the fault's own code
        clean = [0, 0]
        for f, x in chunks.values():
            for fault, v in x["per_fault"].items():
                agg[fault][0] += v["any_rejection"]["k"]
                agg[fault][1] += v["any_rejection"]["n"]
                exact[fault][0] += v["detection"]["k"]
                exact[fault][1] += v["detection"]["n"]
            clean[0] += x["clean"]["false_rejection"]["k"]
            clean[1] += x["clean"]["false_rejection"]["n"]
        f = files[-1][0]
        d = {"domains": [x for _, x in chunks.values()]}
        rows = [[fault.replace("_", " "), n, fmt(rate(k, n), True), fmt(rate(*exact[fault]), True)]
                for fault, (k, n) in sorted(agg.items())]
        tot = rate(sum(k for k, _ in agg.values()), sum(n for _, n in agg.values()))
        tot_exact = rate(sum(k for k, _ in exact.values()), sum(n for _, n in exact.values()))
        rows.append(["\\textbf{All faults}", tot["n"], fmt(tot, True), fmt(tot_exact, True)])
        cr = rate(*clean)
        num_faults[key] = {"files": sorted({ff.name for ff, _ in chunks.values()}), "chunks": len(chunks),
                           "per_fault": {k: rate(*v) for k, v in agg.items()}, "overall": tot,
                           "per_fault_exact": {k: rate(*v) for k, v in exact.items()}, "overall_exact": tot_exact,
                           "clean_false_rejection": cr, "domains": sorted({x["domain"] for x in d["domains"]})}
        name = f"table_faults_{key}"
        write_tex(name, f"Verifier detection of injected faults: {title}. Clean texts: {cr['n']}, false rejections "
                  f"{fmt(cr, True)}. Blocked: any critical violation, so the text is not shown; exact: flagged with the "
                  f"fault's own error code.", f"tab:faults-{key}", ["Fault type", "n", "Blocked", "Exact code"], rows)
        write_csv(name, ["fault", "n", "blocked", "exact_code"], [[r[0].replace("\\textbf{", "").replace("}", ""), r[1], r[2].replace("\\%", "%"), r[3].replace("\\%", "%")] for r in rows])
        md += [f"## Faults: {title}", "", f"Sources: {len(num_faults[key]['files'])} file(s), {len(chunks)} chunk(s); "
               f"domains: {', '.join(num_faults[key]['domains'])}.", ""]
        md += md_table(["Fault type", "n", "Blocked", "Exact code"], [[r[0].replace("\\textbf{", "**").replace("}", "**"), r[1], r[2].replace("\\%", "%"), r[3].replace("\\%", "%")] for r in rows])
        md += ["", f"Clean texts: {cr['n']}, false rejections {fmt(cr)}.", ""]

    # ---------------- caveats
    md += ["## Caveats to state in the paper", "",
           "- Reduced study: finance and health only, \"explain\" questions, novice and expert levels, few cases per "
           "domain; the intervals are wide. HR is evaluated without an LLM (models, fairness, verifier).",
           "- Baselines are our reimplementations of the published methods, run with the same writer model.",
           "- Measurement shares the text and policy rules our pipeline gates on, which favours it; the LLM-scorer "
           "figure and the human labels are the measures without that bias.",
           "- Open-weights models on free-tier APIs (Qwen 3.8 27B and gpt-oss on Groq, Gemma 4 26B on the Gemini API); "
           "the ablations use a different writer from the main comparison.",
           "- Generation failures (empty or error replies from the provider) are excluded and counted per system.", ""]
    numbers.update({"main": num_main, "paired": num_pairs, "errors": num_err, "ablations": num_abl,
                    "faults": num_faults})
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=str), encoding="utf-8")
    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
