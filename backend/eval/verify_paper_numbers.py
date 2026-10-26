"""Independent check of the paper tables: recompute every headline cell from results/paper/records.csv.

    python eval/verify_paper_numbers.py

Uses its own Wilson interval and exact McNemar code (scipy) and its own de-duplication-free counting over the
per-text rows written by eval/paper_tables.py, then compares with results/paper/numbers.json. Prints every
mismatch and exits 1 if there is any.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

PAPER = Path(__file__).resolve().parents[1] / "results" / "paper"
MAIN_WRITER, ABL_WRITER = "qwen/qwen3.8-27b", "gemma-4-26b-a4b-it"


def wilson(k, n, z=1.959964):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def close(a, b, tol=1e-6):
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= tol)


def main() -> int:
    rows = list(csv.DictReader(open(PAPER / "records.csv", encoding="utf-8")))
    num = json.loads((PAPER / "numbers.json").read_text(encoding="utf-8"))
    bad = []

    def check(label, got, want):
        if not close(got, want, 1e-4):
            bad.append(f"{label}: recomputed {got} vs table {want}")

    keys = {(r["writer"], r["domain"], r["case"], r["level"], r["intent"], r["system"]) for r in rows}
    if len(keys) != len(rows):
        bad.append(f"duplicate rows: {len(rows) - len(keys)}")
    groups = (("main", MAIN_WRITER), ("ablations", ABL_WRITER))
    for section, writer in groups:
        sub = [r for r in rows if r["writer"] == writer and r["intent"] == "explain"]
        for s in sorted({r["system"] for r in sub}):
            for dom in ("pooled", "finance", "health"):
                rs = [r for r in sub if r["system"] == s and (dom == "pooled" or r["domain"] == dom)]
                scored = [r for r in rs if r["failed"] == "0" and r["empty"] == "0" and r["error_free_llm"] != ""]
                k, n = sum(int(r["error_free_llm"]) for r in scored), len(scored)
                ref = num[section][s] if section == "ablations" else num[section][s][dom]
                if section == "ablations" and dom != "pooled":
                    continue
                t = ref["error_free_llm"]
                check(f"{section}/{s}/{dom} n", n, t["n"])
                check(f"{section}/{s}/{dom} k", k, t["k"])
                lo, hi = wilson(k, n)
                check(f"{section}/{s}/{dom} lo", lo, t["lo"])
                check(f"{section}/{s}/{dom} hi", hi, t["hi"])
    # paired tests
    sub = [r for r in rows if r["writer"] == MAIN_WRITER and r["intent"] == "explain" and r["failed"] == "0"
           and r["empty"] == "0" and r["error_free_llm"] != ""]
    by = defaultdict(dict)
    for r in sub:
        by[(r["domain"], r["case"], r["level"])][r["system"]] = int(r["error_free_llm"])
    for s, ref in num["paired"].items():
        b = c = n = 0
        for v in by.values():
            if "ours:full" in v and s in v:
                n += 1
                b += v["ours:full"] and not v[s]
                c += v[s] and not v["ours:full"]
        p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        check(f"paired/{s} pairs", n, ref["pairs"])
        check(f"paired/{s} ours_only", b, ref["ours_only"])
        check(f"paired/{s} other_only", c, ref["other_only"])
        check(f"paired/{s} p", round(min(1.0, p), 6), round(ref["p_mcnemar"], 6))
    # LLM-written faults: independent merge of the raw chunk files (latest file per domain and case offset)
    res = PAPER.parent
    chunks = {}
    for f in sorted(res.glob("faults_*.json"), key=lambda p: p.stat().st_mtime):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("created", "") < "2026-10-24":
            continue  # earlier phases (some with a mock fault writer) are not part of this study
        for x in d.get("domains", []):
            if x.get("injector") == "llm" and not str(x.get("fault_llm") or "").startswith("mock"):
                chunks[(x["domain"], d.get("offset", 0))] = x
    if chunks and "llm" in num.get("faults", {}):
        k = sum(v["any_rejection"]["k"] for x in chunks.values() for v in x["per_fault"].values())
        n = sum(v["any_rejection"]["n"] for x in chunks.values() for v in x["per_fault"].values())
        ke = sum(v["detection"]["k"] for x in chunks.values() for v in x["per_fault"].values())
        cn = sum(x["clean"]["n"] for x in chunks.values())
        ck = sum(x["clean"]["false_rejection"]["k"] for x in chunks.values())
        ref = num["faults"]["llm"]
        check("faults/llm n", n, ref["overall"]["n"])
        check("faults/llm blocked k", k, ref["overall"]["k"])
        check("faults/llm exact k", ke, ref["overall_exact"]["k"])
        check("faults/llm clean n", cn, ref["clean_false_rejection"]["n"])
        check("faults/llm clean rejected", ck, ref["clean_false_rejection"]["k"])
    print(f"checked {len(rows)} rows; {len(bad)} mismatch(es)")
    for line in bad:
        print("  MISMATCH", line)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
