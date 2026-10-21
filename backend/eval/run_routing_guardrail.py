"""Routing and guardrail evaluation across all registered domains.

  python eval/run_routing_guardrail.py

Routing (eval/data/routing_cases.json): questions without a domain and without case details.
  - accuracy: routed to the expected domain (or asked to clarify when no domain is expected)
  - misrouting: routed to a WRONG domain (the harmful error; clarifying is safe)
  - clarify rate on domain-specific questions
Also routing by case details alone (a generic question plus a held-out example from each domain).

Guardrail (rules layer only, no LLM): for every domain, the development set (used to write the patterns) and the
held-out set (written before the patterns, not used for tuning). Refusal recall and false-refusal rate, with Wilson
95% intervals. All patterns of all domains are active at once, as in deployment.

Writes results/routing_guardrail_<timestamp>.json and .md
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agents.router import Router  # noqa: E402
from app.config import settings  # noqa: E402
from app.domains.registry import default_registry  # noqa: E402
from app.ml.training import wilson  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"


def rate(k, n):
    lo, hi = wilson(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def guardrail_files(name: str) -> dict[str, Path]:
    dev = DATA / ("guardrail_cases.json" if name == "hr" else f"guardrail_cases_{name}.json")
    held = DATA / f"guardrail_heldout_{name}.json"
    team = DATA / f"team_heldout_{name}.json"  # written by the team without seeing the patterns (docs/GUARDRAIL_SET_GUIDE.md)
    return {k: p for k, p in (("dev", dev), ("heldout", held), ("team_heldout", team)) if p.exists()}


def main() -> int:
    reg = default_registry()
    router = Router(reg, settings.router_confidence_threshold)
    res = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "domains": reg.names()}

    cases = [c for c in json.loads((DATA / "routing_cases.json").read_text())["cases"]
             if c["domain"] is None or reg.has(c["domain"])]
    ok = mis = clar = spec = 0
    errors = []
    for c in cases:
        r = router.route(c["query"], None, None)
        got = r.domain
        if c["domain"] is not None:
            spec += 1
            clar += got is None
        if got == c["domain"]:
            ok += 1
        elif got is not None:
            mis += 1
            errors.append({"query": c["query"], "expected": c["domain"], "got": got})
        else:
            errors.append({"query": c["query"], "expected": c["domain"], "got": None})
    inst_ok = inst_n = 0
    for d in reg.list():
        rt = d.runtime("main")
        for _, row in rt.examples.head(10).iterrows():
            raw = {f: (d.readable(f, row[f]) if d.spec(f).categorical else float(row[f])) for f in rt.features}
            inst_n += 1
            inst_ok += router.route("Why is this case flagged?", raw, None).domain == d.name
    res["routing"] = {"n": len(cases), "accuracy": rate(ok, len(cases)), "misrouting": rate(mis, len(cases)),
                      "clarify_on_specific": rate(clar, spec), "by_case_details": rate(inst_ok, inst_n), "errors": errors}

    res["guardrail"] = {}
    for d in reg.list():
        for split, path in guardrail_files(d.name).items():
            data = json.loads(path.read_text())
            caught = [q for q in data["refuse"] if router.guard_rules(q)]
            false_ref = [q for q in data["allow"] if router.guard_rules(q)]
            res["guardrail"][f"{d.name}/{split}"] = {
                "refusal_recall": rate(len(caught), len(data["refuse"])),
                "false_refusal": rate(len(false_ref), len(data["allow"])),
                "missed": [q for q in data["refuse"] if q not in caught], "wrongly_refused": false_ref}

    RESULTS.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"routing_guardrail_{stamp}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    def ci(r):
        return "n/a" if r["rate"] is None else f"{r['rate']:.1%} [{r['ci95'][0]:.1%}, {r['ci95'][1]:.1%}] ({r['k']}/{r['n']})"

    rt_ = res["routing"]
    md = [f"# Routing and guardrail ({res['created']})", "", f"Domains: {', '.join(res['domains'])}.", "",
          "## Routing (wording only)", "", f"- Accuracy: {ci(rt_['accuracy'])}", f"- Misrouting (wrong domain): {ci(rt_['misrouting'])}",
          f"- Clarification asked on domain-specific questions: {ci(rt_['clarify_on_specific'])}",
          f"- Routing by case details (generic question): {ci(rt_['by_case_details'])}", ""]
    if rt_["errors"]:
        md += ["Errors:", ""] + [f"- {e['query']!r}: expected {e['expected']}, got {e['got']}" for e in rt_["errors"]] + [""]
    md += ["## Guardrail (rules layer)", "", "| Domain / set | Refusal recall | False refusals |", "|---|---|---|"]
    for k, v in res["guardrail"].items():
        md.append(f"| {k} | {ci(v['refusal_recall'])} | {ci(v['false_refusal'])} |")
    md += ["", "Held-out sets were written before each domain's patterns and are not used for tuning. Report them as the "
           "main guardrail result; the development sets show fit, not generalisation."]
    text = "\n".join(md) + "\n"
    (RESULTS / f"routing_guardrail_{stamp}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
