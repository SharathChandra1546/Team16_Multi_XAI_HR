"""Check that everything Phase 3 needs is in place, and say exactly what is missing.

  python scripts/preflight_phase3.py

Checks (each prints OK, MISSING or WARN):
  1. the three real datasets exist in backend/data/ and every model was trained on them (not the stand-in)
  2. LLM settings: generator, reviewer from a DIFFERENT model family, judge, fault generator, API keys present
  3. team-written held-out guardrail sets exist for each domain (eval/data/team_heldout_<domain>.json)
  4. the healthcare clinician sign-off (docs/CLINICIAN_REVIEW.md) is filled in
Exit code 0 only if nothing is MISSING. No network calls are made and no key values are printed.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from app.agents.reviewer import family  # noqa: E402
from app.config import PROVIDERS  # noqa: E402

DATA = {"hr": "WA_Fn-UseC_-HR-Employee-Attrition.csv", "finance": "german.data", "health": "diabetes.csv"}
bad = 0


def report(status: str, msg: str) -> None:
    global bad
    bad += status == "MISSING"
    print(f"{status:8s} {msg}")


def main() -> int:
    print("1. Data and trained models")
    for dom, fn in DATA.items():
        report("OK" if (ROOT / "data" / fn).exists() else "MISSING", f"data/{fn} ({dom})")
        man = ROOT / "models" / f"{dom}_manifest.json"
        if not man.exists():
            report("MISSING", f"models/{dom}_manifest.json: not trained (python scripts/{'train_hr.py' if dom == 'hr' else 'train_domain.py --domain ' + dom})")
            continue
        src = json.loads(man.read_text(encoding="utf-8")).get("dataset", {}).get("source", "unknown")
        report("MISSING" if src == "synthetic_standin" else "OK", f"{dom} model trained on: {src}"
               + (" (retrain after adding the real file)" if src == "synthetic_standin" else ""))

    print("\n2. LLM settings")
    gen = os.getenv("LLM_PROVIDER", "mock")
    rev, judge, fault = (os.getenv(k, "") for k in ("REVIEWER_PROVIDER", "JUDGE_PROVIDER", "FAULT_PROVIDER"))
    report("MISSING" if gen == "mock" else "OK", f"LLM_PROVIDER={gen}" + (" (mock outputs are not results)" if gen == "mock" else ""))
    model = lambda prov, var: os.getenv(var) or PROVIDERS.get(prov, {}).get("default_model", "")
    gid = f"{gen}:{model(gen, 'LLM_MODEL')}"
    for name, val in (("REVIEWER_PROVIDER", rev), ("JUDGE_PROVIDER", judge), ("FAULT_PROVIDER", fault)):
        report("OK" if val and val != "mock" else "MISSING", f"{name}={val or '(unset)'}")
    if rev and gen != "mock":
        rid = f"{rev}:{model(rev, 'REVIEWER_MODEL')}"
        same = family(rid) == family(gid)
        report("MISSING" if same else "OK", f"reviewer family {family(rid)} vs generator family {family(gid)}"
               + (": must differ, or the reviewer shares the generator's blind spots" if same else ""))
    for p in sorted({gen, rev, judge, fault} - {"", "mock"}):
        env = PROVIDERS.get(p, {}).get("key_env")
        if env is None:
            report("MISSING", f"provider {p!r} is not supported; use one of {', '.join(PROVIDERS)}")
        elif p == "ollama":
            report("OK", "ollama runs locally, no key needed (check that it is running)")
        else:
            report("OK" if os.getenv(env) else "MISSING", f"{env} is {'set' if os.getenv(env) else 'not set'} (for {p})")
    report("OK" if os.getenv("LLM_TEMPERATURE", "0") == "0" else "WARN", f"LLM_TEMPERATURE={os.getenv('LLM_TEMPERATURE', '0')} (0 for reproducible runs)")

    print("\n3. Team-written held-out guardrail sets (see docs/GUARDRAIL_SET_GUIDE.md)")
    for dom in DATA:
        p = ROOT / "eval" / "data" / f"team_heldout_{dom}.json"
        if not p.exists():
            report("MISSING", f"eval/data/team_heldout_{dom}.json")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        n_r, n_a = len(d.get("refuse", [])), len(d.get("allow", []))
        ok = n_r >= 40 and n_a >= 25 and d.get("written_by", "").strip() and not re.search(r"\b(claude|ai|gpt|chatgpt|llm|gemini)\b", d["written_by"], re.I)
        report("OK" if ok else "MISSING", f"{dom}: {n_r} refuse, {n_a} allow, written_by={d.get('written_by', '(unset)')!r} (need >=40, >=25, human authors)")

    print("\n4. Clinician review of the healthcare policy")
    cr = ROOT.parent / "docs" / "CLINICIAN_REVIEW.md"
    txt = cr.read_text(encoding="utf-8") if cr.exists() else ""
    m = re.search(r"Reviewed by:\s*(.+)", txt)
    report("OK" if m and m.group(1).strip() and "____" not in m.group(1) else "MISSING", "docs/CLINICIAN_REVIEW.md signed off ('Reviewed by:' filled in)")
    print(f"\n{'Ready for Phase 3.' if not bad else f'{bad} item(s) MISSING. Fix them, then run this again.'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
