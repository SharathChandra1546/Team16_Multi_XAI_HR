"""Quick install check: one explanation per domain with the offline mock model (no API key, no quota).

    python scripts/smoke_check.py
Exit code 0 when every domain ran (an escalation to human review also counts). The full test suite is `python -m pytest -q`.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["LLM_PROVIDER"] = "mock"  # never spend quota on an install check
os.environ["MOCK_FAULT_RATE"] = "0"
os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(prefix="mxai-smoke-"), "smoke.sqlite3"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    from app.domains.registry import default_registry
    from app.schemas import ExplanationRequest
    from app.service import Service

    reg = default_registry()
    svc = Service(registry=reg)
    failed = 0
    for card in svc.domain_cards():
        name = card["name"]
        d = reg.get(name)
        rt = d.runtime("main")
        x = {f: float(rt.examples.iloc[0][f]) for f in rt.features}
        raw = {f: (d.readable(f, v) if d.spec(f).categorical else v) for f, v in x.items()}
        t0 = time.perf_counter()
        try:
            resp = svc.explain(ExplanationRequest(query=d.language.default_question, instance=raw, level="novice",
                                                  domain=name))
            text = resp.explanation.explanation if resp.explanation else ""
            # "escalated" (sent to a person) and "withheld" are correct pipeline outcomes, not install failures
            ok = bool(text.strip()) or resp.status in ("escalated", "withheld")
            source = getattr(d, "data_source", "unknown")
            print(f"[{'OK' if ok else 'FAIL'}] {name:8s} {resp.status:10s} {time.perf_counter() - t0:5.1f}s  "
                  f"data: {source}")
            if text.strip():
                print(f"         {text[:160]}{'...' if len(text) > 160 else ''}")
            failed += not ok
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name:8s} {type(e).__name__}: {e}")
            failed += 1
    print("Smoke check passed." if not failed else f"Smoke check: {failed} domain(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
