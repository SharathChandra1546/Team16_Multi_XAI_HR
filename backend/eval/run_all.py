"""Run every experiment for the paper in one go and time each step.

  python eval/run_all.py                 # sizes suitable for the paper
  python eval/run_all.py --quick         # small sizes to check that everything works

Uses LLM_PROVIDER / JUDGE_PROVIDER from backend/.env. With the mock LLM the LLM-based steps are skipped,
because mock outputs are not results.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings  # noqa: E402


def step(name: str, args: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, *args], cwd=ROOT)
    print(f"--- {name}: {'done' if r.returncode == 0 else 'FAILED'} in {time.time() - t0:.0f} s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n_faults, n_llm, n_phase1 = (8, 3, 3) if a.quick else (40, 10, 30)
    n_cmp = 3 if a.quick else 30
    from app.domains.registry import default_registry

    domains = ",".join(default_registry().names())
    step("Model and fairness tables (every trained domain)", ["eval/export_tables.py", "--all"])
    step("Routing and guardrail (dev and held-out sets)", ["eval/run_routing_guardrail.py"])
    step("Fault injection on template bases (deterministic checks)",
         ["eval/run_faults.py", "--n", str(n_faults), "--domains", domains])
    if settings.llm_provider == "mock":
        print("\nLLM_PROVIDER=mock: skipping the LLM experiments. Set a provider and key in backend/.env to run them.")
        return 0
    step("Fault injection: LLM-written faults on LLM-written bases, with the LLM reviewer",
         ["eval/run_faults.py", "--n", str(n_llm), "--base", "llm", "--faults", "both", "--reviewer", "--domains", domains])
    for d in domains.split(","):
        step(f"Phase I on {d} (raw LLM errors, pipeline outcomes, levels, reviewer agreement)",
             ["eval/run_phase1.py", "--n", str(n_phase1), "--reviewer", "--domain", d])
    step("Baselines vs this framework (all domains, all levels, LLM extractor)",
         ["eval/run_comparison.py", "--preset", "baselines", "--n", str(n_cmp), "--domains", domains,
          "--intents", "explain,recourse", "--judge"])
    step("Ablations (all domains)",
         ["eval/run_comparison.py", "--preset", "ablations", "--n", str(n_cmp), "--domains", domains, "--judge"])
    print("\nResults are in backend/results/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
