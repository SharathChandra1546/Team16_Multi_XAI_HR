"""Deterministic mock LLM so the system runs and is testable without an API key.

- generate: returns the template explanation (marked as LLM output). If MOCK_FAULT_RATE > 0,
  the first attempt is corrupted with a random fault so the retry loop is exercised.
- critic / guard / route / review: fixed, neutral answers.
Results produced with the mock are NOT valid LLM results for the paper.
"""
from __future__ import annotations

import os
import random

from .client import BaseLLM

DOMAIN_RESOLVER = None  # set by the service: name -> DomainAgent


class MockLLM(BaseLLM):
    name = "mock"
    model = "deterministic-template"

    def json(self, task: str, system: str, user: str, payload: dict | None = None) -> dict:
        payload = payload or {}
        if task == "generate":
            return self._generate(payload)
        if task == "critic":
            return {"clarity": 4, "level_fit": 4, "relevance": 4, "feedback": "mock critic"}
        if task == "guard":
            return {"decision_request": False, "reason": "mock: rules layer only"}
        if task == "review":
            return {"faithful": True, "issues": []}
        return {}

    def _generate(self, payload: dict) -> dict:
        from ..agents.fallback import build_template
        from ..faults import FAULTS, inject
        from ..schemas import EvidencePackage, ExplanationPlan

        pkg = EvidencePackage(**payload["package"])
        plan = ExplanationPlan(**payload["plan"])
        domain = DOMAIN_RESOLVER(payload["domain"])
        attempt = int(payload.get("attempt", 1))
        gen = build_template(pkg, plan, domain, variant=(attempt + 1) % 2)
        rate = float(os.getenv("MOCK_FAULT_RATE", "0"))
        rng = random.Random(f"{pkg.package_id}-{attempt}")
        if attempt == 1 and rate > 0 and rng.random() < rate:
            faults = [f for f in os.getenv("MOCK_FAULTS", ",".join(FAULTS)).split(",") if f in FAULTS]
            for f in rng.sample(faults, len(faults)):
                bad = inject(f, gen, pkg, plan, domain, rng)
                if bad is not None:
                    gen = bad
                    break
        return {"explanation": gen.explanation, "claims": [c.model_dump() for c in gen.claims]}
