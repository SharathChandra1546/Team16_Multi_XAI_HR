"""The Claim Fidelity Score (CFS): a formal definition of the metric the verifier computes.

Definition
----------
Given a generated explanation E = (claims C_1 .. C_n) checked against a
Canonical Evidence Package P:

    CFS(E, P) = |{ C_i : C_i has no CRITICAL violation against P }| / n          (n = |C|)

CFS(E, P) = 1.0 when there are no claims (an empty explanation is handled as a
separate, always-failing case upstream, not as a vacuous 1.0 — see
VerificationResult in verifier.py).

A weighted variant additionally discounts, rather than ignores, MINOR
violations (readability, non-blocking terminology notes) on claims that are
otherwise critical-violation-free, so two explanations that both pass can
still be told apart by polish, not just by pass/fail:

    CFS_w(E, P) = ( |clean claims| + 0.5 * |claims with only minor violations| ) / n

Properties
----------
- Bounded: CFS, CFS_w in [0, 1] by construction (a count of claims divided by
  the total count, optionally with a non-negative sub-unit weight).
- Deterministic / reproducible: every check in verifier.py is a closed-form
  comparison against the Evidence Package (string/number equality within a
  fixed tolerance, or a fixed regex), not a model call. Re-running verify()
  on the same (package, plan, explanation) always returns the same score.
- Domain-agnostic: the score is computed from the EvidencePackage/Claim
  contracts in schemas.py, which carry no domain-specific fields. The same
  function runs unchanged for the HR domain and any other DomainAgent
  (exercised by tests/test_graph_domain_agnostic.py).
- Distinct from, and verifiable independently of, a judge-model's opinion:
  CFS never calls an LLM. The critic's optional LLM-judge quality score
  (clarity/level_fit/relevance) is a separate number precisely so that
  fidelity (did it say true things) and quality (is it well written) are
  never conflated into one figure.

This module does not change what verifier.verify() computes; it names the
same computation formally, in one place, so it can be reported and reasoned
about as a metric in its own right rather than read back out of an
implementation detail.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..schemas import Violation

NAME = "Claim Fidelity Score (CFS)"


@dataclass(frozen=True)
class FidelityReport:
    n_claims: int
    n_clean: int  # claims with zero critical violations
    n_minor_only: int  # claims with zero critical but at least one minor violation
    score: float  # CFS: n_clean / n_claims, in [0, 1]
    weighted_score: float  # CFS_w: (n_clean + 0.5 * n_minor_only) / n_claims, in [0, 1]

    def as_dict(self) -> dict:
        return {"name": NAME, "n_claims": self.n_claims, "n_clean": self.n_clean, "n_minor_only": self.n_minor_only,
                "score": self.score, "weighted_score": self.weighted_score}


def claim_fidelity_score(n_claims: int, violations: list[Violation]) -> FidelityReport:
    """Compute CFS and CFS_w from the per-claim violations the verifier already found.

    `violations` is the full violation list from VerificationResult; only
    those with a claim_index (i.e. bound to one specific claim) count toward
    the per-claim tallies, matching how verify() tracks bad_claims.
    """
    critical_claims: set[int] = set()
    minor_claims: set[int] = set()
    for v in violations:
        if v.claim_index is None:
            continue
        if v.severity == "critical":
            critical_claims.add(v.claim_index)
        else:
            minor_claims.add(v.claim_index)
    if n_claims == 0:
        return FidelityReport(0, 0, 0, 0.0, 0.0)
    n_clean = sum(1 for i in range(n_claims) if i not in critical_claims)
    n_minor_only = sum(1 for i in range(n_claims) if i not in critical_claims and i in minor_claims)
    score = round(n_clean / n_claims, 4)
    weighted = round((n_clean + 0.5 * n_minor_only) / n_claims, 4)
    return FidelityReport(n_claims, n_clean, n_minor_only, score, weighted)


def aggregate(reports: list[FidelityReport], n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean CFS and a 95% bootstrap confidence interval across many explanations (for eval/export_tables.py)."""
    import numpy as np

    scores = np.array([r.score for r in reports], dtype=float)
    if scores.size == 0:
        return {"name": NAME, "n": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    boots = [float(rng.choice(scores, size=scores.size, replace=True).mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"name": NAME, "n": int(scores.size), "mean": round(float(scores.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
