"""The Claim Fidelity Score (CFS): the share of an explanation's checkable units that are faithful to the evidence.

Definition (v2)
---------------
An explanation E has claims C_1 .. C_n (structured statements citing evidence ids) and free text. The verifier
reports violations against the Canonical Evidence Package P. Each violation is attached to a unit:

- a CLAIM, if the violation names that claim (claim_index) or its span (the offending words/number) occurs in that
  claim's text;
- an UNBOUND SENTENCE, if it is a text-level violation (unsupported number, causal wording, raw code, a factor
  named without a claim) whose span lies in a sentence that is not a claim. Each such sentence is one extra unit;
- a MISSING CLAIM, for each required evidence id the explanation never cites (missing_required). Each is one
  extra unit: a required statement that should have been made and was not.

    units  = n + u_sentences + u_missing
    CFS    = |clean claims| / units                                   (0.0 when units == 0)
    CFS_w  = (|claims with no violation| + 0.5 * |claims with only minor violations|) / units

A claim is clean when no CRITICAL violation is attached to it (CFS counts it fully); CFS_w gives full credit only
to claims with no violation at all and half credit to claims whose only violations are minor. (v1's code added
the half credit on top of full credit, so CFS_w could exceed 1; fixed in v2.) "Minor" means a non-blocking verifier finding;
currently only not_allowed_evidence (citing an id outside this plan's scope) is minor. Critic findings (policy
wording, level fit, readability) are deliberately NOT part of CFS: CFS measures fidelity to the evidence, and
policy/level are reported separately (app/policy_report.py).

Why v2: v1 counted only violations bound to a claim, so an explanation with correct claims plus an extra sentence
containing an invented number or a causal statement scored 1.0 although the verifier rejected it. v2 makes
"CFS == 1.0" imply "no critical fidelity violation anywhere in the text".

Properties
----------
- Bounded: CFS, CFS_w in [0, 1].
- CFS == 1.0  iff  the verifier found no critical violation and the explanation has at least one claim.
- Deterministic / reproducible: every check in verifier.py is a closed-form comparison against P (string/number
  equality within a fixed tolerance, or a fixed regex), not a model call.
- Domain-agnostic: computed from the EvidencePackage/Claim contracts in schemas.py.
- Independent of any judge model: CFS never calls an LLM.
- An explanation with no claims scores 0.0 (and is rejected upstream as empty_explanation or missing_required).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..agents.textutil import norm, sentences
from ..schemas import Violation

NAME = "Claim Fidelity Score (CFS)"


@dataclass(frozen=True)
class FidelityReport:
    n_claims: int
    n_clean: int  # claims with zero critical violations (includes minor-only claims)
    n_minor_only: int  # claims with zero critical but at least one minor violation
    score: float  # CFS
    weighted_score: float  # CFS_w
    n_unbound_sentences: int = 0  # non-claim sentences with a critical text-level violation
    n_missing_required: int = 0  # required evidence ids never cited
    version: str = "v2"

    @property
    def n_units(self) -> int:
        return self.n_claims + self.n_unbound_sentences + self.n_missing_required

    def as_dict(self) -> dict:
        return {"name": NAME, "version": self.version, "n_claims": self.n_claims, "n_clean": self.n_clean,
                "n_minor_only": self.n_minor_only, "n_unbound_sentences": self.n_unbound_sentences,
                "n_missing_required": self.n_missing_required, "n_units": self.n_units,
                "score": self.score, "weighted_score": self.weighted_score}


def _claim_for_span(span: str, claim_texts: list[str]) -> int | None:
    s = norm(span)
    if not s:
        return None
    for i, ct in enumerate(claim_texts):
        if ct and s in ct:
            return i
    return None


def _sentence_for_span(span: str, sents: list[str]) -> int | None:
    s = norm(span)
    for i, st in enumerate(sents):
        ns = norm(st)
        if s and (s in ns or ns.startswith(s)):
            return i
    return None


def claim_fidelity_score(n_claims: int, violations: list[Violation], claim_texts: list[str] | None = None,
                         text: str = "") -> FidelityReport:
    """Compute CFS v2 from the verifier's violations.

    `claim_texts` and `text` let text-level violations (no claim_index) be attached to the claim whose text
    contains their span, or else to the non-claim sentence that contains it. Without them, every text-level
    critical violation counts as one unbound unit (conservative).
    """
    cts = [norm(t) for t in (claim_texts or [])]
    sents = sentences(text) if text else []
    critical_claims: set[int] = set()
    minor_claims: set[int] = set()
    unbound: set = set()
    missing: set[str] = set()
    for v in violations:
        if v.code == "empty_explanation":
            continue
        idx = v.claim_index
        if idx is None and v.code == "missing_required":
            if v.severity == "critical":
                missing.add(v.span or v.message)
            continue
        if idx is None and v.span:
            idx = _claim_for_span(v.span, cts)
        if idx is not None and 0 <= idx < n_claims:
            (critical_claims if v.severity == "critical" else minor_claims).add(idx)
            continue
        if v.severity == "critical":
            si = _sentence_for_span(v.span or "", sents) if sents else None
            unbound.add(("s", si) if si is not None else ("v", v.code, v.span))
    units = n_claims + len(unbound) + len(missing)
    if n_claims == 0 or units == 0:
        return FidelityReport(n_claims, 0, 0, 0.0, 0.0, len(unbound), len(missing))
    n_clean = sum(1 for i in range(n_claims) if i not in critical_claims)
    n_minor_only = sum(1 for i in range(n_claims) if i not in critical_claims and i in minor_claims)
    score = round(n_clean / units, 4)
    weighted = round((n_clean - n_minor_only + 0.5 * n_minor_only) / units, 4)
    return FidelityReport(n_claims, n_clean, n_minor_only, score, weighted, len(unbound), len(missing))


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
