"""Evidence validator: checks the XAI output before any text is generated."""
from __future__ import annotations

import numpy as np

from ..schemas import EvidencePackage, ValidationCheck

ADDITIVITY_TOL = 0.01


def validate_package(pkg: EvidencePackage, fx: float, phi: np.ndarray, phi_rerun: np.ndarray) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    finite = bool(np.all(np.isfinite(phi)) and np.isfinite(fx) and np.isfinite(pkg.base_value))
    checks.append(ValidationCheck(name="finite_values", passed=finite, detail="All attributions are finite numbers."))
    p = pkg.prediction.probability
    checks.append(ValidationCheck(name="probability_range", passed=0.0 <= p <= 1.0, detail=f"p={p}"))
    add_err = abs(pkg.base_value + float(np.sum(phi)) - fx)
    checks.append(ValidationCheck(
        name="additivity", passed=add_err <= ADDITIVITY_TOL,
        detail=f"|base + sum(attributions) - f(x)| = {add_err:.5f} (tolerance {ADDITIVITY_TOL})"))
    pred_err = abs(fx - p)
    checks.append(ValidationCheck(
        name="prediction_consistency", passed=pred_err <= 0.005,
        detail=f"Explained value matches the served prediction (difference {pred_err:.5f})."))
    vis = pkg.visible
    ranks_ok = all(vis[i].rank == i + 1 for i in range(len(vis))) and all(
        abs(vis[i].attribution) >= abs(vis[i + 1].attribution) - 1e-12 for i in range(len(vis) - 1))
    checks.append(ValidationCheck(name="ranking", passed=ranks_ok, detail="Ranks follow absolute attribution."))
    # stability: same top-3 features (as a set, at least 2 of 3) across two independent Shapley runs
    top_a = set(np.argsort(-np.abs(phi))[:3])
    top_b = set(np.argsort(-np.abs(phi_rerun))[:3])
    overlap = len(top_a & top_b)
    sign_same = float(np.mean(np.sign(phi) == np.sign(phi_rerun)))
    checks.append(ValidationCheck(
        name="stability", passed=overlap >= 2, severity="warning",
        detail=f"Top-3 overlap {overlap}/3 across two runs; sign agreement {sign_same:.0%}."))
    top5 = vis[:5]
    agree = [it.methods_agree for it in top5 if it.methods_agree is not None]
    rate = sum(agree) / len(agree) if agree else 1.0
    checks.append(ValidationCheck(
        name="method_agreement", passed=rate >= 0.6, severity="warning",
        detail=f"Shapley and LIME agree on direction for {rate:.0%} of the top-5 factors."))
    return checks
