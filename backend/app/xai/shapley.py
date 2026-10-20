"""Shapley attributions for a probability function.

Uses shap.PermutationExplainer when the `shap` package is installed. Otherwise a
built-in permutation-sampling estimator with antithetic permutations is used.
Both are interventional (background-based) and exactly additive:
    base_value + sum(attributions) == f(x)   (base_value = mean f over the background rows used)
"""
from __future__ import annotations

from typing import Callable

import numpy as np

try:  # optional dependency
    import shap  # type: ignore

    HAS_SHAP = True
except Exception:  # pragma: no cover - depends on environment
    shap = None
    HAS_SHAP = False


def builtin_permutation_shapley(
    f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, background: np.ndarray, n_perm: int = 32, seed: int = 0
) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    m = x.shape[0]
    half = max(1, n_perm // 2)
    perms = [rng.permutation(m) for _ in range(half)]
    perms += [p[::-1] for p in perms]  # antithetic pairs reduce variance
    nb = background.shape[0]
    rows = np.empty((len(perms), nb, m + 1, m), dtype=float)
    for pi, perm in enumerate(perms):
        cur = background.astype(float).copy()
        rows[pi, :, 0, :] = cur
        for k, j in enumerate(perm):
            cur[:, j] = x[j]
            rows[pi, :, k + 1, :] = cur
    preds = np.asarray(f(rows.reshape(-1, m)), dtype=float).reshape(len(perms), nb, m + 1)
    diffs = preds[:, :, 1:] - preds[:, :, :-1]
    phi = np.zeros(m)
    for pi, perm in enumerate(perms):
        phi[perm] += diffs[pi].mean(axis=0)
    phi /= len(perms)
    base = float(preds[:, :, 0].mean())
    fx = float(preds[:, :, -1].mean())
    return phi, base, fx


def shapley(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    n_perm: int = 32,
    seed: int = 0,
    prefer_library: bool = True,
) -> tuple[np.ndarray, float, float, str]:
    """Returns (attributions, base_value, f(x), method_name)."""
    if prefer_library and HAS_SHAP:
        try:
            masker = shap.maskers.Independent(background, max_samples=background.shape[0])
            explainer = shap.PermutationExplainer(f, masker, seed=seed)
            m = x.shape[0]
            exp = explainer(x.reshape(1, -1), max_evals=max(2 * m + 1, n_perm * (m + 1)), silent=True)
            phi = np.asarray(exp.values, dtype=float).reshape(-1)[:m]
            base = float(np.ravel(exp.base_values)[0])
            fx = float(np.ravel(f(x.reshape(1, -1)))[0])
            return phi, base, fx, "shap.PermutationExplainer"
        except Exception:  # library/version problem: fall back to the built-in estimator (recorded in the method name)
            pass
    phi, base, fx = builtin_permutation_shapley(f, x, background, n_perm=n_perm, seed=seed)
    return phi, base, fx, "permutation-shapley (built-in)"
