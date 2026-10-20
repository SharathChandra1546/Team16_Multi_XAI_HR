"""LIME attributions, used as a second method to check agreement with Shapley values.

Uses the `lime` package when installed; otherwise a built-in local surrogate:
features are switched between the instance value and a background value, samples
are weighted by an exponential kernel on the fraction of switched features, and a
weighted ridge regression gives one weight per feature. A positive weight means
the instance's value raises the estimate, matching the Shapley sign convention.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

try:  # optional dependency
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore

    HAS_LIME = True
except Exception:  # pragma: no cover
    HAS_LIME = False


def builtin_lime(
    f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, background: np.ndarray, n_samples: int = 1000,
    kernel_width: float = 0.75, seed: int = 0,
) -> np.ndarray:
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(seed)
    m = x.shape[0]
    keep = rng.random((n_samples, m)) < 0.5
    keep[0] = True  # include the instance itself
    bg_idx = rng.integers(0, background.shape[0], size=n_samples)
    samples = np.where(keep, x[None, :], background[bg_idx])
    preds = np.asarray(f(samples), dtype=float).reshape(-1)
    dist = np.sqrt((~keep).sum(axis=1) / m)
    weights = np.exp(-(dist**2) / kernel_width**2)
    model = Ridge(alpha=1.0)
    model.fit(keep.astype(float), preds, sample_weight=weights)
    return np.asarray(model.coef_, dtype=float)


def lime_weights(
    f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, background: np.ndarray, categorical_idx: list[int],
    feature_names: list[str], seed: int = 0, prefer_library: bool = True,
) -> tuple[np.ndarray, str]:
    if prefer_library and HAS_LIME:
        try:
            return _lime_library(f, x, background, categorical_idx, feature_names, seed), "lime.LimeTabularExplainer"
        except Exception:  # e.g. degenerate bins on a small background: use the built-in surrogate
            pass
    return builtin_lime(f, x, background, seed=seed), "local-surrogate (built-in LIME)"


def _lime_library(f, x, background, categorical_idx, feature_names, seed):
    explainer = LimeTabularExplainer(
        training_data=background, feature_names=feature_names, categorical_features=categorical_idx,
        mode="classification", discretize_continuous=True, random_state=seed,
    )

    def f2(a: np.ndarray) -> np.ndarray:
        p = np.asarray(f(a), dtype=float).reshape(-1)
        return np.column_stack([1 - p, p])

    exp = explainer.explain_instance(x, f2, num_features=len(feature_names), labels=(1,), num_samples=1000)
    w = np.zeros(len(feature_names))
    for idx, weight in exp.as_map()[1]:
        w[idx] = weight
    return w
