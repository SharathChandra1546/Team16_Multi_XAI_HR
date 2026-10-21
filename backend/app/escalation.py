"""Escalation score: one number per case saying how unreliable an automated explanation would be.

Two components, each in [0, 1]:

- uncertainty: the share of the bootstrap-ensemble interval [lo, hi] that lies OUTSIDE the band of the point
  estimate. 0 when the whole interval sits inside one band; it grows as the interval reaches into a neighbouring
  band. (The interval is the configured percentile range of the ensemble, INTERVAL_LOW_PCT..INTERVAL_HIGH_PCT.)
- ood: how unusual the case is for the training data. 1 if any feature is outside its training range; otherwise
  the isolation-forest score mapped linearly so that the training median scores 0 and the detector's threshold
  (1st percentile of training scores) scores 1, clipped to [0, 1].

score = max(uncertainty, ood).

Two decision modes:

- rule (default, the original behaviour): escalate if uncertainty > 0 (the interval spans two bands) or the case is
  out of distribution. This equals score mode with an infinitesimal uncertainty threshold and ood == 1.
- score: escalate if score >= threshold.

The score is logged for EVERY case, escalated or not, so a coverage-risk curve can be computed afterwards by
sweeping the threshold over logged scores (share of cases explained vs. explanation error among them).
"""
from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-9


def _band_bounds(band: str, low_max: float, high_min: float) -> tuple[float, float]:
    return {"low": (0.0, low_max), "medium": (low_max, high_min), "high": (high_min, 1.0)}[band]


def uncertainty_component(p: float, lo: float, hi: float, band: str, low_max: float, high_min: float) -> float:
    width = hi - lo
    if width <= EPS:
        return 0.0
    b_lo, b_hi = _band_bounds(band, low_max, high_min)
    inside = max(0.0, min(hi, b_hi) - max(lo, b_lo))
    return float(np.clip(1.0 - inside / width, 0.0, 1.0))


def ood_component(runtime, x: dict) -> tuple[float, list[str]]:
    reasons = []
    for f in runtime.features:
        lo, hi = runtime.ood["ranges"][f]
        if x[f] < lo or x[f] > hi:
            reasons.append(f"{f} outside the training range")
    s = float(runtime.ood["iso"].score_samples(runtime.frame(x).values)[0])
    thr = float(runtime.ood["score_threshold"])
    med = _median_score(runtime)
    if reasons:
        return 1.0, reasons
    if s < thr:
        reasons.append("unusual combination of values")
    denom = med - thr
    val = 1.0 if denom <= EPS else float(np.clip((med - s) / denom, 0.0, 1.0))
    return val, reasons


def _median_score(runtime) -> float:
    """Median isolation score of the training data. Stored in new artefacts; estimated from the background
    rows (a training-split sample) for artefacts trained before it was stored."""
    if "score_median" in runtime.ood:
        return float(runtime.ood["score_median"])
    cache = getattr(runtime, "_iso_median", None)
    if cache is None:
        cache = float(np.median(runtime.ood["iso"].score_samples(runtime.background.values.astype(float))))
        runtime._iso_median = cache
    return cache


def escalation_info(runtime, x: dict, pred: dict, mode: str = "rule", threshold: float = 0.5,
                    enabled: bool = True) -> dict[str, Any]:
    p, (lo, hi), band = pred["probability"], pred["interval"], pred["band"]
    unc = uncertainty_component(p, lo, hi, band, runtime.low_max, runtime.high_min)
    ood, ood_reasons = ood_component(runtime, x)
    score = max(unc, ood)
    rule = bool(pred["band_uncertain"] or pred["ood_reasons"])
    decision = rule if mode == "rule" else score >= threshold
    reasons = []
    if pred["band_uncertain"]:
        reasons.append("interval spans two risk bands")
    reasons += pred["ood_reasons"]
    return {
        "score": round(score, 4),
        "uncertainty": round(unc, 4),
        "ood": round(ood, 4),
        "mode": mode,
        "threshold": None if mode == "rule" else threshold,
        "rule_decision": rule,
        "would_escalate": bool(decision),
        "escalate": bool(decision and enabled),
        "enabled": enabled,
        "reasons": reasons or ood_reasons,
    }
