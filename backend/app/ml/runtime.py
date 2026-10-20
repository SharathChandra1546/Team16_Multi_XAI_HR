"""Loads a trained artefact and serves predictions, uncertainty intervals, bands and OOD checks."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import settings
from .training import band_of

BANDS = ["low", "medium", "high"]


class ModelRuntime:
    def __init__(self, artifact: dict, manifest: dict | None = None, variant: str = "main", version: str = ""):
        self.a = artifact
        self.features: list[str] = artifact["features"]
        self.nominal: list[str] = artifact["nominal"]
        self.model = artifact["model"]
        self.ensemble = artifact["ensemble"]
        self.threshold: float = artifact["threshold"]
        self.low_max: float = artifact["band_edges"]["low_max"]
        self.high_min: float = artifact["band_edges"]["high_min"]
        self.ood = artifact["ood"]
        self.background: pd.DataFrame = artifact["background"]
        self.stats: dict = artifact["stats"]
        self.global_importance: list[dict] = artifact["global_importance"]
        self.examples: pd.DataFrame = artifact["examples"]
        self.domain_stats: dict = artifact.get("domain_stats", {})
        self.manifest = manifest or {}
        self.variant = variant
        self.version = version or artifact.get("version", "unversioned")

    @classmethod
    def load(cls, path: Path | str, manifest: dict | None = None, variant: str = "main") -> "ModelRuntime":
        art = joblib.load(path)
        return cls(art, manifest, variant, art.get("version", Path(path).stem))

    # ---------------------------------------------------------------- prediction
    def frame(self, rows: list[dict] | dict | np.ndarray) -> pd.DataFrame:
        if isinstance(rows, dict):
            rows = [rows]
        if isinstance(rows, np.ndarray):
            return pd.DataFrame(rows, columns=self.features).astype(float)
        return pd.DataFrame([[r[f] for f in self.features] for r in rows], columns=self.features).astype(float)

    def proba(self, X) -> np.ndarray:
        X = X if isinstance(X, pd.DataFrame) else self.frame(X)
        return self.model.predict_proba(X[self.features])[:, 1]

    def proba_array(self, a: np.ndarray) -> np.ndarray:
        return self.proba(pd.DataFrame(a, columns=self.features))

    def interval(self, X) -> tuple[np.ndarray, np.ndarray]:
        X = X if isinstance(X, pd.DataFrame) else self.frame(X)
        ens = np.array([m.predict_proba(X[self.features])[:, 1] for m in self.ensemble])
        return (np.percentile(ens, settings.interval_low_pct, axis=0), np.percentile(ens, settings.interval_high_pct, axis=0))

    def band(self, p: float) -> str:
        return band_of(p, self.low_max, self.high_min)

    def band_rank(self, band: str) -> int:
        return BANDS.index(band)

    def check_ood(self, x: dict) -> list[str]:
        reasons = []
        for f in self.features:
            lo, hi = self.ood["ranges"][f]
            if x[f] < lo or x[f] > hi:
                reasons.append(f"{f} outside the training range")
        score = float(self.ood["iso"].score_samples(self.frame(x).values)[0])
        if score < self.ood["score_threshold"]:
            reasons.append("unusual combination of values")
        return reasons

    def predict(self, x: dict) -> dict:
        X = self.frame(x)
        p = float(self.proba(X)[0])
        lo, hi = self.interval(X)
        lo, hi = float(min(lo[0], p)), float(max(hi[0], p))
        band = self.band(p)
        ood = self.check_ood(x)
        return {
            "probability": p,
            "interval": (lo, hi),
            "band": band,
            "band_uncertain": self.band(lo) != self.band(hi),
            "band_low": self.band(lo),
            "band_high": self.band(hi),
            "threshold": self.threshold,
            "ood_reasons": ood,
        }

    def defaults(self) -> dict[str, float]:
        out = {}
        for f in self.features:
            out[f] = self.stats["medians"].get(f, self.stats["modes"][f]) if f not in self.nominal else self.stats["modes"][f]
            out[f] = float(round(out[f]))
        return out
