"""Builds the Canonical Evidence Package for one case (domain-agnostic)."""
from __future__ import annotations

import uuid

import numpy as np

from ..config import settings
from ..domains.base import DomainAgent
from ..recourse import find_recourse
from ..schemas import EvidenceItem, EvidencePackage, PredictionResult
from .lime_local import lime_weights
from .shapley import shapley
from .validator import validate_package

N_PERM = 32  # default; SHAP_PERMUTATIONS overrides it (more permutations = more stable attributions)
N_BACKGROUND = 25


def magnitude_of(share: float) -> str:
    if share >= 0.20:
        return "strong"
    if share >= 0.08:
        return "moderate"
    return "weak"


def build_package(
    domain: DomainAgent,
    instance: dict[str, float],
    level: str,
    intent: str,
    variant: str = "main",
    assumed_fields: list[str] | None = None,
    seed: int = 0,
) -> EvidencePackage:
    rt = domain.runtime(variant)
    feats = rt.features
    x = np.array([instance[f] for f in feats], dtype=float)
    bg = rt.background.values[:N_BACKGROUND].astype(float)
    pred = rt.predict(instance)

    n_perm = settings.shap_permutations or N_PERM
    phi, base, fx, method = shapley(rt.proba_array, x, bg, n_perm=n_perm, seed=seed)
    phi2, _, _, _ = shapley(rt.proba_array, x, bg, n_perm=n_perm, seed=seed + 101)  # stability re-run
    cat_idx = [i for i, f in enumerate(feats) if domain.spec(f) and domain.spec(f).categorical]
    lime_w, lime_method = lime_weights(rt.proba_array, x, bg, cat_idx, feats, seed=seed)

    total = float(np.abs(phi).sum()) or 1.0
    order = np.argsort(-np.abs(phi))
    sensitive = set(domain.policy.sensitive)
    assumed = set(assumed_fields or [])
    items: list[EvidenceItem] = []
    for j in order:
        f = feats[j]
        spec = domain.spec(f)
        protected = f in sensitive
        items.append(EvidenceItem(
            id="",
            feature=f,
            display_name=spec.display if spec else f,
            glossary_id=spec.glossary_id if spec else "",
            value=float(x[j]),
            value_label=domain.readable(f, x[j]),
            attribution=float(phi[j]),
            direction="increases" if phi[j] >= 0 else "decreases",
            rank=0,
            share=float(abs(phi[j]) / total),
            magnitude=magnitude_of(abs(phi[j]) / total),
            lime_attribution=float(lime_w[j]),
            methods_agree=bool(np.sign(lime_w[j]) == np.sign(phi[j]) or abs(phi[j]) < 0.005),
            protected=protected,
            withheld=protected and level != "expert",
            assumed=f in assumed,
        ))
    rank = 0
    for i, it in enumerate(items, start=1):
        it.id = f"E{i}"
        if not it.withheld:
            rank += 1
            it.rank = rank

    prediction = PredictionResult(
        label=domain.positive_label if pred["probability"] >= rt.threshold else domain.negative_label,
        positive_label=domain.positive_label,
        probability=round(pred["probability"], 4),
        interval=(round(pred["interval"][0], 4), round(pred["interval"][1], 4)),
        band=pred["band"],
        band_uncertain=pred["band_uncertain"],
        threshold=rt.threshold,
        out_of_distribution=bool(pred["ood_reasons"]),
        ood_reasons=pred["ood_reasons"],
        model_version=rt.version,
        model_variant=variant,
    )
    pkg = EvidencePackage(
        package_id=uuid.uuid4().hex[:12],
        domain=domain.name,
        method=f"{method}; agreement check: {lime_method}",
        level=level,
        prediction=prediction,
        base_value=round(base, 6),
        items=items,
        limitations=[
            "Contributions describe how the model's estimate moved for this profile. They are associations in the "
            "training data, not causes.",
        ],
        assumptions=[f"{domain.spec(f).display if domain.spec(f) else f}: {domain.readable(f, instance[f])}"
                     for f in feats if f in assumed],
    )
    if domain.data_note():
        pkg.limitations.append(domain.data_note())
    if any(it.withheld for it in items):
        pkg.limitations.append(domain.policy.withheld_note)
    disagree = [it.display_name for it in pkg.visible[:5] if it.methods_agree is False]
    if disagree:
        pkg.limitations.append("Shapley values and LIME disagree on the direction of: " + ", ".join(disagree) + ".")
    if intent == "recourse":
        pkg.recourse, pkg.recourse_note = find_recourse(domain, instance, variant)
    # glossary entries for everything an explanation may cite
    gids = {it.glossary_id for it in pkg.visible if it.glossary_id} | {"G-RISK", "G-BAND", "G-CONTRIBUTION"}
    if intent == "recourse":
        gids.add("G-RECOURSE")
    pkg.glossary = [domain.glossary[g] for g in sorted(gids) if g in domain.glossary]
    pkg.checks = validate_package(pkg, fx, phi, phi2)
    pkg.valid = all(c.passed for c in pkg.checks if c.severity == "critical")
    return pkg
