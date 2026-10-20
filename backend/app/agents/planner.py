"""Adaptive explanation planner: level + intent -> ExplanationPlan (word budget, terminology, evidence scope)."""
from __future__ import annotations

from ..schemas import EvidencePackage, ExplanationPlan

LEVELS = {
    "novice": {
        "top_k": 3, "max_words": 90, "include_numbers": False, "allow_column_names": False,
        "banned_terms": ["shap", "shapley", "lime", "attribution", "log-odds", "calibrated", "calibration",
                         "feature", "features", "probability", "percentile", "coefficient", "threshold"],
        "style": "Plain everyday words, short sentences, no statistics terms. Say 'chance' not 'probability'.",
    },
    "intermediate": {
        "top_k": 5, "max_words": 160, "include_numbers": True, "allow_column_names": False,
        "banned_terms": ["shap", "shapley", "lime", "log-odds"],
        "style": "Clear professional language for an HR manager. Percentages are fine; avoid method names.",
    },
    "expert": {
        "top_k": 8, "max_words": 260, "include_numbers": True, "allow_column_names": True,
        "banned_terms": [],
        "style": "Precise and quantitative. Report contributions in percentage points and name the method if useful.",
    },
}


def make_plan(pkg: EvidencePackage, level: str, intent: str, extra: list[str] | None = None) -> ExplanationPlan:
    cfg = LEVELS[level]
    visible = pkg.visible
    if intent == "recourse":
        top = visible[: min(3, cfg["top_k"])]
        options = [r.id for r in pkg.recourse][: 2 if level == "novice" else 3]
        allowed = ["P", *[i.id for i in top], *options]
        required = ["P", *options[:1]]
    else:
        top = visible[: cfg["top_k"]]
        allowed = ["P", *[i.id for i in top]]
        required = ["P", *([top[0].id] if top else [])]
    allowed += [g.id for g in pkg.glossary]
    return ExplanationPlan(
        level=level,
        intent=intent,
        allowed_ids=allowed,
        required_ids=required,
        max_words=cfg["max_words"],
        include_numbers=cfg["include_numbers"],
        allow_column_names=cfg["allow_column_names"],
        banned_terms=cfg["banned_terms"],
        style=cfg["style"],
        extra_instructions=list(extra or []),
    )
