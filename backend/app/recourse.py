"""Actionable-only recourse with constraints. Domain-agnostic.

Every candidate is (1) checked against the DomainPolicy (immutable features never
change, consequence features change only through the action that declares them,
values stay inside the feature's valid range), then (2) re-predicted with the
model. Only candidates that lower the estimate by at least `min_risk_drop`, or
move it to a lower band, are returned.
"""
from __future__ import annotations

from itertools import combinations

from .domains.base import Action, DomainAgent
from .schemas import RecourseChange, RecourseOption


def validate_candidate(domain: DomainAgent, x: dict, new: dict, actions: list[Action]) -> list[str]:
    errors = []
    declared_primary = {f for a in actions for f in a.primary}
    declared_conseq = {f for a in actions for f in a.consequences}
    for f, v in new.items():
        if v == x[f]:
            continue
        status = domain.policy.status(f)
        if status == "immutable":
            errors.append(f"{f} is immutable")
        elif status == "consequence" and f not in declared_conseq:
            errors.append(f"{f} changed without a declaring action")
        elif status in ("actionable", "partial") and f not in declared_primary and f not in declared_conseq:
            errors.append(f"{f} changed without an action")
        spec = domain.spec(f)
        if spec and spec.min is not None and v < spec.min or spec and spec.max is not None and v > spec.max:
            errors.append(f"{f} out of range")
    return errors


def _apply(x: dict, actions: list[Action]) -> dict | None:
    new = dict(x)
    touched: set[str] = set()
    for a in actions:
        for f, v in a.changes.items():
            if f in touched:
                return None  # conflicting actions (e.g. promotion and a raise both set income)
            touched.add(f)
            new[f] = v
    return new


def find_recourse(domain: DomainAgent, x: dict, variant: str = "main", max_options: int = 3, max_actions: int = 2) -> tuple[list[RecourseOption], str]:
    rt = domain.runtime(variant)
    p0 = float(rt.proba(rt.frame(x))[0])
    b0 = rt.band(p0)
    actions = domain.recourse_actions(x, variant)
    combos: list[list[Action]] = [[a] for a in actions]
    if max_actions >= 2:
        for a, b in combinations(actions, 2):
            if a.id.split("_")[0] != b.id.split("_")[0]:
                combos.append([a, b])
    cands, news = [], []
    for combo in combos:
        new = _apply(x, combo)
        if new is None or validate_candidate(domain, x, new, combo):
            continue
        cands.append(combo)
        news.append(new)
    if not cands:
        return [], "No actionable change is available for this profile."
    probs = rt.proba(rt.frame(news))
    valid = []
    for combo, new, p1 in zip(cands, news, probs):
        p1 = float(p1)
        b1 = rt.band(p1)
        if p1 <= p0 - domain.policy.min_risk_drop or rt.band_rank(b1) < rt.band_rank(b0):
            valid.append((len(combo), sum(a.cost for a in combo), p1, combo, new, b1))
    valid.sort(key=lambda r: (r[0], r[1], r[2]))
    chosen, used_families = [], set()
    for n, c, p1, combo, new, b1 in valid:
        fam = tuple(sorted(a.id.split("_")[0] for a in combo))
        if fam in used_families:
            continue
        used_families.add(fam)
        chosen.append((combo, new, p1, b1, c))
        if len(chosen) == max_options:
            break
    options = []
    for i, (combo, new, p1, b1, c) in enumerate(chosen, start=1):
        changes = []
        conseq = {f for a in combo for f in a.consequences}
        for f in domain.model_features(variant):
            if new[f] != x[f]:
                spec = domain.spec(f)
                changes.append(RecourseChange(
                    feature=f, display_name=spec.display if spec else f,
                    from_label=domain.readable(f, x[f]), to_label=domain.readable(f, new[f]), consequence=f in conseq,
                ))
        options.append(RecourseOption(
            id=f"R{i}", actions=[a.id for a in combo], description=" combined with ".join(a.label for a in combo),
            short_description=" and ".join(a.short or a.label for a in combo),
            changes=changes, probability_before=round(p0, 4), probability_after=round(p1, 4), band_before=b0,
            band_after=b1, assumptions=[s for a in combo for s in a.assumptions], cost=c,
        ))
    note = "" if options else (
        f"No change to actionable factors lowers this estimate by at least {round(domain.policy.min_risk_drop * 100)} "
        "percentage points or moves it to a lower band."
    )
    return options, note


def recheck_option(domain: DomainAgent, x: dict, option: RecourseOption, variant: str = "main") -> bool:
    """Independent re-check used by tests and the verifier: re-apply and re-predict."""
    rt = domain.runtime(variant)
    acts = {a.id: a for a in domain.recourse_actions(x, variant)}
    combo = [acts[a] for a in option.actions if a in acts]
    if len(combo) != len(option.actions):
        return False
    new = _apply(x, combo)
    if new is None or validate_candidate(domain, x, new, combo):
        return False
    p0 = float(rt.proba(rt.frame(x))[0])
    p1 = float(rt.proba(rt.frame(new))[0])
    return (p1 <= p0 - domain.policy.min_risk_drop or rt.band_rank(rt.band(p1)) < rt.band_rank(rt.band(p0))) and abs(p1 - option.probability_after) < 1e-3
