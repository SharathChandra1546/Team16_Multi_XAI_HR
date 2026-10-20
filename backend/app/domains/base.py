"""Domain-agnostic contracts: FeatureSpec, DomainPolicy, Action, DomainAgent.

Everything domain-specific (features, glossary, policy rules, recourse actions)
is data supplied by a DomainAgent subclass. The graph, verifier, critic and
recourse engine only use these interfaces, so a new domain needs no changes
to them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from ..schemas import GlossaryEntry

FeatureKind = Literal["numeric", "ordinal", "nominal", "binary"]
ActionStatus = Literal["actionable", "partial", "immutable", "consequence"]


@dataclass
class FeatureSpec:
    name: str
    display: str
    description: str
    kind: FeatureKind
    glossary_id: str
    group: str = "Other"
    unit: str = ""
    options: dict[int, str] = field(default_factory=dict)  # code -> raw dataset value
    labels: dict[int, str] = field(default_factory=dict)  # code -> readable label
    min: float | None = None
    max: float | None = None
    step: float = 1
    aliases: list[str] = field(default_factory=list)  # lower-case phrases that refer to this feature in text

    def __post_init__(self) -> None:
        if self.options and not self.labels:
            self.labels = dict(self.options)
        if self.options:
            codes = sorted(self.options)
            self.min = codes[0] if self.min is None else self.min
            self.max = codes[-1] if self.max is None else self.max

    @property
    def categorical(self) -> bool:
        return bool(self.options)

    def readable(self, value: Any) -> str:
        """Readable value for explanation text. Never returns a raw code."""
        if self.categorical:
            code = int(round(float(value)))
            return self.labels.get(code, self.options.get(code, str(value)))
        v = float(value)
        txt = f"{int(v):,}" if v.is_integer() else f"{v:,.1f}"
        if self.unit == "%":
            return f"{txt}%"
        unit = self.unit[:-1] if v == 1 and self.unit.endswith("s") else self.unit
        return f"{txt} {unit}".strip()

    def parse(self, raw: Any) -> float:
        """Accept a code, a raw dataset value or a readable label. Raises ValueError."""
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            raise ValueError("is required")
        if self.categorical:
            if isinstance(raw, str):
                s = raw.strip().lower()
                for code, val in self.options.items():
                    if s == str(val).lower() or s == self.labels.get(code, "").lower():
                        return float(code)
                try:
                    raw = float(s)
                except ValueError:
                    allowed = ", ".join(self.labels[c] for c in sorted(self.labels))
                    raise ValueError(f"must be one of: {allowed}") from None
            code = float(raw)
            if not code.is_integer() or int(code) not in self.options:
                allowed = ", ".join(self.labels[c] for c in sorted(self.labels))
                raise ValueError(f"must be one of: {allowed}")
            return code
        try:
            v = float(raw)
        except (TypeError, ValueError):
            raise ValueError("must be a number") from None
        if v != v:  # NaN
            raise ValueError("must be a number")
        if self.min is not None and v < self.min:
            raise ValueError(f"must be at least {self.min:g}")
        if self.max is not None and v > self.max:
            raise ValueError(f"must be at most {self.max:g}")
        if self.step == 1 and not v.is_integer():
            raise ValueError("must be a whole number")
        return v

    def schema(self) -> dict:
        return {
            "name": self.name,
            "display": self.display,
            "description": self.description,
            "kind": self.kind,
            "group": self.group,
            "unit": self.unit,
            "glossary_id": self.glossary_id,
            "options": [{"code": c, "label": self.labels[c]} for c in sorted(self.labels)] if self.categorical else [],
            "min": self.min,
            "max": self.max,
            "step": self.step,
        }


@dataclass
class Actionability:
    status: ActionStatus
    note: str = ""


@dataclass
class TextRule:
    """A deterministic wording rule. Each fault maps to exactly one code."""

    code: str
    patterns: list[str]
    message: str
    negatable: bool = True  # skip matches preceded by a negation in the same sentence
    min_level_allowed: str | None = None  # e.g. "expert": rule does not apply at/above this level

    def __post_init__(self) -> None:
        self._compiled = [re.compile(p, re.I) for p in self.patterns]

    @property
    def compiled(self) -> list[re.Pattern]:
        return self._compiled


@dataclass
class DomainPolicy:
    sensitive: list[str]  # protected attributes: excluded from the main model, used for audit
    actionability: dict[str, Actionability]
    disclosure: str  # appended verbatim to every explanation
    high_risk_notice: str
    escalation_uncertain: str
    escalation_ood: str
    refusal_message: str
    guard_patterns: list[str]  # query patterns that always request a decision (refuse)
    critic_rules: list[TextRule]  # domain wording rules enforced by the critic
    guard_soft: list[str] = field(default_factory=list)  # decision verbs refused only when framed as a decision
    guard_frames: list[str] = field(default_factory=list)  # decision framings (should, who, rank, deserve...)
    causal_extra: list[str] = field(default_factory=list)  # extra causal patterns for the verifier
    protected_text_rule: str = "protected_attribute_inference"
    min_risk_drop: float = 0.05  # recourse must lower probability by at least this much, or lower the band
    withheld_note: str = (
        "Some factors relate to protected personal characteristics and are shown only at expert level."
    )

    def status(self, feature: str) -> ActionStatus:
        a = self.actionability.get(feature)
        return a.status if a else "immutable"


@dataclass
class Action:
    """One recourse lever with its consequences already applied."""

    id: str
    label: str  # readable, e.g. "Remove regular overtime"
    changes: dict[str, float]  # feature -> new model-space value (includes consequences)
    primary: list[str]  # features the action directly changes
    consequences: list[str] = field(default_factory=list)  # features changed as a side effect
    cost: float = 1.0
    assumptions: list[str] = field(default_factory=list)
    short: str = ""  # plain short label for novice readers, e.g. "less overtime"


class DomainAgent:
    name: str = "base"
    title: str = ""
    description: str = ""
    routing_keywords: list[str] = []
    positive_label: str = "positive"
    negative_label: str = "negative"
    features: list[FeatureSpec] = []
    glossary: dict[str, GlossaryEntry] = {}
    policy: DomainPolicy

    # ---- runtime / model access (implemented by subclasses)
    def runtime(self, variant: str = "main"):
        raise NotImplementedError

    @property
    def ready(self) -> bool:
        try:
            self.runtime("main")
            return True
        except Exception:
            return False

    def recourse_actions(self, instance: dict[str, float], variant: str = "main") -> list[Action]:
        return []

    def consistent_defaults(self, filled: dict[str, float], assumed: set[str], variant: str = "main") -> dict[str, float]:
        """Adjust assumed (defaulted) values so they do not contradict the values the user gave."""
        return filled

    # ---- helpers shared by all domains
    def spec(self, name: str) -> FeatureSpec | None:
        return next((f for f in self.features if f.name == name), None)

    def model_features(self, variant: str = "main") -> list[str]:
        return list(self.runtime(variant).features)

    def readable(self, feature: str, value: Any) -> str:
        s = self.spec(feature)
        return s.readable(value) if s else str(value)

    def parse_instance(self, raw: dict[str, Any], variant: str = "main", allow_missing: bool = False) -> tuple[dict[str, float], dict[str, str]]:
        """Validate + convert to model space. Returns (instance, field_errors)."""
        out: dict[str, float] = {}
        errors: dict[str, str] = {}
        for name in self.model_features(variant):
            spec = self.spec(name)
            if name not in raw or raw[name] in (None, ""):
                if not allow_missing:
                    errors[name] = f"{spec.display if spec else name} is required"
                continue
            try:
                out[name] = spec.parse(raw[name]) if spec else float(raw[name])
            except ValueError as e:
                errors[name] = f"{spec.display if spec else name} {e}"
        return out, errors

    def concept_lookup(self, text: str) -> str | None:
        """Find the glossary id a concept question refers to (longest alias match)."""
        t = text.lower()
        best, best_len = None, 0
        for spec in self.features:
            for alias in [spec.display.lower(), spec.name.lower(), *spec.aliases]:
                if alias and re.search(rf"\b{re.escape(alias)}\b", t) and len(alias) > best_len:
                    best, best_len = spec.glossary_id, len(alias)
        for gid, entry in self.glossary.items():
            term = entry.term.lower()
            if re.search(rf"\b{re.escape(term)}\b", t) and len(term) > best_len:
                best, best_len = gid, len(term)
        return best

    def all_aliases(self) -> dict[str, list[str]]:
        return {f.name: sorted({f.display.lower(), *f.aliases}, key=len, reverse=True) for f in self.features}

    def schema(self, variant: str = "main") -> dict:
        try:
            feats = self.model_features(variant)
        except Exception:
            feats = [f.name for f in self.features]
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
            "features": [self.spec(n).schema() | {"actionability": self.policy.status(n)} for n in feats if self.spec(n)],
            "actionable": [n for n in feats if self.policy.status(n) in ("actionable", "partial")],
            "ready": self.ready,
        }
