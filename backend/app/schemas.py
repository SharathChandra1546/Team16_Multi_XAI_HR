"""Typed data contracts shared by all agents.

The Canonical Evidence Package is the single source of truth: every claim in an
explanation must cite an id from it (P = prediction, E# = attribution item,
R# = recourse option, G-* = glossary entry).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Level = Literal["novice", "intermediate", "expert"]
Intent = Literal["explain", "recourse", "concept", "out_of_scope"]
Direction = Literal["increases", "decreases"]
Band = Literal["low", "medium", "high"]
ModelVariant = Literal["main", "audit"]


# ---------------------------------------------------------------- requests
class ExplanationRequest(BaseModel):
    user_id: str = Field(default="anonymous", max_length=64)
    query: str = Field(default="Why is this employee flagged?", max_length=1000)
    domain: str | None = Field(default=None, max_length=32)
    instance: dict[str, Any] | None = None
    level: Level | None = None
    model_variant: ModelVariant = "main"
    consent_cloud: bool = False
    assumed_fields: list[str] = Field(default_factory=list)  # fields filled with defaults by intake
    batch_mode: bool = False  # batch rows: no template fallback, failures are withheld
    website: str = ""  # honeypot, must stay empty


class RefinementRequest(BaseModel):
    trace_id: str = Field(max_length=64)
    user_id: str = Field(default="anonymous", max_length=64)
    action: Literal["simplify", "expand", "set_level"]
    level: Level | None = None
    consent_cloud: bool = False


class FollowupRequest(BaseModel):
    """A follow-up question about an already-shown explanation. Reuses its Evidence Package; does not re-run XAI
    unless the question's intent needs evidence the original did not include (e.g. asking about recourse)."""
    trace_id: str = Field(max_length=64)
    user_id: str = Field(default="anonymous", max_length=64)
    query: str = Field(max_length=1000)
    consent_cloud: bool = False
    website: str = ""  # honeypot


class ChatIntakeRequest(BaseModel):
    domain: str = "hr"
    instance: dict[str, Any] = Field(default_factory=dict)
    skipped: list[str] = Field(default_factory=list)
    message: str = Field(default="", max_length=2000)
    finish: bool = False
    consent_cloud: bool = False  # the message may contain employee details; same consent gate as /explain


class FeedbackRequest(BaseModel):
    trace_id: str = Field(max_length=64)
    user_id: str = Field(default="anonymous", max_length=64)
    rating: int = Field(ge=1, le=5)
    understood: bool | None = None
    trust: int | None = Field(default=None, ge=1, le=5)
    comment: str = Field(default="", max_length=2000)
    website: str = ""  # honeypot


class WhatIfRequest(BaseModel):
    domain: str = "hr"
    instance: dict[str, Any]
    changes: dict[str, Any] = Field(default_factory=dict)
    model_variant: ModelVariant = "main"


class IntakeRequest(BaseModel):
    domain: str = "hr"
    instance: dict[str, Any] = Field(default_factory=dict)
    skipped: list[str] = Field(default_factory=list)
    finish: bool = False


# ---------------------------------------------------------------- context
class UserProfile(BaseModel):
    user_id: str
    expertise: Level = "intermediate"
    # preferences["notes"]: list of {topic, text, updated_at, source_trace_id} — see app/agents/preferences.py.
    # A new note on a topic replaces the old one (the newer statement wins); rendered and appended to every
    # generator prompt for this user so later explanations and follow-ups reflect what they have said so far.
    preferences: dict[str, Any] = Field(default_factory=dict)


class RoutingResult(BaseModel):
    domain: str | None
    confidence: float
    intent: Intent = "explain"
    refusal_reason: str | None = None
    concept_term: str | None = None
    method: str = "rules"


# ---------------------------------------------------------------- prediction + XAI
class PredictionResult(BaseModel):
    label: str
    positive_label: str
    probability: float  # calibrated probability of the positive class
    interval: tuple[float, float]  # bootstrap ensemble interval
    band: Band
    band_uncertain: bool
    threshold: float
    out_of_distribution: bool
    ood_reasons: list[str] = Field(default_factory=list)
    model_version: str
    model_variant: ModelVariant = "main"


class EvidenceItem(BaseModel):
    id: str
    feature: str
    display_name: str
    glossary_id: str
    value: Any  # model-space value (code or number)
    value_label: str  # readable value, safe for text
    attribution: float  # Shapley value, probability units
    direction: Direction
    rank: int  # 1 = largest |attribution| among visible items
    share: float  # |attribution| / sum |attribution| over all items
    magnitude: Literal["strong", "moderate", "weak"]
    lime_attribution: float | None = None
    methods_agree: bool | None = None
    protected: bool = False
    withheld: bool = False  # hidden at this level by DomainPolicy
    assumed: bool = False  # value filled with a default by intake


class RecourseChange(BaseModel):
    feature: str
    display_name: str
    from_label: str
    to_label: str
    consequence: bool = False  # side effect of an action (e.g. promotion resets years since promotion)


class RecourseOption(BaseModel):
    id: str
    actions: list[str]
    description: str
    short_description: str = ""
    changes: list[RecourseChange]
    probability_before: float
    probability_after: float
    band_before: Band
    band_after: Band
    assumptions: list[str] = Field(default_factory=list)
    cost: float = 0.0


class ValidationCheck(BaseModel):
    name: str
    passed: bool
    severity: Literal["critical", "warning"] = "critical"
    detail: str = ""


class GlossaryEntry(BaseModel):
    id: str
    term: str
    definition: str
    values: dict[str, str] = Field(default_factory=dict)
    source: str = ""


class EvidencePackage(BaseModel):
    package_id: str
    domain: str
    method: str
    level: Level
    prediction: PredictionResult
    base_value: float
    items: list[EvidenceItem]
    recourse: list[RecourseOption] = Field(default_factory=list)
    recourse_note: str = ""
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    checks: list[ValidationCheck] = Field(default_factory=list)
    valid: bool = True
    assumptions: list[str] = Field(default_factory=list)

    def item(self, eid: str) -> EvidenceItem | None:
        return next((i for i in self.items if i.id == eid), None)

    def option(self, rid: str) -> RecourseOption | None:
        return next((r for r in self.recourse if r.id == rid), None)

    @property
    def visible(self) -> list[EvidenceItem]:
        return [i for i in self.items if not i.withheld]


# ---------------------------------------------------------------- explanation
class ExplanationPlan(BaseModel):
    level: Level
    intent: Intent
    allowed_ids: list[str]
    required_ids: list[str]
    max_words: int
    include_numbers: bool
    allow_column_names: bool
    banned_terms: list[str]
    style: str
    extra_instructions: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    text: str = ""
    kind: Literal["risk", "attribution", "recourse", "glossary"] = "attribution"
    evidence_ids: list[str] = Field(default_factory=list)
    feature: str | None = None
    direction: str | None = None
    value: Any = None
    magnitude: str | None = None
    is_top: bool = False
    probability: float | None = None  # for risk/recourse claims, as a fraction 0-1
    band: str | None = None


class GeneratedExplanation(BaseModel):
    explanation: str
    claims: list[Claim] = Field(default_factory=list)
    source: Literal["llm", "template", "glossary", "notice"] = "llm"


class Violation(BaseModel):
    code: str
    severity: Literal["critical", "minor"]
    message: str
    claim_index: int | None = None
    span: str | None = None


class VerificationResult(BaseModel):
    passed: bool
    fidelity_score: float  # Claim Fidelity Score (CFS): supported claims / claims. See app/metrics/fidelity.py.
    fidelity: dict[str, Any] = Field(default_factory=dict)  # full FidelityReport (score, weighted_score, counts)
    violations: list[Violation] = Field(default_factory=list)
    n_claims: int = 0


class CriticResult(BaseModel):
    approved: bool
    violations: list[Violation] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    readability: float | None = None
    word_count: int = 0
    feedback: str = ""


class ExplanationResponse(BaseModel):
    trace_id: str
    status: Literal["verified", "fallback", "escalated", "refused", "concept", "clarify", "withheld", "error"]
    domain: str | None = None
    level: Level | None = None
    intent: str | None = None
    message: str = ""
    notices: list[str] = Field(default_factory=list)
    disclosure: str = ""
    prediction: PredictionResult | None = None
    evidence: EvidencePackage | None = None
    explanation: GeneratedExplanation | None = None
    verification: VerificationResult | None = None
    critic: CriticResult | None = None
    attempts: int = 0
    prompt_version: str = ""
    llm: str = ""
    trace: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    parent_trace_id: str | None = None  # set on a follow-up turn: the trace_id of the explanation it followed
    preferences_applied: list[str] = Field(default_factory=list)  # preference notes injected into this prompt


# ---------------------------------------------------------------- audit records
class WorkflowTraceRecord(BaseModel):
    trace_id: str
    created_at: str
    user_id: str
    domain: str | None
    intent: str | None
    level: str | None
    model_variant: str
    prompt_version: str
    llm: str
    status: str
    verdict: str  # passed | failed_then_fallback | escalated | refused | withheld | concept | clarify | error
    attempts: int
    violations: list[dict[str, Any]] = Field(default_factory=list)  # all attempts
    withheld_features: list[str] = Field(default_factory=list)
    pii_masked: int = 0
    cloud_consent: bool = False
    query: str = ""  # PII-scrubbed
    request: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    parent_trace_id: str | None = None


class FeedbackSignalRecord(BaseModel):
    feedback_id: str
    trace_id: str
    created_at: str
    user_id: str
    rating: int
    understood: bool | None = None
    trust: int | None = None
    comment: str = ""
    prompt_version: str = ""
    level: str | None = None
