"""Pipeline switches: every component of the explanation graph can be turned on or off independently.

Used for ablations (Phase 2) and for configuring which LLM agents run. Defaults come from environment variables
(PIPELINE_<NAME>=0/1) so a deployment can enable the optional LLM agents; experiments pass an explicit
PipelineConfig per request instead.

Shadow mode: when a checking component is switched OFF, it still runs and its result is recorded, but it no
longer gates the output. This is what lets an ablation report "errors that would have reached the user" without a
second measurement pass: the verifier's verdict on what was shown is always in the trace.

Safety switches (verifier, policy_rules, escalation) are never exposed through the public HTTP API; only code
(evaluation scripts, tests) can turn them off.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Literal


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None or v == "" else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class PipelineConfig:
    # deterministic gates (on by default)
    verifier: bool = True  # claim-level fidelity gate
    policy_rules: bool = True  # domain wording rules (decision, protected attributes, blame, certainty)
    level_rules: bool = True  # level constraints (terminology, word budget, numbers)
    repair: bool = True  # structured, evidence-bound repair instructions on retry
    escalation: bool = True  # route uncertain / out-of-distribution cases to human review
    fallback: bool = True  # verified template when retries run out (off: withhold)
    # agents
    profile_agent: bool = True  # chooses the reader level from the request, profile and query cues
    llm_critic: bool = True  # LLM quality score (clarity, level fit, relevance); runs only when an LLM is used
    llm_planner: bool = False  # LLM planner may narrow / reorder the deterministic plan, never widen it
    reviewer: bool = False  # second-family LLM reviewer; advisory only, never overrides the verifier
    max_attempts: int | None = None  # None: settings.max_attempts
    escalation_mode: Literal["rule", "score"] = "rule"
    escalation_threshold: float = 0.5  # used in score mode

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        base = cls()
        kw = {}
        for f in fields(cls):
            env = f"PIPELINE_{f.name.upper()}"
            if f.type in ("bool", bool):
                kw[f.name] = _env_bool(env, getattr(base, f.name))
        kw["escalation_mode"] = os.getenv("ESCALATION_MODE", base.escalation_mode) or base.escalation_mode
        kw["escalation_threshold"] = float(os.getenv("ESCALATION_THRESHOLD") or base.escalation_threshold)
        return cls(**kw)

    @classmethod
    def from_dict(cls, d: dict | None) -> "PipelineConfig":
        if not d:
            return cls.from_env()
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def as_dict(self) -> dict:
        return asdict(self)

    def disabled(self) -> list[str]:
        """Names of switches that are off relative to the default configuration (for logs and result tables)."""
        base = PipelineConfig()
        return [f.name for f in fields(self) if isinstance(getattr(base, f.name), bool)
                and getattr(base, f.name) and not getattr(self, f.name)]


# Named configurations used by the ablation runner (Phase 2) and the tests.
ABLATIONS: dict[str, dict] = {
    "full": {},
    "no_verifier": {"verifier": False},
    "no_policy_rules": {"policy_rules": False},
    "no_critic": {"policy_rules": False, "level_rules": False, "llm_critic": False},
    "no_repair": {"repair": False},
    "no_escalation": {"escalation": False},
    "no_fallback": {"fallback": False},
    "with_llm_agents": {"llm_planner": True, "reviewer": True},
}


def ablation(name: str) -> PipelineConfig:
    return PipelineConfig(**ABLATIONS[name])
