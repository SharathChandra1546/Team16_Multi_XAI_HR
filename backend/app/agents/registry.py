"""The role-separated components ("agents") of the explanation graph, described in one place.

"Agent" here means a component with one role, typed inputs and outputs, and a recorded decision. Some are
deterministic code, some call an LLM. The `kind` field says which, so the paper and the trace never blur them:

- deterministic: closed-form code, no model call, reproducible
- llm: calls a language model
- hybrid: deterministic first, optional LLM layer

`authority` says what an agent's output can do:

- gate: can block an output (only deterministic components have this)
- route: decides which branch the graph takes
- produce: writes or rewrites content
- advise: recorded and shown, but cannot change the outcome; LLM agents never override a gate

`switch` is the PipelineConfig field that turns the agent off (None = always on).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AgentSpec:
    name: str
    node: str
    role: str
    kind: str  # deterministic | llm | hybrid
    authority: str  # gate | route | produce | advise
    switch: str | None = None


AGENTS: dict[str, AgentSpec] = {a.node: a for a in [
    AgentSpec("Profile agent", "profile", "Chooses the reader level from the request, stored profile and query cues.",
              "deterministic", "route", "profile_agent"),
    AgentSpec("Router", "route", "Decision-use guardrail, domain routing and intent detection.", "hybrid", "route"),
    AgentSpec("Evidence builder", "xai", "Builds and validates the canonical evidence package (Shapley, LIME check, "
              "recourse) and computes the escalation score.", "deterministic", "produce"),
    AgentSpec("Escalation gate", "escalate", "Routes unreliable cases (uncertain band, out of distribution) to human "
              "review instead of explaining them.", "deterministic", "gate", "escalation"),
    AgentSpec("Planner", "plan", "Deterministic level plan: evidence scope, word budget, terminology, readability.",
              "deterministic", "produce"),
    AgentSpec("LLM planner", "llm_plan", "Optional audience planner: may narrow and reorder the plan's evidence and add "
              "style notes; can never add evidence the deterministic plan excluded.", "llm", "advise", "llm_planner"),
    AgentSpec("Generator", "generate", "Writes the explanation and its structured claims from the evidence.", "llm",
              "produce"),
    AgentSpec("Verifier", "verify", "Claim-level fidelity gate against the evidence package (typed violations).",
              "deterministic", "gate", "verifier"),
    AgentSpec("Critic", "critic", "Domain policy rules, level constraints, and an optional LLM quality score.",
              "hybrid", "gate", "policy_rules"),
    AgentSpec("Repair", "repair", "Turns violations into evidence-bound repair instructions for the next attempt.",
              "deterministic", "produce", "repair"),
    AgentSpec("Reviewer", "review", "Second LLM from a different model family; flags problems, advisory only.",
              "llm", "advise", "reviewer"),
    AgentSpec("Fallback", "fallback", "Verified template when retries run out (or withhold).", "deterministic",
              "produce", "fallback"),
    AgentSpec("Finalizer", "finalize", "Attaches the disclosure and notices; builds provenance and the policy report.",
              "deterministic", "produce"),
]}


def describe() -> list[dict]:
    return [asdict(a) for a in AGENTS.values()]


def meta(node: str) -> dict:
    a = AGENTS.get(node)
    return {"agent": a.name, "kind": a.kind, "authority": a.authority} if a else {}
