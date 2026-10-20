"""DomainRegistry: the only place that knows which domains exist."""
from __future__ import annotations

from .base import DomainAgent


class DomainRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, DomainAgent] = {}

    def register(self, agent: DomainAgent) -> None:
        self._agents[agent.name] = agent

    def get(self, name: str) -> DomainAgent:
        if name not in self._agents:
            raise KeyError(f"Unknown domain '{name}'")
        return self._agents[name]

    def has(self, name: str | None) -> bool:
        return bool(name) and name in self._agents

    def list(self) -> list[DomainAgent]:
        return list(self._agents.values())

    def names(self) -> list[str]:
        return list(self._agents)


def default_registry() -> DomainRegistry:
    from .hr.agent import HrAgent

    reg = DomainRegistry()
    reg.register(HrAgent())
    return reg
