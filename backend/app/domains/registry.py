"""DomainRegistry: the only place that knows which domains exist.

Adding a domain = one line in KNOWN_DOMAINS plus the package app/domains/<name>/. A domain is registered only when
its main model has been trained (scripts/train_domain.py), so the router can never send a request to a domain that
cannot answer it. HR is always registered so a fresh checkout reports "train the model" instead of having no domain.
"""
from __future__ import annotations

import importlib
import logging

from .base import DomainAgent

log = logging.getLogger("mxai.domains")

# name -> "module:Class"
KNOWN_DOMAINS: dict[str, str] = {
    "hr": "app.domains.hr.agent:HrAgent",
}
ALWAYS_REGISTER = {"hr"}


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


def agent_class(name: str) -> type[DomainAgent]:
    module, cls = KNOWN_DOMAINS[name].split(":")
    return getattr(importlib.import_module(module), cls)


def default_registry(names: list[str] | None = None) -> DomainRegistry:
    reg = DomainRegistry()
    for name in names or list(KNOWN_DOMAINS):
        agent = agent_class(name)()
        if agent.ready or name in ALWAYS_REGISTER:
            reg.register(agent)
        else:
            log.info("domain %s not registered: its model is not trained (scripts/train_domain.py --domain %s)", name, name)
    return reg
