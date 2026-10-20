"""Pluggable LLM client.

Every provider (Gemini, Groq, OpenAI, Ollama) is reached through the
OpenAI-compatible /chat/completions endpoint, so one class covers all of them.
`MockLLM` is a deterministic stand-in so the pipeline, tests and error-injection
experiments run without an API key.

All agents call `llm.json(task, system, user, payload)` and receive a dict.
`task` names the agent role (used by the mock and for caching/logging);
`payload` is the structured input (the real client ignores it because it is
already serialised into `user`).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from typing import Any

from ..config import PROVIDERS, settings


class LLMError(RuntimeError):
    pass


class BaseLLM:
    name = "base"
    model = "none"

    def json(self, task: str, system: str, user: str, payload: dict | None = None) -> dict:
        raise NotImplementedError

    @property
    def id(self) -> str:
        return f"{self.name}:{self.model}"


class _Cache:
    """Tiny sqlite response cache: identical prompts at temperature 0 are reused (saves free-tier quota)."""

    def __init__(self, path: str):
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, k: str) -> str | None:
        with self._lock:
            row = self.conn.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def put(self, k: str, v: str) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (k, v))
            self.conn.commit()


def _parse_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


class OpenAICompatLLM(BaseLLM):
    def __init__(self, provider: str, model: str | None = None, temperature: float | None = None):
        if provider not in PROVIDERS:
            raise LLMError(f"Unknown provider '{provider}'. Choose one of {list(PROVIDERS)} or 'mock'.")
        cfg = PROVIDERS[provider]
        self.name = provider
        self.model = model or cfg["default_model"]
        self.base_url = os.getenv(f"{provider.upper()}_BASE_URL", cfg["base_url"])
        self.api_key = os.getenv(cfg["key_env"], "ollama" if provider == "ollama" else "")
        if not self.api_key:
            raise LLMError(f"Set {cfg['key_env']} in .env to use provider '{provider}'.")
        self.temperature = settings.temperature if temperature is None else temperature
        self.cache = _Cache(os.path.join(os.path.dirname(settings.db_path), "llm_cache.sqlite3")) if settings.cache_llm else None

    def json(self, task: str, system: str, user: str, payload: dict | None = None) -> dict:
        import httpx  # local import keeps module importable without httpx

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system + "\nRespond with a single valid JSON object only."},
                {"role": "user", "content": user},
            ],
        }
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if self.cache and self.temperature == 0:
            hit = self.cache.get(key)
            if hit:
                return json.loads(hit)

        last_err: Exception | None = None
        for attempt in range(4):
            try:
                r = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=body,
                    timeout=90,
                )
                if r.status_code == 429 or r.status_code >= 500:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                data = _parse_json(content)
                if self.cache and self.temperature == 0:
                    self.cache.put(key, json.dumps(data))
                return data
            except Exception as e:  # retry with backoff (free tiers rate-limit)
                last_err = e
                time.sleep(2 ** attempt * 2)
        raise LLMError(f"{self.id} failed for task '{task}': {last_err}")


_instances: dict[tuple, BaseLLM] = {}


def get_llm(provider: str | None = None, model: str | None = None) -> BaseLLM:
    provider = provider or settings.llm_provider
    model = model or (settings.llm_model if provider == settings.llm_provider else None)
    k = (provider, model)
    if k not in _instances:
        if provider == "mock":
            from .mock import MockLLM

            _instances[k] = MockLLM()
        else:
            _instances[k] = OpenAICompatLLM(provider, model)
    return _instances[k]


def get_judge() -> BaseLLM:
    return get_llm(settings.judge_provider, settings.judge_model)
