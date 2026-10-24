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

    # ---------------------------------------------------------------- usage accounting (for cost comparisons)
    def _usage(self) -> dict:
        if not hasattr(self, "_usage_counters"):
            self._usage_counters = {"calls": 0, "cached_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                    "estimated": False, "by_task": {}}
        return self._usage_counters

    def record_usage(self, task: str, prompt_tokens: int, completion_tokens: int, cached: bool = False,
                     estimated: bool = False) -> None:
        u = self._usage()
        u["calls"] += 1
        u["cached_calls"] += int(cached)
        u["prompt_tokens"] += int(prompt_tokens)
        u["completion_tokens"] += int(completion_tokens)
        u["estimated"] = u["estimated"] or estimated
        t = u["by_task"].setdefault(task, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        t["calls"] += 1
        t["prompt_tokens"] += int(prompt_tokens)
        t["completion_tokens"] += int(completion_tokens)

    def usage_snapshot(self) -> dict:
        import copy

        return copy.deepcopy(self._usage())


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token) for mock runs and cache hits."""
    return max(1, len(text) // 4)


def usage_delta(before: dict, after: dict) -> dict:
    keys = ("calls", "cached_calls", "prompt_tokens", "completion_tokens")
    return {k: after.get(k, 0) - before.get(k, 0) for k in keys} | {"estimated": after.get("estimated", False)}


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
    # some open models (Gemma 4) put their reasoning in <thought> blocks before the answer
    text = re.sub(r"<(thought|think)>.*?</\1>", "", text.strip(), flags=re.S | re.I).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # take the LAST complete JSON object in the text (the answer follows any reasoning or examples)
        dec, best = json.JSONDecoder(), None
        for i, ch in enumerate(text):
            if ch == "{":
                try:
                    obj, end = dec.raw_decode(text, i)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and (best is None or end > best[1]):
                    best = (obj, end)
        if best is not None:
            return best[0]
        raise


def _reasoning_effort(provider: str, model: str) -> str | None:
    """Reasoning setting for models that think before answering. LLM_REASONING_EFFORT overrides (empty = default).
    Qwen 3 on Groq: thinking makes a request too large for the free 8K-tokens-per-minute limit, so it is off.
    gpt-oss: 'low' keeps extraction and review calls short."""
    if "LLM_REASONING_EFFORT" in os.environ:
        return os.environ["LLM_REASONING_EFFORT"] or None
    m = (model or "").lower()
    if provider in ("groq", "cerebras") and "qwen" in m:
        return "none"
    if "gpt-oss" in m:
        return "low"
    return None


def _retry_after(r) -> float | None:
    """Seconds to wait from a 429 response (Retry-After header or Google's RetryInfo), if given."""
    h = r.headers.get("retry-after")
    if h:
        try:
            return float(h)
        except ValueError:
            pass
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', r.text or "")
    return float(m.group(1)) if m else None


class OpenAICompatLLM(BaseLLM):
    # HTTP requests sent per model ("provider:model") in this process, retries included (what a daily quota counts)
    requests_sent: dict[str, int] = {}
    # tokens used per model in this process (prompt + completion, cached answers excluded)
    tokens_used: dict[str, int] = {}

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
            "messages": [
                {"role": "system", "content": system + "\nRespond with a single valid JSON object only."},
                {"role": "user", "content": user},
            ],
        }
        if not self.model.startswith("gemma"):  # Gemma on the Gemini API rejects JSON mode (HTTP 500)
            body["response_format"] = {"type": "json_object"}
        effort = _reasoning_effort(self.name, self.model)
        if effort:
            body["reasoning_effort"] = effort
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if self.cache and self.temperature == 0:
            hit = self.cache.get(key)
            if hit:
                # a cache hit costs nothing now, but the comparison reports what the call would have cost
                self.record_usage(task, estimate_tokens(system + user), estimate_tokens(hit), cached=True, estimated=True)
                return json.loads(hit)

        last_err: Exception | None = None
        for attempt in range(4):
            try:
                OpenAICompatLLM.requests_sent[self.id] = OpenAICompatLLM.requests_sent.get(self.id, 0) + 1
                r = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=body,
                    timeout=90,
                )
                if r.status_code == 429:
                    # per-minute limits (Groq free tier: 8K tokens/min) say how long to wait; a daily limit does not
                    # clear within a run, so fail instead of sleeping for hours
                    wait = _retry_after(r)
                    if wait is not None and wait <= 90:
                        time.sleep(wait + 1)
                        continue
                    raise LLMError(f"HTTP 429 (quota): {r.text[:300]}")
                if r.status_code >= 500:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                body_json = r.json()
                content = body_json["choices"][0]["message"]["content"]
                data = _parse_json(content)
                usage = body_json.get("usage") or {}
                if usage.get("prompt_tokens") is not None:
                    self.record_usage(task, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                    spent = int(usage.get("total_tokens") or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)))
                else:
                    self.record_usage(task, estimate_tokens(system + user), estimate_tokens(content), estimated=True)
                    spent = estimate_tokens(system + user) + estimate_tokens(content)
                OpenAICompatLLM.tokens_used[self.id] = OpenAICompatLLM.tokens_used.get(self.id, 0) + spent
                if self.cache and self.temperature == 0:
                    self.cache.put(key, json.dumps(data))
                return data
            except Exception as e:  # retry with backoff (free tiers rate-limit)
                last_err = e
                time.sleep(2 ** attempt * 2)
        raise LLMError(f"{self.id} failed for task '{task}': {last_err}")


_instances: dict[tuple, BaseLLM] = {}


def _usage_file():
    import datetime as _dt
    from pathlib import Path as _P

    day = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    return _P(__file__).resolve().parents[2] / "results" / f"llm_usage_{day}.json"


_SAVED: dict[str, dict[str, int]] = {"requests": {}, "tokens": {}}  # what this process already wrote to the file


def _read_usage() -> dict:
    try:
        return json.loads(_usage_file().read_text(encoding="utf-8"))
    except Exception:
        return {"requests": {}, "tokens": {}}


def load_daily_usage() -> None:
    """Count today's (UTC) usage by other runs, so a daily budget holds across restarts and parallel runs."""
    d = _read_usage()
    for kind, counts in (("requests", OpenAICompatLLM.requests_sent), ("tokens", OpenAICompatLLM.tokens_used)):
        for k, v in d.get(kind, {}).items():
            counts[k] = counts.get(k, 0) + int(v)
            _SAVED[kind][k] = _SAVED[kind].get(k, 0) + int(v)


def save_daily_usage() -> None:
    """Add this process's new usage to the file (merge, so parallel runs do not overwrite each other)."""
    import fcntl

    f = _usage_file()
    f.parent.mkdir(exist_ok=True)
    with open(f.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        d = _read_usage()
        for kind, counts in (("requests", OpenAICompatLLM.requests_sent), ("tokens", OpenAICompatLLM.tokens_used)):
            for k, v in counts.items():
                delta = v - _SAVED[kind].get(k, 0)
                if delta:
                    d.setdefault(kind, {})[k] = d.get(kind, {}).get(k, 0) + delta
                    _SAVED[kind][k] = v
            # pick up what parallel runs added meanwhile
            for k, v in d.get(kind, {}).items():
                mine = _SAVED[kind].get(k, 0)
                if v > mine:
                    counts[k] = counts.get(k, 0) + (v - mine)
                    _SAVED[kind][k] = v
        f.write_text(json.dumps(d, indent=1), encoding="utf-8")


def parse_budget(spec: str) -> dict[str, int]:
    """'groq=900,gemini:gemma-4-26b-a4b-it=5000' -> {key: cap}; k/M suffixes allowed (200k)."""
    out = {}
    for part in (spec or "").split(","):
        if "=" in part:
            k, v = part.rsplit("=", 1)
            v = v.strip().lower()
            mult = 1_000_000 if v.endswith("m") else 1_000 if v.endswith("k") else 1
            out[k.strip()] = int(float(v.rstrip("km")) * mult)
    return out


def over_budget(request_budget: dict[str, int], token_budget: dict[str, int], only: set[str] | None = None) -> str | None:
    """First model (provider:model, or a provider key) that reached its request or token budget today.
    `only`: the model ids this run actually uses; other models' budgets do not stop it."""
    for counts, budget, unit in ((OpenAICompatLLM.requests_sent, request_budget, "requests"),
                                 (OpenAICompatLLM.tokens_used, token_budget, "tokens")):
        for mid, n in counts.items():
            if only is not None and mid not in only:
                continue
            for key, cap in budget.items():
                if (mid == key or mid.split(":", 1)[0] == key) and n >= cap:
                    return f"{mid} used {n} {unit} (budget {key}={cap})"
    return None


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
