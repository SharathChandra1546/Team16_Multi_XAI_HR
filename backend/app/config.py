"""Central configuration, read from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:  # pragma: no cover
    pass

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"

# OpenAI-compatible endpoints. Gemini, Groq, OpenAI and Ollama all speak this protocol.
PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": "OLLAMA_API_KEY",
        "default_model": "llama3.1",
    },
}


@dataclass
class Settings:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER") or "mock")
    llm_model: str | None = field(default_factory=lambda: os.getenv("LLM_MODEL") or None)
    # A second, independent model used as LLM-reviewer baseline / narrative extractor in eval.
    judge_provider: str = field(default_factory=lambda: os.getenv("JUDGE_PROVIDER", "") or os.getenv("LLM_PROVIDER") or "mock")
    judge_model: str | None = field(default_factory=lambda: os.getenv("JUDGE_MODEL") or None)
    # Reviewer agent (advisory): should be a different model family from the generator. Falls back to the judge.
    reviewer_provider: str = field(default_factory=lambda: os.getenv("REVIEWER_PROVIDER", ""))
    reviewer_model: str | None = field(default_factory=lambda: os.getenv("REVIEWER_MODEL") or None)
    # Fault generator for detector-independent fault injection (eval/run_faults.py --faults llm). Falls back to judge.
    fault_provider: str = field(default_factory=lambda: os.getenv("FAULT_PROVIDER", ""))
    fault_model: str | None = field(default_factory=lambda: os.getenv("FAULT_MODEL") or None)
    # Shapley permutations per explanation (more = more stable attributions, slower)
    shap_permutations: int = field(default_factory=lambda: int(os.getenv("SHAP_PERMUTATIONS") or "32"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE") or "0"))
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH") or str(BACKEND_DIR / "mxai.sqlite3"))
    max_attempts: int = field(default_factory=lambda: int(os.getenv("MAX_ATTEMPTS") or "3"))
    router_confidence_threshold: float = field(default_factory=lambda: float(os.getenv("ROUTER_THRESHOLD") or "0.6"))
    cache_llm: bool = field(default_factory=lambda: os.getenv("LLM_CACHE", "1") == "1")
    # Bootstrap-interval percentiles used for band uncertainty (escalation rule).
    interval_low_pct: float = field(default_factory=lambda: float(os.getenv("INTERVAL_LOW_PCT") or "10"))
    interval_high_pct: float = field(default_factory=lambda: float(os.getenv("INTERVAL_HIGH_PCT") or "90"))
    # Security / deployment
    admin_token: str = field(default_factory=lambda: os.getenv("ADMIN_TOKEN", ""))
    force_https: bool = field(default_factory=lambda: os.getenv("FORCE_HTTPS", "0") == "1")
    allowed_origins: list[str] = field(
        default_factory=lambda: [o.strip() for o in (os.getenv("ALLOWED_ORIGINS") or "http://localhost:5173").split(",") if o.strip()]
    )
    static_dir: str = field(default_factory=lambda: os.getenv("STATIC_DIR", ""))
    rate_limit_per_min: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MIN") or "30"))
    batch_max_rows: int = field(default_factory=lambda: int(os.getenv("BATCH_MAX_ROWS") or "2000"))

    @property
    def llm_is_cloud(self) -> bool:
        """True when explanations would be sent to an external provider (needs user consent)."""
        return self.llm_provider not in ("mock", "ollama")


settings = Settings()


def reload_settings() -> Settings:
    """Re-read environment (used by tests)."""
    global settings
    new = Settings()
    settings.__dict__.update(new.__dict__)
    return settings
