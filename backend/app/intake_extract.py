"""Extracting structured instance fields from unstructured input: an uploaded PDF or a chat message.

Two extraction paths, same pattern as the rest of the system (LLM optional,
deterministic fallback always available so the app runs with no API key):

- `extract_pdf_text`: pulls raw text out of a PDF (pypdf; optional dependency).
- `extract_fields`: turns free text into {feature_name: raw_value} using the
  domain's own FeatureSpec aliases/options, either via an LLM JSON-extraction
  call or via regex matching against each feature's aliases. Values are
  handed to `DomainAgent.parse_instance` for validation/conversion, so a
  wrong or unparseable extraction never silently corrupts the instance: it is
  simply dropped and the field stays "missing", to be asked for in chat.
"""
from __future__ import annotations

import io
import re

from .domains.base import DomainAgent
from .llm.client import BaseLLM

try:  # optional dependency
    from pypdf import PdfReader

    HAS_PYPDF = True
except Exception:  # pragma: no cover - environment dependent
    HAS_PYPDF = False


def extract_pdf_text(data: bytes) -> str:
    if not HAS_PYPDF:
        raise RuntimeError("PDF reading needs the 'pypdf' package. Install it on the server (pip install pypdf).")
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


# ---------------------------------------------------------------- rule-based fallback
def _regex_for(alias: str, categorical: bool) -> re.Pattern:
    sep = r"(?:\s*[:=\-]\s*|\s+(?:is|of|was)\s+|\s+)"
    val = r"([A-Za-z][A-Za-z \-/]{0,30}|-?\d+(?:\.\d+)?)" if categorical else r"(-?\d+(?:\.\d+)?)"
    return re.compile(rf"\b{re.escape(alias)}\b{sep}{val}", re.I)


def rule_based_extract(domain: DomainAgent, text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for spec in domain.features:
        aliases = sorted({spec.display.lower(), spec.name.lower(), *spec.aliases}, key=len, reverse=True)
        for alias in aliases:
            m = None
            for pat in (_regex_for(alias, spec.categorical),):
                m = pat.search(text)
                if m:
                    break
            if m:
                found[spec.name] = m.group(1).strip()
                break
    return found


def llm_extract(domain: DomainAgent, text: str, llm: BaseLLM) -> dict[str, str]:
    fields = [{"name": f.name, "display": f.display, "description": f.description,
               "type": "categorical" if f.categorical else "numeric",
               "options": list(f.labels.values()) if f.categorical else None}
              for f in domain.features]
    system = (
        f"Extract details about {domain.language.subject} from the text for these fields. Only use values stated or clearly implied in the "
        "text; never guess or invent a value. For a categorical field, use one of its listed options (or null). "
        "For a numeric field, use a plain number (or null). "
        f"Fields: {fields}. Return JSON: {{\"<field name>\": <value or null>, ...}} for every field."
    )
    try:
        out = llm.json("intake_extract", system, text[:4000], {"domain": domain.name})
    except Exception:
        return {}
    return {k: v for k, v in out.items() if v not in (None, "", "null")}


def extract_fields(domain: DomainAgent, text: str, llm: BaseLLM | None) -> dict[str, str]:
    """Best-effort {feature_name: raw_value}. Caller must still validate via domain.parse_instance."""
    if not text.strip():
        return {}
    found = llm_extract(domain, text, llm) if llm is not None else {}
    # fill in anything the LLM missed (or skip entirely if there's no LLM) with the regex fallback
    for k, v in rule_based_extract(domain, text).items():
        found.setdefault(k, v)
    return found
