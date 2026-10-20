"""User-profile agent: decides the reader level for one request.

Order of precedence (first match wins), recorded as `source` in the trace:
1. explicit: the request names a level;
2. query: the question itself asks for plain language or for technical detail ("in simple terms", "show me the
   exact numbers");
3. profile: the user's stored expertise (updated when they press simplify/expand, see Service.refine);
4. default: intermediate.

It also returns the stored preference notes that will be rendered into the generator prompt. Deterministic: the
same request and profile always give the same level.
"""
from __future__ import annotations

from ..schemas import UserProfile
from .preferences import _DETAILED, _SIMPLER, render_preferences

LEVELS = ("novice", "intermediate", "expert")


def choose_level(requested: str | None, profile: UserProfile | None, query: str, enabled: bool = True) -> dict:
    notes = (profile.preferences.get("notes", []) if profile else []) or []
    if requested in LEVELS:
        level, source = requested, "explicit"
    elif not enabled:
        level, source = "intermediate", "default (profile agent off)"
    elif _SIMPLER.search(query or ""):
        level, source = "novice", "query asks for plain language"
    elif _DETAILED.search(query or ""):
        level, source = "expert", "query asks for technical detail"
    elif profile is not None and profile.expertise in LEVELS:
        level, source = profile.expertise, "stored profile"
    else:
        level, source = "intermediate", "default"
    return {"level": level, "source": source, "notes": notes if enabled else [],
            "preferences_text": render_preferences(notes) if enabled else ""}
