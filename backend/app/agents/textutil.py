"""Small text utilities shared by the verifier and critic (no external NLP deps)."""
from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_NEGATION = re.compile(r"\b(not|no|never|without|cannot|nor|neither)\b|n't\b|\brather than\b", re.I)
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[-+]?\d+(?:\.\d+)?")
_COUNT_WORDS = re.compile(
    r"^\s*(?:main\s+|key\s+|other\s+|more\s+|largest\s+|biggest\s+|top\s+)?"
    r"(factors?|reasons?|options?|changes?|ways?|things?|levers?|items?|features?|steps?|actions?|drivers?)\b",
    re.I,
)


def sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9%.,+\- ]", " ", text.lower()).split()).strip(" .")


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'\-]*", text)


def negated(sentence: str, start: int, window_words: int = 6) -> bool:
    """True if a negation appears within `window_words` words before position `start` in the sentence."""
    prefix = sentence[:start]
    tail = " ".join(prefix.split()[-window_words:])
    return bool(_NEGATION.search(tail))


def find_rule_matches(text: str, patterns: list[re.Pattern], negatable: bool = True) -> list[str]:
    """Return matched spans (sentence-scoped negation handling)."""
    hits: list[str] = []
    for sent in sentences(text):
        for pat in patterns:
            for m in pat.finditer(sent):
                if negatable and negated(sent, m.start()):
                    continue
                hits.append(m.group(0))
    return hits


def extract_numbers(text: str) -> list[tuple[float, str, int, int]]:
    """Numbers in text as (value, raw, start, end). Small integers used as counts ('3 factors') are skipped."""
    out = []
    for m in _NUMBER.finditer(text):
        raw = m.group(0)
        try:
            val = float(raw.replace(",", "").replace("+", ""))
        except ValueError:
            continue
        if val.is_integer() and 0 <= val <= 10 and _COUNT_WORDS.match(text[m.end():]):
            continue
        out.append((abs(val), raw, m.start(), m.end()))
    return out


_VOWELS = "aeiouy"


def _syllables(word: str) -> int:
    w = word.lower().strip("'")
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    groups = re.findall(r"[aeiouy]{1,2}", w)
    return max(1, len(groups))


def flesch_reading_ease(text: str) -> float:
    sents = max(1, len(sentences(text)))
    ws = words(text)
    if not ws:
        return 100.0
    syl = sum(_syllables(w) for w in ws)
    return round(206.835 - 1.015 * (len(ws) / sents) - 84.6 * (syl / len(ws)), 1)


def word_count(text: str) -> int:
    return len(words(text))
