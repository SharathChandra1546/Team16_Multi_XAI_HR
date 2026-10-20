"""PII scrubbing for free-text input before anything is sent to an LLM provider.

Masks e-mail addresses, phone numbers, employee IDs, long digit strings and likely
personal names. Name detection is heuristic (capitalised words that are not known
domain vocabulary); it reduces exposure but cannot guarantee every name is caught.
"""
from __future__ import annotations

import re

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"(?<![\w])(?:\+?\d[\d\s\-().]{7,}\d)(?![\w])")
EMP_ID = re.compile(r"\b(?:emp(?:loyee)?[\s_-]*(?:id|no\.?|number|#)?[\s:#_-]*)\d{2,}\b|\b(?:id|ID|Id)\s*[:#]?\s*\d{3,}\b", re.I)
LONG_DIGITS = re.compile(r"\b\d{6,}\b")
HONORIFIC = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof)\.?\s+[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)?")
NAMED = re.compile(r"\b(?:named|called|name is|employee)\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,2})")
CAPWORD = re.compile(r"\b[A-Z][a-z][a-zA-Z'\-]*\b")

COMMON = {
    "i", "why", "what", "how", "is", "are", "can", "could", "would", "should", "the", "this", "that", "these", "those",
    "please", "explain", "tell", "show", "give", "does", "do", "which", "who", "when", "where", "if", "our", "we",
    "they", "their", "hr", "ok", "yes", "no", "and", "or", "for", "about", "with", "a", "an", "in", "on", "of",
    "to", "me", "my", "it", "its", "has", "have", "had", "was", "were", "be", "will", "risk", "employee", "employees",
    "model", "attrition", "flagged", "lower", "reduce", "help", "keep", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "hi", "hello", "thanks", "thank", "also", "then", "so", "but",
    "any", "all", "some", "one", "two", "three", "here", "there", "now", "next", "last", "new", "manager",
}


def domain_vocab(registry) -> set[str]:
    vocab = set(COMMON)
    for d in registry.list():
        for f in d.features:
            for s in [f.display, f.name, *f.labels.values(), *f.options.values(), *f.aliases]:
                vocab |= {w.lower() for w in re.findall(r"[A-Za-z]+", str(s))}
        for g in d.glossary.values():
            vocab |= {w.lower() for w in re.findall(r"[A-Za-z]+", g.term)}
    return vocab


def scrub(text: str, vocab: set[str]) -> tuple[str, int]:
    count = 0

    def sub(pattern, repl, s):
        nonlocal count
        s2, n = pattern.subn(repl, s)
        count += n
        return s2

    t = sub(EMAIL, "[EMAIL]", text)
    t = sub(EMP_ID, "[ID]", t)
    t = sub(PHONE, "[PHONE]", t)
    t = sub(LONG_DIGITS, "[ID]", t)
    t = sub(HONORIFIC, "[NAME]", t)

    def named(m):
        nonlocal count
        words = m.group(1).split()
        if all(w.lower() in vocab for w in words):
            return m.group(0)
        count += 1
        return m.group(0).replace(m.group(1), "[NAME]")

    t = NAMED.sub(named, t)

    # remaining capitalised words that are not domain vocabulary and not at the start of a sentence
    out, last_end = [], 0
    for m in CAPWORD.finditer(t):
        w = m.group(0)
        before = t[:m.start()].rstrip()
        sentence_start = not before or before[-1] in ".!?:\n"
        base = w[:-2] if w.endswith("'s") else w
        if base.lower() in vocab:
            continue
        if sentence_start and not re.match(r"(?:'s\b|\s+(?:is|was|has|works|seems|left|joined|wants)\b)", t[m.end() - (2 if w.endswith("'s") else 0):]):
            continue
        out.append(t[last_end:m.start()])
        out.append("[NAME]")
        last_end = m.end()
        count += 1
    out.append(t[last_end:])
    t = "".join(out)
    t = re.sub(r"\[NAME\](?:\s+\[NAME\])+", "[NAME]", t)
    return t, count
