"""Mention extraction from free-text narratives and the value / sign / rank agreement metrics.

The agreement metrics follow the definitions Musicco et al. (2026) use, which they attribute to Krishna et al.
(TMLR 2024): for the features a narrative mentions,

- Value Agreement (VA): the stated feature value matches the evidence value;
- Sign Agreement (SA): the stated direction (raises / lowers the estimate) matches the sign of the attribution;
- Rank Agreement (RA): the feature's position in the narrative matches its position in the evidence ranking.
  A rank stated explicitly ("largest", "second") is used if present; otherwise the order of first mention.

Two extractors turn a narrative into mentions, as in Musicco et al.:

- deterministic (`extract_mentions`): finds features by display name, column name and aliases; reads a local window
  (up to the next feature mention in the same sentence) for value labels or numbers, direction cues and ordinal
  rank words. Fully reproducible, but misses attribution stated only implicitly.
- LLM (`llm_extract_mentions`): a judge model lists the features the narrative mentions with value, direction and
  rank. More robust to implicit wording; not deterministic.

Neither extractor uses the explanation's own structured claims, so every system (including this framework's
pipeline) is measured the same way, from the text a reader sees.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from ..agents.textutil import extract_numbers, sentences
from ..agents.verifier import DOWN_CUES, UP_CUES
from ..domains.base import DomainAgent
from ..llm.client import BaseLLM
from ..schemas import EvidencePackage

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}
TOP_WORDS = re.compile(r"\b(?:most important|largest|biggest|main|top|strongest|primary|leading|greatest)\b", re.I)
ORDINAL_WORDS = re.compile(r"\b(first|second|third|fourth|fifth|sixth)\b", re.I)
SIGNED = re.compile(r"(?<![\w.])([+−-])\s?\d")


@dataclass
class Mention:
    feature: str  # model feature name
    in_evidence: bool  # a visible evidence item exists for this feature
    value: str | None = None  # stated value (label or number text), if any
    direction: str | None = None  # increases | decreases | None
    stated_rank: int | None = None
    order: int = 0  # 1-based order of first mention


def _alias_index(domain: DomainAgent) -> list[tuple[str, str]]:
    pairs = []
    for f in domain.features:
        for a in {f.display.lower(), f.name.lower(), *[x.lower() for x in f.aliases]}:
            if a:
                pairs.append((a, f.name))
    return sorted(pairs, key=lambda p: -len(p[0]))


def _find(sentence: str, aliases: list[tuple[str, str]]) -> list[tuple[int, int, str]]:
    low = sentence.lower()
    taken: list[tuple[int, int]] = []
    hits = []
    for alias, feat in aliases:
        for m in re.finditer(rf"\b{re.escape(alias)}\b", low):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            hits.append((m.start(), m.end(), feat))
    return sorted(hits)


def _value_in(window: str, feat: str, domain: DomainAgent) -> str | None:
    spec = domain.spec(feat)
    if spec is None:
        return None
    low = window.lower()
    if spec.categorical:
        for lab in sorted(spec.labels.values(), key=len, reverse=True):
            if re.search(rf"(?<![\w]){re.escape(lab.lower())}(?![\w])", low):
                return lab
        return None
    nums = [raw for _, raw, s, e in extract_numbers(window) if not window[e:e + 1] == "%"
            and "percentage point" not in window[e:e + 20].lower()]
    return nums[0] if nums else None


def _direction_in(window: str) -> str | None:
    up, down = bool(UP_CUES.search(window)), bool(DOWN_CUES.search(window))
    if up and not down:
        return "increases"
    if down and not up:
        return "decreases"
    m = SIGNED.search(window)
    if m:
        return "increases" if m.group(1) == "+" else "decreases"
    return None


def _rank_in(window: str) -> int | None:
    m = ORDINAL_WORDS.search(window)
    if m and TOP_WORDS.search(window[m.end():m.end() + 30]):
        return ORDINALS[m.group(1).lower()]
    if TOP_WORDS.search(window) and not m:
        return 1
    return None


def extract_mentions(text: str, pkg: EvidencePackage, domain: DomainAgent) -> list[Mention]:
    """Deterministic extractor. One Mention per distinct feature (first occurrence wins, later windows may add
    a missing value/direction/rank)."""
    aliases = _alias_index(domain)
    visible = {it.feature for it in pkg.visible}
    out: dict[str, Mention] = {}
    for sent in sentences(text or ""):
        hits = _find(sent, aliases)
        for i, (s, e, feat) in enumerate(hits):
            end = hits[i + 1][0] if i + 1 < len(hits) else len(sent)
            window = sent[s:end]
            # a direction/rank phrase often comes before the name ("The largest factor was overtime")
            prefix = sent[hits[i - 1][1] if i else 0:s]
            m = out.get(feat)
            if m is None:
                m = Mention(feature=feat, in_evidence=feat in visible, order=len(out) + 1)
                out[feat] = m
            m.value = m.value or _value_in(window, feat, domain)
            m.direction = m.direction or _direction_in(window)
            m.stated_rank = m.stated_rank or _rank_in(prefix + " " + window)
    return list(out.values())


def llm_extract_mentions(text: str, pkg: EvidencePackage, domain: DomainAgent, llm: BaseLLM) -> list[Mention]:
    """LLM extractor; falls back to the deterministic extractor if the call fails."""
    feats = [{"feature": f.name, "name": f.display} for f in domain.features]
    system = (
        "You read a short explanation of a model's prediction and list every input factor it mentions. For each, "
        "give the feature id from the list, the value the text states for it (or null), whether the text says it "
        "raised or lowered the estimate (\"increases\", \"decreases\" or null), and the rank the text gives it "
        "(1 = most important, or null if not stated). List them in the order they first appear. "
        'Return JSON {"mentions": [{"feature": "...", "value": "...", "direction": "...", "rank": n}]}.'
    )
    user = json.dumps({"features": feats, "explanation": text}, ensure_ascii=False)
    try:
        out = llm.json("extract", system, user, {"text": text, "package": pkg.model_dump(), "domain": domain.name})
    except Exception:
        return extract_mentions(text, pkg, domain)
    if out.get("fallback"):
        return extract_mentions(text, pkg, domain)
    names = {f.name for f in domain.features}
    visible = {it.feature for it in pkg.visible}
    res, seen = [], set()
    for m in out.get("mentions") or []:
        f = str(m.get("feature") or "")
        if f not in names or f in seen:
            continue
        seen.add(f)
        d = str(m.get("direction") or "").lower()
        r = m.get("rank")
        res.append(Mention(feature=f, in_evidence=f in visible,
                           value=None if m.get("value") in (None, "", "null") else str(m.get("value")),
                           direction=d if d in ("increases", "decreases") else None,
                           stated_rank=int(r) if isinstance(r, (int, float)) and r >= 1 else None,
                           order=len(res) + 1))
    return res


def _value_ok(stated: str, it, domain: DomainAgent) -> bool:
    spec = domain.spec(it.feature)
    if spec is not None and spec.categorical:
        return stated.strip().lower() == it.value_label.strip().lower()
    nums = extract_numbers(stated)
    targets = [abs(float(it.value))] + [v for v, *_ in extract_numbers(it.value_label)]
    return bool(nums) and all(any(abs(v - t) <= max(0.5, 0.01 * t) for t in targets) for v, *_ in nums)


def agreement(mentions: list[Mention], pkg: EvidencePackage, domain: DomainAgent) -> dict:
    """VA, SA, RA over mentioned evidence features, plus coverage of the top factors and unsupported mentions."""
    items = {it.feature: it for it in pkg.visible}
    m_ev = [m for m in mentions if m.in_evidence]
    va = [(_value_ok(m.value, items[m.feature], domain)) for m in m_ev if m.value]
    sa = [(m.direction == items[m.feature].direction) for m in m_ev if m.direction]
    # rank: compare the narrative's position (stated rank, else mention order among evidence mentions) with the
    # feature's position among the mentioned features in the evidence ranking
    by_card = sorted(m_ev, key=lambda m: items[m.feature].rank)
    card_pos = {m.feature: i + 1 for i, m in enumerate(by_card)}
    ra = []
    for i, m in enumerate(sorted(m_ev, key=lambda m: m.order), start=1):
        narr = m.stated_rank if m.stated_rank else i
        target = items[m.feature].rank if m.stated_rank else card_pos[m.feature]
        ra.append(narr == target)
    top = [it.feature for it in pkg.visible[:3]]
    mentioned = {m.feature for m in mentions}

    def r(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "n_mentions": len(mentions), "n_evidence_mentions": len(m_ev),
        "value_agreement": r(va), "n_value": len(va),
        "sign_agreement": r(sa), "n_sign": len(sa),
        "rank_agreement": r(ra), "n_rank": len(ra),
        "top3_coverage": round(sum(f in mentioned for f in top) / len(top), 4) if top else None,
        "unsupported_mentions": sorted(m.feature for m in mentions if not m.in_evidence),
        "value_errors": sum(1 for x in va if not x), "sign_errors": sum(1 for x in sa if not x),
        "rank_errors": sum(1 for x in ra if not x),
        "mentions": [asdict(m) for m in mentions],
    }
