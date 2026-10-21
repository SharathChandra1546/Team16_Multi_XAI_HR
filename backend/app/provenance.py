"""Evidence provenance: which evidence ids support each sentence of the shown explanation, plus an audit export.

Each sentence of the final text gets one entry:
- kind "claim": the sentence is (or contains) a claim; evidence_ids are the ids that claim cites
- kind "disclosure": the mandatory disclosure (attached by construction, see graph.n_finalize)
- kind "note": a fixed sentence the system adds (limitations, withheld-factor note, no-recourse note)
- kind "unlinked": any other sentence; it states no checked fact (the verifier rejects sentences that name a
  factor or a number without a claim), so it carries no evidence id

The audit export bundles one trace record with the evidence package, the plan, every attempt's violations,
provenance and the agent trace into a single JSON document a reviewer can archive.
"""
from __future__ import annotations

from typing import Any

from .agents.fallback import template_notes
from .agents.textutil import norm, sentences
from .domains.base import DomainAgent
from .schemas import EvidencePackage, ExplanationPlan, GeneratedExplanation


def _fixed_sentences(pkg: EvidencePackage | None, domain: DomainAgent, plan: ExplanationPlan | None) -> set[str]:
    fixed = {domain.policy.withheld_note}
    if pkg is not None:
        fixed |= set(pkg.limitations)
        if pkg.recourse_note:
            fixed.add(pkg.recourse_note)
        if plan is not None:
            fixed |= set(template_notes(pkg, plan, domain))
    out = set()
    for f in fixed:
        out |= {norm(s) for s in sentences(f)}
    return out


def build_provenance(gen: GeneratedExplanation | dict | None, pkg: EvidencePackage | dict | None,
                     domain: DomainAgent, disclosure: str = "",
                     plan: ExplanationPlan | dict | None = None) -> list[dict[str, Any]]:
    if gen is None:
        return []
    gen = gen if isinstance(gen, GeneratedExplanation) else GeneratedExplanation(**gen)
    pkg = pkg if (pkg is None or isinstance(pkg, EvidencePackage)) else EvidencePackage(**pkg)
    text = gen.explanation or ""
    disc = {norm(s) for s in sentences(disclosure)} if disclosure else set()
    plan = plan if (plan is None or isinstance(plan, ExplanationPlan)) else ExplanationPlan(**plan)
    fixed = _fixed_sentences(pkg, domain, plan)
    claims = [(i, norm(c.text), c) for i, c in enumerate(gen.claims) if c.text]
    out = []
    for idx, sent in enumerate(sentences(text)):
        ns = norm(sent)
        hits = [(i, c) for i, ct, c in claims if ct and (ct in ns or ns in ct)]
        if hits:
            ids = sorted({e for _, c in hits for e in c.evidence_ids if e})
            out.append({"index": idx, "text": sent, "kind": "claim", "claim_indices": [i for i, _ in hits],
                        "evidence_ids": ids})
        elif ns in disc:
            out.append({"index": idx, "text": sent, "kind": "disclosure", "claim_indices": [], "evidence_ids": []})
        elif ns in fixed:
            out.append({"index": idx, "text": sent, "kind": "note", "claim_indices": [], "evidence_ids": []})
        else:
            out.append({"index": idx, "text": sent, "kind": "unlinked", "claim_indices": [], "evidence_ids": []})
    return out


def audit_export(record: dict) -> dict[str, Any]:
    """One archivable JSON document for a stored trace (WorkflowTraceRecord as a dict)."""
    resp = record.get("response", {}) or {}
    return {
        "format": "mxai-audit-v1",
        "trace_id": record.get("trace_id"),
        "created_at": record.get("created_at"),
        "domain": record.get("domain"),
        "intent": record.get("intent"),
        "level": record.get("level"),
        "status": record.get("status"),
        "verdict": record.get("verdict"),
        "model": {"variant": record.get("model_variant"), "version": (resp.get("prediction") or {}).get("model_version"),
                  "llm": record.get("llm"), "prompt_version": record.get("prompt_version")},
        "query": record.get("query"),
        "evidence_id": resp.get("evidence_id"),
        "prediction": resp.get("prediction"),
        "escalation": record.get("escalation") or (resp.get("prediction") or {}).get("escalation"),
        "explanation": (resp.get("explanation") or {}).get("explanation"),
        "claims": (resp.get("explanation") or {}).get("claims", []),
        "provenance": resp.get("provenance", []),
        "verification": resp.get("verification"),
        "critic": resp.get("critic"),
        "review": resp.get("review"),
        "violations_by_attempt": record.get("violations", []),
        "policy_report": record.get("policy_report") or resp.get("policy_report"),
        "withheld_features": record.get("withheld_features", []),
        "pipeline": resp.get("pipeline", {}),
        "agent_trace": resp.get("trace", []),
        "pii_masked": record.get("pii_masked"),
        "cloud_consent": record.get("cloud_consent"),
    }
