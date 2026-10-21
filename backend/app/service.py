"""Application service: ties the graph, store, PII scrubbing, consent and the auxiliary features together."""
from __future__ import annotations

import csv
import io
import json
import uuid
from collections import Counter, defaultdict

import numpy as np

from .agents import preferences as prefs
from .agents.profile_agent import choose_level
from .config import settings
from .domains.registry import DomainRegistry, default_registry
from .graph import ExplanationGraph
from .intake_extract import extract_fields, extract_pdf_text
from .llm import mock as mock_llm
from .llm.client import BaseLLM, get_llm
from .llm.prompts import GUIDANCE_BY_CODE
from .pii import domain_vocab, scrub
from .pipeline import PipelineConfig
from .provenance import audit_export
from .schemas import (
    ChatIntakeRequest, ExplanationRequest, ExplanationResponse, FeedbackRequest, FeedbackSignalRecord,
    FollowupRequest, IntakeRequest, RefinementRequest, WhatIfRequest, WorkflowTraceRecord,
)
from .store import Store, now

LEVEL_ORDER = ["novice", "intermediate", "expert"]
VERDICT = {"verified": "passed", "fallback": "failed_then_fallback", "escalated": "escalated", "refused": "refused",
           "withheld": "withheld", "concept": "concept", "clarify": "clarify", "error": "error",
           "unverified": "shown_unverified (ablation)"}


class SpamError(ValueError):
    pass


class Service:
    def __init__(self, registry: DomainRegistry | None = None, store: Store | None = None, llm: BaseLLM | None = None,
                 reviewer_llm: BaseLLM | None = None, pipeline: PipelineConfig | None = None):
        self.registry = registry or default_registry()
        self.store = store or Store(settings.db_path)
        self.llm = llm or get_llm()
        mock_llm.DOMAIN_RESOLVER = self.registry.get
        self.graph = ExplanationGraph(self.registry, self.llm, reviewer_llm=reviewer_llm)
        self.vocab = domain_vocab(self.registry)
        self.pipeline = pipeline or PipelineConfig.from_env()

    # ---------------------------------------------------------------- info
    def config(self) -> dict:
        doms = []
        for d in self.registry.list():
            doms.append({"name": d.name, "title": d.title, "ready": d.ready,
                         "data_source": getattr(d, "data_source", "unknown")})
        from .agents.registry import describe

        return {"llm": self.llm.id, "llm_is_cloud": settings.llm_is_cloud and self.llm.name != "mock",
                "graph_engine": self.graph.engine, "domains": doms, "max_attempts": settings.max_attempts,
                "admin_enabled": bool(settings.admin_token), "agents": describe(),
                "pipeline": self.pipeline.as_dict()}

    def domain_cards(self) -> list[dict]:
        """Registered domains with the wording the web app needs (selector, labels, example questions)."""
        return [{"name": d.name, "title": d.title, "description": d.description, "ready": d.ready,
                 "data_source": getattr(d, "data_source", "unknown"), "ui": d.ui_card()} for d in self.registry.list()]

    # ---------------------------------------------------------------- explanations
    def _use_llm(self, consent: bool) -> bool:
        cloud = settings.llm_is_cloud and self.llm.name != "mock"
        return (not cloud) or consent

    def explain(self, req: ExplanationRequest, pipeline: PipelineConfig | None = None,
                forced_intent: str | None = None, parent_trace_id: str | None = None,
                extra_notices: list[str] | None = None) -> ExplanationResponse:
        """Run one request through the graph. `pipeline` (switches) is set by code only, never by HTTP callers."""
        import time

        if req.website:
            raise SpamError("rejected")
        cfg = pipeline or self.pipeline
        query, n_masked = scrub(req.query, self.vocab)
        profile = self.store.profile(req.user_id)
        choice = choose_level(req.level, profile, query, enabled=cfg.profile_agent)
        level = choice["level"]
        use_llm = self._use_llm(req.consent_cloud)
        prompt = self.store.active_prompt("generator")
        pref_notes = choice["notes"]
        request = req.model_dump(exclude={"website"}) | {"query": query, "preferences_text": choice["preferences_text"]}
        if forced_intent:
            request["forced_intent"] = forced_intent

        t0 = time.perf_counter()
        s = self.graph.run(request, level, use_llm, prompt, cfg)
        latency = (time.perf_counter() - t0) * 1000
        status = s.get("status") or ("verified" if s.get("generated") else "error")
        pkg = s.get("package")
        trace_id = uuid.uuid4().hex[:16]
        shown = status in ("verified", "fallback", "unverified")
        profile_entry = {"node": "profile", "agent": "Profile agent", "kind": "deterministic", "authority": "route",
                         "level": level, "source": choice["source"], "notes": len(pref_notes)}
        notices = (s.get("notices") or []) + [n for n in (extra_notices or []) if shown]
        resp = ExplanationResponse(
            trace_id=trace_id,
            status=status,
            domain=s.get("domain"),
            level=level if s.get("domain") else None,
            intent=s.get("intent"),
            message=s.get("message", ""),
            notices=notices,
            disclosure=s.get("disclosure", ""),
            prediction=pkg["prediction"] if pkg else None,
            evidence=pkg,
            explanation=s.get("generated") if status not in ("withheld", "escalated", "refused", "error", "clarify") else None,
            verification=s.get("verification") if shown else None,
            critic=s.get("critic") if shown else None,
            attempts=s.get("attempts", 0),
            prompt_version=prompt.get("version", "") if use_llm else "template",
            llm=self.llm.id if use_llm else "none",
            trace=[profile_entry, *s.get("trace", [])],
            latency_ms=round(latency, 1),
            preferences_applied=[n["text"] for n in pref_notes] if use_llm else [],
            parent_trace_id=parent_trace_id,
            level_source=choice["source"],
            review=s.get("review"),
            policy_report=s.get("policy_report"),
            provenance=s.get("provenance") or [],
            pipeline=cfg.as_dict(),
        )
        violations = []
        for h in s.get("history") or []:
            for v in h["violations"]:
                violations.append({"attempt": h["attempt"], "stage": h["stage"], "gated": h.get("gated", True), **v})
        rec = WorkflowTraceRecord(
            trace_id=trace_id, created_at=now(), user_id=req.user_id, domain=resp.domain, intent=resp.intent,
            level=resp.level, model_variant=req.model_variant, prompt_version=resp.prompt_version, llm=resp.llm,
            status=status, verdict=VERDICT.get(status, status), attempts=resp.attempts, violations=violations,
            withheld_features=[i["feature"] for i in (pkg or {}).get("items", []) if i.get("withheld")],
            pii_masked=n_masked, cloud_consent=req.consent_cloud, query=query,
            request=request, response=resp.model_dump(exclude={"evidence"}) | {"evidence_id": (pkg or {}).get("package_id")},
            latency_ms=resp.latency_ms, parent_trace_id=parent_trace_id,
            policy_report=s.get("policy_report") or {},
            escalation=((pkg or {}).get("prediction") or {}).get("escalation") or {},
        )
        self.store.save_trace(rec)
        return resp

    def audit(self, trace_id: str) -> dict:
        rec = self.store.get_trace(trace_id)
        if rec is None:
            raise KeyError("Unknown trace id")
        return audit_export(rec.model_dump())

    def refine(self, req: RefinementRequest) -> ExplanationResponse:
        rec = self.store.get_trace(req.trace_id)
        if rec is None:
            raise KeyError("Unknown trace id")
        base = ExplanationRequest(**{k: v for k, v in rec.request.items() if k in ExplanationRequest.model_fields})
        cur = rec.level or "intermediate"
        i = LEVEL_ORDER.index(cur)
        if req.action == "simplify":
            new = LEVEL_ORDER[max(0, i - 1)]
        elif req.action == "expand":
            new = LEVEL_ORDER[min(2, i + 1)]
        else:
            new = req.level or cur
        # a descriptive, dated preference note replaces the old "two in a row" counter: the level changes
        # immediately, and the note (not a bare level string) is what future generation prompts see.
        if new != cur:
            prof = self.store.profile(req.user_id)
            kind = {"novice": "short, plain-language", "intermediate": "clear professional-level",
                    "expert": "precise, quantitative, technical"}[new]
            note = {"topic": "detail_level", "text": f"Detail level: asked to switch to {kind} explanations."}
            prof.preferences["notes"] = prefs.merge_preferences(prof.preferences.get("notes", []), note, req.trace_id)
            prof.expertise = new
            self.store.save_profile(prof)
        base.level = new
        base.consent_cloud = req.consent_cloud
        base.user_id = req.user_id
        return self.explain(base)

    def feedback(self, req: FeedbackRequest) -> dict:
        if req.website:
            raise SpamError("rejected")
        rec = self.store.get_trace(req.trace_id)
        if rec is None:
            raise KeyError("Unknown trace id")
        fb = FeedbackSignalRecord(
            feedback_id=uuid.uuid4().hex[:16], trace_id=req.trace_id, created_at=now(), user_id=req.user_id,
            rating=req.rating, understood=req.understood, trust=req.trust, comment=scrub(req.comment, self.vocab)[0],
            prompt_version=rec.prompt_version, level=rec.level,
        )
        self.store.save_feedback(fb)
        return {"ok": True, "feedback_id": fb.feedback_id}

    # ---------------------------------------------------------------- follow-up questions
    def followup(self, req: FollowupRequest) -> ExplanationResponse:
        """A new question about an already-shown explanation.

        The evidence package is rebuilt from the stored (already validated) case details. Evidence building is
        seeded, so for the same case, level and intent this reproduces the original attributions exactly; if the
        question needs different evidence (e.g. it asks about recourse after an explain turn), the recourse search
        runs for that intent. No new case data is used. The draft goes through the same graph as a first-time
        explanation (verify -> critic -> retry -> fallback, escalation, agents), so a follow-up carries the same
        guarantees. Every answered turn can also update this user's descriptive preferences
        (see app/agents/preferences.py).
        """
        if req.website:
            raise SpamError("rejected")
        rec = self.store.get_trace(req.trace_id)
        if rec is None:
            raise KeyError("Unknown trace id")
        if not rec.domain or rec.status in ("refused", "clarify", "error"):
            raise ValueError("That explanation has no evidence to ask a follow-up question about.")
        d = self.registry.get(rec.domain)
        variant = rec.request.get("model_variant", "main")
        raw_instance = rec.request.get("instance") or {}
        assumed_fields = rec.request.get("assumed_fields") or []
        _, errors = d.parse_instance(raw_instance, variant)
        if errors:
            raise ValueError("The original request's details could not be re-validated.")
        use_llm = self._use_llm(req.consent_cloud)
        query, _ = scrub(req.query, self.vocab)
        routing = self.graph.router.route(query, raw_instance, rec.domain, self.llm if use_llm else None)
        forced = "recourse" if routing.intent == "recourse" else "explain"
        if routing.intent == "concept":
            forced = None  # the graph answers concept questions from the glossary
        base = ExplanationRequest(user_id=req.user_id, query=req.query, domain=rec.domain, instance=raw_instance,
                                  level=None, model_variant=variant, consent_cloud=req.consent_cloud,
                                  assumed_fields=assumed_fields)
        resp = self.explain(base, forced_intent=forced, parent_trace_id=req.trace_id,
                            extra_notices=["This answer reuses the case details from the original explanation; "
                                           "no new data was used."])
        if resp.status in ("verified", "fallback") and resp.explanation is not None:
            profile = self.store.profile(req.user_id)
            note = prefs.extract_preference(req.query, resp.explanation.explanation, self.llm if use_llm else None,
                                            d.language)
            if note:
                profile.preferences["notes"] = prefs.merge_preferences(profile.preferences.get("notes", []), note,
                                                                       resp.trace_id)
                self.store.save_profile(profile)
        return resp

    def profile_view(self, user_id: str) -> dict:
        p = self.store.profile(user_id)
        return {"user_id": p.user_id, "expertise": p.expertise, "preferences": p.preferences.get("notes", [])}

    # ---------------------------------------------------------------- what-if sliders
    def whatif(self, req: WhatIfRequest) -> dict:
        d = self.registry.get(req.domain)
        rt = d.runtime(req.model_variant)
        base, errors = d.parse_instance(req.instance, req.model_variant)
        if errors:
            raise ValueError("; ".join(errors.values()))
        new = dict(base)
        for f, raw in req.changes.items():
            if f not in rt.features:
                raise ValueError(f"Unknown field {f}")
            if d.policy.status(f) not in ("actionable", "partial"):
                raise ValueError(f"{f} is not an actionable factor")
            val = d.spec(f).parse(raw)
            lo, hi = rt.ood["ranges"][f]
            if val < lo or val > hi:
                raise ValueError(f"{d.spec(f).display} must stay within the training range {lo:g} to {hi:g}")
            new[f] = val
        out = {}
        for key, inst in (("baseline", base), ("changed", new)):
            p = rt.predict(inst)
            out[key] = {"probability": round(p["probability"], 4), "interval": [round(v, 4) for v in p["interval"]],
                        "band": p["band"], "band_uncertain": p["band_uncertain"], "out_of_distribution": bool(p["ood_reasons"])}
        out["changes"] = [{"feature": f, "display_name": d.spec(f).display, "from": d.readable(f, base[f]),
                           "to": d.readable(f, new[f])} for f in rt.features if new[f] != base[f]]
        out["note"] = ("Model estimates only. Changes are hypothetical and are not recommendations. "
                       "Distance changes stand in for remote or hybrid work, which the data does not record.")
        return out

    def slider_specs(self, domain: str, variant: str = "main") -> list[dict]:
        d = self.registry.get(domain)
        rt = d.runtime(variant)
        out = []
        for f in rt.features:
            st = d.policy.status(f)
            if st not in ("actionable", "partial"):
                continue
            lo, hi = rt.ood["ranges"][f]
            out.append(d.spec(f).schema() | {"actionability": st, "note": d.policy.actionability[f].note,
                                             "train_min": lo, "train_max": hi})
        return out

    # ---------------------------------------------------------------- conversational intake
    def intake(self, req: IntakeRequest) -> dict:
        d = self.registry.get(req.domain)
        rt = d.runtime("main")
        order = [g["feature"] for g in rt.global_importance if g["feature"] in rt.features]
        order += [f for f in rt.features if f not in order]
        given, errors = d.parse_instance(req.instance, allow_missing=True)
        missing = [f for f in order if f not in given and f not in req.skipped]
        if errors:
            return {"complete": False, "errors": errors, "next": [], "remaining": len(missing)}
        if missing and not req.finish:
            nxt = missing[:1]
            return {"complete": False, "errors": {}, "remaining": len(missing),
                    "next": [d.spec(f).schema() | {"why": "Asked early because it has a large influence on the model's estimates."}
                             for f in nxt]}
        defaults = rt.defaults()
        assumed = [f for f in order if f not in given]
        filled = {f: (given[f] if f in given else defaults[f]) for f in rt.features}
        filled = d.consistent_defaults(filled, set(assumed))
        return {"complete": True, "errors": {}, "next": [], "remaining": 0,
                "instance": {f: d.readable(f, v) if d.spec(f).categorical else v for f, v in filled.items()},
                "instance_codes": filled,
                "assumed": [{"feature": f, "display_name": d.spec(f).display, "assumed_value": d.readable(f, filled[f])}
                            for f in assumed]}

    # ---------------------------------------------------------------- chat intake (free text + optional PDF)
    def pdf_intake(self, domain: str, data: bytes, consent_cloud: bool = False) -> dict:
        """Extract whatever case details a PDF states. Never blocks: unparseable or missing fields are just
        left for the chat to ask about next."""
        d = self.registry.get(domain)
        text = extract_pdf_text(data)
        use_llm = self._use_llm(consent_cloud)
        raw = extract_fields(d, text, self.llm if use_llm else None)
        given, errors = d.parse_instance(raw, allow_missing=True)
        rt = d.runtime("main")
        order = [g["feature"] for g in rt.global_importance if g["feature"] in rt.features]
        order += [f for f in rt.features if f not in order]
        missing = [f for f in order if f not in given]
        return {
            "instance": {f: d.readable(f, v) if d.spec(f).categorical else v for f, v in given.items()},
            "found": [{"feature": f, "display_name": d.spec(f).display, "value": d.readable(f, v)} for f, v in given.items()],
            "unparsed": errors,  # text was found for these fields but could not be understood as a valid value
            "missing": [{"feature": f, "display_name": d.spec(f).display} for f in missing],
            "n_chars_extracted": len(text),
        }

    def _chat_question(self, d, spec: dict, understood: list[dict]) -> str:
        lead = ""
        if understood:
            names = ", ".join(u["display_name"] for u in understood)
            lead = f"Got it — I noted {names}. "
        hint = ""
        if spec.get("options"):
            hint = " (" + ", ".join(o["label"] for o in spec["options"]) + ")"
        return f"{lead}What is their {spec['display'].lower()}?{hint}"

    def chat_intake(self, req: ChatIntakeRequest) -> dict:
        """One turn of chat-style intake: understand whatever the free-text message states, merge it with what is
        already known, and either ask about the next most influential missing field or report completion (same
        completion payload as `intake()`, which this delegates to once nothing more is missing or `finish` is set)."""
        d = self.registry.get(req.domain)
        use_llm = self._use_llm(req.consent_cloud)
        message, _ = scrub(req.message, self.vocab) if req.message else ("", 0)
        found = extract_fields(d, message, self.llm if use_llm else None) if message.strip() else {}
        merged = dict(req.instance)
        newly_found = {k: v for k, v in found.items() if k not in merged or merged[k] in (None, "")}
        merged |= newly_found
        given, errors = d.parse_instance(merged, allow_missing=True)
        clean_instance = {k: v for k, v in merged.items() if k in given}
        understood = [{"feature": k, "display_name": d.spec(k).display if d.spec(k) else k, "value": d.readable(k, given[k])}
                      for k in newly_found if k in given]
        result = self.intake(IntakeRequest(domain=req.domain, instance=clean_instance, skipped=req.skipped, finish=req.finish))
        unparsed = [k for k in newly_found if k not in given]
        if result["complete"]:
            reply = ("Thanks — I have everything I need." if understood else "All set — here is what I have.")
        elif result["next"]:
            reply = self._chat_question(d, result["next"][0], understood)
        else:
            reply = d.language.chat_more_details
        return result | {"reply": reply, "understood": understood, "unparsed": unparsed}

    # ---------------------------------------------------------------- batch
    def batch_upload(self, domain: str, data: bytes) -> dict:
        d = self.registry.get(domain)
        rt = d.runtime("main")
        if len(data) > 2_000_000:
            raise ValueError("File is larger than 2 MB")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError("File must be UTF-8 CSV") from None
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ValueError("CSV has no header row")
        missing_cols = [f for f in rt.features if f not in reader.fieldnames]
        if missing_cols:
            raise ValueError("CSV is missing columns: " + ", ".join(missing_cols))
        rows, errors = [], []
        for i, row in enumerate(reader, start=1):
            if i > settings.batch_max_rows:
                raise ValueError(f"CSV has more than {settings.batch_max_rows} rows")
            inst, errs = d.parse_instance(row)
            if errs:
                errors.append({"row": i, "errors": list(errs.values())[:3]})
                continue
            ref = (row.get(d.language.reference_column) or "").strip()[:20] if d.language.reference_column else ""
            rows.append({"row": i, "reference": ref, "instance": {f: inst[f] for f in rt.features}})
        if not rows:
            return {"batch_id": None, "n_rows": len(errors), "n_valid": 0, "errors": errors[:50], "rows": []}
        X = rt.frame([r["instance"] for r in rows])
        p = rt.proba(X)
        lo, hi = rt.interval(X)
        iso = rt.ood["iso"].score_samples(X.values) < rt.ood["score_threshold"]
        results = []
        for r, pi, a, b, o in zip(rows, p, lo, hi, iso):
            a, b = min(a, pi), max(b, pi)
            inst = r["instance"]
            out_range = any(inst[f] < rt.ood["ranges"][f][0] or inst[f] > rt.ood["ranges"][f][1] for f in rt.features)
            results.append({"row": r["row"], "reference": r["reference"], "probability": round(float(pi), 4),
                            "interval": [round(float(a), 4), round(float(b), 4)], "band": rt.band(float(pi)),
                            "needs_review": bool(rt.band(float(a)) != rt.band(float(b)) or o or out_range),
                            "explanation_status": "not requested"})
        results.sort(key=lambda r: -r["probability"])
        batch_id = uuid.uuid4().hex[:12]
        self.store.save_batch(batch_id, domain, rows, results)
        bands = Counter(r["band"] for r in results)
        return {"batch_id": batch_id, "n_rows": len(rows) + len(errors), "n_valid": len(rows), "errors": errors[:50],
                "summary": {"bands": {b: bands.get(b, 0) for b in ["low", "medium", "high"]},
                            "mean_estimate": round(float(np.mean(p)), 4),
                            "needs_review": sum(r["needs_review"] for r in results)},
                "rows": results}

    def batch_explain(self, batch_id: str, row: int, level: str, consent: bool, user_id: str) -> ExplanationResponse:
        b = self.store.get_batch(batch_id)
        if b is None:
            raise KeyError("Unknown batch")
        r = next((x for x in b["rows"] if x["row"] == row), None)
        if r is None:
            raise KeyError("Unknown row")
        d = self.registry.get(b["domain"])
        inst = {f: (d.readable(f, v) if d.spec(f).categorical else v) for f, v in r["instance"].items()}
        resp = self.explain(ExplanationRequest(user_id=user_id, query=d.language.batch_question,
                                               domain=b["domain"], instance=inst, level=level, consent_cloud=consent,
                                               batch_mode=True))
        for res in b["results"]:
            if res["row"] == row:
                res["explanation_status"] = resp.status
                res["trace_id"] = resp.trace_id
        self.store.save_batch(batch_id, b["domain"], b["rows"], b["results"])
        return resp

    # ---------------------------------------------------------------- dashboards
    def manifest(self, domain: str) -> dict:
        d = self.registry.get(domain)
        return getattr(d, "manifest", {}) or {}

    def aggregate(self, domain: str) -> dict:
        m = self.manifest(domain)
        rt = self.registry.get(domain).runtime("main")
        d = self.registry.get(domain)
        imp = [{"feature": g["feature"], "display_name": d.spec(g["feature"]).display, "mean_abs": g["mean_abs"],
                "mean": g["mean"], "actionability": d.policy.status(g["feature"])} for g in rt.global_importance]
        return {"aggregate": m.get("aggregate", {}), "global_importance": imp,
                "dataset": m.get("dataset", {}), "threshold": rt.threshold,
                "band_edges": m.get("main", {}).get("band_edges", {})}

    def fairness(self, domain: str) -> dict:
        m = self.manifest(domain)
        keep = ["model_kind", "version", "n_train", "n_test", "cost_matrix", "threshold", "band_edges", "cv", "test",
                "headline", "reliability_test", "reliability_oof_all", "uncertainty", "calibration", "class_weighting"]
        d = self.registry.get(domain)
        fair = json.loads(json.dumps(m.get("fairness", {})))
        for r in fair.get("proxy_check", {}).values():
            for s_ in r.get("top_single_features", []):
                spec = d.spec(s_["feature"])
                s_["display_name"] = spec.display if spec else s_["feature"]
        return {"dataset": m.get("dataset", {}), "fairness": fair,
                "models": {v: {k: m.get(v, {}).get(k) for k in keep} for v in ("main", "audit") if v in m}}

    def glossary(self, domain: str) -> list[dict]:
        return [g.model_dump() for g in self.registry.get(domain).glossary.values()]

    def examples(self, domain: str, n: int = 20) -> list[dict]:
        d = self.registry.get(domain)
        rt = d.runtime("main")
        out = []
        for idx, row in rt.examples.head(n).iterrows():
            inst = {f: (d.readable(f, row[f]) if d.spec(f).categorical else float(row[f])) for f in rt.features}
            out.append({"id": str(idx), "instance": inst})
        return out

    # ---------------------------------------------------------------- admin
    def feedback_summary(self) -> dict:
        fbs = self.store.feedback()
        by: dict[tuple, list] = defaultdict(list)
        for f in fbs:
            by[(f.prompt_version, f.level)].append(f)
        rows = []
        for (pv, lvl), items in sorted(by.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            und = [f.understood for f in items if f.understood is not None]
            tr = [f.trust for f in items if f.trust is not None]
            rows.append({"prompt_version": pv, "level": lvl, "n": len(items),
                         "mean_rating": round(sum(f.rating for f in items) / len(items), 2),
                         "understood_rate": round(sum(und) / len(und), 2) if und else None,
                         "mean_trust": round(sum(tr) / len(tr), 2) if tr else None})
        traces = self.store.trace_records(2000)
        verdicts = Counter(t.verdict for t in traces)
        codes = Counter(v["code"] for t in traces for v in t.violations if v.get("severity") == "critical")
        return {"feedback": rows, "verdicts": dict(verdicts), "violation_codes": dict(codes.most_common(15)),
                "n_traces": len(traces), "n_feedback": len(fbs)}

    def propose_prompt(self) -> dict:
        active = self.store.active_prompt("generator")
        summary = self.feedback_summary()
        codes = [c for c in summary["violation_codes"] if c in GUIDANCE_BY_CODE][:3]
        guidance = list(active.get("guidance", []))
        added = [GUIDANCE_BY_CODE[c] for c in codes if GUIDANCE_BY_CODE[c] not in guidance]
        low_novice = [r for r in summary["feedback"] if r["level"] == "novice" and r["n"] >= 5 and r["mean_rating"] < 3]
        if low_novice:
            added.append("For novice readers, use at most two short sentences per factor and no percentages beyond the risk.")
        if not added:
            raise ValueError("No change suggested: no frequent violations or low ratings in the log.")
        notes = (f"Based on {summary['n_traces']} traces and {summary['n_feedback']} ratings. Most frequent violations: "
                 + ", ".join(f"{c} ({summary['violation_codes'][c]})" for c in codes))
        return self.store.add_candidate("generator", active["template"], guidance + added, notes)

    def approve_prompt(self, prompt_id: int, by: str) -> dict:
        return self.store.approve(prompt_id, by)
