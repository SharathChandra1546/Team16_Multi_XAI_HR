# Requirements, decisions and acceptance evidence

Scope: HR attrition only (finance deferred). Perspective: HR manager. Decisions agreed before the build:

- **Escalation:** "route to human review" replaces the explanation when the risk band is uncertain (the bootstrap range spans two bands) or the profile is out of distribution. High-risk cases get the verified explanation plus a mandatory review notice.
- **Protected attributes:** age, gender and marital status are excluded from the main model. They are kept for the fairness audit and for one audit model used for comparison.
- **Employee-facing view:** dropped. The levels are novice, intermediate and expert, all for managers.
- **Defaults:**
  - 5:1 cost matrix (missed leaver : false flag)
  - Platt calibration
  - a recourse option must lower the estimate by at least 5 points or move it to a lower band
  - column names allowed only at expert level
  - disclosure appended from a fixed template
  - each fault maps to exactly one violation code

| # | Requirement | Where | Acceptance evidence |
|---|---|---|---|
| 1 | Validated HR model pipeline | `scripts/fetch_hr.py`, `scripts/train_hr.py`, `app/ml/training.py` | `models/hr_manifest.json` → `headline` and `test.ci95`: ROC-AUC, PR-AUC, recall and cost per employee with 95% bootstrap CIs. Also repeated stratified CV and accuracy as a secondary metric. `eval/export_tables.py` prints them. |
| 2 | HrAgent in DomainRegistry; four intents; no HR branches in the graph | `app/domains/hr/agent.py`, `app/domains/registry.py`, `app/agents/router.py`, `app/graph.py` | `tests/test_graph_domain_agnostic.py`: a second toy domain runs through the same graph and verifier unchanged, and a source scan finds no HR-specific code in the generic modules. |
| 3 | HR DomainPolicy: sensitive features, actionability table, distance partially actionable | `app/domains/hr/policy.py` | `test_main_model_excludes_protected`, `test_protected_withheld_below_expert`, `test_distance_is_partial_with_assumption`. The evidence chart greys out withheld items (`EvidenceChart.jsx`). |
| 4 | Actionable-only recourse; re-predicted; tenure and age fixed; bounded salary | `app/recourse.py`, `HrAgent.recourse_actions` | `tests/test_recourse.py`: every option re-checked by `recheck_option`, immutables never change, salary at most +20% and within the job level's observed range. |
| 5 | HR critic rules: decision_wording, protected_attribute_inference, blame_wording, prediction_as_fact | `app/domains/hr/policy.py`, `app/agents/critic.py` | `tests/test_critic_rules.py`: at least 5 failing and 5 passing examples per rule (6 to 7 of each). |
| 6 | Decision-use guardrail, 100% refusal on a labelled adversarial set | `app/agents/router.py` + policy patterns (+ optional LLM classifier) | `eval/data/guardrail_cases.json` (70 refuse incl. rephrasings, 45 allow). `tests/test_guardrail.py`: 70/70 refused, 0/45 false refusals. |
| 7 | HR glossary, citable IDs, no raw codes | `app/domains/hr/spec.py`, verifier `raw_code_leak` | Every feature has a `G-*` entry (`test_every_feature_has_glossary_entry`). `test_no_raw_codes_in_text` and `test_raw_code_rule` cover `OverTime=Yes`, `Travel_Frequently` and coded ratings. |
| 8 | Mandatory disclosure, enforced | `DomainPolicy.disclosure`, `graph.n_finalize`, `critic.check_disclosure` | `test_disclosure_on_every_explanation`, `test_disclosure_check_catches_missing`. |
| 9 | HR fault injection and Phase I run | `app/faults.py`, `eval/run_faults.py`, `eval/run_phase1.py` | Detection rate per fault type, false-rejection rate on clean text, optional LLM-reviewer column. `--domains hr,finance` produces the side-by-side table once finance exists (deferred). |
| 10 | Fairness audit | `app/ml/fairness.py`, Fairness page | Error and flag rates by gender, age band and marital status. Disparate-impact ratio with the four-fifths rule. Proxy check (all-feature and single-feature AUC). Main vs audit model. |
| 11 | Risk bands with uncertainty; reliability curve | `app/ml/training.py`, `app/ml/runtime.py` | Bands low/medium/high from calibrated probability. Bootstrap ensemble range. `models/reliability_hr.png` for the thesis; also drawn on the Fairness page. |
| 12 | Audience views via ExplanationPlan | `app/agents/planner.py` | `test_levels_differ`: word budgets 90/160/260 and different banned-term lists; mean words rise with level. |
| 13 | Audit log and feedback capture | `app/store.py`, `WorkflowTraceRecord`, `FeedbackSignalRecord` | `test_trace_and_feedback_records`: each request writes prompt version and verdict, and ratings are stored with the prompt version. |
| 14 | Human-review escalation | `graph.c_gate`, `n_escalate` | `test_escalation_replaces_explanation`, `test_ood_input_escalates`. |
| 15 | What-if sliders, actionable only, live | `service.whatif`, `WhatIf.jsx` | `test_whatif_rejects_immutable`; browser test confirms live updates. |
| 16 | Batch CSV, ranked, per-row verified, failures withheld | `service.batch_upload`, `batch_explain` (`batch_mode`) | `test_batch_upload`; graph routes exhausted retries to `withheld` in batch mode. |
| 17 | PII scrubbing; cloud mode needs consent | `app/pii.py`, `Service._use_llm` | `test_pii_masks_names_ids_contacts`, `test_cloud_mode_requires_consent`. The stored query is masked. |
| 18 | Conversational intake, most influential first, defaults flagged | `service.intake`, Intake page | `test_intake_orders_by_influence_and_flags_defaults`: asks in global-importance order; skipped fields become assumptions listed in the result and kept consistent (role within department, years within tenure). |
| 19 | Cross-domain consistency test | `tests/test_graph_domain_agnostic.py` | The same verifier and critic run on HR and on the toy domain. The finance comparison is deferred until finance exists. |
| 20 | Chat-style intake with PDF parsing; missing fields asked for in chat | `app/intake_extract.py`, `service.chat_intake`, `service.pdf_intake`, `ChatIntake.jsx` | `extract_fields` combines a regex-over-feature-aliases pass with an optional LLM JSON slot-filler (same cloud-consent gate as `/explain`); the chat endpoint asks for exactly the fields still missing, in influence order, and the page offers "skip" and "finish with typical values" so it always terminates. |
| 21 | Verified conversational follow-up on a shown explanation, used to refine preferences | `service.followup`, `FollowupRequest`, `FollowupChat.jsx` | `followup()` re-derives the same Evidence Package from the trace's stored raw instance (`build_package` is deterministic for a fixed seed) and runs the identical generate -> verify -> critic -> retry -> fallback loop as a first-time explanation, so a follow-up answer carries the same fidelity guarantee; each answer records `preferences_applied`. |
| 22 | Descriptive, contradiction-aware user preferences appended to every generator prompt | `app/agents/preferences.py`, `service.refine`, `service.followup`, `render_generator` (`app/llm/prompts.py`) | One dated, topic-scoped note per topic (e.g. `detail_level`, capped at `MAX_NOTES`); a new note on a topic replaces the older one on that topic rather than accumulating, so the newest statement always wins on contradiction; `profile_view` and the "What this system has learned" panel show the live set; a smoke test confirms a note written by `refine()` is picked up by a later `explain()` call for the same user end to end. |
| 23 | Formalized Claim Fidelity Score (CFS) and structured, violation-specific repair instructions | `app/metrics/fidelity.py`, `docs/FIDELITY_METRIC.md`, `app/agents/repair.py`, `app/graph.py::n_generate` | `claim_fidelity_score(n_claims, violations)` returns a bounded, deterministic, judge-independent score plus a weighted variant and the clean/minor/critical counts, surfaced as `VerificationResult.fidelity`; `build_repair_instructions` turns a prior attempt's violations into one instruction per violation code, pulling the correct value straight from the evidence package when the violation names an evidence id, and `n_generate` passes those instructions (not a raw violation dump) into the next retry. |

## Known limits to state in the paper

- **Template-based fault injection is an upper bound.** Report detection on LLM-written base texts (`--base llm`) as the main result.
- **The verifier both detects errors and defines fidelity.** Add a small human-labelled sample and the LLM-reviewer comparison.
- **Name masking is heuristic.** Capitalised words that are not domain vocabulary are masked. Lower-case names are not.
- **The guardrail is pattern-based, with an optional LLM second layer.** The 100% figure holds for the labelled set, not for every possible rephrasing.
- **The dataset is fictional and small** (237 leavers). Confidence intervals are wide; report them.
- **The Claim Fidelity Score is produced by the same verifier that decides pass/fail.** It is a deterministic, reproducible count of clean vs. flagged claims, not an independent judge; back it with the LLM-reviewer comparison and a human-labelled sample before citing it as the sole fidelity measure (see `docs/FIDELITY_METRIC.md`).
- **Preference notes are extracted by a small rule-based/LLM classifier over the follow-up question and level changes**, not a learned user model; they change what is appended to the next prompt, not the model or the evidence.
- **Deliberately out of scope for this build pass** (feasible only with new experiments or labelled data, not as pure code changes): a fail-fast meta-classifier on drafts, counterfactual fairness via the audit model, proxy-leakage auditing of the explanation layer, an adversarial-robustness red-team study, and the human-labelled gold set / LLM-reviewer agreement study. These remain good next steps for strengthening the paper's research contribution.
