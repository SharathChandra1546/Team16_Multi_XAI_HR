# Phase 1: verified, policy-constrained explanation pipeline

This document describes what Phase 1 added to the system, where each piece lives, and how it is tested. It is the
build half of the paper plan. Phase 2 (baselines, finance and healthcare domains) and Phase 3 (experiments, human
labels, paper) use what is described here.

## Contribution this supports

A domain-agnostic framework for LLM explanations that are:

1. checked against the model's evidence by a deterministic, claim-level gate;
2. constrained by domain policy written as code;
3. escalated to human review when the case is unreliable, with a logged score that supports a coverage-risk curve.

Audience adaptation, a planner/generator/reviewer loop, a reviewer from another model family, an immutable evidence
card and trace logging are **not** claimed as novel (Musicco et al. 2026 has all of them). Phase 1 builds them so
the system can be compared against that design, not to claim them.

## What was added

### A. Novelty features

| Item | What it does | Where |
|---|---|---|
| A1 Claim-level verifier, hardened | Besides the structured fields an LLM fills in, the verifier now reads each claim's own wording: magnitude words ("strongly", "(weak)"), ordinal rank words ("second largest"), comparisons between two cited factors ("X more than Y"), the band named in a risk sentence, and the headline risk percentage (ignoring percentages qualified as a range, average or threshold). Recourse sentences must state the option's re-predicted risk. | `app/agents/verifier.py` |
| A2 Verifier as inline gate | The verifier decides the branch before any LLM agent runs. LLM agents can advise but cannot turn a rejection into an approval. Tested with an LLM that scores every draft 5/5 and calls every draft faithful while the generator is always wrong: the output is the verified template, never the faulty text. | `app/graph.py`, `tests/test_phase1_pipeline.py::test_llm_agents_cannot_override_the_verifier` |
| A2 Policy as code + violation report | Every violation is put in a category (fidelity, policy, level, quality). Each response carries a report: violations in the first draft, across all attempts, and in the text actually shown. | `app/policy_report.py`, `ExplanationResponse.policy_report` |
| A3 Escalation score | One score per case in [0, 1] from two parts: how much of the uncertainty interval lies outside the estimate's band, and how unusual the case is (isolation-forest score scaled between the training median and the detector threshold; 1 if any feature is out of range). `rule` mode reproduces the original rule exactly; `score` mode escalates above a threshold. The score is logged for every case, escalated or not. | `app/escalation.py`, `PredictionResult.escalation` |
| A4 Verified recourse | Each option records its risk drop and is re-checked independently (rebuild the actions, re-apply, re-validate, re-predict); an option that fails the re-check is dropped. | `app/recourse.py` |
| A5 Evidence provenance | Every sentence of the shown text is labelled `claim` (with its evidence ids), `note` (a fixed system sentence), `disclosure`, or `unlinked`. Admins can export one archivable audit document per trace. | `app/provenance.py`, `GET /api/admin/traces/{id}/audit` |
| A6 Verified reader levels | Each level's constraints are checkable: banned terms, word budget, a readability floor stored in the plan, and a new `level_numbers` rule (no contribution figures or raw probabilities at novice level). | `app/agents/planner.py`, `app/agents/critic.py` |
| A7 Fault taxonomy and detector-independent faults | 15 fault types, each with a plain-language description written without reference to any rule. An LLM fault generator receives only that description, the evidence and the correct text; it never sees the verifier's patterns or codes. Template faults remain as an upper bound. Three new template faults exercise the new checks. | `app/faults.py`, `app/faults_llm.py`, `eval/run_faults.py --faults llm` |
| A8/A9 Domain plug-in contract | All domain wording moved out of generic modules into `DomainLanguage` (subject, reader, outcome, decision kind, protected attributes, outcome terms for causal patterns, fault sentences, data notes). `validate_domain()` checks a domain package against the contract. The generator prompt is now domain-neutral (`gen-v2`); databases still on the untouched HR-only `gen-v1` seed are migrated automatically, keeping their guidance. | `app/domains/base.py`, `app/domains/hr/policy.py`, `app/llm/prompts.py`, `app/store.py` |
| A9 Transfer effort | Counts lines changed outside `app/domains/<d>/` between two revisions. Run it when finance and healthcare are added. | `scripts/transfer_effort.py` |

### B. Agents and switches

| Agent | Kind | Authority | Switch |
|---|---|---|---|
| Profile agent | deterministic | route | `profile_agent` |
| Router | hybrid | route | — |
| Evidence builder | deterministic | produce | — |
| Escalation gate | deterministic | gate | `escalation` |
| Planner | deterministic | produce | — |
| LLM planner | llm | advise | `llm_planner` |
| Generator | llm | produce | — |
| Verifier | deterministic | gate | `verifier` |
| Critic (rules + LLM score) | hybrid | gate | `policy_rules`, `level_rules`, `llm_critic` |
| Repair | deterministic | produce | `repair` |
| Reviewer | llm | advise | `reviewer` |
| Fallback | deterministic | produce | `fallback` |

- No LLM agent has gate authority (tested).
- The **LLM planner** may only narrow and reorder the deterministic plan's evidence and add up to three style notes. Unknown, withheld or out-of-level ids are rejected, required ids are always kept, and notes that would loosen a rule (or match any of the domain's policy rules) are rejected. Everything rejected is logged.
- The **reviewer** should come from a different model family (`REVIEWER_PROVIDER`). The trace records whether it shares a family with the generator.
- The **profile agent** picks the level: explicit request, then the question's own cues ("in simple terms", "exact numbers"), then the stored profile, then intermediate.
- Every trace entry names the agent, whether it is deterministic or an LLM, its authority and, for LLM agents, the model id.

**Switches.** `PipelineConfig` turns each component on or off. A switched-off check still runs in **shadow mode**: its verdict is recorded but does not gate. An output shown with a shadow violation gets the status `unverified`. This is how ablations report errors that reached the user. Named configurations are in `app.pipeline.ABLATIONS`. Safety switches are never exposed through the HTTP API; only code can set them.

### C. Fixes

| Fix | Detail |
|---|---|
| CFS v2 | Text-level violations now count: a violation is attached to the claim whose text contains its span, else its non-claim sentence becomes an extra failed unit; each missing required citation is an extra failed unit. CFS = 1.0 exactly when the verifier passes (tested). An explanation with no claims scores 0.0 (the old doc said 1.0; the code already returned 0.0). The weighted variant could exceed 1.0 in v1; fixed. See `docs/FIDELITY_METRIC.md`. |
| Training-split statistics | Recourse bounds (`domain_stats`) are computed from the training split only, and recorded as such in the artefact and manifest. |
| Escalation rate in the manifest | Uses the configured interval percentiles, the same ones the runtime uses. |
| OOD median | Stored in new artefacts (old artefacts estimate it from the background sample). |
| Fairness rule | One rule for every attribute and model: groups under 30 rows are listed but excluded from the disparate-impact ratio and the TPR/FPR gaps, at the model's own threshold. Excluded groups and the threshold are recorded. |
| Follow-up | Now runs through the same graph as a first explanation (so switches, escalation and agents apply). The docstring states honestly that evidence is rebuilt from stored case details, reproducing the original attributions because evidence building is seeded. |
| Disclosure | Described honestly in code as attached by construction; the check is a guard against a code change, not an independent test of generated text. |
| Shapley stability | Stability and method-agreement warnings are logged per request and counted by `eval/run_phase1.py`. `SHAP_PERMUTATIONS` raises the permutation count if the warning rate is high. |
| Prompt versions | Candidate version numbers no longer collide with seeded versions. |

## How to run

```bash
cd backend
python scripts/train_hr.py                      # real IBM data in data/, or --standin for software tests
python -m pytest -q                             # full suite
python eval/run_faults.py --n 40 --faults both --reviewer   # template (upper bound) + LLM-written faults
python eval/run_phase1.py --n 30 --reviewer     # raw LLM errors, pipeline outcomes, warnings, escalation scores
python scripts/transfer_effort.py --domain finance --base <commit before finance>
```

Environment variables added: `REVIEWER_PROVIDER`, `REVIEWER_MODEL`, `FAULT_PROVIDER`, `FAULT_MODEL`,
`SHAP_PERMUTATIONS`, `ESCALATION_MODE`, `ESCALATION_THRESHOLD`, and `PIPELINE_<SWITCH>` (e.g.
`PIPELINE_LLM_PLANNER=1`, `PIPELINE_REVIEWER=1`).

## What is still open (Phase 2 and 3)

- The ablation **runner** and the baselines (direct SHAP-to-LLM, Explingo-style, Musicco-style B1–B4, critic loop)
  are Phase 2. The switches and the `ABLATIONS` table they need exist.
- Finance and healthcare packages are Phase 2. The contract (`validate_domain`) and `transfer_effort.py` are ready.
- The frontend still sends `domain: "hr"` and the intake request schemas default to `"hr"`; making the UI multi-domain
  is Phase 2.
- All numbers produced in this build environment came from the mock LLM and synthetic stand-in data. They test the
  software only.
