# Phase 2: baselines, ablations and three domains

Phase 2 builds the comparison the paper needs: every baseline and every ablation runs on the same evidence and is
measured the same way, in three domains. Phase 3 runs it with real LLMs and real data and adds human labels.

## 1. Baselines (`app/baselines.py`)

All systems receive the same evidence package for a case (same model output, Shapley values, recourse options).

| Name | What it reproduces | Checker | Budget |
|---|---|---|---|
| `direct` | Bello-style SHAP-to-LLM translation for a reader level | none | 1 call |
| `b1` | Musicco et al. B1: one generic narrative from the explanation card | none | 1 call |
| `b2` | Musicco et al. B2: B1 + reader role description | none | 1 call |
| `b3` | Musicco et al. B3: B1 + reviewer (1–5 score, accept/revise, comment) | LLM | 2 revisions |
| `b4` | Musicco et al. B4: B3 + format verifier (numbers present, no list/JSON, ≥2 sentences) | LLM + format rules | 2 revisions |
| `explingo` | Zytek et al.: narrator with 3 few-shot exemplars + grader (accuracy, completeness, fluency; conciseness formula) | LLM | 2 rewrites |
| `critic_loop` | He & Martens Critic Design: narrator + faithfulness critic | LLM | 3 rounds |
| `musicco` | Musicco et al. full pipeline: Framer → Narrator → routing Reviewer | LLM | 2 revision rounds |

These are re-implementations from the papers' descriptions, not the authors' code. State this in the paper. In
particular, Explingo's conciseness formula is approximated, and the exemplars are verified templates from three
fixed cases of the same domain. Musicco et al. released code (github.com/sisinflab/XstrAI); adapting it to the same
evidence would be a stronger comparison if time allows.

## 2. Ablations

`ours:<config>` runs this framework through the full graph with a `PipelineConfig` from `app.pipeline.ABLATIONS`:
`full`, `no_verifier`, `no_policy_rules`, `no_critic`, `no_repair`, `no_escalation`, `no_fallback`,
`with_llm_agents`. Switched-off checks run in shadow mode, so the text shown is still measured.

## 3. One measurement for every system (`app/metrics/measure.py`, `app/metrics/agreement.py`)

The comparison never uses what a system says about itself (including this framework's claims). It measures the
shown text against the shared evidence:

- **Value / sign / rank agreement** (VA, SA, RA) and top-3 coverage, defined as in Musicco et al. (after Krishna et
  al.), with a deterministic extractor and, with `--judge`, an LLM extractor. Explain answers only; recourse answers
  state hypothetical values.
- **Unsupported mentions**: factors that are not in the visible evidence (withheld or invented).
- **Text checks**: unsupported numbers, causal wording, raw codes.
- **Policy checks**: each domain's wording rules.
- **Level checks**: reading ease, word count, level-rule violations.
- **Error-free**: none of the fidelity or policy problems above.
- **Cost**: LLM calls and tokens per explanation (from the API's usage field; estimated for cache hits and the mock).

**Bias to report.** The text and policy checks are the rules this framework gates on, so its outputs are measured
by the rules they were filtered with. The LLM-extractor agreement and the Phase 3 human labels are the measures
without that bias. Report them side by side.

## 4. Runner (`eval/run_comparison.py`)

```bash
python eval/run_comparison.py --preset baselines --domains hr,finance,health --n 30 --intents explain,recourse --judge
python eval/run_comparison.py --preset ablations --domains hr,finance,health --n 30 --judge
```

The output includes a table per domain with Wilson 95% intervals, policy codes per system, and a **coverage-risk**
table. Escalation runs in shadow mode for every system; the escalation score is logged for every case, so the table
shows, for each threshold, the share of cases explained and the error rate among them.

## 5. Domains

| | HR | Finance | Healthcare |
|---|---|---|---|
| Data | IBM HR Attrition (Kaggle) | Statlog German Credit (UCI, CC BY 4.0) | Pima Indians Diabetes (NIDDK; Kaggle copy used by Musicco et al.) |
| File in `backend/data/` | `WA_Fn-UseC_-HR-Employee-Attrition.csv` | `german.data` | `diabetes.csv` |
| Train | `scripts/train_hr.py` | `scripts/train_domain.py --domain finance` | `scripts/train_domain.py --domain health` |
| Reader | HR manager | credit analyst | clinician |
| Positive class | leaves | bad credit risk | diabetes (WHO criteria in the study) |
| Cost FN:FP | 5:1 | 5:1 (the dataset's documented cost matrix) | 5:1 (clinical rationale in the manifest; needs clinician review and a sensitivity analysis) |
| Excluded from the main model | age, gender, marital status | age, personal status/sex, foreign worker | none (age is a clinical factor) |
| Fairness groups | gender, age band, marital status | age band, sex, foreign worker | age band |
| Levers | overtime, travel, pay, stock, training, promotion, role, hybrid/remote | smaller amount, shorter term, guarantor/co-applicant, savings, checking balance | BMI, blood pressure (framed as the model's behaviour, not advice) |
| Refused | employment decisions | credit decisions, terms, rate/limit changes | diagnosis, treatment, medication, referral, coverage, prioritisation |

Every domain passes `validate_domain()` and the same cross-domain test suite (`tests/test_phase2_domains.py`):
clean templates, every fault detected with exactly its code, no raw codes, disclosure, protected attributes withheld
below expert level, recourse re-checked, guardrail sets, routing.

**Healthcare data notes.** The dataset records missing measurements as 0. These are replaced by the median of observed
values over all rows (Insulin is missing in about half the rows). This leaks a little test information into imputed
values; the counts are in the manifest. For fairness, age is deliberately in the model and diabetes prevalence rises
with age, so flag-rate ratios by age band are expected to differ. Report the TPR/FPR gaps (equalised odds) as the
fairness measure for this domain, not the four-fifths rule.

**Finance data notes.** The data are from the 1970s–80s in Deutsche Mark. "Personal status and sex" combines sex and
marital status in one attribute; it is protected and excluded from the main model.

## 6. Domain-transfer effort (`scripts/transfer_effort.py`)

Each domain was added in its own commit on top of the Phase 2a infrastructure commit:

| Domain | Lines in the domain package | Registry | Tests/eval data | Lines changed in shared code | Shared change |
|---|---|---|---|---|---|
| Finance | +610 | +1 | +330 | 12 | verifier: a categorical value must match the evidence label (a shared number was accepted) |
| Healthcare | +457 | +1 | +159 | 18 | verifier: locate a number by position, not by searching for its digits |

Both shared changes were **bugs in the verifier that the new domain exposed**, not domain-specific code. Report them
as such: the domain plug-in contract held, and adding domains improved the shared checker. Making routing work
across three domains also required one shared router change in the following commit: domain-neutral words ("risk",
"estimate", "band") no longer count as evidence for a domain.

## 7. Routing and guardrail (`eval/run_routing_guardrail.py`)

| Set | Refusal recall | False refusals |
|---|---|---|
| HR dev | 70/70 | 0/45 |
| Finance dev | 45/45 | 0/30 |
| **Finance held-out** | **46/50 (92%)** | **0/30** |
| Health dev | 40/40 | 0/30 |
| **Health held-out (first run)** | **39/45 (87%)** | **1/30** |
| Health held-out (after fixing one over-broad pattern) | 39/45 | 0/30 |

These numbers come from the rules layer on the guardrail sets. They don't depend on the stand-in data.

- **How the held-out sets were made.** Each domain's held-out set was written before that domain's guard patterns
  existed and was not used to tune them.
- **One fix after a held-out result.** The health false refusal ("What does the glucose value **refer** to?") came
  from an over-broad referral pattern. It was fixed after the held-out run, so the health held-out set is no longer
  untouched for that item. Report the first-run figure.
- **Caveat.** The same author (Claude) wrote both the sets and the patterns. A held-out set written by the team (for
  healthcare, ideally with a clinician) is still needed for the paper.
- **HR.** HR has no held-out set yet, because its patterns existed before Phase 2. The team should write one
  without looking at `app/domains/hr/policy.py`.
- **Routing.** On the mixed-domain routing set (questions with no domain and no case details), routing is 35/35 with
  0 misroutes, and vague questions ask for clarification. Routing by case details alone is 30/30. The routing set
  was used to find and fix the generic-words problem, so treat it as a development set.

## 8. What still needs you

1. **Real data.** Put the three files in `backend/data/` and train (the commands are in section 5). All numbers
   produced here are from synthetic stand-ins.
2. **Real LLMs.** Set `LLM_PROVIDER`, `REVIEWER_PROVIDER` (different family), `JUDGE_PROVIDER` (extractor) and
   `FAULT_PROVIDER`. Then run `python eval/run_all.py`.
3. **Held-out guardrail sets written by the team**, for all three domains.
4. **A clinician review of the healthcare policy and cost matrix**, before the paper claims clinical
   appropriateness.
5. **Frontend.** The web app still sends `domain: "hr"`. A domain selector is needed to demo finance and
   healthcare. The API already serves all three.
