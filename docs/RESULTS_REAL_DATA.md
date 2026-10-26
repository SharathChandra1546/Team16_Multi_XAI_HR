# Real-data results without an LLM (Phase 3, part 1)

Date: 2026-10-21. Data: IBM HR attrition (Kaggle, 1,470 rows), UCI Statlog German credit (1,000 rows),
Pima Indians diabetes (NIDDK, 768 rows). Every number below comes from a file in `backend/results/`; regenerate with
the commands at the end. **No LLM was called.** All explanation texts here are the deterministic templates, so these
results cover the models, the verifier, the guardrail and routing. They do not show how well an LLM pipeline compares
with the baselines (that still needs a reachable LLM, see "Not yet measured").

## 1. Models (Table 1, `results/real_data_summary.md`)

| Domain | Model | n test (positives) | ROC-AUC [95% CI] | PR-AUC | Recall | Precision | Brier | ECE | Threshold |
|---|---|---|---|---|---|---|---|---|---|
| HR attrition | logreg | 294 (47) | 0.798 [0.718, 0.871] | 0.565 | 0.638 | 0.484 | 0.101 | 0.026 | 0.25 |
| Credit risk | hgb | 200 (60) | 0.794 [0.724, 0.857] | 0.652 | 0.850 | 0.415 | 0.160 | 0.026 | 0.20 |
| Diabetes | logreg | 154 (54) | 0.811 [0.740, 0.872] | 0.672 | 0.981 | 0.515 | 0.173 | 0.029 | 0.14 |

Thresholds minimise a 5:1 cost (missed positive : false flag). The model beats both trivial policies on cost in every
domain (Table 2). Table 2b gives the sensitivity analysis for cost ratios 1, 2, 3, 5, 10 and 20.

**Flags for the team:**
- At 5:1 the health model flags **66.9%** of test patients and finance flags **61.5%** of applicants. That ratio is a
  placeholder. A clinician has to choose the health ratio (`docs/CLINICIAN_REVIEW.md`), and the paper should report
  the sensitivity table, not only 5:1.
- The finance model sends **59.5%** of test cases to human review (uncertain band 59.0%, out-of-distribution 1.0%).
  The band is too wide to be useful as it is. Narrow it, or report it as a limitation.

## 2. Fairness (Table 3)

Four-fifths rule failures: HR age band (0.52) and marital status (0.56); finance age band (0.68) and foreign worker
(0.66); health age band (0.60, but age is a clinical factor there, so compare error rates rather than flag rates).
Proxy AUC of 0.70 or more (the remaining features can predict the attribute): HR age band and marital status; finance
age band, sex and foreign worker; health age band. The pipeline withholds protected attributes from non-expert
explanations, but the models still show these gaps. The paper must state them, not hide them.

## 3. Routing and guardrail (`results/routing_guardrail_20261021-105400.md`)

- Routing accuracy 100.0% [90.1%, 100.0%] (35/35); routing by case details 30/30.
- Guardrail refusal recall on **held-out** sets: finance 92.0% [81.2%, 96.8%] (46/50), health 86.7% [73.8%, 93.7%]
  (39/45). False refusals 0/30 in each. The development sets score 100% but only show fit.
- Caveat: Claude wrote the held-out sets, not the team. The paper's guardrail number should come from team-written
  sets (`docs/GUARDRAIL_SET_GUIDE.md`, template `backend/eval/data/team_heldout_TEMPLATE.json`).

## 4. Template-fault detection (`results/faults_20261021-130004.md`)

30 cases per domain × 3 levels × 2 intents. There are 15 fault types, injected by templates.

| Domain | Faulty texts | Detection [95% CI] | Clean texts | False rejection [95% CI] |
|---|---|---|---|---|
| HR | see file (15 types, 60–180 each) | 100.0% [99.8%, 100.0%] | 719 | 0.0% [0.0%, 0.5%] |
| Credit | see file | 100.0% [99.8%, 100.0%] | 705 | 0.0% [0.0%, 0.5%] |
| Diabetes | see file | 100.0% [99.8%, 100.0%] | 711 | 0.0% [0.0%, 0.5%] |

**How to read this:** template faults are an **upper bound**. The verifier was written knowing what the templates
look like. Also, the bugs in Section 5 were found on these same cases and then fixed, so the 100% is a post-fix score
on development data, **not** a held-out result. The run before the fixes, on the same data, scored: finance raw_code
98.3%, finance risk_text_swap 97.8%, health risk_text_swap 98.9%, and finance clean false rejection 4/705 (0.6%).
The detector-independent LLM fault generator and the LLM-written base texts measure the realistic case. Neither has
been run yet.

## 5. Bugs that only showed up on real data (all fixed, each with a regression test that fails without the fix)

1. **Injector: risk_text_swap substring.** An estimate of 1.3% made the injector search for "1%", which matched
   inside "2.1%". The fault was then not a real fault.
2. **Measurement extractor: value labels.** "Overtime (works overtime)" lost its value. A finance label that contains
   another factor's name ("Credit history (existing credits repaid duly so far)") was counted as a mention of that
   other factor.
3. **Injector: risk_text_swap hit the threshold.** 19.7% rounds to "20%", which is also the review threshold, so the
   injector changed the threshold instead of the estimate.
4. **Verifier: cue words inside value labels.** "Employment length (less than 1 year) moves the estimate up" was
   rejected as a direction error, because "less" in the label was read as "lowers". Correct LLM text could hit the
   same bug, so this was a real false-rejection source.
5. **Verifier: raw codes with several underscores.** "other_installment_plans=A141" slipped through, because both
   raw-code patterns allowed only one underscore. This was a real miss.

Bugs 4 and 5 were in the verifier itself, which is the paper's main component. Report them in the paper as findings
from moving off synthetic data.

## 6. Not yet measured

- Pipeline vs baselines (direct, B1–B4, Explingo, critic loop, Critic-Rule, Musicco), ablations, LLM-generated
  faults, LLM-extractor agreement, cost/latency. These need a reachable LLM and, for an independent reviewer, a second
  provider family.
- Human labels (two annotators, Cohen's kappa, checker-vs-human agreement).
- Team-written guardrail sets; clinician sign-off on the health cost ratio and wording.
- Transfer effort (`scripts/transfer_effort.py --domain <d> --base <commit>`) has not yet been rerun for this write-up.

## Regenerate

```bash
cd backend
python eval/summarise_models.py                       # Tables 1, 2, 2b, 3
python eval/run_routing_guardrail.py
LLM_PROVIDER=mock python eval/run_faults.py --n 30 --domains hr,finance,health
```
