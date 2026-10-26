# Phase 3 results: real-LLM experiments (reduced study, finance + health)

Status: **v2, 2026-10-26** (v1 plus a top-up run that added one credit case to the main comparison and one credit
case to the LLM-fault test). Every number below comes from `backend/results/paper/numbers.json`, built by `eval/paper_tables.py` from the raw
run files (backed up in `backend/results_phase3/`). The non-LLM results (models, fairness, cost sensitivity, routing,
guardrail) are in `docs/RESULTS_REAL_DATA.md`.

## Paper-ready files (`backend/results/paper/`)

| Use in the paper | LaTeX (`\input{...}`) | Also |
|---|---|---|
| Main comparison | `table_main_comparison.tex` | `.csv`, `fig_main_comparison.pdf/.png` |
| Paired tests vs ours | `table_paired_tests.tex` | `.csv` |
| Error types | `table_error_types.tex` | `.csv` |
| Ablations | `table_ablations.tex` | `.csv`, `fig_ablation.pdf/.png` |
| LLM-written faults | `table_faults_llm.tex` | `.csv`, `fig_faults_llm.pdf/.png` |
| Template faults (upper bound) | `table_faults_template.tex` | `.csv` |

The `.tex` tables use `booktabs` (`\usepackage{booktabs}`). `records.csv` holds one row per generated text (with
the text), and `numbers.json` holds every figure with its interval.

## What was run

| | Main comparison | Ablations | LLM-written faults |
|---|---|---|---|
| Domains | Credit risk (UCI Statlog German Credit), diabetes (Pima, NIDDK) | same | same |
| Cases | Credit: first 4 held-out test cases; diabetes: first 3 (+1 partial) | First 10 per domain | Credit: first 3 cases; diabetes: first 2 (+1 fault from a later case) |
| Levels / question | novice and expert / "why this estimate" | same | same |
| Writer | Qwen 3.8 27B (Groq), the same for every system, including each baseline's own review steps | Gemma 4 26B (Gemini API), the same for every variant | gpt-oss-20b (Groq) writes the faults |
| Scorer | gpt-oss-120b (Groq): LLM extractor of factors, values, directions, ranks | same | the verifier itself (deterministic) |
| Size | 6 systems, 15 to 16 texts each (93 in total) | 5 variants × 40 texts (200) | 141 faults, 93 clean texts |

Dates: 2026-10-24 to 2026-10-26. Free-tier API quotas set the size. HR was left out of the LLM runs to fit them; HR
is covered by the non-LLM results.

Error-free means the text has no wrong value, direction or stated rank, no unsupported factor or number, no causal
wording, no raw code and no policy violation. The primary figure uses the independent LLM scorer; the rule-based
figure (the deterministic extractor) is shown beside it. Intervals are 95% Wilson; paired tests are exact McNemar
on the same case and level.

## Table 1. Main comparison

| System | n | Error-free | Finance | Health | Error-free (rule-based) | Policy viol. | Calls | Tokens |
|---|---|---|---|---|---|---|---|---|
| Ours (full pipeline) | 16 | 100% [81, 100] | 100% [68, 100] | 100% [68, 100] | 100% [81, 100] | 0% [0, 19] | 4.2 | 6.0k |
| Direct prompting | 16 | 44% [23, 67] | 38% [14, 69] | 50% [22, 78] | 31% [14, 56] | 0% [0, 19] | 1.0 | 0.3k |
| B3 (LLM reviewer loop) | 16 | 75% [51, 90] | 75% [41, 93] | 75% [41, 93] | 75% [51, 90] | 0% [0, 19] | 2.1 | 3.1k |
| Explingo-style | 15 | 47% [25, 70] | 38% [14, 69] | 57% [25, 84] | 47% [25, 70] | 0% [0, 20] | 2.3 | 1.6k |
| Critic-Rule (He & Martens) | 15 | 67% [42, 85] | 50% [22, 78] | 86% [49, 97] | 67% [42, 85] | 0% [0, 20] | 4.1 | 4.8k |
| Musicco et al.-style | 15 | 40% [20, 64] | 38% [14, 69] | 43% [16, 75] | 40% [20, 64] | 7% [1, 30] | 4.3 | 9.0k |

Paired with our pipeline on the same case and level (error-free, LLM scorer):

| Baseline | Pairs | Only ours correct | Only baseline correct | Both | Exact McNemar p |
|---|---|---|---|---|---|
| Direct prompting | 16 | 9 | 0 | 7 | 0.004 |
| B3 (LLM reviewer loop) | 16 | 4 | 0 | 12 | 0.125 |
| Explingo-style | 15 | 8 | 0 | 7 | 0.008 |
| Critic-Rule (He & Martens) | 15 | 5 | 0 | 10 | 0.063 |
| Musicco et al.-style | 15 | 9 | 0 | 6 | 0.004 |

Texts with at least one error of each kind:

| System | Texts | Value | Stated rank | Unsupported number | Causal wording | Raw code | Policy |
|---|---|---|---|---|---|---|---|
| Ours | 16 | 0 | 0 | 0 | 0 | 0 | 0 |
| Direct | 16 | 0 | 1 | 9 | 1 | 2 | 0 |
| B3 | 16 | 0 | 4 | 0 | 1 | 0 | 0 |
| Explingo-style | 15 | 3 | 0 | 8 | 0 | 0 | 0 |
| Critic-Rule | 15 | 0 | 2 | 2 | 3 | 0 | 0 |
| Musicco-style | 15 | 0 | 1 | 5 | 3 | 0 | 1 |

No system made a direction error or mentioned a factor outside the evidence.

**Findings to state (worded for the paper):**
- Our pipeline produced no error in any of its 16 texts. No baseline was correct on a case where ours was wrong.
- The difference is significant against direct prompting (p = 0.004), the Explingo-style pipeline (p = 0.008) and
  the Musicco-style multi-agent pipeline (p = 0.004). Against the LLM-reviewer loop (B3, p = 0.125) and the
  rule-based critic of He & Martens (p = 0.063) the sample is too small to show a difference at the 5% level.
- The baselines' errors are mostly unsupported numbers (direct prompting restates raw SHAP values and codes;
  Explingo-style texts copy figures from their few-shot examples), causal wording and wrong stated ranks.
- Cost: our pipeline used 4.2 LLM calls and about 6.0k tokens per explanation, against 1.0 call for direct
  prompting and 4.3 calls / 9.0k tokens for the Musicco-style pipeline.

## Table 2. Ablations (writer Gemma 4 26B)

| Variant | Texts | Error-free | Any fidelity error | Template fallback | LLM text shown unverified |
|---|---|---|---|---|---|
| Full pipeline | 40 | 98% [87, 100] | 2% [0, 13] | 18% | 0 |
| without claim verifier | 40 | 98% [87, 100] | 2% [0, 13] | 10% | 5 |
| without critic | 40 | 100% [91, 100] | 0% [0, 9] | 2% | 19 |
| without repair loop | 40 | 100% [91, 100] | 0% [0, 9] | 62% | 0 |
| without policy rules | 40 | 98% [87, 100] | 2% [0, 13] | 18% | 0 |

**Findings to state:** with this writer, removing a component did not change correctness measurably: its drafts
were rarely wrong, and every fallback is a correct template. The components show up in what the reader gets. The
repair loop keeps the model's own text: without it 62% of answers fall back to the template, against 18% with it.
Without the verifier or the critic, unchecked LLM text reaches the reader (5 and 19 of 40 texts). Say that this
sample cannot show that each component prevents errors; it shows the trade-off between gating and fallback.

## Table 3. Verifier against injected faults

| Fault source | Faults | Blocked | Exact error type | False rejections on clean texts |
|---|---|---|---|---|
| Template faults, 3 domains (upper bound) | 6,390 | 100% [100, 100] | 100% [100, 100] | 0 of 2,135 |
| LLM-written faults, finance + health | 141 | 82% [75, 88] | 66% [58, 73] | 0 of 93 [0, 4] |

LLM-written faults by type (blocked): causal claim, decision wording, direction flip, fake factor, missing
citation, protected inference, risk swap, value swap and wrong top factor 100%; unsupported number 89%; blame wording
78%; magnitude swap 56%; prediction stated as fact 38%; raw code 20%; wrong ordinal 17% (6 to 11 faults per type,
so each per-type rate is rough).

**Verifier gaps found by the LLM-written faults** (genuine misses, read by hand; the verifier was frozen for this
study, so the gaps are reported, not fixed):
1. Internal evidence ids in the text ("G-SAVINGS") pass; the raw-code check covers dataset codes only.
2. "the second most influential factor" passes; the ordinal check knows "largest" and "most important".
3. A wrong magnitude ("-5.0 percentage points" for -10.5) passes when the number happens to be valid elsewhere in
   the evidence (5, 50); numbers are not checked against the factor they describe.
4. Blame phrasing outside the list passes ("shows a lack of responsibility").
5. "The model predicts that the applicant will be a bad credit risk" drops the uncertainty but passes.

Template faults are an upper bound (the checks were written knowing those patterns); report the LLM-written figure
as the realistic one.

## Caveats (state them)

- Small reduced study: 13 to 14 texts per system in the main comparison; wide intervals. Two domains, novice and
  expert levels, "why" questions only. HR has no LLM runs.
- Baselines are our reimplementations, run with the same writer model and given the same evidence card.
- The rule-based measurement shares the text and policy rules our pipeline gates on, which favours it. The primary
  figure therefore uses an independent LLM scorer (gpt-oss-120b, a different model family from the writer); human
  labels are still to come (`eval/make_annotation_sheets.py`).
- Open-weights models on free API tiers (Qwen 3.8 27B, gpt-oss-120b, gpt-oss-20b on Groq; Gemma 4 26B on the Gemini
  API). The ablations use a different writer from the main comparison.
- Texts whose generation hit an API error (quota, server error) are excluded for every system alike and counted.
- Measurement fixes made during the study (before the final scoring, applied to every system): paraphrased category
  values, hedged "chance that ... has diabetes", a derived "points above the average" number, and mention order no
  longer counted as a stated rank. Each has a regression test.

## How the numbers were checked

1. `eval/verify_paper_numbers.py` recomputes every error-free count, Wilson interval and paired test from
   `records.csv`, and the LLM-fault totals from the raw fault files, with independent code (scipy binomial test):
   296 texts and the fault chunks checked, 0 mismatches. This pass caught one real error before release: early fault
   files were counted twice (fixed; the totals above are the corrected ones).
2. A sample of texts was read by hand (3 marked correct, 4 marked wrong); all 7 verdicts were right on reading.

## Regenerate

```bash
cd backend
python eval/paper_tables.py --judge     # merges all runs, re-scores every text (scorer answers come from the cache)
python eval/paper_figures.py
python eval/verify_paper_numbers.py
```
