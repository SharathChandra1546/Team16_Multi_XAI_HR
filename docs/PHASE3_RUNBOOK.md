# Phase 3 runbook: real experiments, human labels, paper

Run `python scripts/preflight_phase3.py` (from `backend/`) first. It lists what is still missing. Nothing below should
be run for results until it prints "Ready for Phase 3."

## 0. Setup (once)

1. Put the three data files in `backend/data/`: `WA_Fn-UseC_-HR-Employee-Attrition.csv`, `german.data`, `diabetes.csv`.
2. Copy `backend/.env.example` to `backend/.env` and set, with different model families for generator and reviewer
   (providers supported: gemini, groq, openai, ollama):
   ```
   LLM_PROVIDER=groq            # generator, for example a Llama model
   REVIEWER_PROVIDER=gemini     # reviewer, a different family
   JUDGE_PROVIDER=openai        # LLM extractor for value/sign/rank agreement
   FAULT_PROVIDER=openai        # writes the faults the verifier is tested on
   LLM_TEMPERATURE=0
   ```
   plus the matching `*_API_KEY` values. Keep `.env` out of git (it already is).
3. Train: `python scripts/train_hr.py`, `python scripts/train_domain.py --domain finance`,
   `python scripts/train_domain.py --domain health`. Check that each manifest says the real source, not the stand-in.
4. Team tasks that can run in parallel: held-out guardrail sets (`docs/GUARDRAIL_SET_GUIDE.md`), clinician review
   (`docs/CLINICIAN_REVIEW.md`), two annotators booked (`docs/ANNOTATION_GUIDE.md`).

## 1. Dry run on a small size

`python eval/run_all.py --quick` and read the output. Fix anything that fails before spending on the full run.

## 2. Experiments (each writes to `backend/results/`)

| Question for the paper | Command | Report |
|---|---|---|
| Does the verifier catch faults, and without sharing the faults' rules? | `python eval/run_faults.py` and the LLM-fault runner in `run_all.py` | detection rate per fault type, template (upper bound) vs LLM-written faults |
| Our framework vs every baseline | `python eval/run_comparison.py --preset baselines --domains hr,finance,health --n 30 --intents explain,recourse --judge` | error-free rate, agreement (value, sign, rank), unsupported mentions, calls and tokens, with Wilson intervals |
| Which component matters | `python eval/run_comparison.py --preset ablations --domains hr,finance,health --n 30 --judge` | effect of removing each component |
| Coverage vs risk | in the comparison output | share explained against error rate at each escalation threshold |
| Guardrail and routing | `python eval/run_routing_guardrail.py` | team held-out sets first, development sets second |
| Transfer effort | `python scripts/transfer_effort.py` | lines added per domain, lines changed in shared code |
| Model quality and fairness | `python eval/export_tables.py --all` | metrics with intervals; age-band gaps for healthcare reported as error rates |
| Cost-ratio sensitivity | run the training script with other FN:FP ratios (for example 2:1, 5:1, 10:1) | how thresholds and flag rates move |

Run the comparison with at least `--n 30` cases per domain. Use the same `--offset` for every system so all systems
see the same cases. Set a fixed seed where a script offers one and record the model names and dates in the results.

## 3. Human evaluation

1. `python eval/make_annotation_sheets.py --results results/comparison_<stamp>.json --per-system 24 --annotators A,B`
2. Send each annotator only `annotation/<tag>/sheet_A.csv` (or `_B`) and `docs/ANNOTATION_GUIDE.md`. Keep
   `items_private.json` away from them.
3. Both annotators first practise on 10 rows together, then label all rows alone.
4. `python eval/score_annotations.py --dir annotation/<tag>` gives kappa per label, human error-free rate per system,
   and how well the automatic checker matches the humans.
5. Resolve the rows in `adjudication.csv` together, save as `adjudicated.csv`, run again with `--adjudicated`.
6. Rough sizes: 24 items per system per domain with 10 systems and 3 domains is about 720 items per annotator. Reduce
   `--systems` to the ones that matter for the human study (for example ours, musicco, critic_rule, explingo, direct)
   to bring that down to about 360.

## 4. What to write

- Results tables from the files above, with the sizes and intervals. Say that baselines are re-implementations.
- The bias note: text and policy checks are the rules our pipeline gates on; the LLM-extractor agreement and the human
  labels are the measures that do not share that bias.
- Related work from `docs/RELATED_WORK.md` after re-checking each cell against the papers.
- Limitations: one dataset per domain, public research data, small human study, healthcare not clinically validated beyond
  the sign-off, guardrail sets small.

## 5. Optional, if time allows

- Run the authors' released XstrAI code on the same evidence as an extra Musicco et al. comparison.
- A second generator model to show the result does not depend on one LLM.
