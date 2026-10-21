# Related work: how this framework differs

Status of each source: **checked** means the abstract page was read on 2026-10-21 for this table. **From earlier
reading** means it comes from the paper the team uploaded or from notes made earlier in the project and must be
re-checked against the paper before submission. **Not re-checked** means the cells are blank on purpose.

## Closest prior work

| Work | What it does | Evidence it uses | Checker | Datasets | Human study |
|---|---|---|---|---|---|
| **He & Martens 2026**, "An Agentic Approach to Generating XAI-Narratives" (arXiv 2603.20003) — checked | Narrator writes, critic agents give feedback, the narrator revises (up to 3 rounds). Five designs, five LLMs. | SHAP table, top four features | **Critic Design**: an LLM extracts each factor's rank, sign and value, an LLM critic writes fixes. **Critic-Rule Design**: the critic is rule-based (code), the extractor is still an LLM. | Student Performance, FIFA, **German Credit, Diabetes**, Stroke; 20 test instances each; Random Forest | None reported; a qualitative manual review of examples |
| **Musicco et al. 2026**, "Who Are You Explaining To?" (arXiv 2608.11033; code github.com/sisinflab/XstrAI) — from earlier reading | Framer plans, Narrator writes, Reviewer routes revisions; reader personas. Baselines B1 to B4. | Explanation card from SHAP-style attributions | LLM reviewer plus a format verifier (B4). Agreement measured with value/sign/rank. | Includes the Pima diabetes data | Re-check the paper for any user study |
| **Zytek et al. 2024**, "Explingo" (arXiv 2412.05145) — checked | Narrator (GPT-4o, few-shot exemplars) and Grader (accuracy, completeness, fluency, conciseness). | SHAP top-N features | LLM grader; conciseness by a length formula. Grader validated against 50 human-labelled pairs per metric (96% agreement). | Ames Housing, Student Performance, Mushroom, PDF Malware | No user study |
| **Bello et al.** (arXiv 2506.05887), the base paper of this project | SHAP to LLM translation for HR attrition on IBM HR data | | | IBM HR | Not re-checked |
| Miyaji and Corrêa 2025; Yamaguchi 2026 (listing 2512.21066, number unconfirmed) | | | | | Not re-checked |

## What this framework has that the closest works do not (as far as checked)

| Feature | He & Martens (Critic-Rule) | Musicco et al. | Explingo | This framework |
|---|---|---|---|---|
| Fact check done by code against a fixed evidence package, with an id for every claim | extraction is by LLM, comparison by code | LLM reviewer | LLM grader | claims carry evidence ids; the verifier is code and returns typed violation codes |
| Deterministic check of wording rules (decisions, blame, protected attributes, prediction stated as fact) | not reported | not reported | not reported | policy-as-code per domain |
| Refusal of decision requests before any LLM call | not reported | not reported | not reported | router guardrail, three domains |
| Escalation to a human when the estimate is uncertain or the case is unusual, with a score for coverage-risk curves | not reported | not reported | not reported | yes |
| Reader level (novice, intermediate, expert) with level rules that are checked | not reported | personas | exemplars for style | yes, checked by code |
| Verified fallback text when the checks fail | not reported | not reported | not reported | yes |
| New domain by adding a package, with measured effort | not reported | not reported | not reported | `scripts/transfer_effort.py` |
| Switches to turn each component off, with the shown text still measured | designs compared | B1 to B4 | not applicable | ablation presets |

"Not reported" means it was not in the abstract or the description that was read, not that the work lacks it.
Re-check each cell against the full paper before it goes into the submission.

## Things to state in the paper

1. **Datasets are not new.** German Credit and Diabetes are in He & Martens; Pima diabetes is in Musicco et al. The
   contribution is the verifier, the policy layer, the escalation and the transfer across three domains, not the data.
2. **A rule-based critic already exists** (He & Martens, Critic-Rule). It is a baseline here (`critic_rule`). The claim to
   defend is the difference: code-checked claims with evidence ids, policy and level rules, refusal, escalation, fallback.
3. **Our measurement favours us.** The text and policy checks are the same rules the pipeline gates on. Report them next
   to the LLM-extractor agreement and the human labels, which do not share that bias.
4. **The baselines are re-implementations** from the descriptions, not the authors' code. For Musicco et al. the authors'
   code exists and a run of it on the same evidence is the stronger comparison.
5. **Sizes.** He & Martens use 20 instances per dataset; plan for at least that and report intervals.
