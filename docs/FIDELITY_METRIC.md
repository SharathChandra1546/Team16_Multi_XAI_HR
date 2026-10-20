# The Claim Fidelity Score (CFS), version 2

The verifier (`verifier.verify()`) computes this metric on every explanation. This note defines it so it can be
reported and discussed as a metric.

## Definition

An explanation `E` has claims `C_1 .. C_n`, each citing one or more ids from a Canonical Evidence Package `P`
(`P` the prediction, `E#` an attribution item, `R#` a recourse option, `G-#` a glossary entry), plus free text.
The deterministic verifier reports violations against `P`. Each violation is attached to one **unit**:

- **a claim**, if the violation names that claim, or if its span (the offending words or number) occurs in that
  claim's text;
- **an unbound sentence**, for a text-level violation (unsupported number, causal wording, raw code, a factor named
  without a claim) whose span lies in a sentence that is not a claim. Each such sentence is one extra unit;
- **a missing claim**, for each required evidence id the explanation never cites (`missing_required`). Each is one
  extra unit: a required statement that should have been made and was not.

```
units  = n + u_sentences + u_missing
CFS    = |clean claims| / units                                          (0.0 when there are no claims)
CFS_w  = (|claims with no violation| + 0.5 * |claims with only minor violations|) / units
```

A claim is **clean** when no critical violation is attached to it. **Minor** means a non-blocking verifier finding;
currently only `not_allowed_evidence` (citing an id outside the plan's scope) is minor.

Critic findings (policy wording, level fit, readability, the LLM quality score) are deliberately **not** part of CFS.
CFS measures fidelity to the evidence. Policy and level violations are reported separately by category
(`app/policy_report.py`).

Implementation: `app/metrics/fidelity.py::claim_fidelity_score`. `verify()` calls it with the claim texts and the
full text, so the score and the pass/fail decision come from the same violation list.

## Changes from version 1

| | v1 | v2 |
|---|---|---|
| Text-level violations (unsupported number, causal wording, raw code, uncovered factor) | ignored | counted (attached to a claim or an extra unit) |
| Missing required citation | ignored | one extra failed unit per id |
| Explanation with correct claims plus an invented statistic | CFS = 1.0 although rejected | CFS < 1.0 |
| No claims | doc said 1.0, code returned 0.0 | 0.0 (doc and code agree) |
| Weighted variant | could exceed 1.0 (half credit added on top of full credit) | bounded in [0, 1] |

Version 2 makes `CFS == 1.0` equivalent to "the verifier found no critical violation" (tested in
`tests/test_phase1_verifier.py::test_cfs_is_one_exactly_when_verifier_passes`).

## Properties

- **Bounded.** `CFS, CFS_w ∈ [0, 1]`.
- **Consistent with the gate.** `CFS == 1.0` exactly when the explanation has at least one claim and no critical
  verifier violation anywhere in the text.
- **Deterministic and reproducible.** Every check behind CFS is a closed-form comparison against `P` (string or
  number equality within a fixed tolerance, or a fixed regular expression), never a model call.
- **Domain-agnostic.** CFS is computed from the `EvidencePackage`/`Claim` contracts in `app/schemas.py`. The same
  function runs unchanged on any registered domain (`tests/test_graph_domain_agnostic.py`).
- **Separate from explanation quality.** CFS never calls a language model. The LLM critic's quality score is a
  separate number, so "is it true to the evidence" and "is it well written" are never combined.

## Relationship to existing XAI faithfulness metrics

Most faithfulness metrics (deletion/insertion curves, infidelity, comprehensiveness) test whether an attribution
method reflects the model. CFS tests an earlier link: whether the natural-language text reflects the evidence
package it was given, claim by claim. A problem in the Shapley or LIME attributions themselves shows up upstream,
in the evidence validator (`app/xai/validator.py`), not here. The per-claim value, sign and rank checks are close to
the agreement metrics of Krishna et al., which Musicco et al. (2026) use for post-hoc evaluation; here they run
inline, as a gate, before anything is shown.

## Known limitation

CFS is computed by the same verifier that decides pass/fail, so a report of "100% detection" on template faults is
partly circular: it shows internal consistency, not agreement with human judgement. Two things address this:

1. detection on **LLM-written faults** (`eval/run_faults.py --faults llm`), produced from rule-free descriptions
   by a model that never sees the verifier's patterns;
2. a **human-labelled sample** (verifier vs LLM reviewer vs two human annotators, with Cohen's kappa), planned for
   Phase 3.
