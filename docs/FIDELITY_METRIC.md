# The Claim Fidelity Score (CFS)

This note formalises a metric the system already computes on every
explanation (`verifier.verify()`), so it can be reported and discussed as a
contribution in its own right rather than read back out of the code.

## Definition

An explanation `E` is a set of claims `C_1 .. C_n`, each citing one or more
ids from a Canonical Evidence Package `P` (`P` the prediction, `E#` an
attribution item, `R#` a recourse option, `G-#` a glossary entry). The
deterministic verifier checks each claim against `P`: the cited id must
exist and not be withheld, and (depending on claim kind) the stated feature,
direction, value, rank, probability or band must match the corresponding
field in `P` exactly or within a fixed numeric tolerance (0.5 percentage
points). A claim with no such mismatch, and whose exact text appears in the
rendered explanation, is *clean*.

    CFS(E, P) = | { C_i : C_i is clean } | / n,  n = |C|

A weighted variant additionally gives half credit to a claim that is clean of
*critical* violations but still carries a *minor* one (currently: a
readability note), so two explanations that both pass verification can still
be distinguished by polish rather than collapsing to the same score:

    CFS_w(E, P) = ( |clean| + 0.5 * |minor-only| ) / n

Implementation: `app/metrics/fidelity.py::claim_fidelity_score`. `verify()`
in `app/agents/verifier.py` calls it directly, so the reported score and the
pass/fail decision are computed from the same violation list — the score is
never approximated or recomputed by a second method.

## Properties

- **Bounded.** `CFS, CFS_w ∈ [0, 1]` by construction: a count of claims
  divided by the total claim count, with a non-negative sub-unit weight on
  the weighted variant. An explanation with no claims is handled upstream as
  an unconditional failure (`empty_explanation`), not folded into a vacuous
  `CFS = 1.0`.
- **Deterministic and reproducible.** Every check behind CFS is a closed-form
  comparison against `P` — string/number equality within a fixed tolerance,
  or a fixed regular expression — never a model call. Re-running `verify()`
  on the same `(package, plan, explanation)` triple always returns the same
  score, byte for byte. This is what makes CFS usable as ground truth in the
  fault-injection benchmark (`eval/run_faults.py`), not just as a monitoring
  number.
- **Domain-agnostic.** CFS is computed entirely from the
  `EvidencePackage`/`Claim` contracts in `app/schemas.py`, which carry no
  HR-specific fields. `tests/test_graph_domain_agnostic.py` runs the same
  verifier, unmodified, against a second toy domain and confirms the graph
  and the score both work unchanged.
- **Separate from explanation quality.** CFS never calls a language model.
  The critic's optional LLM-judge pass (clarity / level fit / relevance,
  `app/agents/critic.py`) is a distinct score, explicitly instructed not to
  judge facts, so "is this explanation true to the evidence" (CFS) and "is
  this explanation well written" (the judge score) are never conflated into
  one number.

## Relationship to existing XAI fidelity/faithfulness metrics

Most faithfulness metrics in the XAI literature (deletion/insertion curves,
infidelity, comprehensiveness) measure whether an *attribution method*
reflects the *underlying model* — they test the explainer against the model.
CFS measures something earlier in the pipeline: whether the *natural-language
text* a language model wrote reflects the *evidence package it was given* —
it tests the writer against the explainer's own output, at the level of
individual factual claims rather than a single aggregate correlation. The two
are complementary: a low CFS means the written explanation misrepresents
correct evidence; a faithfulness problem in the underlying Shapley/LIME
attributions would show up upstream, in the evidence validator
(`app/xai/validator.py`), not here.

## Known limitation, stated plainly

CFS is computed by the same verifier that fault-injection experiments use as
ground truth, so a report of "100% detection" on those experiments is
partly circular — it shows the verifier is internally consistent, not that it
agrees with human judgement. The project's `--base llm` experiment plus a
small human-labelled sample (verifier vs. an independent LLM reviewer vs.
human raters, reporting agreement/kappa) is the planned way to validate CFS
against an external standard; see `README.md`'s "Notes for honest reporting."
