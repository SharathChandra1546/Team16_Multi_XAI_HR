# Phase 3 results (reduced real-LLM study): finance and health

Generated 2026-10-24T17:01:54+00:00 by `backend/eval/paper_tables.py` from 10 run files. Every text was re-measured with the current measurement code. Do not edit numbers by hand.

## Table 1. Main comparison

Writer for every system: Qwen 3.8 27B (Groq). Each system's own review/critique steps use the same writer. Scorer: gpt-oss-120b (Groq, a different model family). Questions: "why this estimate" (explain); levels: novice and expert.

| System | n | Error-free (pooled) | Finance | Health | Error-free (rule-based) | Policy viol. | VA | SA | RA | Calls | Tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours (full pipeline) | 6 | n/a | n/a | n/a | 100% [61, 100] | 0% [0, 39] | n/a | n/a | n/a | 4.0 | 6.9k |
| Direct prompting | 6 | n/a | n/a | n/a | 50% [19, 81] | 0% [0, 39] | n/a | n/a | n/a | 1.0 | 0.4k |
| B3 (LLM reviewer loop) | 6 | n/a | n/a | n/a | 83% [44, 97] | 0% [0, 39] | n/a | n/a | n/a | 2.3 | 4.1k |
| Explingo-style | 6 | n/a | n/a | n/a | 33% [10, 70] | 0% [0, 39] | n/a | n/a | n/a | 2.5 | 1.9k |
| Critic-Rule (He & Martens) | 6 | n/a | n/a | n/a | 50% [19, 81] | 0% [0, 39] | n/a | n/a | n/a | 4.2 | 5.4k |
| Musicco et al.-style | 5 | n/a | n/a | n/a | 20% [4, 62] | 20% [4, 62] | n/a | n/a | n/a | 3.5 | 7.8k |

### Paired tests (same case and level)

| Baseline | Pairs | Only ours correct | Only baseline correct | Both | Neither | Exact McNemar p |
|---|---|---|---|---|---|---|
| Direct prompting | 0 | 0 | 0 | 0 | 0 | 1 |
| B3 (LLM reviewer loop) | 0 | 0 | 0 | 0 | 0 | 1 |
| Explingo-style | 0 | 0 | 0 | 0 | 0 | 1 |
| Critic-Rule (He & Martens) | 0 | 0 | 0 | 0 | 0 | 1 |
| Musicco et al.-style | 0 | 0 | 0 | 0 | 0 | 1 |

### Error types (texts with at least one)

| System | Texts | Value | Direction | Stated rank | Unsupported factor | Unsupported number | Causal wording | Raw code | Policy |
|---|---|---|---|---|---|---|---|---|---|
| Ours (full pipeline) | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Direct prompting | 6 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 |
| B3 (LLM reviewer loop) | 6 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| Explingo-style | 6 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 |
| Critic-Rule (He & Martens) | 6 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| Musicco et al.-style | 5 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 1 |

## Table 2. Ablations

Writer: Gemma 4 26B (Gemini API) for every variant; scorer gpt-oss-120b.

| Variant | Texts | Error-free | Any fidelity error | Policy viol. | Template fallback | Withheld/escalated | Calls |
|---|---|---|---|---|---|---|---|
| Full pipeline | 12 | n/a | n/a | 0% [0, 24] | 17% | 0% | 4.7 |
| -- claim verifier | 12 | n/a | n/a | 0% [0, 24] | 17% | 0% | 4.7 |
| -- critic | 12 | n/a | n/a | 0% [0, 24] | 0% | 0% | 2.0 |
| -- repair loop | 12 | n/a | n/a | 0% [0, 24] | 67% | 0% | 5.7 |
| -- policy rules | 12 | n/a | n/a | 0% [0, 24] | 17% | 0% | 4.7 |

## Faults: LLM-written faults (gpt-oss-20b; detector-independent)

Sources: 1 file(s), 1 chunk(s); domains: finance.

| Fault type | n | Blocked | Exact code |
|---|---|---|---|
| blame wording | 1 | 0% [0, 79] | 0% [0, 79] |
| causal claim | 2 | 100% [34, 100] | 100% [34, 100] |
| decision wording | 2 | 100% [34, 100] | 100% [34, 100] |
| direction flip | 2 | 100% [34, 100] | 100% [34, 100] |
| fake feature | 2 | 100% [34, 100] | 0% [0, 66] |
| magnitude swap | 2 | 50% [9, 91] | 0% [0, 66] |
| missing citation | 2 | 100% [34, 100] | 100% [34, 100] |
| prediction as fact | 2 | 50% [9, 91] | 50% [9, 91] |
| protected inference | 2 | 100% [34, 100] | 100% [34, 100] |
| raw code | 2 | 0% [0, 66] | 0% [0, 66] |
| risk text swap | 2 | 100% [34, 100] | 100% [34, 100] |
| unsupported number | 2 | 100% [34, 100] | 50% [9, 91] |
| value swap | 2 | 100% [34, 100] | 100% [34, 100] |
| wrong ordinal | 1 | 100% [21, 100] | 100% [21, 100] |
| wrong top | 2 | 100% [34, 100] | 100% [34, 100] |
| **All faults** | 28 | 82% [64, 92] | 68% [49, 82] |

Clean texts: 8, false rejections 0% [0, 32].

## Faults: Template faults (upper bound)

Sources: 1 file(s), 3 chunk(s); domains: finance, health, hr.

| Fault type | n | Blocked | Exact code |
|---|---|---|---|
| blame wording | 540 | 100% [99, 100] | 100% [99, 100] |
| causal claim | 540 | 100% [99, 100] | 100% [99, 100] |
| decision wording | 540 | 100% [99, 100] | 100% [99, 100] |
| direction flip | 270 | 100% [99, 100] | 100% [99, 100] |
| fake feature | 540 | 100% [99, 100] | 100% [99, 100] |
| magnitude swap | 180 | 100% [98, 100] | 100% [98, 100] |
| missing citation | 270 | 100% [99, 100] | 100% [99, 100] |
| prediction as fact | 540 | 100% [99, 100] | 100% [99, 100] |
| protected inference | 540 | 100% [99, 100] | 100% [99, 100] |
| raw code | 540 | 100% [99, 100] | 100% [99, 100] |
| risk text swap | 540 | 100% [99, 100] | 100% [99, 100] |
| unsupported number | 540 | 100% [99, 100] | 100% [99, 100] |
| value swap | 270 | 100% [99, 100] | 100% [99, 100] |
| wrong ordinal | 270 | 100% [99, 100] | 100% [99, 100] |
| wrong top | 270 | 100% [99, 100] | 100% [99, 100] |
| **All faults** | 6390 | 100% [100, 100] | 100% [100, 100] |

Clean texts: 2135, false rejections 0% [-0, 0].

## Caveats to state in the paper

- Reduced study: finance and health only, "explain" questions, novice and expert levels, few cases per domain; the intervals are wide. HR is evaluated without an LLM (models, fairness, verifier).
- Baselines are our reimplementations of the published methods, run with the same writer model.
- Measurement shares the text and policy rules our pipeline gates on, which favours it; the LLM-scorer figure and the human labels are the measures without that bias.
- Open-weights models on free-tier APIs (Qwen 3.8 27B and gpt-oss on Groq, Gemma 4 26B on the Gemini API); the ablations use a different writer from the main comparison.
- Generation failures (empty or error replies from the provider) are excluded and counted per system.
