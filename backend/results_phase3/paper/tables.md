# Phase 3 results (reduced real-LLM study): finance and health

Generated 2026-10-24T21:41:39+00:00 by `backend/eval/paper_tables.py` from 90 run files. Every text was re-measured with the current measurement code. Do not edit numbers by hand.

## Table 1. Main comparison

Writer for every system: Qwen 3.8 27B (Groq). Each system's own review/critique steps use the same writer. Scorer: gpt-oss-120b (Groq, a different model family). Questions: "why this estimate" (explain); levels: novice and expert.

| System | n | Error-free (pooled) | Finance | Health | Error-free (rule-based) | Policy viol. | VA | SA | RA | Calls | Tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours (full pipeline) | 12 | 100% [76, 100] | 100% [61, 100] | 100% [61, 100] | 100% [76, 100] | 0% [0, 24] | 1.00 | 1.00 | 1.00 | 2.8 | 3.6k |
| Direct prompting | 12 | 42% [19, 68] | 33% [10, 70] | 50% [19, 81] | 33% [14, 61] | 0% [0, 24] | 1.00 | 1.00 | 0.78 | 1.0 | 0.3k |
| B3 (LLM reviewer loop) | 8 | 75% [41, 93] | 75% [30, 95] | 75% [30, 95] | 88% [53, 98] | 0% [0, 32] | 1.00 | 1.00 | 0.63 | 2.2 | 3.2k |
| Explingo-style | 7 | 43% [16, 75] | 40% [12, 77] | 50% [9, 91] | 43% [16, 75] | 0% [0, 35] | 0.87 | 1.00 | 1.00 | 2.8 | 1.3k |
| Critic-Rule (He & Martens) | 6 | 50% [19, 81] | 50% [15, 85] | 50% [9, 91] | 50% [19, 81] | 0% [0, 39] | 1.00 | 1.00 | 0.89 | 2.7 | 2.8k |
| Musicco et al.-style | 5 | 20% [4, 62] | 33% [6, 79] | 0% [0, 66] | 20% [4, 62] | 20% [4, 62] | 1.00 | 1.00 | 0.74 | 3.6 | 7.1k |

### Paired tests (same case and level)

| Baseline | Pairs | Only ours correct | Only baseline correct | Both | Neither | Exact McNemar p |
|---|---|---|---|---|---|---|
| Direct prompting | 12 | 7 | 0 | 5 | 0 | 0.0156 |
| B3 (LLM reviewer loop) | 8 | 2 | 0 | 6 | 0 | 0.5 |
| Explingo-style | 7 | 4 | 0 | 3 | 0 | 0.125 |
| Critic-Rule (He & Martens) | 6 | 3 | 0 | 3 | 0 | 0.25 |
| Musicco et al.-style | 5 | 4 | 0 | 1 | 0 | 0.125 |

### Error types (texts with at least one)

| System | Texts | Value | Direction | Stated rank | Unsupported factor | Unsupported number | Causal wording | Raw code | Policy |
|---|---|---|---|---|---|---|---|---|---|
| Ours (full pipeline) | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Direct prompting | 12 | 0 | 0 | 1 | 0 | 7 | 1 | 1 | 0 |
| B3 (LLM reviewer loop) | 8 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 |
| Explingo-style | 7 | 2 | 0 | 0 | 0 | 4 | 0 | 0 | 0 |
| Critic-Rule (He & Martens) | 6 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 |
| Musicco et al.-style | 5 | 0 | 0 | 1 | 0 | 2 | 1 | 0 | 1 |

## Table 2. Ablations

Writer: Gemma 4 26B (Gemini API) for every variant; scorer gpt-oss-120b.

| Variant | Texts | Error-free | Any fidelity error | Policy viol. | Template fallback | Withheld/escalated | Calls |
|---|---|---|---|---|---|---|---|
| Full pipeline | 40 | 98% [87, 100] | 2% [0, 13] | 0% [-0, 9] | 18% | 0% | 4.4 |
| -- claim verifier | 40 | 98% [87, 100] | 2% [0, 13] | 0% [-0, 9] | 10% | 0% | 4.3 |
| -- critic | 40 | 100% [91, 100] | 0% [-0, 9] | 0% [-0, 9] | 2% | 0% | 2.1 |
| -- repair loop | 40 | 100% [91, 100] | 0% [-0, 9] | 0% [-0, 9] | 62% | 0% | 5.1 |
| -- policy rules | 40 | 98% [87, 100] | 2% [0, 13] | 0% [-0, 9] | 18% | 0% | 4.4 |

## Faults: LLM-written faults (gpt-oss-20b; detector-independent)

Sources: 3 file(s), 3 chunk(s); domains: finance, health.

| Fault type | n | Blocked | Exact code |
|---|---|---|---|
| blame wording | 3 | 33% [6, 79] | 33% [6, 79] |
| causal claim | 4 | 100% [51, 100] | 100% [51, 100] |
| decision wording | 4 | 100% [51, 100] | 100% [51, 100] |
| direction flip | 4 | 100% [51, 100] | 100% [51, 100] |
| fake feature | 4 | 100% [51, 100] | 0% [0, 49] |
| magnitude swap | 4 | 50% [15, 85] | 0% [0, 49] |
| missing citation | 4 | 100% [51, 100] | 100% [51, 100] |
| prediction as fact | 3 | 33% [6, 79] | 33% [6, 79] |
| protected inference | 4 | 100% [51, 100] | 75% [30, 95] |
| raw code | 4 | 0% [0, 49] | 0% [0, 49] |
| risk text swap | 4 | 100% [51, 100] | 100% [51, 100] |
| unsupported number | 4 | 100% [51, 100] | 50% [15, 85] |
| value swap | 4 | 100% [51, 100] | 100% [51, 100] |
| wrong ordinal | 2 | 50% [9, 91] | 50% [9, 91] |
| wrong top | 4 | 100% [51, 100] | 100% [51, 100] |
| **All faults** | 56 | 80% [68, 89] | 64% [51, 76] |

Clean texts: 16, false rejections 0% [0, 19].

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
