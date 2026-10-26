# Phase 3 results (reduced real-LLM study): finance and health

Generated 2026-10-26T01:12:13+00:00 by `backend/eval/paper_tables.py` from 116 run files. Every text was re-measured with the current measurement code. Do not edit numbers by hand.

## Table 1. Main comparison

Writer for every system: Qwen 3.8 27B (Groq). Each system's own review/critique steps use the same writer. Scorer: gpt-oss-120b (Groq, a different model family). Questions: "why this estimate" (explain); levels: novice and expert.

| System | n | Error-free (pooled) | Finance | Health | Error-free (rule-based) | Policy viol. | VA | SA | RA | Calls | Tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours (full pipeline) | 16 | 100% [81, 100] | 100% [68, 100] | 100% [68, 100] | 100% [81, 100] | 0% [0, 19] | 1.00 | 1.00 | 1.00 | 4.2 | 6.0k |
| Direct prompting | 16 | 44% [23, 67] | 38% [14, 69] | 50% [22, 78] | 31% [14, 56] | 0% [0, 19] | 1.00 | 1.00 | 0.84 | 1.0 | 0.3k |
| B3 (LLM reviewer loop) | 16 | 75% [51, 90] | 75% [41, 93] | 75% [41, 93] | 75% [51, 90] | 0% [0, 19] | 1.00 | 1.00 | 0.75 | 2.1 | 3.1k |
| Explingo-style | 15 | 47% [25, 70] | 38% [14, 69] | 57% [25, 84] | 47% [25, 70] | 0% [0, 20] | 0.91 | 1.00 | 1.00 | 2.3 | 1.6k |
| Critic-Rule (He & Martens) | 15 | 67% [42, 85] | 50% [22, 78] | 86% [49, 97] | 67% [42, 85] | 0% [0, 20] | 1.00 | 1.00 | 0.90 | 4.1 | 4.8k |
| Musicco et al.-style | 15 | 40% [20, 64] | 38% [14, 69] | 43% [16, 75] | 40% [20, 64] | 7% [1, 30] | 1.00 | 1.00 | 0.81 | 4.3 | 9.0k |

### Paired tests (same case and level)

| Baseline | Pairs | Only ours correct | Only baseline correct | Both | Neither | Exact McNemar p |
|---|---|---|---|---|---|---|
| Direct prompting | 16 | 9 | 0 | 7 | 0 | 0.00391 |
| B3 (LLM reviewer loop) | 16 | 4 | 0 | 12 | 0 | 0.125 |
| Explingo-style | 15 | 8 | 0 | 7 | 0 | 0.00781 |
| Critic-Rule (He & Martens) | 15 | 5 | 0 | 10 | 0 | 0.0625 |
| Musicco et al.-style | 15 | 9 | 0 | 6 | 0 | 0.00391 |

### Error types (texts with at least one)

| System | Texts | Value | Direction | Stated rank | Unsupported factor | Unsupported number | Causal wording | Raw code | Policy |
|---|---|---|---|---|---|---|---|---|---|
| Ours (full pipeline) | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Direct prompting | 16 | 0 | 0 | 1 | 0 | 9 | 1 | 2 | 0 |
| B3 (LLM reviewer loop) | 16 | 0 | 0 | 4 | 0 | 0 | 1 | 0 | 0 |
| Explingo-style | 15 | 3 | 0 | 0 | 0 | 8 | 0 | 0 | 0 |
| Critic-Rule (He & Martens) | 15 | 0 | 0 | 2 | 0 | 2 | 3 | 0 | 0 |
| Musicco et al.-style | 15 | 0 | 0 | 1 | 0 | 5 | 3 | 0 | 1 |

## Table 2. Ablations

Writer: Gemma 4 26B (Gemini API) for every variant; scorer gpt-oss-120b.

| Variant | Texts | Error-free | Any fidelity error | Policy viol. | Template fallback | Withheld/escalated | Calls |
|---|---|---|---|---|---|---|---|
| Full pipeline | 40 | 98% [87, 100] | 2% [0, 13] | 0% [0, 9] | 18% | 0% | 4.4 |
| -- claim verifier | 40 | 98% [87, 100] | 2% [0, 13] | 0% [0, 9] | 10% | 0% | 4.3 |
| -- critic | 40 | 100% [91, 100] | 0% [0, 9] | 0% [0, 9] | 2% | 0% | 2.1 |
| -- repair loop | 40 | 100% [91, 100] | 0% [0, 9] | 0% [0, 9] | 62% | 0% | 5.1 |
| -- policy rules | 40 | 98% [87, 100] | 2% [0, 13] | 0% [0, 9] | 18% | 0% | 4.4 |

## Faults: LLM-written faults (gpt-oss-20b; detector-independent)

Sources: 12 file(s), 12 chunk(s); domains: finance, health.

| Fault type | n | Blocked | Exact code |
|---|---|---|---|
| blame wording | 9 | 78% [45, 94] | 67% [35, 88] |
| causal claim | 10 | 100% [72, 100] | 100% [72, 100] |
| decision wording | 10 | 100% [72, 100] | 90% [60, 98] |
| direction flip | 10 | 100% [72, 100] | 100% [72, 100] |
| fake feature | 10 | 100% [72, 100] | 0% [0, 28] |
| magnitude swap | 9 | 56% [27, 81] | 0% [0, 30] |
| missing citation | 10 | 100% [72, 100] | 100% [72, 100] |
| prediction as fact | 8 | 38% [14, 69] | 38% [14, 69] |
| protected inference | 10 | 100% [72, 100] | 90% [60, 98] |
| raw code | 10 | 20% [6, 51] | 20% [6, 51] |
| risk text swap | 9 | 100% [70, 100] | 100% [70, 100] |
| unsupported number | 9 | 89% [56, 98] | 33% [12, 65] |
| value swap | 10 | 100% [72, 100] | 100% [72, 100] |
| wrong ordinal | 6 | 17% [3, 56] | 17% [3, 56] |
| wrong top | 11 | 100% [74, 100] | 100% [74, 100] |
| **All faults** | 141 | 82% [75, 88] | 66% [58, 73] |

Clean texts: 93, false rejections 0% [0, 4].

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

Clean texts: 2135, false rejections 0% [0, 0].

## Caveats to state in the paper

- Reduced study: finance and health only, "explain" questions, novice and expert levels, few cases per domain; the intervals are wide. HR is evaluated without an LLM (models, fairness, verifier).
- Baselines are our reimplementations of the published methods, run with the same writer model.
- Measurement shares the text and policy rules our pipeline gates on, which favours it; the LLM-scorer figure and the human labels are the measures without that bias.
- Open-weights models on free-tier APIs (Qwen 3.8 27B and gpt-oss on Groq, Gemma 4 26B on the Gemini API); the ablations use a different writer from the main comparison.
- Generation failures (empty or error replies from the provider) are excluded and counted per system.
