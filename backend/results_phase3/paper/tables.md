# Phase 3 results (reduced real-LLM study): finance and health

Generated 2026-10-25T18:32:58+00:00 by `backend/eval/paper_tables.py` from 108 run files. Every text was re-measured with the current measurement code. Do not edit numbers by hand.

## Table 1. Main comparison

Writer for every system: Qwen 3.8 27B (Groq). Each system's own review/critique steps use the same writer. Scorer: gpt-oss-120b (Groq, a different model family). Questions: "why this estimate" (explain); levels: novice and expert.

| System | n | Error-free (pooled) | Finance | Health | Error-free (rule-based) | Policy viol. | VA | SA | RA | Calls | Tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours (full pipeline) | 13 | 100% [77, 100] | 100% [65, 100] | 100% [61, 100] | 100% [77, 100] | 0% [0, 23] | 1.00 | 1.00 | 1.00 | 4.2 | 5.4k |
| Direct prompting | 14 | 43% [21, 67] | 38% [14, 69] | 50% [19, 81] | 36% [16, 61] | 0% [0, 22] | 1.00 | 1.00 | 0.81 | 1.0 | 0.3k |
| B3 (LLM reviewer loop) | 13 | 85% [58, 96] | 86% [49, 97] | 83% [44, 97] | 85% [58, 96] | 0% [0, 23] | 1.00 | 1.00 | 0.72 | 2.2 | 3.1k |
| Explingo-style | 13 | 46% [23, 71] | 43% [16, 75] | 50% [19, 81] | 46% [23, 71] | 0% [0, 23] | 0.90 | 1.00 | 1.00 | 2.3 | 1.6k |
| Critic-Rule (He & Martens) | 13 | 69% [42, 87] | 57% [25, 84] | 83% [44, 97] | 69% [42, 87] | 0% [0, 23] | 1.00 | 1.00 | 0.91 | 4.0 | 4.5k |
| Musicco et al.-style | 13 | 38% [18, 64] | 43% [16, 75] | 33% [10, 70] | 38% [18, 64] | 8% [1, 33] | 1.00 | 1.00 | 0.85 | 4.4 | 9.2k |

### Paired tests (same case and level)

| Baseline | Pairs | Only ours correct | Only baseline correct | Both | Neither | Exact McNemar p |
|---|---|---|---|---|---|---|
| Direct prompting | 13 | 7 | 0 | 6 | 0 | 0.0156 |
| B3 (LLM reviewer loop) | 13 | 2 | 0 | 11 | 0 | 0.5 |
| Explingo-style | 13 | 7 | 0 | 6 | 0 | 0.0156 |
| Critic-Rule (He & Martens) | 13 | 4 | 0 | 9 | 0 | 0.125 |
| Musicco et al.-style | 13 | 8 | 0 | 5 | 0 | 0.00781 |

### Error types (texts with at least one)

| System | Texts | Value | Direction | Stated rank | Unsupported factor | Unsupported number | Causal wording | Raw code | Policy |
|---|---|---|---|---|---|---|---|---|---|
| Ours (full pipeline) | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Direct prompting | 14 | 0 | 0 | 1 | 0 | 8 | 1 | 2 | 0 |
| B3 (LLM reviewer loop) | 13 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 |
| Explingo-style | 13 | 3 | 0 | 0 | 0 | 7 | 0 | 0 | 0 |
| Critic-Rule (He & Martens) | 13 | 0 | 0 | 2 | 0 | 1 | 2 | 0 | 0 |
| Musicco et al.-style | 13 | 0 | 0 | 1 | 0 | 4 | 3 | 0 | 1 |

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
| blame wording | 8 | 75% [41, 93] | 62% [31, 86] |
| causal claim | 9 | 100% [70, 100] | 100% [70, 100] |
| decision wording | 9 | 100% [70, 100] | 89% [56, 98] |
| direction flip | 9 | 100% [70, 100] | 100% [70, 100] |
| fake feature | 9 | 100% [70, 100] | 0% [0, 30] |
| magnitude swap | 8 | 62% [31, 86] | 0% [0, 32] |
| missing citation | 9 | 100% [70, 100] | 100% [70, 100] |
| prediction as fact | 7 | 43% [16, 75] | 43% [16, 75] |
| protected inference | 9 | 100% [70, 100] | 89% [56, 98] |
| raw code | 9 | 11% [2, 44] | 11% [2, 44] |
| risk text swap | 8 | 100% [68, 100] | 100% [68, 100] |
| unsupported number | 9 | 89% [56, 98] | 33% [12, 65] |
| value swap | 9 | 100% [70, 100] | 100% [70, 100] |
| wrong ordinal | 5 | 20% [4, 62] | 20% [4, 62] |
| wrong top | 10 | 100% [72, 100] | 100% [72, 100] |
| **All faults** | 127 | 83% [75, 88] | 65% [57, 73] |

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
