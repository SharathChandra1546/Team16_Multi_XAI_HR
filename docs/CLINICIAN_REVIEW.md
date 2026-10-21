# Clinician review of the healthcare domain

The healthcare domain estimates diabetes risk (Pima Indians Diabetes data) and explains it for a clinician. Before the
paper claims clinical appropriateness, a clinician reviews the items below. Fill in the sign-off at the end.

## What to review

1. **Policy** (`backend/app/domains/health/policy.py`): the refusal patterns (diagnosis, treatment, medication, referral,
   coverage, prioritisation), the wording rules (no diagnosis stated as fact, no advice, no ethnicity or sex), the
   disclosure text. Anything missing or wrong?
2. **Cost matrix** (missed case : false alarm = 5 : 1): is the ratio defensible for a screening-style flag? What range
   would you consider reasonable? (The paper reports a sensitivity analysis over that range.)
3. **Levers** (BMI, blood pressure): are the modelled changes described as the model's behaviour and not advice? Is there
   any wording a clinician could read as advice?
4. **Sample explanations** (20 rows chosen by the annotation script): any statement that is clinically misleading even
   though it matches the evidence?
5. **Dataset limits**: the data are 768 women aged 21 or older from one population. Is the disclosure enough?

## Findings

| # | Item | Finding | Change made |
|---|---|---|---|
| 1 | | | |

## Sign-off

Reviewed by: ____
Role and institution: ____
Date: ____
Scope: policy wording, cost ratio range, levers, sample explanations (strike out what was not reviewed)
