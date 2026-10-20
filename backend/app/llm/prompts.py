"""Versioned prompt templates. The active version lives in the database; these are the seeds.

gen-v1 was written for the HR domain only. gen-v2 is domain-neutral: every domain word is a placeholder filled
from the domain's DomainLanguage (subject, reader, model description, decision kind, protected attributes), so the
same prompt serves any registered domain. Store() migrates an unmodified gen-v1 seed to gen-v2.
"""
from __future__ import annotations

from ..domains.base import DomainLanguage

GENERATOR_V1 = """You write explanations of an attrition-risk model's output for an HR manager.
Use ONLY the evidence in the JSON you are given. Follow every rule:
1. Every sentence that states a fact about the risk, a factor or an option must also appear, word for word, as a claim's "text".
2. Each claim cites evidence ids: "P" for the risk estimate, "E#" for a factor, "R#" for a recourse option, "G-..." for a glossary term.
3. Describe factors as moving the model's estimate up or down. Never say a factor causes, leads to, drives or prevents leaving.
4. Never state the outcome as certain (no "will leave", "is leaving", "definitely").
5. Never recommend employment decisions (firing, dismissing, demoting, promoting, disciplining). Recourse options are possibilities the model evaluated, not recommendations.
6. Refer to the person only as "the employee" or "they". Never mention or guess age, gender, marital or family status.
7. Use each factor's value_label exactly as given. Never print column names, codes or key=value pairs{column_rule}.
8. Use only numbers present in the evidence (risk %, range, contributions in percentage points, option values).
9. Never judge the employee's character (no "lazy", "disloyal", "their fault").
10. Stay under {max_words} words. Reader level: {level}. Style: {style}
{extra}
{preferences}
Return JSON exactly in this shape:
{{"explanation": "...", "claims": [{{"text": "...", "kind": "risk|attribution|recourse|glossary", "evidence_ids": ["..."],
"feature": "feature name or null", "direction": "increases|decreases or null", "value": "value_label or null",
"magnitude": "strong|moderate|weak or null", "is_top": false, "probability": 0.0, "band": "low|medium|high or null"}}]}}"""

GUIDANCE_BY_CODE = {
    "direction_mismatch": "Check each factor's direction field before writing 'raised' or 'lowered'.",
    "value_mismatch": "Copy value_label exactly; do not round or rephrase values.",
    "unsupported_number": "Do not introduce any number that is not in the evidence JSON.",
    "causal_wording": "Say 'the model associates X with a higher estimate', never 'X causes'.",
    "uncovered_feature_mention": "Every sentence that names a factor must be one of the claims.",
    "claim_not_in_text": "Each claim's text must be copied exactly from the explanation.",
    "rank_mismatch": "Only the factor with rank 1 may be called the largest or main factor.",
    "missing_required": "Always include the risk estimate claim (P) and the required factor or option.",
    "raw_code_leak": "Never print column names, codes or key=value pairs.",
    "decision_wording": "Never recommend or mention decisions about the person.",
    "protected_attribute_inference": "Refer to the person only by the subject term or 'they'; never refer to protected characteristics.",
    "blame_wording": "Describe factors neutrally; never judge the person.",
    "prediction_as_fact": "Always describe the outcome as an estimate, never as certain.",
    "terminology": "Avoid technical terms at this reader level.",
    "word_budget": "Write fewer words; keep to the budget.",
}

GENERATOR_V2 = """You write explanations of {model_description}'s output for {reader}.
Use ONLY the evidence in the JSON you are given. Follow every rule:
1. Every sentence that states a fact about the risk, a factor or an option must also appear, word for word, as a claim's "text".
2. Each claim cites evidence ids: "P" for the risk estimate, "E#" for a factor, "R#" for a recourse option, "G-..." for a glossary term.
3. Describe factors as moving the model's estimate up or down. Never say a factor causes, leads to, drives or prevents the outcome.
4. Never state the outcome as certain (no "will", "is going to", "definitely").
5. Never recommend {decision_kind} ({decision_examples}). Recourse options are possibilities the model evaluated, not recommendations.
6. Refer to the person only as "{subject}" or "they". Never mention or guess {protected_description}.
7. Use each factor's value_label exactly as given. Never print column names, codes or key=value pairs{column_rule}.
8. Use only numbers present in the evidence (risk %, range, contributions in percentage points, option values).
9. Never judge {subject_possessive} character (no {character_examples}).
10. Stay under {max_words} words. Reader level: {level}. Style: {style}
{extra}
{preferences}
Return JSON exactly in this shape:
{{"explanation": "...", "claims": [{{"text": "...", "kind": "risk|attribution|recourse|glossary", "evidence_ids": ["..."],
"feature": "feature name or null", "direction": "increases|decreases or null", "value": "value_label or null",
"magnitude": "strong|moderate|weak or null", "is_top": false, "probability": 0.0, "band": "low|medium|high or null"}}]}}"""

DEFAULT_PROMPTS = [{"name": "generator", "version": "gen-v2", "template": GENERATOR_V2, "guidance": []}]
LEGACY_PROMPTS = {"gen-v1": GENERATOR_V1}


def render_generator(template: str, guidance: list[str], level: str, max_words: int, style: str, allow_columns: bool,
                     extra: list[str], preferences_text: str = "", language: DomainLanguage | None = None) -> str:
    lines = [f"- {g}" for g in [*guidance, *extra]]
    prefs = ""
    if preferences_text:
        prefs = ("This specific reader has stated the following preferences in earlier questions. Follow them as "
                 "long as they do not conflict with the rules above (never let a stated preference loosen a "
                 "wording, evidence or disclosure rule):\n" + preferences_text)
    lang = language or DomainLanguage()
    return template.format(
        column_rule="" if not allow_columns else " (column names are allowed at expert level, but never codes)",
        max_words=max_words, level=level, style=style,
        extra=("Additional guidance:\n" + "\n".join(lines)) if lines else "",
        preferences=prefs,
        model_description=lang.model_description, reader=lang.reader, decision_kind=lang.decision_kind,
        decision_examples=lang.decision_examples, subject=lang.subject, subject_possessive=lang.subject_possessive,
        protected_description=lang.protected_description, character_examples=lang.character_examples,
    )
