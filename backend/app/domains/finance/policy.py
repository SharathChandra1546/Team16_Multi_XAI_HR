"""Finance DomainPolicy: protected attributes, actionability, wording rules and the credit-decision guardrail.

Creditworthiness assessment of natural persons is a high-risk use under the EU AI Act (Annex III, point 5(b)). Lenders
must also be able to give the principal reasons for adverse credit decisions (e.g. ECOA / Regulation B adverse-action
notices in the US). This tool therefore explains risk estimates but never makes, recommends or ranks a credit
decision, never sets terms, and never uses or infers age, sex, marital status or nationality in the main model or in
explanations.
"""
from __future__ import annotations

from ..base import Actionability, DomainLanguage, DomainPolicy, TextRule
from .spec import PROTECTED

A = Actionability

LANGUAGE = DomainLanguage(
    subject="the applicant",
    subject_indefinite="an applicant",
    subject_plural="applicants",
    subject_possessive="the applicant's",
    reader="a credit analyst",
    outcome_verb="is a bad credit risk",
    outcome_gerund="being a bad credit risk",
    risk_name="credit risk",
    model_description="a credit-risk model",
    decision_kind="a credit decision",
    decision_examples="approving, declining or changing the terms of credit",
    protected_description="age, sex, marital status or nationality",
    character_examples='"irresponsible", "reckless with money", "their fault"',
    outcome_terms=["defaults?", "defaulting", "bad credit", "non-?payment", "missed payments"],
    subject_terms=["applicants", "borrowers", "customers", "them", "the applicant", "the borrower", "him", "her"],
    outcome_verbs=["default", "repay", "miss payments"],
    actor_terms=["the bank", "the lender"],
    lever_terms=["guarantor", "co-applicant", "smaller loan", "shorter loan", "shorter term", "lower amount",
                 "borrowing less", "more savings"],
    missing_instance_message="Provide the applicant's details to explain a risk estimate.",
    batch_question="Why is this applicant's credit risk estimated this way?",
    default_question="Why is this applicant flagged as a credit risk?",
    chat_more_details="Could you tell me a bit more about the applicant and the loan?",
    reference_column="application_id",
    fault_sentences={
        "causal_claim": "A long loan like this causes default.",
        "decision_wording": "The application should be declined.",
        "protected_inference": "As a younger applicant, they may be less stable.",
        "blame_wording": "The applicant seems irresponsible with money.",
        "prediction_as_fact": "The applicant will default.",
        "raw_code": "Profile code: checking_status.",
    },
    data_notes={
        "synthetic_standin": "The model was trained on synthetic stand-in data for software testing.",
        "uci_statlog": ("The model was trained on the Statlog German Credit dataset (1000 applicants from the 1970s-80s, "
                        "amounts in Deutsche Mark)."),
    },
)

ACTIONABILITY: dict[str, Actionability] = {
    # levers: the terms of the requested credit, and security for it
    "credit_amount": A("actionable", "The applicant can request a smaller amount."),
    "duration": A("actionable", "The loan term can be shortened."),
    "other_debtors": A("actionable", "A guarantor or co-applicant can be added."),
    "savings": A("partial", "Savings can grow over time; modelled as the next savings category."),
    "checking_status": A("partial", "A checking balance can recover over time; modelled as the next category."),
    # side effect
    "installment_rate": A("consequence", "Changes when the amount or the term changes."),
    # not changed in recourse
    "credit_history": A("immutable", "Past repayment history cannot change."),
    "existing_credits": A("immutable", "Not changed in recourse."),
    "other_installment_plans": A("immutable", "Not changed in recourse."),
    "purpose": A("immutable", "The purpose of the credit is given."),
    "employment": A("immutable", "Employment history is not changed in recourse."),
    "job": A("immutable", "Not changed in recourse."),
    "property": A("immutable", "Not changed in recourse."),
    "housing": A("immutable", "Not changed in recourse."),
    "residence_since": A("immutable", "Not changed in recourse."),
    "num_dependents": A("immutable", "Not changed in recourse."),
    "telephone": A("immutable", "Not changed in recourse."),
    "age": A("immutable", "Protected attribute; never changed."),
    "personal_status_sex": A("immutable", "Protected attribute; never changed."),
    "foreign_worker": A("immutable", "Protected attribute; never changed."),
}

# ---------------------------------------------------------------- critic wording rules (one code per fault type)
_DECIDE = r"(?:approv(?:e|ed|es|ing|al)|declin(?:e|ed|es|ing)|den(?:y|ied|ies|ying)|reject(?:ed|s|ing|ion)?|refus(?:e|ed|al))"
DECISION_WORDING = TextRule(
    code="decision_wording",
    message="Recommends or implies a credit decision about the applicant.",
    patterns=[
        rf"\b(?:should|must|ought to|needs? to|recommend(?:ed|s)?|advis(?:e|ed|able)|suggest(?:ed|s)?)\b(?:\W+\w+){{0,5}}?\W+{_DECIDE}\b",
        rf"\b(?:application|applicant|loan|credit|request)\s+(?:should|must|ought to)\s+be\s+{_DECIDE}\b",
        r"\b(?:grant|extend|refuse|withhold)\s+(?:the\s+|them\s+|him\s+|her\s+)?(?:credit|the loan|a loan)\b",
        r"\b(?:raise|increase|lower|cut|reduce)\s+(?:the\s+|their\s+|his\s+|her\s+)?(?:interest rate|credit limit)\b",
        r"\b(?:not\s+)?creditworthy\b|\b(?:worth|safe to)\s+lend(?:ing)?\b",
        r"\b(?:send|refer)\b.{0,20}\bcollections?\b|\bblacklist\w*\b",
    ],
)
PROTECTED_INFERENCE = TextRule(
    code="protected_attribute_inference",
    message="Mentions or infers a protected characteristic (age, sex, marital status or nationality).",
    negatable=False,
    patterns=[
        r"\b(?:age|aged|ages|young|younger|youngest|old|older|oldest|elderly|retire\w*|pensioners?)\b",
        r"\b(?:sex|gender|male|female|man|woman|men|women|he|she|him|his|hers|her|himself|herself)\b",
        r"\b(?:marital|married|unmarried|divorced|separated|widowed|spouse|husband|wife|single (?:man|woman|mother|father))\b",
        r"\b(?:foreign(?:er|ers)?|immigrant\w*|migrant\w*|nationality|citizenship|native|ethnic\w*|race|religio\w*)\b",
    ],
)
BLAME_WORDING = TextRule(
    code="blame_wording",
    message="Blames or judges the applicant's character.",
    patterns=[
        r"\b(?:irresponsible|reckless|careless|spendthrift|untrustworthy|unreliable|lazy|deadbeat|greedy|feckless)\b",
        r"\b(?:bad|terrible|poor) with money\b|\blives? beyond (?:their|his|her) means\b",
        r"\b(?:their|his|her|the applicant's) (?:own )?fault\b|\b(?:is|are) to blame\b|\bblam(?:e|es|ed|ing)\b",
    ],
)
PREDICTION_AS_FACT = TextRule(
    code="prediction_as_fact",
    message="States the predicted outcome as a fact instead of an estimate.",
    patterns=[
        r"\b(?:will|is going to|are going to|is certain to|is sure to|is bound to)\s+(?:definitely\s+|certainly\s+|probably\s+)?"
        r"(?:default|repay|pay (?:it )?back|miss payments|fail to (?:pay|repay))\b",
        r"\b(?:definitely|certainly|undoubtedly|for sure|without (?:a )?doubt|guaranteed)\b",
        r"\b(?:is|are) (?:a )?(?:certain|guaranteed|sure) (?:default|bad risk)\b",
    ],
)

# ---------------------------------------------------------------- credit-decision guardrail (queries)
_WHO = r"(?:this (?:applicant|person|customer|borrower|one|application|loan|request)|the (?:applicant|loan|application|credit|customer|borrower)|him|her|them|someone|people)"
GUARD_HARD = [
    rf"\b(?:approve|decline|deny|reject|refuse|turn down|fund)\s+(?:the\s+)?(?:{_WHO}|loans?|credit|applications?|mortgage|request)\b",
    rf"\b(?:should|shall|do|can|would)\s+(?:we|i|you|the bank)\s+(?:approve|decline|deny|reject|refuse|lend|grant|fund|turn down)\b",
    r"\b(?:approve|reject|decline|deny)\s*(?:or|/)\s*(?:approve|reject|decline|deny)\b",
    r"\b(?:who|which (?:applicants?|borrowers?|customers?|ones?|people))\s+(?:should|to|do we|shall we|gets?|deserves?|is)\b.{0,40}\b(?:approve|reject|decline|deny|lend|loans?|credit|fund|turn down)\w*",
    r"\b(?:rank|shortlist|pick|choose|select|prioriti[sz]e)\b.{0,40}\b(?:applicants?|borrowers?|customers?|applications?)\b",
    r"\b(?:interest rate|credit limit|deposit|collateral)\b.{0,40}\b(?:should|set|give|charge|raise|increase|lower|cut|reduce|demand)\b",
    r"\b(?:should|set|give|charge|raise|increase|lower|cut|reduce|demand|ask for)\b.{0,40}\b(?:interest rate|credit limit|deposit|collateral)\b",
    r"\b(?:collections?|blacklist\w*|close (?:his|her|their|this|the) (?:customer's )?account)\b",
    r"\b(?:rejection|refusal|decline) (?:reason|letter|notice)\b",
    r"\b(?:make|take) (?:the|a) (?:credit|lending|loan) decision\b",
    r"\b(?:go ahead and|please)\s+(?:approve|reject|decline|deny)\b|^\s*(?:approve|reject|decline|deny)\b",
    r"\b(?:sign off on|extend credit to|lend (?:this|him|her|them|the))\b",
    r"\bjustify (?:declining|rejecting|denying|refusing)\b",
    r"\b(?:smaller loan|guarantor|co-?applicant)\b.{0,30}\bbefore approv\w*",
    r"\boffer (?:this customer|him|her|them|the applicant) a\b",
    r"\bflag\b.{0,30}\b(?:as )?(?:a )?reject\b",
]
GUARD_SOFT = [  # refused only when framed as a decision
    r"\b(?:approv\w*|lend\w*|grant\w*|credit\s*worth\w*|creditworthy|eligible|eligibility)\b",
    r"\b(?:reject\w*|declin\w*|den(?:y|ying|ied)|refus\w*|turn(?:ing)? down)\b",
    r"\b(?:loan|credit|mortgage|credit card)\b",
]
GUARD_FRAMES = [
    r"\b(?:should|shall|must|ought|deserves?|deserving|worth|safe to|wise to)\b",
    r"\bis (?:it|this|he|she|they|the applicant) (?:ok|okay|fine|wise|right|smart|fair|a good idea|creditworthy|eligible|worth|a good candidate|denied)\b",
    r"\b(?:do you|would you) (?:recommend|advise|suggest|think we should|give|lend|approve)\b|\brecommend\b|\badvise\b|\bgood idea\b",
    r"\b(?:yes or no|right call|good candidate|be denied|be approved|be rejected)\b",
    r"\b(?:can|may)\s+(?:we|i)\s+(?:approve|lend|give)\b",
    r"\bcreditworthy enough\b",
    r"\bis (?:this|the) (?:applicant|person|customer|borrower) (?:creditworthy|eligible|worth|a good (?:candidate|risk))\b",
]

POLICY = DomainPolicy(
    sensitive=list(PROTECTED),
    actionability=ACTIONABILITY,
    disclosure=(
        "This is a model-based credit-risk estimate, not a credit decision. It reflects patterns in historical lending "
        "data, and a person must review it before any decision is made."
    ),
    high_risk_notice="High estimated credit risk: a person must review this application before any decision is made.",
    escalation_uncertain=(
        "Routed to human review. The model's estimate for this applicant is too uncertain to explain automatically: "
        "it ranges from {low}% to {high}%, which spans the {band_a} and {band_b} risk bands."
    ),
    escalation_ood=(
        "Routed to human review. This applicant's profile is unlike the data the model was trained on ({reasons}), "
        "so an automated explanation would not be reliable."
    ),
    refusal_message=(
        "This request asks for a credit decision, which this tool does not make or recommend. It estimates credit risk "
        "and explains that estimate. Approving, declining or setting the terms of credit needs people, lending policy "
        "and information the model does not have."
    ),
    guard_patterns=GUARD_HARD,
    critic_rules=[DECISION_WORDING, PROTECTED_INFERENCE, BLAME_WORDING, PREDICTION_AS_FACT],
    guard_soft=GUARD_SOFT,
    guard_frames=GUARD_FRAMES,
    causal_extra=[r"\b(?:makes?|making)\s+(?:them|the applicant|applicants|borrowers)\s+(?:default|miss payments)\b"],
    min_risk_drop=0.05,
    language=LANGUAGE,
)
