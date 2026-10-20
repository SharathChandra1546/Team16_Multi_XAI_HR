"""HR DomainPolicy: protected attributes, actionability table, wording rules and the decision-use guardrail.

Employment decisions are listed as high-risk uses under the EU AI Act (Annex III), so this
domain refuses decision requests and requires human review of every estimate.
"""
from __future__ import annotations

from ..base import Actionability, DomainPolicy, TextRule
from .spec import PROTECTED

A = Actionability

ACTIONABILITY: dict[str, Actionability] = {
    # levers an organisation can change
    "OverTime": A("actionable", "Regular overtime can be reduced or removed."),
    "BusinessTravel": A("actionable", "Travel requirements can be reduced."),
    "MonthlyIncome": A("actionable", "Pay can change, bounded to +20% and the observed range for the job level."),
    "StockOptionLevel": A("actionable", "Stock option level can be raised by one step at a time."),
    "TrainingTimesLastYear": A("actionable", "Training sessions can be added."),
    "JobRole": A("actionable", "A role change within the same department."),
    "JobLevel": A("actionable", "Promotion by one level, with its consequences applied."),
    "DistanceFromHome": A(
        "partial",
        "Remote or hybrid work can change the effective commute. The dataset has no remote-work column, "
        "so lower distance is a modelling assumption.",
    ),
    # side effects of actions
    "YearsSinceLastPromotion": A("consequence", "Resets to 0 when a promotion is applied."),
    "YearsInCurrentRole": A("consequence", "Resets to 0 when a role change is applied."),
    # not changeable
    "YearsAtCompany": A("immutable", "Tenure never changes in recourse."),
    "TotalWorkingYears": A("immutable", "Work history never changes in recourse."),
    "NumCompaniesWorked": A("immutable", "Work history never changes in recourse."),
    "YearsWithCurrManager": A("immutable", "Not changed in recourse."),
    "Education": A("immutable", "Education is not changed in recourse."),
    "EducationField": A("immutable", "Education is not changed in recourse."),
    "Department": A("immutable", "Department changes are out of scope."),
    "PercentSalaryHike": A("immutable", "Records a past salary review."),
    "PerformanceRating": A("immutable", "A rating, not a lever."),
    "JobInvolvement": A("immutable", "A rating, not a direct lever."),
    "JobSatisfaction": A("immutable", "A rating, not a direct lever."),
    "EnvironmentSatisfaction": A("immutable", "A rating, not a direct lever."),
    "RelationshipSatisfaction": A("immutable", "A rating, not a direct lever."),
    "WorkLifeBalance": A("immutable", "A rating, not a direct lever."),
    "Age": A("immutable", "Protected attribute; never changed."),
    "Gender": A("immutable", "Protected attribute; never changed."),
    "MaritalStatus": A("immutable", "Protected attribute; never changed."),
}

# ---------------------------------------------------------------- critic wording rules (one code per fault type)
DECISION_WORDING = TextRule(
    code="decision_wording",
    message="Recommends or implies an employment decision about the person.",
    patterns=[
        r"\b(terminat(?:e|es|ed|ing|ion)|dismiss(?:es|ed|ing|al)?|sack(?:ed|ing)?|demot(?:e|es|ed|ing|ion))\b",
        r"\bfir(?:e|es|ed|ing)\b(?!\s+(?:drill|alarm|safety|hazard|exit))",
        r"\b(?:lay(?:s|ing)?\s+(?:them|him|her|the employee|this employee)\s+off|laid off|layoffs?|lay off)\b",
        r"\blet(?:ting)?\s+(?:them|him|her|the employee|this employee)\s+go\b",
        r"\b(?:should|must|ought to|needs? to|recommend(?:ed|s)?|advis(?:e|ed|able)|suggest(?:ed|s)?)\b"
        r"(?:\W+\w+){0,5}?\W+(?:promot(?:e|ed|ing)|replac(?:e|ed|ing)|disciplin(?:e|ed|ary)|reassign(?:ed)?|"
        r"transfer(?:red)?|penali[sz](?:e|ed)|punish(?:ed)?|hire|monitor(?:ed)?|warn(?:ed)?)\b",
        r"\b(?:deserv(?:e|es|ed|ing)|worth)\s+(?:a\s+)?(?:promotion|raise|keeping|firing|retaining)\b",
        r"\bcandidates?\s+for\s+(?:promotion|termination|dismissal|layoff|demotion|redundancy)\b",
        r"\bredundan(?:t|cy|cies)\b",
    ],
)
PROTECTED_INFERENCE = TextRule(
    code="protected_attribute_inference",
    message="Mentions or infers a protected characteristic (age, gender, marital or family status).",
    negatable=False,
    patterns=[
        r"\b(?:age|aged|ages|young|younger|youngest|old|older|oldest|elderly|millennials?|boomers?|gen z|retirement age)\b",
        r"\b(?:gender|male|female|man|woman|men|women|he|she|him|his|hers|her|himself|herself)\b",
        r"\b(?:mother|father|maternity|paternity|pregnan\w*|motherhood|fatherhood)\b",
        r"\b(?:marital|married|unmarried|divorced|divorcee|spouse|husband|wife|family plans|children|kids|childcare|parental)\b",
        r"\b(?:is|are|being|as an?|a)\s+single\b(?!\s+(?:factor|feature|reason|change|action|largest|biggest|most|driver|lever|option|step))",
    ],
)
BLAME_WORDING = TextRule(
    code="blame_wording",
    message="Blames or judges the employee's character.",
    patterns=[
        r"\b(?:lazy|laziness|disloyal(?:ty)?|ungrateful|entitled|unmotivated|unreliable|careless|troublemakers?|"
        r"slackers?|complainers?|whiny|whining)\b",
        r"\b(?:problem employee|bad attitude|poor attitude|attitude problem|weak link|not a team player)\b",
        r"\b(?:uncommitted|lacks? (?:of )?commitment|lack of loyalty)\b",
        r"\b(?:their|his|her|the employee's) (?:own )?fault\b",
        r"\b(?:is|are) to blame\b|\bblam(?:e|es|ed|ing)\b",
    ],
)
# Same code, but these phrases contain their own negation, so negation handling must not skip them.
BLAME_SELF_NEGATED = TextRule(
    code="blame_wording",
    message="Blames or judges the employee's character.",
    negatable=False,
    patterns=[r"\b(?:does not|doesn't|do not|don't) (?:care|try|bother)\b|\bnot committed\b|\bnot loyal\b"],
)
PREDICTION_AS_FACT = TextRule(
    code="prediction_as_fact",
    message="States the predicted outcome as a fact instead of an estimate.",
    patterns=[
        r"\b(?:will|is going to|are going to|is about to|is certain to|is sure to|is bound to|is set to)\s+"
        r"(?:definitely\s+|certainly\s+|soon\s+|probably\s+)?(?:leave|quit|resign|depart|stay|remain|walk out|move on)\b",
        r"\b(?:is|are)\s+(?:currently\s+|already\s+)?(?:leaving|quitting|resigning|departing)\b",
        r"\b(?:definitely|certainly|undoubtedly|for sure|without (?:a )?doubt|guaranteed)\b",
        r"\bhas decided to (?:leave|quit|resign)\b",
    ],
)

# ---------------------------------------------------------------- decision-use guardrail (queries)
GUARD_HARD = [
    r"\b(?:fire|fired|firing|terminate|terminating|termination|dismiss|dismissing|dismissal|sack|sacking|axe|demote|demoting|demotion|downsize|downsizing|redundanc(?:y|ies)|make (?:him|her|them|this employee|the employee) redundant)\b",
    r"\b(?:lay(?:ing)? off|laid off|layoffs?)\b",
    r"\blay(?:ing)?\s+(?:him|her|them|this (?:employee|person)|the employee|someone|people|staff)\s+off\b",
    r"\bredundant\b",
    r"\bwho (?:should|shall|must|do we|should we|to)\s+(?:\w+\s+){0,3}(?:keep|go|let go|cut|stay|retain)\b",
    r"\blet(?:ting)?\s+(?:him|her|them|this (?:employee|person|guy|woman|man)|the employee|someone|people)\s+go\b",
    r"\b(?:get(?:ting)? rid of|show (?:him|her|them) the door|boot (?:him|her|them)|push (?:him|her|them) out|force (?:him|her|them) out|manage (?:him|her|them) out)\b",
    r"\b(?:replace|discipline|punish|penali[sz]e)\s+(?:him|her|them|this|the employee|someone|people|staff)\b",
    r"\bcut (?:his|her|their|the employee's) (?:pay|salary|wages|hours|bonus)\b",
    r"\b(?:performance improvement plan|put (?:him|her|them) on a pip)\b",
]
GUARD_SOFT = [  # promotion-type decisions: refused when framed as a decision
    r"\bpromot(?:e|es|ed|ing|ion|ions)\b",
    r"\b(?:raise|bonus|pay rise)\b",
    r"\b(?:keep|keeping|retain|retaining)\s+(?:him|her|them|this (?:employee|person)|the employee|someone)\b",
]
GUARD_FRAMES = [
    r"\b(?:should|shall|must|ought|deserves?|deserving|worth|eligible|candidates?|ready for|merit)\b",
    r"\b(?:who|whom|which (?:employee|employees|person|people|one|ones|staff|worker|workers|candidate|candidates))\b",
    r"\b(?:rank|ranking|shortlist|pick|choose|select|prioriti[sz]e)\b",
    r"\bis it (?:ok|okay|fine|wise|right|smart|fair|a good idea|reasonable|advisable|justified)\b",
    r"\b(?:do you|would you) (?:recommend|advise|suggest|think we should)\b|\brecommend\b|\badvise\b|\bgood idea\b|\bgo ahead\b",
    r"\b(?:can|could|may|shall)\s+(?:we|i|you|they|management|hr)\s+(?:promote|give (?:him|her|them) a (?:promotion|raise|bonus))\b",
    r"\b(?:decide|decision|approve|approval|justify|sign off)\b",
]

POLICY = DomainPolicy(
    sensitive=list(PROTECTED),
    actionability=ACTIONABILITY,
    disclosure=(
        "This is a model-based risk estimate, not a decision. It reflects patterns in the training data, "
        "and a person must review it before any action is taken."
    ),
    high_risk_notice="High estimated risk: a person must review this case before any action is taken.",
    escalation_uncertain=(
        "Routed to human review. The model's estimate for this employee is too uncertain to explain automatically: "
        "it ranges from {low}% to {high}%, which spans the {band_a} and {band_b} risk bands."
    ),
    escalation_ood=(
        "Routed to human review. This employee's profile is unlike the data the model was trained on ({reasons}), "
        "so an automated explanation would not be reliable."
    ),
    refusal_message=(
        "This request asks for an employment decision, which this tool does not make or recommend. "
        "It estimates attrition risk and explains that estimate. Decisions such as firing, promoting or demoting "
        "someone need people, policy and context the model does not have."
    ),
    guard_patterns=GUARD_HARD,
    critic_rules=[DECISION_WORDING, PROTECTED_INFERENCE, BLAME_WORDING, BLAME_SELF_NEGATED, PREDICTION_AS_FACT],
    guard_soft=GUARD_SOFT,
    guard_frames=GUARD_FRAMES,
    causal_extra=[r"\bburn(?:s|ed)?\s+(?:them|him|her|employees|people)\s+out\b"],
    min_risk_drop=0.05,
)
