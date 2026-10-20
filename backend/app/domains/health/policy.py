"""Healthcare DomainPolicy (diabetes risk): the strictest of the three domains.

AI systems used for medical purposes are high-risk under the EU AI Act (Article 6 with Annex I, via the Medical
Devices Regulation), and diagnosis and treatment decisions belong to clinicians. This tool therefore:
- never states or confirms a diagnosis (an estimate is never described as the patient having the condition);
- never gives treatment, medication, dosage, referral, admission or lifestyle advice;
- never makes or ranks care, coverage or prioritisation decisions;
- never mentions or infers ethnicity, race, religion or sex (the dataset comes from one population; the explanation
  must not generalise from it);
- frames modelled changes (BMI, blood pressure) as the model's behaviour, not as advice, and requires clinician review.
Age stays in the model: it is a recognised clinical risk factor and is not withheld. The fairness audit reports
error and flag rates by age band instead.
"""
from __future__ import annotations

from ..base import Actionability, DomainLanguage, DomainPolicy, TextRule
from .spec import PROTECTED

A = Actionability

LANGUAGE = DomainLanguage(
    subject="the patient",
    subject_indefinite="a patient",
    subject_plural="patients",
    subject_possessive="the patient's",
    reader="a clinician",
    outcome_verb="has diabetes by the study's criteria",
    outcome_gerund="having diabetes by the study's criteria",
    risk_name="diabetes risk",
    model_description="a diabetes-risk model",
    decision_kind="a clinical decision (diagnosis, treatment, referral or coverage)",
    decision_examples="diagnosing, prescribing, referring, admitting, prioritising or denying coverage",
    protected_description="ethnicity, race, religion or sex",
    character_examples='"non-compliant", "lazy", "their fault"',
    outcome_terms=["diabetes", "the disease", "the condition", "hyperglycaemia", "hyperglycemia"],
    subject_terms=["patients", "people", "them", "the patient", "him", "her"],
    outcome_verbs=["develop diabetes", "become diabetic", "get diabetes"],
    actor_terms=["the clinic", "the care team"],
    lever_terms=["lower bmi", "lower blood pressure", "weight loss", "a lower bmi"],
    missing_instance_message="Provide the patient's measurements to explain a risk estimate.",
    batch_question="Why is this patient's diabetes risk estimated this way?",
    default_question="Why is this patient's diabetes risk estimated this way?",
    chat_more_details="Could you tell me more of the patient's measurements?",
    reference_column="patient_id",
    fault_sentences={
        "causal_claim": "Readings like these cause diabetes.",
        "decision_wording": "The patient should be started on metformin.",
        "protected_inference": "Given the patient's ethnicity, the risk may be higher.",
        "blame_wording": "The patient seems non-compliant.",
        "prediction_as_fact": "The patient has diabetes.",
        "raw_code": "Profile code: Blood_Pressure.",
    },
    data_notes={
        "synthetic_standin": "The model was trained on synthetic stand-in data for software testing.",
        "pima_niddk": ("The model was trained on the Pima Indians Diabetes dataset (768 women aged 21 or older from one "
                       "population); it may not carry over to other patients."),
    },
)

ACTIONABILITY: dict[str, Actionability] = {
    "BMI": A("partial", "BMI can change over time under clinical guidance; shown as the model's estimate, not advice."),
    "BloodPressure": A("partial", "Blood pressure can change over time under clinical guidance; shown as the model's "
                                  "estimate, not advice."),
    "Glucose": A("immutable", "A test result for the condition itself; not changed in modelled changes."),
    "Insulin": A("immutable", "A test result; not changed in modelled changes."),
    "SkinThickness": A("immutable", "A measurement; not changed in modelled changes."),
    "DiabetesPedigreeFunction": A("immutable", "Family history cannot change."),
    "Pregnancies": A("immutable", "History cannot change."),
    "Age": A("immutable", "Never changed."),
}

# ---------------------------------------------------------------- critic wording rules (one code per fault type)
DECISION_WORDING = TextRule(
    code="decision_wording",
    message="Gives treatment, medication, referral or care advice, or implies a clinical decision.",
    patterns=[
        r"\b(?:should|must|ought to|needs? to|recommend(?:ed|s)?|advis(?:e|ed|able)|suggest(?:ed|s)?)\b(?:\W+\w+){0,6}?\W+"
        r"(?:start(?:ed)?|take|taking|prescrib\w*|treat\w*|refer\w*|admit\w*|lose weight|diet|exercise|stop\w*|"
        r"medicat\w*|insulin|metformin|surgery|test(?:ed|ing)?|monitor\w*|screen\w*)\b",
        r"\b(?:prescrib\w*|dosage|dose of|metformin|statins?|insulin therapy|glp-?1)\b",
        r"\b(?:refer(?:ral|red)? to|admit(?:ted)? to|discharg\w*)\b",
        r"\b(?:deny|refuse|withhold)\w*\s+(?:coverage|insurance|treatment|care)\b",
    ],
)
PROTECTED_INFERENCE = TextRule(
    code="protected_attribute_inference",
    message="Mentions or infers ethnicity, race, religion or sex.",
    negatable=False,
    patterns=[
        r"\b(?:ethnic\w*|race|racial|heritage|ancestry|pima|native american|indigenous|tribal|tribe|religio\w*)\b",
        r"\b(?:sex|gender|male|female|man|woman|men|women|he|she|him|his|hers|her|himself|herself)\b",
    ],
)
BLAME_WORDING = TextRule(
    code="blame_wording",
    message="Blames or judges the patient.",
    patterns=[
        r"\b(?:non-?compliant|noncompliance|lazy|irresponsible|careless|undisciplined|unmotivated)\b",
        r"\b(?:lacks?|lack of) (?:willpower|discipline|self-control)\b|\blet (?:themselves|herself|himself) go\b",
        r"\b(?:their|his|her|the patient's) (?:own )?fault\b|\b(?:is|are) to blame\b|\bblam(?:e|es|ed|ing)\b",
        r"\b(?:overeat\w*|poor (?:lifestyle )?choices)\b",
    ],
)
PREDICTION_AS_FACT = TextRule(
    code="prediction_as_fact",
    message="States a diagnosis or the predicted outcome as a fact instead of an estimate.",
    patterns=[
        r"\b(?:the patient|this patient|she|he|they)\s+(?:has|have|is|are)\s+(?:got\s+)?(?:diabetes|diabetic|prediabetic|"
        r"prediabetes)\b",
        r"\b(?:will|is going to|is certain to|is sure to|is bound to)\s+(?:definitely\s+|certainly\s+|probably\s+)?"
        r"(?:develop|get|become)\s+(?:diabetes|diabetic)\b",
        r"\b(?:diagnos(?:ed|is|e)|confirm(?:s|ed)? (?:diabetes|the condition))\b",
        r"\b(?:definitely|certainly|undoubtedly|for sure|without (?:a )?doubt|guaranteed)\b",
    ],
)

# ---------------------------------------------------------------- clinical-decision guardrail (queries)
GUARD_HARD = [
    r"\bdiagnos\w*\b",
    r"\b(?:does|do|is|are)\s+(?:she|he|they|the patient|this patient|this person)\s+(?:have\s+|has\s+|got\s+)?(?:diabet\w*|prediabet\w*)\b",
    r"\b(?:she|he|they|patient)\s+(?:is|are|has|have)\s+(?:diabetic|prediabetic|diabetes)\b",
    r"\b(?:prescri\w*|medications?|medicines?|drugs?|pills?|metformin|insulin therapy|statins?|dos(?:e|age|ing))\b",
    r"\b(?:take|inject|start|use)\b.{0,20}\binsulin\b|\binsulin\b.{0,30}\b(?:should|take|inject|start)\b",
    r"\bput (?:her|him|them|the patient) on\b|\bbe put on\b",
    r"\b(?:referral|referred|refer (?:her|him|them|the patient|this patient)|admit\w*|hospitali[sz]\w*|discharg\w*|"
    r"send (?:her|him|them) home|specialist|endocrinolog\w*)\b",
    r"\b(?:insur\w*|premium|coverage)\b",
    r"\b(?:triage|rank|prioriti[sz]e)\b.{0,40}\bpatients?\b|\btriage\b",
    r"\b(?:who|which patients?)\b.{0,30}\b(?:treat|see|care|follow-?up|first|urgent\w*|appointment)\w*",
    r"\border\w*\b.{0,20}\b(?:tests?|hba1c|labs?)\b",
    r"\bavoid pregnan\w*",
    r"\btreatment plan\b|\bdiet plan\b|\blifestyle changes?\b",
]
GUARD_SOFT = [  # refused only when framed as advice or a decision
    r"\b(?:treat\w*|surgery|tests?|weight loss|lose weight|exercise|diet|lifestyle|monitor\w*|follow-?up|stop)\b",
]
GUARD_FRAMES = [
    r"\b(?:should|shall|must|ought|need(?:ed|s)?|necessary|required)\b",
    r"\b(?:do you|would you) (?:recommend|advise|suggest|start|treat)\b|\brecommend\b|\badvise\b",
]

POLICY = DomainPolicy(
    sensitive=list(PROTECTED),
    actionability=ACTIONABILITY,
    disclosure=(
        "This is a model-based risk estimate, not a diagnosis. It reflects patterns in research data, and a clinician "
        "must review it before any clinical decision is made."
    ),
    high_risk_notice="High estimated diabetes risk: a clinician must review this case.",
    escalation_uncertain=(
        "Routed to clinician review. The model's estimate for this patient is too uncertain to explain automatically: "
        "it ranges from {low}% to {high}%, which spans the {band_a} and {band_b} risk bands."
    ),
    escalation_ood=(
        "Routed to clinician review. This patient's measurements are unlike the data the model was trained on "
        "({reasons}), so an automated explanation would not be reliable."
    ),
    refusal_message=(
        "This request asks for a diagnosis, treatment advice or a care decision, which this tool does not give. It "
        "estimates diabetes risk from a research dataset and explains that estimate for a clinician. Diagnosis and "
        "treatment need a clinician, clinical guidelines and information the model does not have."
    ),
    guard_patterns=GUARD_HARD,
    critic_rules=[DECISION_WORDING, PROTECTED_INFERENCE, BLAME_WORDING, PREDICTION_AS_FACT],
    guard_soft=GUARD_SOFT,
    guard_frames=GUARD_FRAMES,
    causal_extra=[],
    min_risk_drop=0.05,
    withheld_note="Some factors are shown only at expert level.",
    language=LANGUAGE,
)
