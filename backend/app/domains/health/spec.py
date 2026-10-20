"""Pima Indians Diabetes data: feature specs and glossary.

Source: Smith, J. W. et al. (1988), "Using the ADAP learning algorithm to forecast the onset of diabetes mellitus",
Proc. Symposium on Computer Applications in Medical Care, 261-265; originally from the National Institute of
Diabetes and Digestive and Kidney Diseases. 768 women aged 21 or older of Pima Indian heritage; outcome = diabetes
according to WHO criteria. The Kaggle copy used by Musicco et al. (2026) is
https://www.kaggle.com/datasets/mathchi/diabetes-data-set (same 768 rows).

Units follow the dataset documentation: plasma glucose 2 hours into an oral glucose tolerance test (mg/dL),
diastolic blood pressure (mm Hg), triceps skin-fold thickness (mm), 2-hour serum insulin (mu U/ml), BMI (kg/m^2).
The dataset records several missing measurements as 0; the loader treats those as missing (see data.py).
"""
from __future__ import annotations

from ...schemas import GlossaryEntry
from ..base import FeatureSpec

SOURCE = "Pima Indians Diabetes dataset (Smith et al. 1988; NIDDK), dataset documentation"
TARGET = "Outcome"
PROTECTED: list[str] = []  # age is a clinical risk factor here and stays in the model; the fairness audit uses age bands

FEATURES: list[FeatureSpec] = [
    FeatureSpec("Glucose", "Glucose", "Plasma glucose 2 hours into an oral glucose tolerance test.", "numeric",
                "G-GLUCOSE", "Measurements", unit="mg/dL", min=40, max=250,
                aliases=["glucose", "plasma glucose", "blood sugar", "glucose level"]),
    FeatureSpec("BMI", "BMI", "Body mass index (weight in kg divided by height in m squared).", "numeric", "G-BMI",
                "Measurements", unit="kg/m2", min=15, max=70, step=0.1, aliases=["bmi", "body mass index"]),
    FeatureSpec("BloodPressure", "Blood pressure", "Diastolic blood pressure.", "numeric", "G-BP", "Measurements",
                unit="mm Hg", min=30, max=130, aliases=["blood pressure", "diastolic blood pressure"]),
    FeatureSpec("SkinThickness", "Skin-fold thickness", "Triceps skin-fold thickness.", "numeric", "G-SKIN",
                "Measurements", unit="mm", min=5, max=100, aliases=["skin-fold thickness", "skin thickness", "skinfold"]),
    FeatureSpec("Insulin", "Insulin", "Serum insulin 2 hours into an oral glucose tolerance test.", "numeric",
                "G-INSULIN", "Measurements", unit="mu U/ml", min=10, max=900, aliases=["insulin", "serum insulin"]),
    FeatureSpec("DiabetesPedigreeFunction", "Diabetes pedigree", "A score summarising diabetes in the family "
                "history (higher means more, and closer, relatives with diabetes).", "numeric", "G-PEDIGREE", "History",
                min=0.05, max=2.5, step=0.001, aliases=["diabetes pedigree", "pedigree function", "family history"]),
    FeatureSpec("Pregnancies", "Pregnancies", "Number of times pregnant.", "numeric", "G-PREGNANCIES", "History",
                min=0, max=17, aliases=["pregnancies", "number of pregnancies"]),
    FeatureSpec("Age", "Age", "Age in years.", "numeric", "G-AGE", "History", unit="years", min=21, max=90,
                aliases=["age"]),
]


def _feature_entry(f: FeatureSpec) -> GlossaryEntry:
    return GlossaryEntry(id=f.glossary_id, term=f.display, definition=f.description, source=SOURCE)


GLOSSARY: dict[str, GlossaryEntry] = {f.glossary_id: _feature_entry(f) for f in FEATURES}
GLOSSARY.update({
    "G-OUTCOME": GlossaryEntry(
        id="G-OUTCOME", term="Diabetes (outcome)",
        definition="Whether the person had diabetes according to WHO criteria in the original study. The model is "
                   "trained on this label. It is not a diagnosis of the person being explained.",
        source=SOURCE),
    "G-RISK": GlossaryEntry(
        id="G-RISK", term="Risk estimate",
        definition=("The model's calibrated estimate of how likely a patient with this profile is to have diabetes by "
                    "the study's criteria, based on patterns in the training data. It is not a diagnosis."),
        source="This system"),
    "G-BAND": GlossaryEntry(
        id="G-BAND", term="Risk band",
        definition=("Low, medium or high. Low is below the review threshold chosen from the cost matrix; high is an "
                    "estimate of 50% or more; medium is in between."),
        source="This system"),
    "G-CONTRIBUTION": GlossaryEntry(
        id="G-CONTRIBUTION", term="Contribution",
        definition=("How much one factor moved the model's estimate up or down for this profile, compared with the "
                    "average estimate. Contributions describe the model's behaviour, not causes."),
        source="Shapley values (SHAP)"),
    "G-RECOURSE": GlossaryEntry(
        id="G-RECOURSE", term="Modelled change",
        definition=("A change to a measurement that can change over time (BMI, blood pressure) and the model's estimate "
                    "after it, re-checked by running the model again. It shows the model's behaviour; it is not "
                    "treatment advice."),
        source="This system"),
})
