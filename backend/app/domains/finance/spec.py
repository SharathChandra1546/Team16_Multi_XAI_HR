"""Statlog (German Credit Data): feature specs and glossary.

Source: Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository,
https://doi.org/10.24432/C5NC77 (CC BY 4.0). 1000 applicants, 20 attributes, target good (1) / bad (2) credit risk.
Category codes (A11, A12, ...) follow the dataset's documentation. Two ordinal attributes (installment rate and
present residence) are documented only as levels 1-4, so they are shown as levels rather than given invented bands.
Amounts are in Deutsche Mark (DM), as recorded; the data are from the 1970s-80s.
"""
from __future__ import annotations

from ...schemas import GlossaryEntry
from ..base import FeatureSpec

SOURCE = "Statlog (German Credit Data), UCI Machine Learning Repository (Hofmann 1994), dataset documentation"

TARGET = "class"  # 1 = good credit risk, 2 = bad credit risk (the positive class here)
PROTECTED = ["age", "personal_status_sex", "foreign_worker"]
LEVEL_4 = {1: "level 1", 2: "level 2", 3: "level 3", 4: "level 4"}

FEATURES: list[FeatureSpec] = [
    # ---- accounts and history
    FeatureSpec("checking_status", "Checking account", "Balance of the applicant's checking account at this bank.",
                "nominal", "G-CHECKING", "Accounts",
                options={0: "A11", 1: "A12", 2: "A13", 3: "A14"},
                labels={0: "below 0 DM", 1: "0 to 200 DM", 2: "200 DM or more", 3: "no checking account"},
                aliases=["checking account", "checking balance", "checking status"]),
    FeatureSpec("savings", "Savings", "Balance of savings accounts or bonds.", "nominal", "G-SAVINGS", "Accounts",
                options={0: "A61", 1: "A62", 2: "A63", 3: "A64", 4: "A65"},
                labels={0: "below 100 DM", 1: "100 to 500 DM", 2: "500 to 1000 DM", 3: "1000 DM or more",
                        4: "unknown or no savings account"},
                aliases=["savings", "savings balance", "savings account"]),
    FeatureSpec("credit_history", "Credit history", "How the applicant has repaid credit in the past.", "nominal",
                "G-HISTORY", "Accounts",
                options={0: "A30", 1: "A31", 2: "A32", 3: "A33", 4: "A34"},
                labels={0: "no credits taken or all repaid duly", 1: "all credits at this bank repaid duly",
                        2: "existing credits repaid duly so far", 3: "delays in repaying in the past",
                        4: "critical account or credits at other banks"},
                aliases=["credit history", "repayment history"]),
    FeatureSpec("existing_credits", "Existing credits", "Number of existing credits at this bank.", "numeric",
                "G-EXISTING", "Accounts", unit="credits", min=1, max=4, aliases=["existing credits", "credits at this bank"]),
    FeatureSpec("other_installment_plans", "Other installment plans", "Installment plans with other lenders.",
                "nominal", "G-OTHERPLANS", "Accounts", options={0: "A141", 1: "A142", 2: "A143"},
                labels={0: "with a bank", 1: "with stores", 2: "none"},
                aliases=["other installment plans", "other loans"]),
    # ---- the loan
    FeatureSpec("duration", "Loan duration", "Length of the loan.", "numeric", "G-DURATION", "Loan", unit="months",
                min=4, max=72, aliases=["loan duration", "duration", "loan term"]),
    FeatureSpec("credit_amount", "Credit amount", "Amount of credit requested.", "numeric", "G-AMOUNT", "Loan",
                unit="DM", min=250, max=20000, aliases=["credit amount", "loan amount", "amount borrowed"]),
    FeatureSpec("purpose", "Loan purpose", "What the credit is for.", "nominal", "G-PURPOSE", "Loan",
                options={0: "A40", 1: "A41", 2: "A42", 3: "A43", 4: "A44", 5: "A45", 6: "A46", 7: "A47", 8: "A48",
                         9: "A49", 10: "A410"},
                labels={0: "new car", 1: "used car", 2: "furniture or equipment", 3: "radio or television",
                        4: "domestic appliances", 5: "repairs", 6: "education", 7: "vacation", 8: "retraining",
                        9: "business", 10: "other purpose"},
                aliases=["loan purpose", "purpose"]),
    FeatureSpec("installment_rate", "Installment rate", "Installment as a share of disposable income, recorded as a "
                "level from 1 to 4 in the dataset.", "ordinal", "G-INSTALLMENT", "Loan",
                options={1: "1", 2: "2", 3: "3", 4: "4"}, labels=LEVEL_4, aliases=["installment rate"]),
    FeatureSpec("other_debtors", "Other debtors", "Whether a co-applicant or guarantor shares the credit.", "nominal",
                "G-DEBTORS", "Loan", options={0: "A101", 1: "A102", 2: "A103"},
                labels={0: "no co-applicant or guarantor", 1: "co-applicant", 2: "guarantor"},
                aliases=["other debtors", "guarantor", "co-applicant"]),
    # ---- circumstances
    FeatureSpec("employment", "Employment length", "Time in the present employment.", "ordinal", "G-EMPLOYMENT",
                "Circumstances", options={0: "A71", 1: "A72", 2: "A73", 3: "A74", 4: "A75"},
                labels={0: "unemployed", 1: "less than 1 year", 2: "1 to 4 years", 3: "4 to 7 years",
                        4: "7 years or more"},
                aliases=["employment length", "time in employment", "employment"]),
    FeatureSpec("job", "Job type", "Type of job.", "nominal", "G-JOB", "Circumstances",
                options={0: "A171", 1: "A172", 2: "A173", 3: "A174"},
                labels={0: "unemployed or unskilled non-resident", 1: "unskilled resident",
                        2: "skilled employee or official", 3: "management, self-employed or highly qualified"},
                aliases=["job type", "occupation"]),
    FeatureSpec("property", "Property", "Most valuable property the applicant holds.", "nominal", "G-PROPERTY",
                "Circumstances", options={0: "A121", 1: "A122", 2: "A123", 3: "A124"},
                labels={0: "real estate", 1: "savings agreement or life insurance", 2: "car or other property",
                        3: "no known property"},
                aliases=["property"]),
    FeatureSpec("housing", "Housing", "Housing situation.", "nominal", "G-HOUSING", "Circumstances",
                options={0: "A151", 1: "A152", 2: "A153"}, labels={0: "rents", 1: "owns", 2: "lives for free"},
                aliases=["housing"]),
    FeatureSpec("residence_since", "Present residence", "Time at the present residence, recorded as a level from 1 "
                "to 4 in the dataset.", "ordinal", "G-RESIDENCE", "Circumstances",
                options={1: "1", 2: "2", 3: "3", 4: "4"}, labels=LEVEL_4,
                aliases=["present residence", "time at residence"]),
    FeatureSpec("num_dependents", "Dependents", "Number of people the applicant provides maintenance for.", "numeric",
                "G-DEPENDENTS", "Circumstances", unit="people", min=1, max=2, aliases=["dependents", "number of dependents"]),
    FeatureSpec("telephone", "Telephone", "Whether a telephone is registered in the applicant's name.", "binary",
                "G-TELEPHONE", "Circumstances", options={0: "A191", 1: "A192"},
                labels={0: "no registered telephone", 1: "registered telephone"}, aliases=["telephone"]),
    # ---- protected (audit model only)
    FeatureSpec("age", "Age", "Age in years.", "numeric", "G-AGE", "Protected", unit="years", min=18, max=80,
                aliases=["age"]),
    FeatureSpec("personal_status_sex", "Personal status and sex", "Sex and marital status, combined in one "
                "attribute in the dataset.", "nominal", "G-PERSONAL", "Protected",
                options={0: "A91", 1: "A92", 2: "A93", 3: "A94", 4: "A95"},
                labels={0: "male, divorced or separated", 1: "female, divorced, separated or married",
                        2: "male, single", 3: "male, married or widowed", 4: "female, single"},
                aliases=["personal status", "marital status"]),
    FeatureSpec("foreign_worker", "Foreign worker", "Whether the applicant is a foreign worker.", "binary",
                "G-FOREIGN", "Protected", options={0: "A201", 1: "A202"},
                labels={0: "foreign worker", 1: "not a foreign worker"}, aliases=["foreign worker"]),
]


def _feature_entry(f: FeatureSpec) -> GlossaryEntry:
    values = {str(c): f.labels[c] for c in sorted(f.labels)} if f.categorical else {}
    return GlossaryEntry(id=f.glossary_id, term=f.display, definition=f.description, values=values, source=SOURCE)


GLOSSARY: dict[str, GlossaryEntry] = {f.glossary_id: _feature_entry(f) for f in FEATURES}
GLOSSARY.update({
    "G-CREDITRISK": GlossaryEntry(
        id="G-CREDITRISK", term="Bad credit risk",
        definition="The dataset labels each past applicant as a good or bad credit risk. The model is trained on this label.",
        source=SOURCE),
    "G-RISK": GlossaryEntry(
        id="G-RISK", term="Risk estimate",
        definition=("The model's calibrated estimate of how likely an applicant with this profile is to be a bad credit "
                    "risk, based on patterns in historical lending data. It is not a credit decision."),
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
        id="G-RECOURSE", term="Recourse option",
        definition=("A change to factors that can realistically change (such as the amount or term of the loan, or "
                    "adding a guarantor) that the model says would lower its estimate. Each option is re-checked by "
                    "running the model again."),
        source="This system"),
})
