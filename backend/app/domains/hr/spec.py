"""IBM HR Analytics Employee Attrition dataset: feature specs and glossary.

Source: "IBM HR Analytics Employee Attrition & Performance" (Kaggle,
pavansubhasht). The dataset is fictional, created by IBM data scientists.
Code tables below follow the dataset's published data dictionary.
"""
from __future__ import annotations

from ...schemas import GlossaryEntry
from ..base import FeatureSpec

SOURCE = "IBM HR Analytics Employee Attrition & Performance dataset (fictional, created by IBM data scientists), Kaggle data dictionary"

RATING_4 = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}

TARGET = "Attrition"
# Columns removed before modelling, with the reason recorded in the training manifest.
DROPPED = {
    "EmployeeCount": "constant column",
    "Over18": "constant column",
    "StandardHours": "constant column",
    "EmployeeNumber": "identifier",
    "DailyRate": "meaning not documented in the data dictionary",
    "HourlyRate": "meaning not documented in the data dictionary",
    "MonthlyRate": "meaning not documented in the data dictionary",
}
PROTECTED = ["Age", "Gender", "MaritalStatus"]

FEATURES: list[FeatureSpec] = [
    # ---- job
    FeatureSpec("Department", "Department", "Department the employee works in.", "nominal", "G-DEPARTMENT", "Job",
                options={0: "Human Resources", 1: "Research & Development", 2: "Sales"}, aliases=["department"]),
    FeatureSpec("JobRole", "Job role", "The employee's job role.", "nominal", "G-JOBROLE", "Job",
                options={0: "Healthcare Representative", 1: "Human Resources", 2: "Laboratory Technician", 3: "Manager",
                         4: "Manufacturing Director", 5: "Research Director", 6: "Research Scientist",
                         7: "Sales Executive", 8: "Sales Representative"},
                aliases=["job role"]),
    FeatureSpec("JobLevel", "Job level", "Seniority level of the job, from 1 (entry) to 5.", "ordinal", "G-JOBLEVEL", "Job",
                options={1: "1", 2: "2", 3: "3", 4: "4", 5: "5"},
                labels={1: "level 1", 2: "level 2", 3: "level 3", 4: "level 4", 5: "level 5"},
                aliases=["job level", "seniority level"]),
    FeatureSpec("JobInvolvement", "Job involvement", "Rating of how involved the employee is in their job.", "ordinal",
                "G-JOBINVOLVEMENT", "Ratings", options={1: "1", 2: "2", 3: "3", 4: "4"}, labels=RATING_4,
                aliases=["job involvement", "involvement"]),
    FeatureSpec("PerformanceRating", "Performance rating", "Most recent performance rating.", "ordinal",
                "G-PERFORMANCERATING", "Ratings", options={1: "1", 2: "2", 3: "3", 4: "4"},
                labels={1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"},
                aliases=["performance rating", "performance"]),
    FeatureSpec("OverTime", "Overtime", "Whether the employee regularly works overtime.", "binary", "G-OVERTIME", "Workload",
                options={0: "No", 1: "Yes"}, labels={0: "no overtime", 1: "works overtime"},
                aliases=["overtime", "over time", "extra hours"]),
    FeatureSpec("BusinessTravel", "Business travel", "How often the employee travels for work.", "ordinal",
                "G-BUSINESSTRAVEL", "Workload",
                options={0: "Non-Travel", 1: "Travel_Rarely", 2: "Travel_Frequently"},
                labels={0: "does not travel", 1: "travels rarely", 2: "travels frequently"},
                aliases=["business travel", "travel", "travels"]),
    FeatureSpec("DistanceFromHome", "Distance from home", "Distance between home and work (unit not stated in the dataset).",
                "numeric", "G-DISTANCEFROMHOME", "Workload", min=0, max=100,
                aliases=["distance from home", "commute", "distance"]),
    # ---- pay
    FeatureSpec("MonthlyIncome", "Monthly income", "Monthly income (currency not stated in the dataset).", "numeric",
                "G-MONTHLYINCOME", "Pay", unit="a month", min=0, max=100000,
                aliases=["monthly income", "income", "salary", "pay"]),
    FeatureSpec("PercentSalaryHike", "Last salary increase", "Percentage increase at the last salary review.", "numeric",
                "G-PERCENTSALARYHIKE", "Pay", unit="%", min=0, max=100,
                aliases=["last salary increase", "salary hike", "salary increase", "percent salary hike"]),
    FeatureSpec("StockOptionLevel", "Stock option level", "Stock option level, from 0 (none) to 3.", "ordinal",
                "G-STOCKOPTIONLEVEL", "Pay", options={0: "0", 1: "1", 2: "2", 3: "3"},
                labels={0: "level 0 (none)", 1: "level 1", 2: "level 2", 3: "level 3"},
                aliases=["stock option level", "stock options", "stock option"]),
    # ---- time
    FeatureSpec("TotalWorkingYears", "Total working years", "Total years of work experience.", "numeric",
                "G-TOTALWORKINGYEARS", "Time", unit="years", min=0, max=60,
                aliases=["total working years", "years of experience", "work experience"]),
    FeatureSpec("YearsAtCompany", "Years at the company", "Years at the current company.", "numeric", "G-YEARSATCOMPANY",
                "Time", unit="years", min=0, max=60, aliases=["years at the company", "years at company", "tenure"]),
    FeatureSpec("YearsInCurrentRole", "Years in current role", "Years in the current role.", "numeric",
                "G-YEARSINCURRENTROLE", "Time", unit="years", min=0, max=60,
                aliases=["years in current role", "years in the current role", "time in role"]),
    FeatureSpec("YearsSinceLastPromotion", "Years since last promotion", "Years since the last promotion.", "numeric",
                "G-YEARSSINCELASTPROMOTION", "Time", unit="years", min=0, max=60,
                aliases=["years since last promotion", "years since the last promotion", "last promotion"]),
    FeatureSpec("YearsWithCurrManager", "Years with current manager", "Years with the current manager.", "numeric",
                "G-YEARSWITHCURRMANAGER", "Time", unit="years", min=0, max=60,
                aliases=["years with current manager", "years with the current manager", "current manager"]),
    FeatureSpec("NumCompaniesWorked", "Previous companies", "Number of companies worked at before.", "numeric",
                "G-NUMCOMPANIESWORKED", "Time", min=0, max=50,
                aliases=["previous companies", "number of companies", "companies worked"]),
    FeatureSpec("TrainingTimesLastYear", "Training last year", "Number of training sessions in the last year.", "numeric",
                "G-TRAININGTIMESLASTYEAR", "Development", unit="times", min=0, max=50,
                aliases=["training last year", "training sessions", "training"]),
    FeatureSpec("Education", "Education level", "Highest education level.", "ordinal", "G-EDUCATION", "Development",
                options={1: "1", 2: "2", 3: "3", 4: "4", 5: "5"},
                labels={1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"},
                aliases=["education level", "education"]),
    FeatureSpec("EducationField", "Field of education", "Field of education.", "nominal", "G-EDUCATIONFIELD", "Development",
                options={0: "Human Resources", 1: "Life Sciences", 2: "Marketing", 3: "Medical", 4: "Other",
                         5: "Technical Degree"},
                aliases=["field of education", "education field", "field of study"]),
    # ---- ratings
    FeatureSpec("JobSatisfaction", "Job satisfaction", "Rating of satisfaction with the job.", "ordinal",
                "G-JOBSATISFACTION", "Ratings", options={1: "1", 2: "2", 3: "3", 4: "4"}, labels=RATING_4,
                aliases=["job satisfaction"]),
    FeatureSpec("EnvironmentSatisfaction", "Environment satisfaction", "Rating of satisfaction with the work environment.",
                "ordinal", "G-ENVIRONMENTSATISFACTION", "Ratings", options={1: "1", 2: "2", 3: "3", 4: "4"},
                labels=RATING_4, aliases=["environment satisfaction", "work environment"]),
    FeatureSpec("RelationshipSatisfaction", "Relationship satisfaction", "Rating of satisfaction with workplace relationships.",
                "ordinal", "G-RELATIONSHIPSATISFACTION", "Ratings", options={1: "1", 2: "2", 3: "3", 4: "4"},
                labels=RATING_4, aliases=["relationship satisfaction", "workplace relationships"]),
    FeatureSpec("WorkLifeBalance", "Work-life balance", "Rating of work-life balance.", "ordinal", "G-WORKLIFEBALANCE",
                "Ratings", options={1: "1", 2: "2", 3: "3", 4: "4"},
                labels={1: "Bad", 2: "Good", 3: "Better", 4: "Best"},
                aliases=["work-life balance", "work life balance"]),
    # ---- protected (audit model only)
    FeatureSpec("Age", "Age", "Age in years.", "numeric", "G-AGE", "Protected", unit="years", min=16, max=80, aliases=["age"]),
    FeatureSpec("Gender", "Gender", "Gender as recorded in the dataset.", "binary", "G-GENDER", "Protected",
                options={0: "Female", 1: "Male"}, aliases=["gender"]),
    FeatureSpec("MaritalStatus", "Marital status", "Marital status as recorded in the dataset.", "nominal",
                "G-MARITALSTATUS", "Protected", options={0: "Divorced", 1: "Married", 2: "Single"},
                aliases=["marital status"]),
]


def _feature_entry(f: FeatureSpec) -> GlossaryEntry:
    values = {}
    if f.categorical:
        values = {str(c): f.labels[c] for c in sorted(f.labels)}
    return GlossaryEntry(id=f.glossary_id, term=f.display, definition=f.description, values=values, source=SOURCE)


GLOSSARY: dict[str, GlossaryEntry] = {f.glossary_id: _feature_entry(f) for f in FEATURES}
GLOSSARY.update(
    {
        "G-ATTRITION": GlossaryEntry(
            id="G-ATTRITION", term="Attrition",
            definition="Whether the employee left the company. The model is trained on this column (Yes or No).",
            source=SOURCE),
        "G-RISK": GlossaryEntry(
            id="G-RISK", term="Risk estimate",
            definition=("The model's calibrated estimate of how likely an employee with this profile is to leave, "
                        "based on patterns in the training data. It is not a prediction about a specific person's choices."),
            source="This system"),
        "G-BAND": GlossaryEntry(
            id="G-BAND", term="Risk band",
            definition=("Low, medium or high. Low is below the review threshold chosen from the cost matrix; "
                        "high is an estimate of 50% or more; medium is in between."),
            source="This system"),
        "G-CONTRIBUTION": GlossaryEntry(
            id="G-CONTRIBUTION", term="Contribution",
            definition=("How much one factor moved the model's estimate up or down for this profile, compared with the "
                        "average estimate. Contributions describe the model's behaviour, not causes."),
            source="Shapley values (SHAP)"),
        "G-RECOURSE": GlossaryEntry(
            id="G-RECOURSE", term="Recourse option",
            definition=("A change to factors that can realistically be changed (such as overtime or training) that the "
                        "model says would lower its estimate. Each option is re-checked by running the model again."),
            source="This system"),
    }
)
