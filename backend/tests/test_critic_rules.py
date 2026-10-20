"""HR critic rules: each code has at least 5 failing and 5 passing examples (acceptance criterion)."""
import pytest

from app.agents.critic import rule_violations

CASES = {
    "decision_wording": {
        "fail": [
            "They should be dismissed.",
            "We recommend terminating the contract.",
            "Management should promote them now.",
            "This employee is a candidate for termination.",
            "Consider letting them go before they quit.",
            "It may be time to lay them off.",
            "Demotion would be appropriate here.",
        ],
        "pass": [
            "A promotion to level 3 would lower the model's estimate from 40% to 25%.",
            "The model's estimate would fall with less overtime.",
            "This is not a basis for dismissal.",
            "The largest factor was overtime, which raised the estimate.",
            "Option 1: with a role change, the estimate would drop.",
            "Monthly income lowered the estimate slightly.",
        ],
    },
    "protected_attribute_inference": {
        "fail": [
            "As a younger employee, they may be more mobile.",
            "She has worked overtime regularly.",
            "Being single, the employee may feel less tied to the company.",
            "His work-life balance is rated Bad.",
            "Employees with children often prefer stable roles.",
            "The employee is married, which lowers the estimate.",
            "Older staff tend to stay longer.",
        ],
        "pass": [
            "The employee works overtime, which raised the estimate.",
            "Overtime was the single largest factor.",
            "They have been in the current role for 3 years.",
            "Job satisfaction (Low) raised the estimate moderately.",
            "The estimate is in the medium band.",
            "Years at the company (2 years) raised the estimate.",
        ],
    },
    "blame_wording": {
        "fail": [
            "The employee seems lazy.",
            "This looks like a bad attitude problem.",
            "Their lack of commitment is the issue.",
            "It is their own fault that satisfaction is low.",
            "The employee does not care about the job.",
            "They are disloyal to the team.",
        ],
        "pass": [
            "Job involvement (Low) raised the estimate.",
            "This is not the employee's fault.",
            "Satisfaction ratings are associated with the estimate.",
            "The employee travels frequently.",
            "Low environment satisfaction pushed the estimate up.",
            "No one is to blame for a model estimate.",
        ],
    },
    "prediction_as_fact": {
        "fail": [
            "The employee will leave.",
            "They are going to quit soon.",
            "This employee is leaving.",
            "They will definitely stay.",
            "The employee has decided to resign.",
            "They will certainly leave next year.",
        ],
        "pass": [
            "The model estimates a 40% chance that an employee with this profile leaves.",
            "This does not mean they will leave.",
            "The estimate is a probability, not a prediction of what the employee will do.",
            "The risk estimate is in the high band.",
            "The estimated attrition risk is 22%.",
            "Employees with similar profiles left more often in the training data.",
        ],
    },
}


@pytest.mark.parametrize("code", list(CASES))
def test_rule_has_enough_examples(code):
    assert len(CASES[code]["fail"]) >= 5 and len(CASES[code]["pass"]) >= 5


@pytest.mark.parametrize("code,text", [(c, t) for c in CASES for t in CASES[c]["fail"]])
def test_rule_fails(hr, code, text):
    codes = {v.code for v in rule_violations(text, hr)}
    assert code in codes, f"{code} not raised for: {text}"


@pytest.mark.parametrize("code,text", [(c, t) for c in CASES for t in CASES[c]["pass"]])
def test_rule_passes(hr, code, text):
    codes = {v.code for v in rule_violations(text, hr)}
    assert code not in codes, f"{code} wrongly raised for: {text} ({codes})"
