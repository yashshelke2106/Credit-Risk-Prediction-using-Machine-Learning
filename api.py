"""FastAPI scoring endpoint for the calibrated PD model.

Run:   uvicorn api:app --port 8000
Call:  POST /score  with one applicant -> PD, decision, reason codes.

Uses the same pd_model.score() as the Streamlit app, so the API and the demo
cannot disagree about a decision.
"""
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

import pd_model as pm


class Applicant(BaseModel):
    person_age: Optional[float] = Field(None, ge=18, le=100,
                                        description="Not used by the model; accepted for record-keeping")
    person_income: float = Field(..., gt=0)
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: Optional[float] = Field(None, ge=0, le=60)
    loan_intent: Literal["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
                         "MEDICAL", "PERSONAL", "VENTURE"]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(..., gt=0)
    loan_int_rate: Optional[float] = Field(None, gt=0, lt=100)
    loan_percent_income: Optional[float] = Field(None, ge=0,
                                                 description="Derived from amount/income if omitted")
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: float = Field(..., ge=0)

    model_config = {"json_schema_extra": {"examples": [{
        "person_age": 27, "person_income": 42000, "person_home_ownership": "RENT",
        "person_emp_length": 3, "loan_intent": "MEDICAL", "loan_grade": "D",
        "loan_amnt": 15000, "loan_int_rate": 15.2, "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4}]}}


COLUMNS = ["person_age", "person_income", "person_home_ownership", "person_emp_length",
           "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
           "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"]

MODEL, META, SOURCE = pm.load_or_train()
app = FastAPI(title="Credit PD scoring", version=META.get("model_version", "1"))


def to_frame(a: Applicant) -> pd.DataFrame:
    row = a.model_dump()
    if row["loan_percent_income"] is None:
        row["loan_percent_income"] = round(row["loan_amnt"] / row["person_income"], 2)
    df = pd.DataFrame([row])[COLUMNS]
    return df.astype({"person_age": "float64", "person_emp_length": "float64",
                      "loan_int_rate": "float64"})


@app.get("/health")
def health():
    return {"status": "ok", "model_source": SOURCE, "cutoff": META["cutoff"],
            "lgd": pm.LGD, "excluded_features": pm.EXCLUDED}


@app.post("/score")
def score(applicant: Applicant):
    return pm.score(MODEL, META, to_frame(applicant))[0]
