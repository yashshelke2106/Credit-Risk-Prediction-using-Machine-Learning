"""The deployed probability-of-default (PD) model, decision rule and reason codes.

Imported by evaluate.py, api.py and app1.py so all three score an applicant
identically. Nothing here is fitted on import.

Model:   LightGBM inside the leak-free preprocessing pipeline (credit_data.py),
         wrapped in Platt (sigmoid) calibration fitted on held-out folds, with
         the Basel 0.03% PD floor. Isotonic was tested and rejected: equal Brier,
         but it emitted PD = 0 or 1 for ~22% of applicants and only ~360 distinct
         PD levels - unusable for pricing. See reports/calibration_method.csv.
Why calibrated: the class-weighted LightGBM ranks well but predicts a mean PD
         of ~0.31 against a 0.22 default rate. Ranking metrics do not care;
         expected loss, pricing and per-loan break-even decisions do.
Decision: decline when calibrated PD >= the profit-maximising cutoff chosen in
         evaluate.py (stored in models/pd_model_meta.json).
Reasons: exact TreeSHAP contributions from LightGBM's own pred_contrib, summed
         back from one-hot columns to the applicant-facing field.
"""
import json
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

import credit_data as cd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "pd_model.joblib")
META_PATH = os.path.join(HERE, "models", "pd_model_meta.json")

# ---- business assumptions (stated, not estimated) ---------------------------
LGD = 0.45            # loss given default: Basel foundation-IRB senior unsecured
PD_FLOOR = 0.0003     # Basel minimum PD (0.03%): no loan is ever scored risk-free
CALIBRATION = "sigmoid"
# EAD = loan_amnt     # exposure at default: full principal, no amortisation
# Revenue on a repaid loan = loan_int_rate/100 x loan_amnt (one period, gross).

# ---- feature set ------------------------------------------------------------
# person_age is excluded on purpose: IV 0.0065, and dropping it costs 0.0024
# AUC (~1 fold std). Excluding it means age can never be an adverse-action
# reason and makes the age-band fairness check a clean disparate-impact test.
# To revert, set EXCLUDED = [].
EXCLUDED = ["person_age"]

LABELS = {
    "loan_percent_income": "Loan is a large share of income",
    "loan_grade": "Lender risk grade",
    "loan_int_rate": "Interest rate on the loan",
    "person_income": "Income level",
    "person_home_ownership": "Home ownership status",
    "loan_intent": "Stated loan purpose",
    "cb_person_default_on_file": "Prior default on credit file",
    "person_emp_length": "Length of employment",
    "loan_amnt": "Loan amount",
    "cb_person_cred_hist_length": "Length of credit history",
    "person_age": "Age",
}


def select(X: pd.DataFrame) -> pd.DataFrame:
    return X.drop(columns=[c for c in EXCLUDED if c in X.columns])


def make_lgbm():
    # Same hyperparameters as the benchmark in compare_models.py / recheck_models.py.
    return lgb.LGBMClassifier(n_estimators=100, max_depth=7, learning_rate=0.1,
                              class_weight="balanced", random_state=cd.RANDOM_STATE,
                              verbose=-1)


def calibrate(estimator, method=CALIBRATION):
    """Calibration fitted on held-out inner folds of whatever data .fit() gets,
    so inside an outer CV loop it never sees the outer test fold.
    ensemble=False keeps ONE underlying model, which reason codes need."""
    inner = StratifiedKFold(3, shuffle=True, random_state=cd.RANDOM_STATE)
    return CalibratedClassifierCV(estimator, method=method, cv=inner, ensemble=False)


def build(X: pd.DataFrame):
    Xs = select(X)
    return calibrate(cd.make_pipeline(Xs, make_lgbm()))


# ---- revenue / profit helpers --------------------------------------------------
def rate_fill_table(X: pd.DataFrame) -> dict:
    """Median interest rate per loan grade, used ONLY to fill missing rates in
    the profit accounting (3,095 rows). Not a model input transformation."""
    return X.groupby("loan_grade")["loan_int_rate"].median().round(4).to_dict()


def revenue_rate(X: pd.DataFrame, fill: dict) -> np.ndarray:
    r = X["loan_int_rate"].copy()
    miss = r.isna()
    r[miss] = X.loc[miss, "loan_grade"].map(fill)
    return (r.fillna(np.nanmedian(list(fill.values()))) / 100.0).to_numpy()


def break_even_pd(rate: np.ndarray) -> np.ndarray:
    """PD at which expected profit is zero: (1-p)*rate = p*LGD."""
    return rate / (rate + LGD)


# ---- reason codes ----------------------------------------------------------------
def _inner_pipeline(model):
    return model.calibrated_classifiers_[0].estimator


def contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Per-applicant TreeSHAP contributions (log-odds) per source feature.
    Positive = pushes towards default. Sigmoid calibration is monotone, so it
    preserves the direction and ranking of these contributions."""
    pipe = _inner_pipeline(model)
    pre = pipe.named_steps["preprocessing"]
    booster = pipe.named_steps["model"].booster_
    Xs = select(X)
    Xt = pre.transform(Xs)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    raw = booster.predict(Xt, pred_contrib=True)[:, :-1]
    names = pre.get_feature_names_out()
    # select_dtypes, not dtype == object: pandas 3 reports string columns as "str".
    cats = Xs.select_dtypes(exclude="number").columns.tolist()
    src = []
    for n in names:
        body = n.split("__", 1)[1]
        match = next((c for c in cats if body.startswith(c + "_")), None)
        src.append(match or body)
    return pd.DataFrame(raw, columns=names, index=X.index).T.groupby(src).sum().T


def reason_codes(model, X: pd.DataFrame, top_k: int = 3):
    contrib = contributions(model, X)
    out = []
    for idx, row in contrib.iterrows():
        top = row[row > 0].sort_values(ascending=False).head(top_k)
        out.append([{"feature": f, "reason": LABELS.get(f, f),
                     "value": _fmt(X.loc[idx, f]), "impact": round(float(v), 4)}
                    for f, v in top.items()])
    return out


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v


# ---- persistence -------------------------------------------------------------------
def save(model, meta: dict):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def load_meta() -> dict:
    with open(META_PATH) as f:
        return json.load(f)


def load_or_train():
    """Load the committed artifact; retrain if missing or unloadable (e.g. a
    scikit-learn version mismatch on a fresh Streamlit Cloud container).
    The cutoff always comes from the committed JSON, never from a refit."""
    meta = load_meta()
    try:
        return joblib.load(MODEL_PATH), meta, "loaded"
    except Exception:
        X, y, _ = cd.load_raw()
        t0 = time.perf_counter()
        model = build(X).fit(select(X), y)
        meta = {**meta, "retrained_seconds": round(time.perf_counter() - t0, 2)}
        return model, meta, "retrained"


def predict_pd(model, X: pd.DataFrame) -> np.ndarray:
    """Calibrated PD with the regulatory floor applied. The ONLY way any caller
    should turn the model into a PD."""
    return np.clip(model.predict_proba(select(X))[:, 1], PD_FLOOR, 1 - PD_FLOOR)


def score(model, meta: dict, X: pd.DataFrame, top_k: int = 3) -> list:
    pds = predict_pd(model, X)
    rates = revenue_rate(X, meta["rate_fill"])
    reasons = reason_codes(model, X, top_k)
    cut = meta["cutoff"]
    out = []
    for i, p in enumerate(pds):
        ead = float(X.iloc[i]["loan_amnt"])
        out.append({
            "pd": round(float(p), 4),
            "decision": "DECLINE" if p >= cut else "APPROVE",
            "cutoff": cut,
            "break_even_pd": round(float(break_even_pd(rates[i])), 4),
            "expected_loss": round(float(p) * LGD * ead, 2),
            "expected_profit": round((1 - float(p)) * rates[i] * ead - float(p) * LGD * ead, 2),
            "reason_codes": reasons[i],
        })
    return out
