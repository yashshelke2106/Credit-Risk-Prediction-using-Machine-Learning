"""
Synthetic multi-table loan-portfolio dataset for the collections/vintage analytics
project. Extends the flat credit_risk_dataset.csv (one row per loan, no time series)
into a proper star-schema-ready set of raw tables:

  dim_borrower.csv        one row per borrower
  dim_loan.csv            one row per loan (origination terms)
  fact_loan_payments.csv  one row per loan per month since origination (DPD bucket,
                           balance, payment) -- this is what makes vintage curves and
                           roll-rate matrices possible; the source dataset has no
                           repayment history at all.

Known simplification (flag this in interviews): each borrower has exactly one loan
here, because the source dataset is one row per applicant. Real bureau data has
many loans per borrower -- the schema keeps borrower_id and loan_id as separate
keys so it *could* support that, it just isn't populated that way yet.

The monthly delinquency path is a stochastic Markov simulation calibrated so that:
  - roll-forward risk increases with loan_grade (A safest -> G riskiest), matching
    the real default-rate-by-grade pattern already found in EDA.ipynb.
  - a loan's simulated path is forced to be consistent with the ORIGINAL, real
    loan_status label (1 = actually defaulted, 0 = actually repaid), so downstream
    analytics can be checked against ground truth.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

SRC = Path.home() / "mnt" / "ML_project" / "clean_data.csv"
OUT = Path.home() / "mnt" / "ML_project" / "loan-portfolio-analytics" / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

AS_OF_MONTH = pd.Period("2026-09", freq="M")   # "today" in the synthetic world
VINTAGE_WINDOW_MONTHS = 36                      # origination spread: last 36 months

# monthly Current -> 30dpd roll probability, by loan_grade
BASE_ROLL_FWD = {"A": 0.010, "B": 0.020, "C": 0.035, "D": 0.060, "E": 0.090, "F": 0.130, "G": 0.170}
# monthly cure probability back to Current, by current bucket
CURE_PROB = {"DPD30": 0.42, "DPD60": 0.20, "DPD90": 0.08}
# monthly further roll-forward probability once delinquent
FURTHER_ROLL = {"DPD30": 0.46, "DPD60": 0.52, "DPD90": 0.38}   # DPD90 -> Default

TERM_CHOICES = np.array([12, 24, 36, 48, 60])

def pick_term(loan_amnt: float) -> int:
    # bigger loans skew toward longer terms
    if loan_amnt < 5000:
        weights = [0.45, 0.35, 0.15, 0.04, 0.01]
    elif loan_amnt < 15000:
        weights = [0.15, 0.30, 0.35, 0.15, 0.05]
    else:
        weights = [0.05, 0.15, 0.30, 0.30, 0.20]
    return int(rng.choice(TERM_CHOICES, p=weights))

def monthly_installment(principal: float, annual_rate_pct: float, term: int) -> float:
    r = (annual_rate_pct / 100.0) / 12.0
    if r <= 0:
        return principal / term
    return principal * r / (1 - (1 + r) ** (-term))

def simulate_path(term: int, grade: str, truly_defaulted: bool):
    """Return list of (month_index, dpd_bucket) for months_on_book = 0..min(term,observed)-1
    plus a flag for whether/when the loan closes (paid off) or defaults."""
    roll_fwd = BASE_ROLL_FWD[grade]
    order = ["Current", "DPD30", "DPD60", "DPD90"]
    path = []
    state = "Current"
    # if this loan is a true defaulter, bias risk upward so it's plausible within term
    fwd_mult = 2.6 if truly_defaulted else 1.0
    cure_mult = 0.4 if truly_defaulted else 1.0

    default_month = None
    close_month = None
    for m in range(term):
        path.append(state)
        if state == "Default":
            break
        if state == "Current":
            if rng.random() < min(roll_fwd * fwd_mult, 0.9):
                state = "DPD30"
        else:
            if rng.random() < CURE_PROB[state] * cure_mult:
                state = "Current"
            elif rng.random() < FURTHER_ROLL[state]:
                nxt = {"DPD30": "DPD60", "DPD60": "DPD90", "DPD90": "Default"}[state]
                state = nxt
                if state == "Default":
                    default_month = m + 1

    # force ground-truth consistency: real defaulters that never hit Default in the
    # stochastic sim are forced to default near the back half of the term.
    if truly_defaulted and "Default" not in path:
        force_at = rng.integers(int(term * 0.5), max(int(term * 0.5) + 1, term))
        path = path[:force_at] + ["Default"]
        default_month = force_at
    # real non-defaulters that drifted into Default (rare) are pulled back to DPD90/cure
    if (not truly_defaulted) and "Default" in path:
        idx = path.index("Default")
        path = path[:idx] + ["DPD90"] * (len(path) - idx)

    observed_len = len(path)
    if not truly_defaulted and observed_len == term:
        close_month = term  # paid off in full at term end
    return path[:observed_len], default_month, close_month


def main():
    df = pd.read_csv(SRC).drop(columns=["age_group"], errors="ignore").reset_index(drop=True)
    df["borrower_id"] = np.arange(1, len(df) + 1)
    df["loan_id"] = np.arange(1, len(df) + 1)  # 1:1 borrower:loan simplification (see module docstring)

    # ---- dim_borrower -----------------------------------------------------
    dim_borrower = df[[
        "borrower_id", "person_age", "person_income", "person_home_ownership",
        "person_emp_length", "cb_person_default_on_file", "cb_person_cred_hist_length",
    ]].copy()
    dim_borrower.to_csv(OUT / "dim_borrower.csv", index=False)

    # ---- origination terms --------------------------------------------------
    months_ago = rng.integers(1, VINTAGE_WINDOW_MONTHS + 1, size=len(df))
    origination_month = [AS_OF_MONTH - int(k) for k in months_ago]
    term = df["loan_amnt"].apply(pick_term).to_numpy()
    installment = [
        round(monthly_installment(a, r, t), 2)
        for a, r, t in zip(df["loan_amnt"], df["loan_int_rate"], term)
    ]

    dim_loan = df[[
        "loan_id", "borrower_id", "loan_intent", "loan_grade", "loan_amnt",
        "loan_int_rate", "loan_percent_income", "loan_status",
    ]].copy()
    dim_loan["origination_month"] = [str(p) for p in origination_month]
    dim_loan["term_months"] = term
    dim_loan["monthly_installment"] = installment
    dim_loan.rename(columns={"loan_status": "ever_defaulted_actual"}, inplace=True)
    dim_loan.to_csv(OUT / "dim_loan.csv", index=False)

    # ---- fact_loan_payments (the new time-series table) --------------------
    rows = []
    for loan_id, orig_m, t, grade, defaulted, bal0, inst in zip(
        dim_loan["loan_id"], origination_month, term, dim_loan["loan_grade"],
        dim_loan["ever_defaulted_actual"].astype(bool), df["loan_amnt"], installment,
    ):
        months_elapsed = int((AS_OF_MONTH - orig_m).n) + 1
        # simulate against the FULL contractual term so forced-default timing is
        # realistic, then censor to what would actually be observable as-of-now
        full_path, default_month, close_month = simulate_path(t, grade, defaulted)
        observed_len = min(len(full_path), max(months_elapsed, 1))
        path = full_path[:observed_len]
        if close_month is not None and close_month > observed_len:
            close_month = None  # hasn't actually reached term yet as-of-now
        balance = float(bal0)
        for m_idx, bucket in enumerate(path):
            pay_month = orig_m + m_idx
            if bucket == "Current" and balance > 0:
                balance = max(balance - inst, 0.0)
                payment = inst
            elif bucket == "Default":
                payment = 0.0
            else:
                payment = 0.0  # missed payment while delinquent
            rows.append((
                loan_id, str(pay_month), m_idx, bucket, round(balance, 2), payment,
                bucket == "Default", (close_month == m_idx + 1),
            ))

    fact = pd.DataFrame(rows, columns=[
        "loan_id", "payment_month", "months_on_book", "dpd_bucket",
        "balance_outstanding", "payment_amount", "is_default_event", "is_closed",
    ])
    fact.to_csv(OUT / "fact_loan_payments.csv", index=False)

    print(f"dim_borrower: {len(dim_borrower):,} rows")
    print(f"dim_loan:     {len(dim_loan):,} rows")
    print(f"fact_loan_payments: {len(fact):,} rows")
    print(f"actual default rate honored: {dim_loan['ever_defaulted_actual'].mean():.4f}")
    sim_default_rate = fact.groupby("loan_id")["is_default_event"].any().mean()
    print(f"simulated path default rate: {sim_default_rate:.4f}")

if __name__ == "__main__":
    main()
