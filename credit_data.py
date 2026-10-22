"""Shared data loading + leak-free pipeline construction for the credit risk project.

Single source of truth so the audit, the CV evaluation and the scorecard cannot
silently diverge in how they prepare data.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW = "credit_risk_dataset.csv"
CLEAN = "clean_data.csv"
TARGET = "loan_status"
RANDOM_STATE = 42

# Values that cannot be real. Converting them to NaN is a row-wise rule with no
# fitted statistic, so applying it before the split leaks nothing - unlike the
# mean imputation, which must happen inside each fold.
IMPLAUSIBLE = {"person_age": lambda s: s > 100, "person_emp_length": lambda s: s > 60}


def load_raw(dedupe: bool = True, sanitize: bool = True):
    """Load the raw Kaggle file. No imputation here, by design."""
    df = pd.read_csv(RAW)
    report = {"rows_loaded": len(df)}

    if sanitize:
        n = 0
        for col, rule in IMPLAUSIBLE.items():
            mask = rule(df[col])
            n += int(mask.sum())
            df.loc[mask, col] = np.nan          # becomes NaN, row is KEPT
        report["cells_nulled"] = n

    if dedupe:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        report["duplicates_removed"] = before - len(df)

    report["rows_final"] = len(df)
    report["default_rate"] = float(df[TARGET].mean())
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y, report


def load_clean_legacy():
    """The pre-imputed clean_data.csv, used ONLY to reproduce the leaky baseline."""
    df = pd.read_csv(CLEAN).drop(columns=["age_group"], errors="ignore")
    return df.drop(columns=[TARGET]), df[TARGET]


def build_preprocessor(X):
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), num),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ]
    )


def make_pipeline(X, estimator):
    """Every fitted statistic lives inside this object, so cross_val_score refits
    it per fold and no test-fold information reaches the imputer or the scaler."""
    return Pipeline([("preprocessing", build_preprocessor(X)), ("model", estimator)])


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov: max separation between the good and bad score CDFs."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    y_true = y_true[order]
    bad = np.cumsum(y_true) / max(y_true.sum(), 1)
    good = np.cumsum(1 - y_true) / max((1 - y_true).sum(), 1)
    return float(np.max(np.abs(good - bad)))
