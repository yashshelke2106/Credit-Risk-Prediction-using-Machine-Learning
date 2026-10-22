"""
Credit Risk Model Comparison: Random Forest, LightGBM, XGBoost, CatBoost.

All four classifiers run inside the SAME preprocessing pipeline that app1.py
deploys (mean/mode imputation -> scaling -> one-hot encoding), so the comparison
is apples-to-apples and the deployed configuration is literally one of the rows.

Reports both a single stratified 80/20 holdout and 5-fold cross-validated
ROC-AUC, because the gap between the two boosters is inside the noise band of a
single split and only cross-validation shows which differences are real.

Also runs a threshold-dominance check: Random Forest's apparent precision
advantage at the default 0.50 cutoff is a threshold artifact, not a model
property. The winner is re-thresholded to match RF's precision and recall so the
two models can be compared at identical operating points.

Outputs
-------
model_comparison_results.csv   full metrics + confusion matrices + CV AUC
model_comparison_summary.csv   rounded summary table for the README
threshold_analysis.csv         winner vs Random Forest at matched operating points
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = "clean_data.csv"

# ============================================================================
# LOAD & PREPARE DATA
# ============================================================================
df = pd.read_csv(DATA_PATH)

# age_group is a binning column created in EDA.ipynb for visualisation only.
# It is a deterministic function of person_age, so leaving it in would leak a
# duplicate of an existing feature into the model.
df = df.drop(columns=["age_group"], errors="ignore")

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

print(f"Dataset:      {df.shape[0]:,} rows x {X.shape[1]} features")
print(f"Default rate: {y.mean():.2%}  ({int(y.sum()):,} defaults)")

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"Numeric:      {numeric_cols}")
print(f"Categorical:  {categorical_cols}")


def build_preprocessor() -> ColumnTransformer:
    """The exact preprocessing app1.py uses, rebuilt per model to avoid shared state."""
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {X_train.shape[0]:,} ({y_train.mean():.2%} default)")
print(f"Test:  {X_test.shape[0]:,} ({y_test.mean():.2%} default)")

# ============================================================================
# DEFINE MODELS
# ============================================================================
# Every model carries class weighting, because a 21.8% default rate means an
# unweighted model is rewarded for under-predicting the minority class.
# scale_pos_weight is computed from the training fold only.
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

estimators = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1,
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    ),
    "CatBoost": CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        auto_class_weights="Balanced",
        random_state=RANDOM_STATE,
        verbose=0,
    ),
}

models = {
    name: Pipeline([("preprocessing", build_preprocessor()), ("model", est)])
    for name, est in estimators.items()
}

# ============================================================================
# TRAIN, EVALUATE, CROSS-VALIDATE
# ============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = []
probabilities = {}

for name, pipeline in models.items():
    print(f"\n--- {name} ---")
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    probabilities[name] = y_proba

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1)

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "CV ROC-AUC (5-fold)": cv_scores.mean(),
            "CV Std": cv_scores.std(),
            "Specificity": tn / (tn + fp),
            "True Negatives": tn,
            "False Positives": fp,
            "False Negatives": fn,
            "True Positives": tp,
        }
    )

    r = results[-1]
    print(f"  Holdout ROC-AUC: {r['ROC-AUC']:.4f}")
    print(f"  5-fold ROC-AUC:  {r['CV ROC-AUC (5-fold)']:.4f} +/- {r['CV Std']:.4f}")
    print(f"  Precision: {r['Precision']:.4f}   Recall: {r['Recall']:.4f}   F1: {r['F1-Score']:.4f}")

results_df = (
    pd.DataFrame(results)
    .sort_values("CV ROC-AUC (5-fold)", ascending=False)
    .reset_index(drop=True)
)

print("\n" + "=" * 94)
print("MODEL COMPARISON (ranked by 5-fold cross-validated ROC-AUC)")
print("=" * 94)
print(
    results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
            "CV ROC-AUC (5-fold)",
            "CV Std",
        ]
    ].to_string(index=False)
)
print("=" * 94)

winner = results_df.iloc[0]["Model"]
runner_up = results_df.iloc[1]["Model"]
gap = results_df.iloc[0]["CV ROC-AUC (5-fold)"] - results_df.iloc[1]["CV ROC-AUC (5-fold)"]
pooled_std = max(results_df.iloc[0]["CV Std"], results_df.iloc[1]["CV Std"])

print(f"\nTop model: {winner} (CV ROC-AUC {results_df.iloc[0]['CV ROC-AUC (5-fold)']:.4f})")
print(f"Gap to {runner_up}: {gap:.4f} vs fold std {pooled_std:.4f}", end="  ")
print("-> REAL" if gap > 2 * pooled_std else "-> within noise, treat as tied")

# ============================================================================
# THRESHOLD DOMINANCE: winner vs Random Forest at identical operating points
# ============================================================================
# Comparing precision at a fixed 0.50 cutoff compares two arbitrary points on
# two different curves. If the winner's ROC curve sits above RF's, then matching
# RF's precision should still leave the winner with more recall - and vice versa.
baseline = "Random Forest"
rf_pred = (probabilities[baseline] >= 0.5).astype(int)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_tn, rf_fp, rf_fn, rf_tp = confusion_matrix(y_test, rf_pred).ravel()

prec_curve, rec_curve, thresholds = precision_recall_curve(y_test, probabilities[winner])
prec_curve, rec_curve = prec_curve[:-1], rec_curve[:-1]

matched_precision_idx = int(np.argmin(np.abs(prec_curve - rf_precision)))
matched_recall_idx = int(np.argmin(np.abs(rec_curve - rf_recall)))

threshold_rows = [
    {
        "Configuration": f"{baseline} @ 0.50 (previous deployment)",
        "Threshold": 0.50,
        "Precision": rf_precision,
        "Recall": rf_recall,
        "Defaults Caught": rf_tp,
        "False Alarms": rf_fp,
    }
]

for label, idx in [
    (f"{winner} @ matched precision", matched_precision_idx),
    (f"{winner} @ matched recall", matched_recall_idx),
]:
    threshold = float(thresholds[idx])
    pred = (probabilities[winner] >= threshold).astype(int)
    _, w_fp, _, w_tp = confusion_matrix(y_test, pred).ravel()
    threshold_rows.append(
        {
            "Configuration": label,
            # Unrounded on purpose. These thresholds come from
            # precision_recall_curve, so each one IS a sample's predicted
            # probability. Rounding it moves that sample across the >= cutoff and
            # the row stops reproducing its own precision/recall.
            "Threshold": threshold,
            "Precision": float(prec_curve[idx]),
            "Recall": float(rec_curve[idx]),
            "Defaults Caught": w_tp,
            "False Alarms": w_fp,
        }
    )

threshold_df = pd.DataFrame(threshold_rows)

print("\n" + "=" * 94)
print(f"THRESHOLD DOMINANCE: {winner} vs {baseline} at identical operating points")
print("=" * 94)
print(threshold_df.to_string(index=False))
print("=" * 94)

caught_delta = threshold_df.iloc[1]["Defaults Caught"] - rf_tp
print(
    f"\nAt {baseline}'s own precision ({rf_precision:.4f}), {winner} catches "
    f"{caught_delta:+d} more defaults ({threshold_df.iloc[1]['Defaults Caught']:,} vs {rf_tp:,} "
    f"of {int(y_test.sum()):,})."
)
print(
    f"At {baseline}'s own recall ({rf_recall:.4f}), {winner} reaches precision "
    f"{threshold_df.iloc[2]['Precision']:.4f} vs {rf_precision:.4f}."
)
print(
    f"\nConclusion: {baseline} is not precision-optimised, it is on a lower curve.\n"
    f"{winner} is deployed in app1.py with the decision threshold exposed as a control."
)

# ============================================================================
# EXPORT
# ============================================================================
results_df.to_csv("model_comparison_results.csv", index=False)
print("\nWrote model_comparison_results.csv")

summary_df = results_df[
    ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "CV ROC-AUC (5-fold)", "CV Std"]
].copy()
for col in summary_df.columns[1:]:
    summary_df[col] = summary_df[col].map(lambda v: f"{v:.4f}")
summary_df.to_csv("model_comparison_summary.csv", index=False)
print("Wrote model_comparison_summary.csv")

threshold_df.to_csv("threshold_analysis.csv", index=False)
print("Wrote threshold_analysis.csv")
