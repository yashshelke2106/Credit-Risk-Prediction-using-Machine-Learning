"""Leakage audit for the credit risk model, plus corrected 5-fold CV.

Run:  python leakage_audit.py
Writes (to reports/): leakage_ablation.csv, feature_audit.csv

Three checks, in the order they matter:
  A. Was any fitted preprocessing applied before the train/test split?
  B. Do duplicate rows let the same record sit in train and test?
  C. Is any feature recorded after the loan outcome is known?
Then an ablation that prices each confirmed leak in ROC-AUC, so the fix is
reported as a number rather than an assertion.
"""
import os
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
import credit_data as cd

RS = cd.RANDOM_STATE
os.makedirs("reports", exist_ok=True)
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
raw = pd.read_csv(cd.RAW)
clean = pd.read_csv(cd.CLEAN)

print("=" * 78); print("A. PRE-SPLIT PREPROCESSING"); print("=" * 78)
leak_a = False
for col in ["person_emp_length", "loan_int_rate"]:
    full_mean = raw[col].mean()
    n_at_mean = int(clean[col].sub(full_mean).abs().lt(1e-6).sum())
    n_missing = int(raw[col].isna().sum())
    hit = n_at_mean > 0
    leak_a |= hit
    print(f"  {col}: {n_missing} missing in raw; {n_at_mean} rows in clean_data.csv hold the "
          f"FULL-DATA mean {full_mean:.6f} -> {'LEAK' if hit else 'clean'}")
print(f"  VERDICT: {'CONFIRMED - imputation fitted on all rows before any split' if leak_a else 'none found'}")

print("\n" + "=" * 78); print("B. DUPLICATE ROWS ACROSS TRAIN/TEST"); print("=" * 78)
n_dup = int(raw.duplicated().sum())
feat = [c for c in raw.columns if c != cd.TARGET]
grp = raw[raw.duplicated(subset=feat, keep=False)].groupby(feat, dropna=False)[cd.TARGET].nunique()
print(f"  exact duplicate rows: {n_dup} ({n_dup/len(raw):.2%} of data)")
print(f"  duplicate groups: {len(grp)}; with conflicting target: {int((grp>1).sum())}")
# Row-level hash is dtype-safe, unlike string concatenation over mixed columns.
key = pd.util.hash_pandas_object(raw[feat], index=False)
idx = np.arange(len(raw)); spill = []
for tr, te in cv.split(idx, raw[cd.TARGET]):
    spill.append(int(key.iloc[te].isin(set(key.iloc[tr])).sum()))
print(f"  test-fold rows with an identical twin in the train fold: {spill} (mean {np.mean(spill):.1f})")
print(f"  VERDICT: {'CONFIRMED - same record appears in train and test' if n_dup else 'none found'}")

print("\n" + "=" * 78); print("C. FEATURE TIMING"); print("=" * 78)
AUDIT = [
 ("person_age","Application","Stated by applicant at application. No outcome information.","OK"),
 ("person_income","Application","Stated/verified at application, before any repayment.","OK"),
 ("person_home_ownership","Application","Applicant status at application.","OK"),
 ("person_emp_length","Application","Employment history at application.","OK"),
 ("loan_intent","Application","Stated purpose at application.","OK"),
 ("loan_amnt","Origination","Amount requested/approved at origination, before outcome.","OK"),
 ("loan_percent_income","Derived","loan_amnt / person_income; both known at origination. Redundant, not leaky.","OK"),
 ("cb_person_default_on_file","Bureau, pre-application","PRIOR defaults from credit bureau. Strictly before this loan.","OK"),
 ("cb_person_cred_hist_length","Bureau, pre-application","History length at pull. No outcome information.","OK"),
 ("loan_grade","Underwriting output","Lender's own risk verdict, produced BY underwriting. Pre-outcome, but target-adjacent.","CAUTION"),
 ("loan_int_rate","Underwriting output","Priced off loan_grade i.e. off expected default. Pre-outcome, but target-adjacent.","CAUTION"),
 ("loan_status","TARGET","Outcome being predicted.","TARGET"),
]
fa = pd.DataFrame(AUDIT, columns=["Feature","Recorded","Verdict","Flag"])
fa.to_csv("reports/feature_audit.csv", index=False)
for _, r in fa.iterrows(): print(f"  [{r.Flag:8s}] {r.Feature:27s} {r.Verdict}")
print("  VERDICT: no feature is recorded after the outcome. Two are underwriting outputs.")

print("\n" + "=" * 78); print("ABLATION - what each confirmed leak is worth"); print("=" * 78)
def ev(X, y, label, models=("Random Forest",)):
    est = {"Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RS),
           "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=15,
                                                   class_weight="balanced", random_state=RS, n_jobs=-1)}
    out = []
    for m in models:
        p = cd.make_pipeline(X, est[m])
        a = cross_val_score(p, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        o = cross_val_predict(p, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        out.append({"Configuration": label, "Model": m, "CV ROC-AUC": a.mean(), "CV Std": a.std(),
                    "KS": cd.ks_statistic(y, o), "Rows": len(y)})
        print(f"  {label:42s} {m:20s} AUC {a.mean():.4f} +/- {a.std():.4f}  KS {out[-1]['KS']:.4f}")
    return out

rows = []
Xl, yl = cd.load_clean_legacy()
rows += ev(Xl, yl, "1. Leaky baseline (pre-imputed + dupes)")
X2, y2, _ = cd.load_raw(dedupe=False)
rows += ev(X2, y2, "2. In-fold imputation, dupes remain")
X3, y3, _ = cd.load_raw(dedupe=True)
rows += ev(X3, y3, "3. CORRECTED (in-fold + deduped)", ("Logistic Regression", "Random Forest"))
Xa = X3.drop(columns=["loan_grade", "loan_int_rate"])
rows += ev(Xa, y3, "4. Corrected, application-only", ("Logistic Regression", "Random Forest"))
df = pd.DataFrame(rows); df.to_csv("reports/leakage_ablation.csv", index=False)

rf = df[df.Model == "Random Forest"].reset_index(drop=True)
print(f"\n  RF leaky -> corrected: {rf.loc[0,'CV ROC-AUC']:.4f} -> {rf.loc[2,'CV ROC-AUC']:.4f} "
      f"({rf.loc[2,'CV ROC-AUC']-rf.loc[0,'CV ROC-AUC']:+.4f}), fold std ~{rf.loc[2,'CV Std']:.4f}")
print(f"    from pre-split imputation: {rf.loc[1,'CV ROC-AUC']-rf.loc[0,'CV ROC-AUC']:+.4f}")
print(f"    from duplicate spillover : {rf.loc[2,'CV ROC-AUC']-rf.loc[1,'CV ROC-AUC']:+.4f}")
print(f"    removing underwriting features: {rf.loc[3,'CV ROC-AUC']-rf.loc[2,'CV ROC-AUC']:+.4f}  <-- the real driver")

o = cross_val_predict(cd.make_pipeline(X3, RandomForestClassifier(n_estimators=300, max_depth=15,
        class_weight="balanced", random_state=RS, n_jobs=-1)), X3, y3, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
fpr, tpr, _ = roc_curve(y3, o)
print("\n  ROC shape (corrected model), explains the near-vertical rise:")
for f in (0.001, 0.01, 0.05, 0.10):
    print(f"    TPR @ FPR={f:<5.3f} = {np.interp(f, fpr, tpr):.4f}")
