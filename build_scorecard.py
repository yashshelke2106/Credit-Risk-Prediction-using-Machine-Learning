"""WoE credit scorecard: IV table, logistic regression on WoE, points, cut-off.

Run:  python build_scorecard.py
Writes (to reports/): iv_table.csv, scorecard.csv, cutoff_analysis.csv, scorecard_meta.json

Scaling: base score 600 at 50:1 odds (good:bad), PDO 20.
    factor = PDO / ln(2)
    offset = base - factor * ln(odds)
    points_ij = -(beta_i * woe_ij + alpha/n) * factor + offset/n
Summing points over characteristics gives offset + factor*ln(odds), asserted below.

All model selection and cut-off numbers come from OUT-OF-FOLD predictions with
the WoE bins refit inside each fold, because bin edges and WoE values are fitted
from the target and would otherwise leak exactly like the imputation this
project was audited for.
"""
import json
import os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import credit_data as cd
from woe import WoEEncoder

BASE, PDO, ODDS = 600, 20, 50
FACTOR = PDO / np.log(2)
OFFSET = BASE - FACTOR * np.log(ODDS)
RS = cd.RANDOM_STATE
os.makedirs("reports", exist_ok=True)
cv = StratifiedKFold(5, shuffle=True, random_state=RS)

X, y, rep = cd.load_raw()
print(f"Rows {len(X):,} | default rate {y.mean():.4f}")
print(f"factor {FACTOR:.4f} | offset {OFFSET:.4f}\n")

def pipe():
    return Pipeline([("woe", WoEEncoder()), ("model", LogisticRegression(max_iter=2000, random_state=RS))])

aucs = cross_val_score(pipe(), X, y, cv=cv, scoring="roc_auc", n_jobs=1)
oof = cross_val_predict(pipe(), X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
ks = cd.ks_statistic(y, oof)
print(f"Scorecard CV ROC-AUC {aucs.mean():.4f} +/- {aucs.std():.4f} | OOF KS {ks:.4f}")

# ---------- final scorecard, fitted on all rows ----------
enc = WoEEncoder().fit(X, y)
W = enc.transform(X)
model = LogisticRegression(max_iter=2000, random_state=RS).fit(W, y)
alpha, betas = float(model.intercept_[0]), model.coef_[0]
feats = list(enc.woe_maps_.keys()); n = len(feats)

iv = enc.iv_table()
iv["Flag"] = iv.IV.map(lambda v: "WEAK (<0.02)" if v < 0.02 else ("SUSPICIOUS (>0.5)" if v > 0.5 else "ok"))
iv.to_csv("reports/iv_table.csv", index=False)
print("\n--- INFORMATION VALUE ---"); print(iv.to_string(index=False))

rows = []
for i, f in enumerate(feats):
    for b, r in enc.bin_tables_[f].iterrows():
        rows.append({"Feature": f, "Bin": b, "Count": int(r.Count), "Bads": int(r.Bads),
                     "BadRate": round(float(r.BadRate), 4), "WoE": round(float(r.WoE), 4),
                     "IV_contrib": round(float(r.IV_contrib), 4), "Beta": round(float(betas[i]), 4),
                     "Points": int(round(-(betas[i] * r.WoE + alpha / n) * FACTOR + OFFSET / n))})
pd.DataFrame(rows).to_csv("reports/scorecard.csv", index=False)
print(f"\nscorecard.csv: {len(rows)} bins across {n} characteristics")

pts = sum(-(betas[i] * W[:, i] + alpha / n) * FACTOR + OFFSET / n for i in range(n))
assert np.allclose(pts, OFFSET + FACTOR * (-(alpha + W @ betas)), atol=1e-6), "points algebra broken"
p600 = 1 / (1 + np.exp((600 - OFFSET) / FACTOR))
print(f"Points algebra verified | score 600 => odds {(1-p600)/p600:.1f}:1 (target {ODDS}:1)")
print(f"Score range {pts.min():.0f}-{pts.max():.0f}, mean {pts.mean():.0f}")

# ---------- cut-off, from OOF scores ----------
score = OFFSET + FACTOR * (-np.log(oof / (1 - oof)))
tot, totbad, totgood = len(y), int(y.sum()), int((y == 0).sum())
grid = []
for c in range(int(score.min()), int(score.max()) + 1):
    ap = score >= c
    na = int(ap.sum())
    if na in (0, tot): continue
    ba = int(y[ap].sum()); gd = int(((~ap) & (y == 0)).sum())
    grid.append({"Cutoff": c, "Approval rate": na / tot, "Defaults captured": (totbad - ba) / totbad,
                 "Approved book bad rate": ba / na, "Good applicants declined": gd / totgood,
                 "KS": abs((totbad - ba) / totbad - gd / totgood)})
g = pd.DataFrame(grid); g.to_csv("reports/cutoff_analysis.csv", index=False)
best = g.loc[g.KS.idxmax()]
print("\n--- CUT-OFF GRID (every 20 points) ---")
sh = g[g.Cutoff % 20 == 0].copy()
for c in sh.columns[1:]: sh[c] = sh[c].map(lambda v: f"{v:.4f}")
print(sh.to_string(index=False))
print(f"\nSelected cut-off (KS-optimal): {int(best.Cutoff)}")
print(f"  approval rate            {best['Approval rate']:.4f}")
print(f"  defaults captured        {best['Defaults captured']:.4f}")
print(f"  approved-book bad rate   {best['Approved book bad rate']:.4f} vs {y.mean():.4f} baseline")
print(f"  good applicants declined {best['Good applicants declined']:.4f}")

json.dump({"cv_auc": float(aucs.mean()), "cv_std": float(aucs.std()), "ks": float(ks),
           "rows": int(len(X)), "default_rate": float(y.mean()), "factor": FACTOR, "offset": OFFSET,
           "cutoff": int(best.Cutoff), "approval_rate": float(best["Approval rate"]),
           "defaults_captured": float(best["Defaults captured"]),
           "approved_bad_rate": float(best["Approved book bad rate"]),
           "good_declined": float(best["Good applicants declined"]),
           "top3_iv": iv.head(3)[["Feature", "IV"]].to_dict("records")},
          open("reports/scorecard_meta.json", "w"), indent=2)
print("\nWrote iv_table.csv, scorecard.csv, cutoff_analysis.csv, scorecard_meta.json")
