"""Full evaluation of the credit decision system. Regenerates every number in
the README's results block.

Run:   python evaluate.py          (or: make eval)
Writes reports/*.csv, reports/metrics.json, reports/figures/*.png,
       models/pd_model.joblib + models/pd_model_meta.json,
       and rewrites the README block between <!-- RESULTS:START/END -->.

Every fitted thing - imputation, scaling, encoding, WoE bins, the model, and
the calibrator - is refit inside each outer fold. Nothing in the
reported numbers is fitted on the rows it is scored on.
"""
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

import credit_data as cd
import pd_model as pm
from woe import WoEEncoder

T_START = time.perf_counter()
RS = cd.RANDOM_STATE
OUT, FIG = "reports", os.path.join("reports", "figures")
os.makedirs(FIG, exist_ok=True)
CV = StratifiedKFold(5, shuffle=True, random_state=RS)

# palette: dataviz reference slots 1-3 (validated all-pairs), light surface
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

X, y, DATA = cd.load_raw()
yv = y.to_numpy()
BASE_RATE = float(y.mean())
print(f"Data: {len(X):,} rows, default rate {BASE_RATE:.4f}")


# =============================================================================
# 1. Models, raw vs calibrated, per-fold
# =============================================================================
def all_cols(X_):
    return X_


MODELS = {
    "Logistic Regression": (lambda Xf: cd.make_pipeline(Xf, LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RS)), all_cols),
    "WoE Scorecard": (lambda Xf: Pipeline([("woe", WoEEncoder()), ("model", LogisticRegression(
        max_iter=2000, random_state=RS))]), all_cols),
    "Random Forest": (lambda Xf: cd.make_pipeline(Xf, RandomForestClassifier(
        n_estimators=300, max_depth=15, class_weight="balanced", random_state=RS,
        n_jobs=-1)), all_cols),
    "LightGBM (all features)": (lambda Xf: cd.make_pipeline(Xf, pm.make_lgbm()), all_cols),
    "LightGBM deployed (excl. age)": (lambda Xf: cd.make_pipeline(Xf, pm.make_lgbm()), pm.select),
}
DEPLOYED = "LightGBM deployed (excl. age)"


def ks(y_true, p):
    return cd.ks_statistic(y_true, p)


def ece(y_true, p, bins=10):
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    d = pd.DataFrame({"y": y_true, "p": p, "b": q})
    g = d.groupby("b").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g.n * (g.y - g.p).abs()).sum() / g.n.sum())


oof = {}
rows = []
for name, (factory, cols) in MODELS.items():
    t0 = time.perf_counter()
    Xm = cols(X)
    raw, cal = np.zeros(len(y)), np.zeros(len(y))
    fold = {"raw": [], "calibrated": []}
    for tr, te in CV.split(Xm, y):
        Xtr, Xte, ytr, yte = Xm.iloc[tr], Xm.iloc[te], y.iloc[tr], y.iloc[te]
        raw[te] = factory(Xtr).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        cal[te] = np.clip(pm.calibrate(factory(Xtr)).fit(Xtr, ytr).predict_proba(Xte)[:, 1],
                          pm.PD_FLOOR, 1 - pm.PD_FLOOR)
        for k, p in (("raw", raw[te]), ("calibrated", cal[te])):
            fold[k].append((roc_auc_score(yte, p), average_precision_score(yte, p),
                            brier_score_loss(yte, p)))
    oof[(name, "raw")], oof[(name, "calibrated")] = raw, cal
    for k, p in (("raw", raw), ("calibrated", cal)):
        f = np.array(fold[k])
        rows.append({"Model": name, "Variant": k,
                     "ROC-AUC": f[:, 0].mean(), "ROC-AUC std": f[:, 0].std(),
                     "Gini": 2 * f[:, 0].mean() - 1,
                     "PR-AUC": f[:, 1].mean(), "PR-AUC std": f[:, 1].std(),
                     "Brier": f[:, 2].mean(), "Brier std": f[:, 2].std(),
                     "KS": ks(yv, p), "Mean PD": float(p.mean()), "ECE": ece(yv, p)})
    print(f"  {name:32s} done in {time.perf_counter()-t0:5.1f}s | "
          f"raw AUC {rows[-2]['ROC-AUC']:.4f} Brier {rows[-2]['Brier']:.4f} meanPD {rows[-2]['Mean PD']:.3f} | "
          f"cal Brier {rows[-1]['Brier']:.4f} meanPD {rows[-1]['Mean PD']:.3f}")

# majority-class baseline: predicts the training-fold default rate for everyone
maj = np.zeros(len(y)); mf = []
for tr, te in CV.split(X, y):
    maj[te] = y.iloc[tr].mean()
    mf.append((0.5, average_precision_score(y.iloc[te], maj[te]),
               brier_score_loss(y.iloc[te], maj[te])))
mf = np.array(mf)
rows.insert(0, {"Model": "Majority class (baseline)", "Variant": "-", "ROC-AUC": 0.5,
                "ROC-AUC std": 0.0, "Gini": 0.0, "PR-AUC": mf[:, 1].mean(),
                "PR-AUC std": mf[:, 1].std(), "Brier": mf[:, 2].mean(),
                "Brier std": mf[:, 2].std(), "KS": 0.0, "Mean PD": float(maj.mean()),
                "ECE": ece(yv, maj + np.random.default_rng(RS).normal(0, 1e-9, len(maj)), 1)})
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(os.path.join(OUT, "model_metrics.csv"), index=False)

# ---- reliability figure --------------------------------------------------------
def reliability(p, bins=10):
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    d = pd.DataFrame({"y": yv, "p": p, "b": q}).groupby("b").mean()
    return d.p.to_numpy(), d.y.to_numpy()

panels = ["Logistic Regression", "Random Forest", DEPLOYED]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURF, sharey=True)
for ax, name in zip(axes, panels):
    ax.set_facecolor(SURF)
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, zorder=1)
    for variant, col in (("raw", ORANGE), ("calibrated", BLUE)):
        px_, py_ = reliability(oof[(name, variant)])
        r = metrics_df[(metrics_df.Model == name) & (metrics_df.Variant == variant)].iloc[0]
        ax.plot(px_, py_, color=col, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, zorder=3,
                label=f"{variant}  (Brier {r.Brier:.4f})")
    ax.set_title(name.replace(" deployed (excl. age)", "\n(deployed, excl. age)"), color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Mean predicted PD (decile)", color=INK2, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values(): s.set_color(GRID)
    ax.grid(color=GRID, lw=0.6, alpha=0.6)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
axes[0].set_ylabel("Observed default rate", color=INK2, fontsize=9)
fig.suptitle("Reliability: class-weighted models over-predict PD; Platt (sigmoid) calibration fixes it",
             color=INK, fontsize=12, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "reliability.png"), dpi=130, facecolor=SURF)
plt.close(fig)


# ---- calibration method: isotonic vs sigmoid on the deployed model ------------
cm_rows = []
Xd = pm.select(X)
for method in ("isotonic", "sigmoid"):
    p = np.zeros(len(y)); f = []
    for tr, te in CV.split(Xd, y):
        m = pm.calibrate(cd.make_pipeline(Xd.iloc[tr], pm.make_lgbm()), method).fit(Xd.iloc[tr], y.iloc[tr])
        p[te] = m.predict_proba(Xd.iloc[te])[:, 1]
        f.append((roc_auc_score(y.iloc[te], p[te]), brier_score_loss(y.iloc[te], p[te])))
    f = np.array(f)
    cm_rows.append({"Method": method, "ROC-AUC": f[:, 0].mean(), "Brier": f[:, 1].mean(),
                    "Brier std": f[:, 1].std(), "ECE": ece(yv, p),
                    "PD exactly 0 (<=1e-4)": int((p <= 1e-4).sum()),
                    "PD exactly 1 (>=0.9999)": int((p >= 0.9999).sum()),
                    "Distinct PD levels": int(len(np.unique(p.round(6))))})
cal_method = pd.DataFrame(cm_rows)
cal_method.to_csv(os.path.join(OUT, "calibration_method.csv"), index=False)
print(cal_method.round(4).to_string(index=False))

# =============================================================================
# 2. Profit-based cut-off on the deployed model's calibrated OOF PDs
# =============================================================================
p_dep = oof[(DEPLOYED, "calibrated")]
fill = pm.rate_fill_table(X)
rate = pm.revenue_rate(X, fill)
ead = X["loan_amnt"].to_numpy(dtype=float)
rev, loss = rate * ead, pm.LGD * ead
GOOD, BAD = (yv == 0), (yv == 1)
os.makedirs("models", exist_ok=True)
# out-of-fold arrays for the Streamlit threshold slider (honest held-out numbers)
np.savez_compressed(os.path.join("models", "oof_deployed.npz"), y=yv.astype(np.int8),
                    pd=p_dep.astype(np.float32), rev=rev.astype(np.float32), loss=loss.astype(np.float32))


def policy(approve, p=None, label=""):
    a = approve
    tn, fp, fn, tp = confusion_matrix(yv, (~a).astype(int), labels=[0, 1]).ravel()
    # positive class = default; "predicted positive" = declined
    out = {"Policy": label,
           "Approval rate": a.mean(),
           "Defaults declined (recall)": tp / BAD.sum(),
           "Precision (declined who default)": tp / max(tp + fp, 1),
           "Good borrowers declined": fp / GOOD.sum(),
           "Bad rate in approved book": yv[a].mean() if a.any() else 0.0,
           "Realised loss": float(loss[a & BAD].sum()),
           "Realised revenue": float(rev[a & GOOD].sum()),
           "Realised profit": float(rev[a & GOOD].sum() - loss[a & BAD].sum()),
           "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    if p is not None:
        out["Expected loss (model)"] = float((p[a] * loss[a]).sum())
    return out


grid = np.round(np.arange(0.02, 0.981, 0.005), 3)
curve = []
for t in grid:
    a = p_dep < t
    curve.append({"cutoff": t,
                  "expected_profit": float(((1 - p_dep[a]) * rev[a] - p_dep[a] * loss[a]).sum()),
                  "realised_profit": float(rev[a & GOOD].sum() - loss[a & BAD].sum()),
                  "approval_rate": float(a.mean())})
curve = pd.DataFrame(curve)
curve.to_csv(os.path.join(OUT, "profit_curve.csv"), index=False)
# chosen WITHOUT looking at outcomes: maximise the model's own expected profit
CUT = float(curve.loc[curve.expected_profit.idxmax(), "cutoff"])
CUT_ORACLE = float(curve.loc[curve.realised_profit.idxmax(), "cutoff"])

p_old = oof[("LightGBM (all features)", "raw")]
policies = pd.DataFrame([
    policy(np.ones(len(y), bool), p_dep, "Approve everyone"),
    policy(p_old < 0.5, None, "Old app: uncalibrated LightGBM @ 0.50"),
    policy(p_dep < CUT, p_dep, f"Deployed: calibrated PD < {CUT:.3f} (expected-profit cutoff)"),
    policy(p_dep < pm.break_even_pd(rate), p_dep, "Per-loan break-even: PD < rate/(rate+LGD)"),
    policy(p_dep < CUT_ORACLE, p_dep, f"Oracle: deployed model, best realised cutoff {CUT_ORACLE:.3f} (hindsight)"),
])
policies.to_csv(os.path.join(OUT, "policy_comparison.csv"), index=False)
chosen = policies.iloc[2]

fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURF)
ax.set_facecolor(SURF)
ax.plot(curve.cutoff, curve.realised_profit / 1e6, color=BLUE, lw=2, label="Realised profit (OOF outcomes)")
ax.plot(curve.cutoff, curve.expected_profit / 1e6, color=ORANGE, lw=2, ls="--", label="Expected profit (model PDs)")
ax.axvline(CUT, color=INK2, lw=1, ls=":")
ax.annotate(f"chosen cutoff {CUT:.3f}\napproves {chosen['Approval rate']:.1%}",
            xy=(CUT, chosen["Realised profit"] / 1e6), xytext=(CUT + 0.08, chosen["Realised profit"] / 1e6 * 0.55),
            color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.axhline(0, color=GRID, lw=1)
ax.set_xlabel("Decline when calibrated PD >= cutoff", color=INK2, fontsize=9)
ax.set_ylabel("Portfolio profit (millions, dataset currency units)", color=INK2, fontsize=9)
ax.tick_params(colors=INK2, labelsize=8)
for s in ax.spines.values(): s.set_color(GRID)
ax.grid(color=GRID, lw=0.6, alpha=0.6)
ax.legend(frameon=False, fontsize=9, labelcolor=INK)
ax.set_title(f"Profit vs cut-off (LGD {pm.LGD:.0%}, EAD = loan amount, revenue = rate x amount)",
             color=INK, fontsize=11, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "profit_vs_cutoff.png"), dpi=130, facecolor=SURF)
plt.close(fig)

# =============================================================================
# 3. Fairness by age band (age is NOT a model input)
# =============================================================================
bands = pd.cut(X["person_age"], [0, 25, 35, 45, 55, 200],
               labels=["20-25", "26-35", "36-45", "46-55", "56+"])
approve = p_dep < CUT
fair = (pd.DataFrame({"band": bands, "y": yv, "p": p_dep, "a": approve})
        .dropna(subset=["band"])
        .groupby("band", observed=True)
        .apply(lambda g: pd.Series({
            "Applicants": len(g),
            "Actual default rate": g.y.mean(),
            "Mean predicted PD": g.p.mean(),
            "Approval rate": g.a.mean(),
            "Approval rate among good borrowers": g.a[g.y == 0].mean(),
            "Default rate in approved": g.y[g.a].mean() if g.a.any() else np.nan}),
            include_groups=False)
        .reset_index())
big = fair[fair.Applicants >= 300]
gaps = {c: float(big[c].max() - big[c].min()) for c in
        ["Approval rate", "Actual default rate", "Approval rate among good borrowers",
         "Default rate in approved"]}
fair.to_csv(os.path.join(OUT, "fairness_age.csv"), index=False)

# =============================================================================
# 4. Holdout split: stability (PSI/CSI) and out-of-sample reason codes
# =============================================================================
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
hold = pm.build(Xtr).fit(pm.select(Xtr), ytr)
s_tr = pm.predict_pd(hold, Xtr)
s_te = pm.predict_pd(hold, Xte)


def psi(expected, actual, bins=None, categorical=False):
    eps = 1e-6
    if categorical:
        cats = sorted(set(expected.dropna()) | set(actual.dropna()))
        e = expected.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        a = actual.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, 11)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected[~np.isnan(expected)], edges)[0] / np.sum(~np.isnan(expected))
        a = np.histogram(actual[~np.isnan(actual)], edges)[0] / np.sum(~np.isnan(actual))
    e, a = e + eps, a + eps
    return float(np.sum((a - e) * np.log(a / e)))


def band(v):
    return "stable" if v < 0.10 else ("monitor" if v < 0.25 else "RETRAIN")


# =============================================================================
# 5. Final model: fit on all rows, time, size, global importance
# =============================================================================
t0 = time.perf_counter()
final = pm.build(X).fit(pm.select(X), y)
TRAIN_SECONDS = time.perf_counter() - t0
contrib = pm.contributions(final, X)
imp = contrib.abs().mean().sort_values(ascending=False)
imp_df = pd.DataFrame({"Feature": imp.index, "Mean |SHAP| (log-odds)": imp.values})
imp_df.to_csv(os.path.join(OUT, "shap_importance.csv"), index=False)
top_feats = imp.index[:5].tolist()

stab = [{"Variable": "Model score (PD)", "Type": "score", "PSI": psi(s_tr, s_te)}]
for f in top_feats:
    cat = X[f].dtype.kind not in "biuf"
    v = psi(Xtr[f], Xte[f], categorical=True) if cat else psi(Xtr[f].to_numpy(float), Xte[f].to_numpy(float))
    stab.append({"Variable": f, "Type": "CSI", "PSI": v})
stab = pd.DataFrame(stab); stab["Status"] = stab.PSI.map(band)
stab.to_csv(os.path.join(OUT, "stability.csv"), index=False)

# reason codes on HOLDOUT applicants, scored by the holdout model
meta = {"model_version": "2.0", "cutoff": CUT, "lgd": pm.LGD, "rate_fill": fill,
        "pd_floor": pm.PD_FLOOR,
        "excluded_features": pm.EXCLUDED, "calibration": f"{pm.CALIBRATION}, 3-fold inner CV",
        "train_rows": int(len(X)), "default_rate": BASE_RATE,
        "sklearn": sklearn.__version__, "lightgbm": lgb.__version__}
te_scored = pd.DataFrame(pm.score(hold, meta, Xte))
te_scored.index = Xte.index
dec = te_scored[te_scored.decision == "DECLINE"].sort_values("pd", ascending=False)
pick = pd.concat([dec.iloc[[0, len(dec) // 3, 2 * len(dec) // 3, -1]],
                  te_scored[te_scored.decision == "APPROVE"].iloc[[0]]])
rc_rows = []
for idx, r in pick.iterrows():
    rc = r.reason_codes
    rc_rows.append({"Applicant (holdout row)": int(idx), "Actual": "default" if y.loc[idx] else "repaid",
                    "PD": r.pd, "Decision": r.decision,
                    **{f"Reason {k+1}": (f"{rc[k]['reason']} ({rc[k]['feature']}={rc[k]['value']})"
                                         if k < len(rc) else "") for k in range(3)}})
rc_df = pd.DataFrame(rc_rows)
rc_df.to_csv(os.path.join(OUT, "reason_codes_sample.csv"), index=False)
reason_freq = (pd.Series([c["feature"] for rcs in te_scored[te_scored.decision == "DECLINE"].reason_codes
                          for c in rcs[:1]]).value_counts(normalize=True))

pm.save(final, meta)
MODEL_MB = os.path.getsize(pm.MODEL_PATH) / 1e6

# =============================================================================
# 6. Latency: in-process, FastAPI in-process, and real HTTP via uvicorn
# =============================================================================
sample = Xte.iloc[[0]]
ROUNDS = 400


def timeit(fn, n=ROUNDS, warm=30):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 95))


lat = {}
lat["PD only (predict_pd)"] = timeit(lambda: pm.predict_pd(final, sample))
lat["PD + decision + reason codes"] = timeit(lambda: pm.score(final, meta, sample))

payload = {k: (None if (isinstance(v, float) and np.isnan(v)) else
               (v.item() if hasattr(v, "item") else v)) for k, v in sample.iloc[0].to_dict().items()}
from fastapi.testclient import TestClient
import importlib
import api as api_mod
importlib.reload(api_mod)
client = TestClient(api_mod.app)
assert client.post("/score", json=payload).status_code == 200
lat["FastAPI /score (in-process)"] = timeit(lambda: client.post("/score", json=payload))

import httpx
with socket.socket() as s_:
    s_.bind(("127.0.0.1", 0)); PORT = s_.getsockname()[1]
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app", "--port", str(PORT),
                        "--log-level", "warning"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    url = f"http://127.0.0.1:{PORT}"
    for _ in range(120):
        try:
            if httpx.get(url + "/health", timeout=1).status_code == 200: break
        except Exception:
            time.sleep(0.5)
    with httpx.Client(base_url=url, timeout=5) as hc:
        lat["FastAPI /score over HTTP (uvicorn, localhost)"] = timeit(lambda: hc.post("/score", json=payload), n=300)
finally:
    srv.terminate(); srv.wait(10)
lat_df = pd.DataFrame([{"Path": k, "p50 ms": v[0], "p95 ms": v[1]} for k, v in lat.items()])
lat_df.to_csv(os.path.join(OUT, "latency.csv"), index=False)

# =============================================================================
# 7. metrics.json + README results block
# =============================================================================
cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")),
           platform.processor()) if os.path.exists("/proc/cpuinfo") else platform.processor()
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
dep_cal = metrics_df[(metrics_df.Model == DEPLOYED) & (metrics_df.Variant == "calibrated")].iloc[0]
system = {"cpu": cpu, "cores": os.cpu_count(), "python": platform.python_version(),
          "sklearn": sklearn.__version__, "lightgbm": lgb.__version__,
          "train_seconds_final_model": TRAIN_SECONDS, "model_artifact_mb": MODEL_MB,
          "peak_rss_mb": peak_mb, "raw_csv_mb": os.path.getsize(cd.RAW) / 1e6,
          "eval_runtime_seconds": None}

M = {"data": DATA, "cv": "stratified 5-fold, seed 42", "metrics": rows,
     "cutoff": CUT, "cutoff_oracle": CUT_ORACLE, "policies": policies.to_dict("records"),
     "fairness": fair.to_dict("records"), "fairness_gaps_pp": {k: v * 100 for k, v in gaps.items()},
     "stability": stab.to_dict("records"), "shap_importance": imp_df.to_dict("records"),
     "top_decline_reason_share": reason_freq.to_dict(), "latency_ms": lat_df.to_dict("records"),
     "system": system}


def f4(v): return f"{v:.4f}"
def pct(v): return f"{v*100:.1f}%"
def money(v): return f"{v/1e6:,.2f}M"


def fmt_table(df, cols, fmts):
    head = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    return head + "".join("| " + " | ".join(fmts.get(c, str)(r[c]) for c in cols) + " |\n"
                          for _, r in df.iterrows())


def model_rows():
    out = "| Model | ROC-AUC | Gini | PR-AUC | KS | Brier (raw -> calibrated) | Mean PD (raw -> cal.) |\n|---|---|---|---|---|---|---|\n"
    for name in ["Majority class (baseline)"] + list(MODELS):
        if name.startswith("Majority"):
            r = metrics_df.iloc[0]
            out += f"| {name} | 0.5000 | 0.0000 | {f4(r['PR-AUC'])} | 0.0000 | {f4(r.Brier)} | {r['Mean PD']:.3f} |\n"
            continue
        a = metrics_df[(metrics_df.Model == name) & (metrics_df.Variant == "raw")].iloc[0]
        b = metrics_df[(metrics_df.Model == name) & (metrics_df.Variant == "calibrated")].iloc[0]
        bold = "**" if name == DEPLOYED else ""
        out += (f"| {bold}{name}{bold} | {f4(b['ROC-AUC'])} ± {b['ROC-AUC std']:.4f} | {f4(b.Gini)} | "
                f"{f4(b['PR-AUC'])} | {f4(b.KS)} | {f4(a.Brier)} -> {f4(b.Brier)} | "
                f"{a['Mean PD']:.3f} -> {b['Mean PD']:.3f} |\n")
    return out


pol_cols = ["Policy", "Approval rate", "Defaults declined (recall)", "Precision (declined who default)",
            "Good borrowers declined", "Bad rate in approved book", "Realised loss", "Realised profit"]
pol_fmt = {c: pct for c in pol_cols[1:6]} | {"Realised loss": money, "Realised profit": money}
cm = chosen
block = f"""<!-- RESULTS:START - generated by evaluate.py; do not edit by hand -->
## Results

All figures: **stratified 5-fold cross-validation, out-of-fold predictions**, {DATA['rows_final']:,} loans after deduplication, **default rate {BASE_RATE:.2%}**. Every transform, the model and the Platt (sigmoid) calibrator are refit inside each fold. PDs carry the Basel 0.03% floor. Regenerate with `make eval`.

### Headline

| Metric | Value | Split |
|---|---|---|
| ROC-AUC (deployed, calibrated LightGBM) | {f4(dep_cal['ROC-AUC'])} ± {dep_cal['ROC-AUC std']:.4f} | 5-fold CV |
| Gini | {f4(dep_cal.Gini)} | 5-fold CV |
| KS | {f4(dep_cal.KS)} | 5-fold OOF |
| PR-AUC (baseline = default rate {BASE_RATE:.4f}) | {f4(dep_cal['PR-AUC'])} | 5-fold CV |
| Brier score (baseline {f4(metrics_df.iloc[0].Brier)}) | {f4(dep_cal.Brier)} | 5-fold CV |
| Cut-off (decline if PD >=) | {CUT:.3f} | chosen on expected profit, no outcomes used |
| Approval rate at cut-off | {pct(cm['Approval rate'])} | 5-fold OOF |
| Defaults declined (recall) / precision | {pct(cm['Defaults declined (recall)'])} / {pct(cm['Precision (declined who default)'])} | 5-fold OOF |
| Realised profit vs approve-everyone | {money(cm['Realised profit'])} vs {money(policies.iloc[0]['Realised profit'])} | 5-fold OOF, LGD {pm.LGD:.0%} |
| Expected loss of approved book | {money(cm['Expected loss (model)'])} | model PD x LGD x EAD |
| Scoring latency p50 / p95 (HTTP) | {lat_df.iloc[3]['p50 ms']:.1f} / {lat_df.iloc[3]['p95 ms']:.1f} ms | 1 applicant, localhost |

Money is in the dataset's own currency units: the Kaggle file does not label its currency, so no rupee or dollar conversion is applied.

### Models vs baselines

{model_rows()}
ROC-AUC, Gini, PR-AUC and KS are shown for the calibrated variant; Platt (sigmoid) calibration is monotone, so ranking metrics barely move. Raw `class_weight="balanced"` models rank well but overstate PD, which the Brier and Mean-PD columns expose.

![Reliability curves](reports/figures/reliability.png)

**Why Platt, not isotonic** (deployed model, 5-fold CV):

{fmt_table(cal_method, list(cal_method.columns), {'ROC-AUC': f4, 'Brier': f4, 'Brier std': f4, 'ECE': f4, 'PD exactly 0 (<=1e-4)': lambda v: f'{int(v):,}', 'PD exactly 1 (>=0.9999)': lambda v: f'{int(v):,}', 'Distinct PD levels': lambda v: f'{int(v):,}'})}
Brier is a tie within one standard deviation. Isotonic's step function assigns PD = 0 or 1 to thousands of applicants, which cannot be priced or explained; sigmoid gives a continuous PD.

### Decision policy

{fmt_table(policies, pol_cols, pol_fmt)}
Confusion matrix at the deployed cut-off (positive = default, predicted positive = declined):

| | Approved | Declined |
|---|---|---|
| **Repaid** | {cm.TN:,} | {cm.FP:,} |
| **Defaulted** | {cm.FN:,} | {cm.TP:,} |

![Profit vs cut-off](reports/figures/profit_vs_cutoff.png)

### Governance

**Fairness by age band** (age is not a model input; bands with < 300 applicants excluded from gaps):

{fmt_table(fair, list(fair.columns), {c: (pct if c not in ('band', 'Applicants') else (lambda v: f'{int(v):,}' if not isinstance(v, str) else v)) for c in fair.columns})}
Max gap across bands: approval rate **{gaps['Approval rate']*100:.1f} pp**{' (FLAG > 10 pp)' if gaps['Approval rate'] > 0.10 else ''}, actual default rate {gaps['Actual default rate']*100:.1f} pp{' (FLAG > 10 pp)' if gaps['Actual default rate'] > 0.10 else ''}, approval among good borrowers {gaps['Approval rate among good borrowers']*100:.1f} pp{' (FLAG > 10 pp)' if gaps['Approval rate among good borrowers'] > 0.10 else ''}.

**Stability** (holdout 80/20 split; PSI < 0.10 stable, 0.10-0.25 monitor, > 0.25 retrain):

{fmt_table(stab, ['Variable', 'Type', 'PSI', 'Status'], {'PSI': f4})}
These are near zero by construction: the split is random, not time-ordered. They prove the monitoring code works, not that the model is stable over time.

**Top reason codes** (most common first reason among declined holdout applicants): {', '.join(f'{k} {v:.0%}' for k, v in reason_freq.head(3).items())}. Sample in `reports/reason_codes_sample.csv`.

### System

| Factor | Value |
|---|---|
| Training time, final calibrated model | {TRAIN_SECONDS:.1f} s ({DATA['rows_final']:,} rows, {pm.select(X).shape[1]} features) |
| Model artifact | {MODEL_MB:.2f} MB (joblib, compressed) |
{''.join(f"| Latency p50 / p95: {r['Path']} | {r['p50 ms']:.2f} / {r['p95 ms']:.2f} ms |{chr(10)}" for _, r in lat_df.iterrows())}| Peak memory, full evaluation | {peak_mb:,.0f} MB |
| Hardware | {cpu}, {os.cpu_count()} cores |
| Data | {system['raw_csv_mb']:.1f} MB CSV, fits in memory; no GPU |
<!-- RESULTS:END -->"""

EVAL_SECONDS = time.perf_counter() - T_START
system["eval_runtime_seconds"] = EVAL_SECONDS
block = block.replace("<!-- RESULTS:END -->", f"| `evaluate.py` runtime (the full `run_eval.py` also runs the audit and scorecard) | {EVAL_SECONDS/60:.1f} min |\n<!-- RESULTS:END -->")
with open(os.path.join(OUT, "RESULTS.md"), "w") as f:
    f.write(block + "\n")
with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(M, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

def inject(path, start, end, body):
    if not os.path.exists(path):
        return
    md = open(path, encoding="utf-8").read()
    i, j = md.find(start), md.find(end)
    if i == -1 or j == -1:
        print(f"WARNING: markers missing in {path}; not updated")
        return
    md = md[:i] + body + md[j + len(end):]
    open(path, "w", encoding="utf-8").write(md)
    print(f"{path} updated")


inject("README.md", "<!-- RESULTS:START", "<!-- RESULTS:END -->", block)

card = f"""<!-- MODELCARD:START - generated by evaluate.py -->
Stratified 5-fold cross-validation with out-of-fold predictions on {DATA['rows_final']:,} loans (default rate {BASE_RATE:.2%}).

| Metric | Value |
|---|---|
| ROC-AUC / Gini | {f4(dep_cal['ROC-AUC'])} ± {dep_cal['ROC-AUC std']:.4f} / {f4(dep_cal.Gini)} |
| KS | {f4(dep_cal.KS)} |
| PR-AUC (no-skill = {BASE_RATE:.4f}) | {f4(dep_cal['PR-AUC'])} |
| Brier (no-skill = {f4(metrics_df.iloc[0].Brier)}) | {f4(dep_cal.Brier)} |
| Mean PD vs observed default rate | {dep_cal['Mean PD']:.4f} vs {BASE_RATE:.4f} |
| Cut-off / approval rate | {CUT:.3f} / {pct(cm['Approval rate'])} |
| Defaults declined / good borrowers declined | {pct(cm['Defaults declined (recall)'])} / {pct(cm['Good borrowers declined'])} |
| Max approval-rate gap across age bands (n ≥ 300) | {gaps['Approval rate']*100:.1f} pp |
| Scoring latency p50 / p95 over HTTP | {lat_df.iloc[3]['p50 ms']:.1f} / {lat_df.iloc[3]['p95 ms']:.1f} ms |

The full tables and figures are in the README results section and in `reports/`.
<!-- MODELCARD:END -->"""
inject("MODEL_CARD.md", "<!-- MODELCARD:START", "<!-- MODELCARD:END -->", card)

print(f"\nDeployed: AUC {dep_cal['ROC-AUC']:.4f} | KS {dep_cal.KS:.4f} | Brier {dep_cal.Brier:.4f} | cutoff {CUT:.3f} "
      f"| approval {cm['Approval rate']:.1%} | profit {money(cm['Realised profit'])} | eval {EVAL_SECONDS/60:.1f} min")
