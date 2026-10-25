"""Credit decision demo: data exploration plus applicant scoring.

Scoring uses pd_model.py - the same calibrated model, cut-off and reason codes
as the FastAPI endpoint (api.py) - so the demo and the API cannot disagree.
Performance figures are OUT-OF-FOLD predictions written by evaluate.py, never
in-sample numbers from the deployed model.
"""
import json
import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve

import credit_data as cd  # shared leak-free loader - see leakage_audit.py
import pd_model as pm     # deployed PD model, decision rule, reason codes

HERE = os.path.dirname(os.path.abspath(__file__))
OOF_PATH = os.path.join(HERE, "models", "oof_deployed.npz")
METRICS_PATH = os.path.join(HERE, "reports", "metrics.json")
SHAP_PATH = os.path.join(HERE, "reports", "shap_importance.csv")

# ---------------------------------------------------------------- page setup
st.set_page_config(
    page_title="Credit Risk Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#4F8BF9"
GOOD = "#2ECC71"
BAD = "#E74C3C"
PALETTE = [ACCENT, BAD, GOOD, "#F39C12", "#9B59B6", "#1ABC9C", "#E67E22"]
PLOTLY_TEMPLATE = "plotly_white"

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(79,139,249,.08), rgba(79,139,249,.02));
        border: 1px solid rgba(79,139,249,.25);
        border-radius: 12px;
        padding: 14px 18px;
    }
    div[data-testid="stMetric"] label { color: #6b7a90; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)



# ---------------------------------------------------------------- data layer
@st.cache_data
def load_data():
    """Raw data: impossible values nulled (not dropped), exact duplicates
    removed. NO imputation here - that happens inside the model pipeline, fit
    on the training fold only. Missing cells shown on Data Overview are real."""
    X, y, report = cd.load_raw()
    df = X.copy()
    df["loan_status"] = y.values
    return df, report


@st.cache_resource(show_spinner="Loading the calibrated PD model…")
def get_model():
    return pm.load_or_train()


@st.cache_data
def load_oof():
    """Out-of-fold PDs and per-loan revenue/loss from evaluate.py, so the
    threshold slider re-scores honest held-out predictions instantly."""
    d = np.load(OOF_PATH)
    return {k: d[k] for k in d.files}


@st.cache_data
def load_reports():
    with open(METRICS_PATH) as f:
        m = json.load(f)
    return m, pd.read_csv(SHAP_PATH)


def score_at_threshold(oof: dict, threshold: float) -> dict:
    """Decline when PD >= threshold. Positive class = default."""
    y, p = oof["y"], oof["pd"]
    approve = p < threshold
    good, bad = y == 0, y == 1
    tp = int((~approve & bad).sum()); fp = int((~approve & good).sum())
    fn = int((approve & bad).sum()); tn = int((approve & good).sum())
    return {
        "approval": approve.mean(),
        "recall": tp / max(bad.sum(), 1),
        "precision": tp / max(tp + fp, 1),
        "good_declined": fp / max(good.sum(), 1),
        "bad_rate_approved": y[approve].mean() if approve.any() else 0.0,
        "profit": float(oof["rev"][approve & good].sum() - oof["loss"][approve & bad].sum()),
        "confusion": np.array([[tn, fp], [fn, tp]]),
        "caught": tp, "missed": fn, "false_alarms": fp, "defaults": int(bad.sum()),
    }


# ------------------------------------------------------------------- sidebar
MODEL, META, MODEL_SOURCE = get_model()

with st.sidebar:
    st.markdown("## 💳 Credit Risk Analytics")

    page = st.radio("Navigate", ["🏠 Home", "📋 Data Overview", "📊 EDA", "🎯 Predict Risk"],
                    label_visibility="collapsed")

    st.divider()
    st.markdown("**Decision cut-off**")
    threshold = st.slider(
        "Decline when calibrated PD ≥", 0.02, 0.90, float(round(META["cutoff"], 3)), 0.005,
        format="%.3f",
        help=f"Default {META['cutoff']:.3f} maximises the model's own expected profit "
             f"(LGD {pm.LGD:.0%}, EAD = loan amount, revenue = rate × amount), chosen "
             "without looking at outcomes. Move it to see the trade-off on "
             "out-of-fold predictions.",
    )
    st.caption(f"Model: **calibrated LightGBM** ({META.get('calibration', '')}) · "
               f"policy cut-off `{META['cutoff']:.3f}`")
    st.caption("Source: `credit_risk_dataset.csv` (deduplicated, in-fold imputation)")
    if MODEL_SOURCE == "retrained":
        st.caption("⚠️ Saved model could not be loaded in this environment; retrained on startup.")

df, data_report = load_data()


# ---------------------------------------------------------------------- home
if page == "🏠 Home":
    st.title("💳 Credit Risk Prediction")
    st.markdown("Interactive analytics and default-risk scoring for loan applicants. "
                "Explore the data in **EDA**, then score an applicant in **Predict Risk**.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicants", f"{len(df):,}")
    c2.metric("Default rate", f"{df['loan_status'].mean():.1%}")
    c3.metric("Avg loan amount", f"${df['loan_amnt'].mean():,.0f}")
    c4.metric("Avg interest rate", f"{df['loan_int_rate'].mean():.2f}%")

    st.divider()
    left, right = st.columns(2)

    with left:
        by_intent = (df.groupby("loan_intent")["loan_status"].mean()
                       .sort_values(ascending=False).reset_index())
        fig = px.bar(by_intent, x="loan_intent", y="loan_status",
                     color="loan_status", color_continuous_scale="Reds",
                     labels={"loan_status": "Default rate", "loan_intent": ""},
                     title="Default rate by loan intent", template=PLOTLY_TEMPLATE)
        fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        counts = df["loan_status"].map({0: "Repaid", 1: "Default"}).value_counts().reset_index()
        counts.columns = ["status", "count"]
        fig = px.pie(counts, names="status", values="count", hole=0.5,
                     color="status", color_discrete_map={"Repaid": GOOD, "Default": BAD},
                     title="Portfolio outcome split", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------- data overview
elif page == "📋 Data Overview":
    st.title("📋 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", f"{int(df.isnull().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    tab_preview, tab_stats, tab_missing = st.tabs(["Preview", "Statistics", "Missing values"])

    with tab_preview:
        n = st.slider("Rows to show", 5, 200, 25)
        st.dataframe(df.head(n), use_container_width=True)

    with tab_stats:
        st.dataframe(df.describe().T.round(2), use_container_width=True)

    with tab_missing:
        miss = df.isnull().sum()
        if miss.sum() == 0:
            st.success("No missing values 🎉")
        else:
            miss = miss[miss > 0].sort_values(ascending=False).reset_index()
            miss.columns = ["column", "missing"]
            fig = px.bar(miss, x="missing", y="column", orientation="h",
                         color_discrete_sequence=[ACCENT], template=PLOTLY_TEMPLATE,
                         title="Missing values per column")
            st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------- eda
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    # --- global filters ---
    with st.container(border=True):
        f1, f2, f3 = st.columns([2, 2, 3])
        intents = f1.multiselect("Loan intent", sorted(df["loan_intent"].unique()))
        homes = f2.multiselect("Home ownership", sorted(df["person_home_ownership"].unique()))
        lo, hi = int(df["person_age"].min()), int(df["person_age"].max())
        age_rng = f3.slider("Age range", lo, hi, (lo, hi))

    fdf = df[df["person_age"].between(*age_rng)]
    if intents:
        fdf = fdf[fdf["loan_intent"].isin(intents)]
    if homes:
        fdf = fdf[fdf["person_home_ownership"].isin(homes)]
    st.caption(f"Filtered: **{len(fdf):,}** of {len(df):,} applicants")

    if fdf.empty:
        st.warning("No rows match the current filters.")
        st.stop()

    fdf = fdf.assign(status=fdf["loan_status"].map({0: "Repaid", 1: "Default"}))
    status_colors = {"Repaid": GOOD, "Default": BAD}

    num_cols = fdf.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = fdf.select_dtypes(include=["object"]).drop(columns=["status"], errors="ignore").columns.tolist()

    tab_uni, tab_bi, tab_corr = st.tabs(["Univariate", "Bivariate", "Correlation"])

    with tab_uni:
        left, right = st.columns(2)
        with left:
            counts = fdf["status"].value_counts().reset_index()
            counts.columns = ["status", "count"]
            fig = px.bar(counts, x="status", y="count", color="status",
                         color_discrete_map=status_colors, text_auto=True,
                         title="Class balance", template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            sel_cat = st.selectbox("Categorical column", cat_cols)
            order = fdf[sel_cat].value_counts().index.tolist()
            fig = px.histogram(fdf, x=sel_cat, color="status",
                               color_discrete_map=status_colors, barmode="group",
                               category_orders={sel_cat: order},
                               title=f"{sel_cat} by outcome", template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)

        sel_num = st.selectbox("Numeric column", [c for c in num_cols if c != "loan_status"])
        fig = px.histogram(fdf, x=sel_num, color="status", marginal="box", nbins=50,
                           color_discrete_map=status_colors, opacity=0.75,
                           title=f"Distribution of {sel_num}", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    with tab_bi:
        left, right = st.columns(2)
        with left:
            dti = (fdf.groupby(["loan_intent", "status"])["loan_percent_income"]
                     .mean().reset_index())
            fig = px.bar(dti, x="loan_intent", y="loan_percent_income", color="status",
                         barmode="group", color_discrete_map=status_colors,
                         title="Avg debt-to-income by intent & outcome",
                         labels={"loan_percent_income": "avg DTI"},
                         template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.box(fdf, x="loan_intent", y="loan_int_rate", color="status",
                         color_discrete_map=status_colors,
                         title="Interest rate by intent & outcome", template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)

        sample = fdf.sample(min(len(fdf), 3000), random_state=42)
        fig = px.scatter(sample, x="person_income", y="loan_amnt", color="status",
                         color_discrete_map=status_colors, opacity=0.5,
                         hover_data=["loan_intent", "loan_int_rate"],
                         title="Income vs loan amount (sampled)", template=PLOTLY_TEMPLATE)
        fig.update_xaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

    with tab_corr:
        enc = fdf.drop(columns=["status"]).copy()
        for col in enc.select_dtypes(include=["object"]).columns:
            enc[col] = enc[col].astype("category").cat.codes
        corr = enc.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlation heatmap (categoricals label-encoded)",
                        template=PLOTLY_TEMPLATE)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        if "loan_grade" in fdf.columns:
            grade = (fdf.groupby("loan_grade")["loan_status"].mean()
                        .reset_index().sort_values("loan_grade"))
            fig = px.bar(grade, x="loan_grade", y="loan_status",
                         color="loan_status", color_continuous_scale="Reds",
                         labels={"loan_status": "Default rate"},
                         title="Default rate by loan grade", template=PLOTLY_TEMPLATE)
            fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)



# ------------------------------------------------------------------- predict
elif page == "🎯 Predict Risk":
    st.title("🎯 Credit Decision")

    oof = load_oof()
    metrics, shap_imp = load_reports()
    s = score_at_threshold(oof, threshold)
    auc = roc_auc_score(oof["y"], oof["pd"])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC-AUC", f"{auc:.3f}", help="Out-of-fold; independent of the cut-off.")
    m2.metric("Approval rate", f"{s['approval']:.1%}")
    m3.metric("Defaults declined", f"{s['recall']:.1%}")
    m4.metric("Good borrowers declined", f"{s['good_declined']:.1%}")
    m5.metric("Portfolio profit", f"{s['profit']/1e6:,.2f}M",
              help="Realised on out-of-fold predictions, dataset currency units.")
    st.caption(
        f"Out-of-fold predictions on {len(oof['y']):,} loans at cut-off {threshold:.3f}: "
        f"declines **{s['caught']:,} of {s['defaults']:,}** defaulters and "
        f"**{s['false_alarms']:,}** good borrowers. Approve-everyone profit for comparison: "
        f"{(oof['rev'][oof['y']==0].sum() - oof['loss'][oof['y']==1].sum())/1e6:,.2f}M.")

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Applicant details")
        with st.form("applicant"):
            c1, c2 = st.columns(2)
            income = c1.number_input("Annual income", min_value=1000, value=55000, step=1000)
            amount = c2.number_input("Loan amount", min_value=500, value=10000, step=500)
            rate = c1.number_input("Interest rate (%)", min_value=1.0, max_value=40.0, value=11.0, step=0.25)
            emp = c2.number_input("Employment length (years)", min_value=0.0, max_value=60.0, value=4.0, step=1.0)
            hist = c1.number_input("Credit history length (years)", min_value=0, max_value=60, value=4, step=1)
            home = c2.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
            intent = c1.selectbox("Loan purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
                                                   "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
            grade = c2.selectbox("Loan grade", list("ABCDEFG"), index=1)
            prior = c1.selectbox("Prior default on file", ["N", "Y"])
            st.caption("Age is not collected: it is excluded from the model (see README, Governance).")
            submitted = st.form_submit_button("Assess applicant", type="primary",
                                              use_container_width=True)

    with right:
        st.subheader("Decision")
        if submitted:
            row = pd.DataFrame([{
                "person_age": np.nan, "person_income": income, "person_home_ownership": home,
                "person_emp_length": float(emp), "loan_intent": intent, "loan_grade": grade,
                "loan_amnt": amount, "loan_int_rate": float(rate),
                "loan_percent_income": round(amount / income, 2),
                "cb_person_default_on_file": prior, "cb_person_cred_hist_length": hist}])
            r = pm.score(MODEL, {**META, "cutoff": threshold}, row)[0]

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=r["pd"] * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                title={"text": "Probability of default (calibrated)"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2c3e50"},
                       "steps": [{"range": [0, threshold * 100], "color": "#d5f5e3"},
                                 {"range": [threshold * 100, 100], "color": "#fadbd8"}],
                       "threshold": {"line": {"color": BAD, "width": 4}, "thickness": 0.9,
                                     "value": threshold * 100}}))
            fig.update_layout(height=250, margin=dict(t=50, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

            if r["decision"] == "DECLINE":
                st.error(f"**DECLINE** — PD {r['pd']:.1%} is at or above the {threshold:.1%} cut-off.")
            else:
                st.success(f"**APPROVE** — PD {r['pd']:.1%} is below the {threshold:.1%} cut-off.")
            st.caption(f"Expected loss {r['expected_loss']:,.0f} · expected profit "
                       f"{r['expected_profit']:,.0f} · break-even PD at this rate "
                       f"{r['break_even_pd']:.1%}")

            reasons = r["reason_codes"]
            label = "Principal reasons for decline" if r["decision"] == "DECLINE" else "Main risk factors"
            st.markdown(f"**{label}**")
            if reasons:
                for i, c in enumerate(reasons, 1):
                    st.markdown(f"{i}. {c['reason']} — `{c['feature']} = {c['value']}`")
            else:
                st.markdown("_No factor increased risk above the portfolio average._")
        else:
            st.info("Enter the applicant's details and click **Assess applicant**.")

    st.divider()
    left, right = st.columns(2)
    with left:
        fpr, tpr, _ = roc_curve(oof["y"], oof["pd"])
        fig = px.area(x=fpr, y=tpr, labels={"x": "False positive rate", "y": "True positive rate"},
                      title=f"ROC curve, out-of-fold (AUC = {auc:.3f})",
                      color_discrete_sequence=[ACCENT], template=PLOTLY_TEMPLATE)
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        imp = shap_imp.sort_values(shap_imp.columns[1])
        fig = px.bar(imp, x=imp.columns[1], y="Feature", orientation="h",
                     color_discrete_sequence=[ACCENT], template=PLOTLY_TEMPLATE,
                     title="What drives PD: mean |SHAP| contribution")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Confusion matrix at cut-off {threshold:.3f}** (out-of-fold, "
                f"{len(oof['y']):,} loans; declined = predicted default)")
    st.dataframe(pd.DataFrame(s["confusion"], index=["Actual: repaid", "Actual: default"],
                              columns=["Approved", "Declined"]), use_container_width=True)
