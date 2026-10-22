import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
import lightgbm as lgb

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
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # drop notebook-derived helper columns if they leaked into the saved CSV
    df = df.drop(columns=["age_group"], errors="ignore")

    if "person_age" in df.columns:
        df["person_age"] = pd.to_numeric(df["person_age"], errors="coerce").clip(1, 100)
        if df["person_age"].isna().any():
            med = df["person_age"].median(skipna=True)
            df["person_age"] = df["person_age"].fillna(30 if np.isnan(med) else int(med))
        df["person_age"] = df["person_age"].astype(int)

    for col in ["person_income", "loan_amnt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            med = df[col].median(skipna=True)
            df[col] = df[col].fillna(0 if np.isnan(med) else int(med)).round().astype(int)

    return df


@st.cache_resource(show_spinner="Training LightGBM (first run only — cached afterwards)…")
def train_model(path: str):
    """Train once per dataset, cache the fitted pipeline + evaluation artifacts.

    LightGBM is the deployed model: it beats Random Forest by 0.013 cross-validated
    ROC-AUC (0.948 vs 0.935, 5-fold) and dominates it at every decision threshold.
    See compare_models.py and threshold_analysis.csv for the benchmark.

    Threshold-dependent metrics are deliberately NOT computed here — the raw
    test-set probabilities are cached instead, so moving the threshold slider
    re-scores instantly rather than retraining.
    """
    df = load_data(path)
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")),
                          ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        # Matches compare_models.py exactly, so the published benchmark numbers
        # describe the model actually running here.
        ("model", lgb.LGBMClassifier(
            n_estimators=100, max_depth=7, learning_rate=0.1,
            class_weight="balanced", random_state=42, verbose=-1)),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline.fit(X_tr, y_tr)

    proba = pipeline.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)

    evaluation = {
        "auc": roc_auc_score(y_te, proba),
        "y_true": y_te.to_numpy(),
        "proba": proba,
        "fpr": fpr, "tpr": tpr,
    }

    feat_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
    importances = (
        pd.DataFrame({
            "feature": [f.split("__", 1)[-1] for f in feat_names],
            "importance": pipeline.named_steps["model"].feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .head(10)
    )

    return pipeline, evaluation, importances, num_cols, cat_cols


def score_at_threshold(evaluation: dict, threshold: float) -> dict:
    """Re-score the cached test-set probabilities at an arbitrary cutoff.

    Precision and recall are properties of a (model, threshold) pair, not of a
    model alone, so the threshold has to be an explicit input everywhere they
    are reported.
    """
    y_true, proba = evaluation["y_true"], evaluation["proba"]
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "confusion": np.array([[tn, fp], [fn, tp]]),
        "caught": int(tp), "missed": int(fn), "false_alarms": int(fp),
        "defaults": int(tp + fn),
    }


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 💳 Credit Risk Analytics")

    clean_path, orig_path = "clean_data.csv", "credit_risk_dataset.csv"
    options = [p for p in (clean_path, orig_path) if os.path.exists(p)] or [orig_path]
    data_path = st.selectbox("Dataset", options, format_func=lambda p: p.replace(".csv", ""))

    page = st.radio("Navigate", ["🏠 Home", "📋 Data Overview", "📊 EDA", "🎯 Predict Risk"],
                    label_visibility="collapsed")

    st.divider()
    st.markdown("**Decision threshold**")
    threshold = st.slider(
        "Flag as default when probability ≥", 0.05, 0.95, 0.50, 0.01,
        help="A lender's real choice. Lower it to catch more defaults at the cost "
             "of rejecting more good borrowers; raise it to do the reverse. "
             "Around 0.58 lands near the old Random Forest's precision with ~3 points "
             "more recall; around 0.67 matches its recall at far higher precision. "
             "Either way LightGBM is ahead, which is the point: Random Forest was on "
             "a lower curve, not a different trade-off.",
    )
    st.caption(f"Model: **LightGBM** · cutoff `{threshold:.2f}`")
    st.caption(f"Source: `{data_path}`")

df = load_data(data_path)


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
    st.title("🎯 Predict Credit Risk")

    pipeline, evaluation, importances, num_cols, cat_cols = train_model(data_path)
    scored = score_at_threshold(evaluation, threshold)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC-ROC", f"{evaluation['auc']:.3f}",
              help="Threshold-independent: ranks all applicants, so it does not "
                   "move when you change the cutoff.")
    m2.metric("Accuracy", f"{scored['accuracy']:.1%}")
    m3.metric("Precision (default)", f"{scored['precision']:.1%}")
    m4.metric("Recall (default)", f"{scored['recall']:.1%}")
    st.caption(
        f"LightGBM on a 20% held-out test set (stratified), cutoff {threshold:.2f} — "
        f"catches **{scored['caught']:,} of {scored['defaults']:,}** defaults "
        f"({scored['missed']:,} missed) with **{scored['false_alarms']:,}** good "
        f"borrowers wrongly flagged. Only AUC-ROC is independent of the cutoff."
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Applicant details")
        with st.form("applicant"):
            user_input = {}
            cols = st.columns(2)
            for i, col in enumerate(num_cols):
                target = cols[i % 2]
                series = df[col].dropna()
                if col == "person_age":
                    user_input[col] = target.number_input(
                        "Age", 18, 100, int(series.median()), step=1)
                elif pd.api.types.is_integer_dtype(df[col]):
                    user_input[col] = target.number_input(
                        col, min_value=0, value=int(series.median()), step=1)
                else:
                    user_input[col] = target.number_input(
                        col, min_value=0.0, value=float(round(series.median(), 2)))
            for i, col in enumerate(cat_cols):
                target = cols[(len(num_cols) + i) % 2]
                user_input[col] = target.selectbox(col, sorted(df[col].dropna().unique()))

            submitted = st.form_submit_button("Assess risk", type="primary",
                                              use_container_width=True)

    with right:
        st.subheader("Risk assessment")
        if submitted:
            input_df = pd.DataFrame([user_input])
            proba = float(pipeline.predict_proba(input_df)[0, 1])

            cut = threshold * 100
            caution = max(cut * 0.6, 1.0)  # watch-list band below the decline cutoff

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%"},
                title={"text": "Probability of default"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2c3e50"},
                    "steps": [
                        {"range": [0, caution], "color": "#d5f5e3"},
                        {"range": [caution, cut], "color": "#fdebd0"},
                        {"range": [cut, 100], "color": "#fadbd8"},
                    ],
                    "threshold": {
                        "line": {"color": BAD, "width": 4},
                        "thickness": 0.9,
                        "value": cut,
                    },
                },
            ))
            fig.update_layout(height=280, margin=dict(t=60, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

            if proba >= threshold:
                st.error(f"**Decline** — {proba:.1%} default probability, at or above "
                         f"the {threshold:.0%} cutoff.")
            elif proba >= caution / 100:
                st.warning(f"**Refer for review** — {proba:.1%} default probability, "
                           f"below the {threshold:.0%} cutoff but not clean.")
            else:
                st.success(f"**Approve** — {proba:.1%} default probability, well below "
                           f"the {threshold:.0%} cutoff.")
            st.caption("Bands follow the sidebar threshold, so the recommendation "
                       "changes with the lending policy rather than being hardcoded.")
        else:
            st.info("Fill in the applicant details and click **Assess risk**.")

    st.divider()
    left, right = st.columns(2)

    with left:
        fig = px.area(x=evaluation["fpr"], y=evaluation["tpr"],
                      labels={"x": "False positive rate", "y": "True positive rate"},
                      title=f"ROC curve (AUC = {evaluation['auc']:.3f})",
                      color_discrete_sequence=[ACCENT], template=PLOTLY_TEMPLATE)
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.bar(importances.sort_values("importance"), x="importance", y="feature",
                     orientation="h", color_discrete_sequence=[ACCENT],
                     title="Top 10 feature importances (LightGBM split count)",
                     template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    cm = scored["confusion"]
    st.markdown(f"**Confusion matrix at cutoff {threshold:.2f}** (test set, "
                f"{len(evaluation['y_true']):,} applicants)")
    st.dataframe(
        pd.DataFrame(cm,
                     index=["Actual: repaid", "Actual: default"],
                     columns=["Predicted: repaid", "Predicted: default"]),
        use_container_width=True,
    )
