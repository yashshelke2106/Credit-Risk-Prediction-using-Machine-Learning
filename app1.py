import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)

    if 'person_age' in df.columns:
        df['person_age'] = pd.to_numeric(df['person_age'], errors='coerce')
        df['person_age'] = df['person_age'].clip(lower=1, upper=100)
        
        if df['person_age'].isna().any():
            median_age = int(df['person_age'].median(skipna=True)) if not np.isnan(df['person_age'].median(skipna=True)) else 30
            df['person_age'].fillna(median_age, inplace=True)
        df['person_age'] = df['person_age'].astype(int)

    for col in ['person_income', 'loan_amnt']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median(skipna=True)
            if np.isnan(median_val):
                median_val = 0
            df[col].fillna(int(median_val), inplace=True)
            df[col] = df[col].round().astype(int)

    return df

# Sidebar dataset selector (prefer cleaned file if available)
clean_path = "clean_data.csv"
orig_path = "credit_risk_dataset.csv"
default_idx = 0 if os.path.exists(clean_path) else 1
dataset_choice = st.sidebar.selectbox(
    "Dataset",
    ("Cleaned (clean_data.csv)", "Original (credit_risk_dataset.csv)"),
    index=default_idx
)
data_path = clean_path if dataset_choice.startswith("Cleaned") and os.path.exists(clean_path) else orig_path
st.sidebar.markdown(f"**Loaded:** `{data_path}`")

df = load_data(data_path) 

st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Go to",
    ["Home", "Data Overview", "EDA", "Predict Risk"]
)

if option == "Home":
    st.title("Credit Risk Prediction App")
    st.write("This app analyzes credit risk and predicts loan default.")

elif option == "Data Overview":
    st.title("Dataset Overview")
    st.write("Shape of Dataset:", df.shape)

    if 'person_age' in df.columns:
        st.write(f"person_age range: {int(df['person_age'].min())} - {int(df['person_age'].max())}")
    if 'person_income' in df.columns:
        st.write(f"person_income range: {int(df['person_income'].min())} - {int(df['person_income'].max())}")
    if 'loan_amnt' in df.columns:
        st.write(f"loan_amnt range: {int(df['loan_amnt'].min())} - {int(df['loan_amnt'].max())}")

    st.write(df.head())
    st.write("Missing Values:")
    st.write(df.isnull().sum())

elif option == "EDA":
    st.title("Exploratory Data Analysis")

    st.markdown("### EDA — Overview\nThis page follows the notebook structure: **Univariate**, **Bivariate**, and **Correlation / Multivariate** analyses. Open each section to inspect the related plots.")

    if st.checkbox("Show dataset sample (EDA)"):
        st.dataframe(df.head(200))

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    with st.expander("Univariate — distributions & counts", expanded=True):
        st.subheader("Target distribution")
        if 'loan_status' in df.columns:
            fig, ax = plt.subplots()
            sns.countplot(x='loan_status', data=df, palette='Set2', ax=ax)
            ax.set_title('Loan status counts (0 = Non-default, 1 = Default)')
            st.pyplot(fig)

        if cat_cols:
            sel_cat = st.selectbox('Categorical column (counts)', options=cat_cols, key='uni_cat')
            fig, ax = plt.subplots(figsize=(8,4))
            sns.countplot(x=sel_cat, data=df, order=df[sel_cat].value_counts().index, palette='viridis', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right')
            st.pyplot(fig)

        if num_cols:
            sel_num = st.selectbox('Numerical column (distribution)', options=num_cols, key='uni_num')
            fig, ax = plt.subplots()
            sns.histplot(df[sel_num].dropna(), kde=True, ax=ax)
            ax.set_title(f'Distribution of {sel_num}')
            st.pyplot(fig)

    with st.expander("Bivariate — relationships", expanded=False):
        st.subheader("Debt-to-income by loan intent & loan status")
        if {'loan_intent','loan_percent_income','loan_status'}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(12,7))
            sns.barplot(x='loan_intent', y='loan_percent_income', hue='loan_status', data=df, estimator=lambda x: sum(x)/len(x), palette='Set2', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig)

        st.subheader("Interest rate distribution by loan intent & status")
        if {'loan_int_rate','loan_intent','loan_status'}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(12,7))
            sns.boxplot(x='loan_intent', y='loan_int_rate', hue='loan_status', data=df, palette='muted', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig)

    with st.expander("Correlation / Multivariate", expanded=False):
        st.subheader("Correlation (selected features)")
        features = [c for c in ['loan_grade','loan_int_rate','cb_person_default_on_file','cb_person_cred_hist_length','loan_status'] if c in df.columns]
        if features:
            df_corr = df[features].copy()
            if 'loan_grade' in df_corr.columns:
                try:
                    df_corr['loan_grade'] = df_corr['loan_grade'].astype('category').cat.codes
                except Exception:
                    pass
            if 'cb_person_default_on_file' in df_corr.columns:
                df_corr['cb_person_default_on_file'] = df_corr['cb_person_default_on_file'].map({'Y':1,'N':0})

            corr = df_corr.corr()
            fig, ax = plt.subplots(figsize=(6,5))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)

    st.markdown("**Tip:** use the expanders above to navigate Univariate / Bivariate / Correlation analyses.")

elif option == "Predict Risk":
    st.title("Predict Credit Risk")

    target_column = "loan_status"
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    user_input = {}

    st.subheader("Enter Applicant Details")

    input_valid = True

    for col in num_cols:
        if col == 'person_age':
            default_age = int(df[col].median()) if df[col].notna().any() else 30
            user_input[col] = st.number_input(f"{col}", min_value=1, max_value=100, value=default_age, step=1, format="%d")
        elif col in ('person_income', 'loan_amnt'):
            default_v = int(df[col].median()) if df[col].notna().any() else 0
            raw = st.text_input(f"{col} (integer)", value=str(default_v), key=f"txt_{col}")
            try:
                parsed = int(str(raw).replace(",", "").strip())
                user_input[col] = parsed
            except Exception:
                st.warning(f"'{col}' must be an integer (no commas). Example: {default_v}")
                input_valid = False
        else:
            default_v = df[col].mean() if df[col].notna().any() else 0.0
            raw = st.text_input(f"{col} (numeric)", value=f"{default_v:.2f}", key=f"txt_{col}")
            try:
                parsed = float(str(raw).replace(",", "").strip())
                user_input[col] = parsed
            except Exception:
                st.warning(f"'{col}' must be numeric. Example: {default_v:.2f}")
                input_valid = False

    for col in cat_cols:
        user_input[col] = st.selectbox(f"{col}", df[col].unique())

    input_df = pd.DataFrame([user_input])

    for _c in ['person_age', 'person_income', 'loan_amnt']:
        if _c in input_df.columns:
            try:
                input_df[_c] = input_df[_c].astype(int)
            except Exception:
                input_valid = False

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    model = RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=42)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X, y)

    if st.button("Predict"):
        if not input_valid:
            st.error("One or more inputs are invalid. Fix highlighted fields above before predicting.")
        else:
            prediction = pipeline.predict(input_df)[0]

            if prediction == 1:
                st.error("High Risk - Likely to Default")
            else:
                st.success("Low Risk - Safe Applicant")




                