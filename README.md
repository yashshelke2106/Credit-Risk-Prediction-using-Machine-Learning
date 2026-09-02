# Credit Risk Prediction

An end-to-end machine learning project that predicts the probability that a loan applicant will default. It pairs an exploratory analysis notebook with an interactive **Streamlit** dashboard that trains a model on the fly and scores individual applicants in real time.

## Overview

The project uses a credit-risk dataset of ~32,500 loan applications to answer one question: **how likely is this borrower to default?** A `RandomForestClassifier` is wrapped in a scikit-learn pipeline that handles imputation, scaling, and categorical encoding automatically, so raw applicant details can be scored without manual preprocessing.

The app has four sections:

- **Home** — project summary and headline metrics.
- **Data Overview** — dataset preview, summary statistics, and missing-value inspection.
- **EDA** — interactive univariate, bivariate, and correlation charts.
- **Predict Risk** — enter an applicant's details and get an estimated default probability with a risk band (low / moderate / high), alongside the model's ROC curve and feature importances.

## Dataset

Two CSVs are included:

| File | Description |
|------|-------------|
| `credit_risk_dataset.csv` | Raw source data (~32,580 rows) |
| `clean_data.csv` | Cleaned dataset produced by `EDA.ipynb`, used by the app |

Each row describes a loan application with the following fields:

| Column | Meaning |
|--------|---------|
| `person_age` | Applicant age |
| `person_income` | Annual income |
| `person_home_ownership` | RENT / OWN / MORTGAGE / OTHER |
| `person_emp_length` | Employment length (years) |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan grade (A–G) |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Interest rate |
| `loan_percent_income` | Loan amount as a fraction of income |
| `cb_person_default_on_file` | Prior default on record (Y / N) |
| `cb_person_cred_hist_length` | Length of credit history (years) |
| `loan_status` | **Target** — 1 = default, 0 = no default |

## Project structure

```
.
├── app1.py                    # Streamlit dashboard (EDA + live prediction)
├── EDA.ipynb                  # Exploratory analysis & data cleaning
├── credit_risk_dataset.csv    # Raw dataset
├── clean_data.csv             # Cleaned dataset used by the app
├── catboost_info/             # CatBoost training logs (from notebook experiments)
└── README.md
```

## Visualizations

The app (`app1.py`) is built with **Plotly** for interactive, hover-enabled charts, while the notebook (`EDA.ipynb`) uses **Matplotlib** and **Seaborn** for static exploratory plots.

**In the Streamlit app (Plotly):**

- **Bar charts** — target class distribution, categorical value counts, category-vs-default rates, and feature importances
- **Pie chart** — overall loan status (default vs. non-default) breakdown
- **Histograms** — numeric feature distributions, with an overlay by loan status
- **Box plot** — numeric features split by default outcome to spot outliers and spread
- **Scatter plot** — relationship between two numeric variables, colored by loan status
- **Correlation heatmap** (`px.imshow`) — pairwise correlations across numeric features
- **Area chart** — ROC curve with the AUC score
- **Gauge indicator** (`go.Indicator`) — live default-probability dial on the prediction page, color-banded low / moderate / high

**In the notebook (Matplotlib + Seaborn):**

- **Count plots** — frequency of categorical variables
- **Bar plots** — grouped comparisons across categories
- **Box plot** — distribution and outlier inspection
- **Correlation heatmap** — feature correlation matrix
- **Line plots** — trend inspection during analysis

## Model

The prediction pipeline (`app1.py`) is built with scikit-learn:

- **Numeric features** — mean imputation + standard scaling
- **Categorical features** — most-frequent imputation + one-hot encoding
- **Classifier** — `RandomForestClassifier` (300 trees, max depth 15, `class_weight="balanced"`)

The model trains on an 80/20 stratified split and is cached after the first run, so the app stays responsive. Evaluation reports AUC, accuracy, precision, recall, a confusion matrix, and the top feature importances.

## Getting started

### Requirements

Python 3.9+ and the following packages:

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

### Run the app

From the project folder:

```bash
streamlit run app1.py
```

The app opens in your browser. Use the sidebar to switch datasets and navigate between pages. On the **Predict Risk** page, fill in the applicant details and click **Assess risk** to see the estimated default probability.

### Explore the analysis

Open `EDA.ipynb` in Jupyter to review the data cleaning and exploratory analysis behind `clean_data.csv`:

```bash
jupyter notebook EDA.ipynb
```

## Notes

- The model is trained fresh from the selected CSV each session (cached afterward); no serialized model file is shipped.
- `loan_status` is the target and is excluded from the applicant input form.
- Predictions are for demonstration and educational purposes and should not be used for actual lending decisions.
