# Credit Risk Prediction using Machine Learning

Predicts loan default on the public Kaggle credit risk dataset, with a **leakage audit**, cross-validated model comparison, and a **WoE credit scorecard** scaled to points.

**Live app:** [credit-risk-prediction-ml1.streamlit.app](https://credit-risk-prediction-ml1.streamlit.app/)

---

## Dataset

| | |
|---|---|
| **Source** | Kaggle *Credit Risk Dataset* (public; no proprietary data) — `credit_risk_dataset.csv` |
| **Raw** | 32,581 rows x 12 columns |
| **Modelled** | 32,416 rows after removing 165 exact duplicates and nulling 7 impossible values |
| **Target** | `loan_status` — 1 = default, 0 = repaid |
| **Default rate** | 21.87% |
| **Missing in raw** | `loan_int_rate` 3,116 · `person_emp_length` 895 (no other column) |

7 cells held impossible values (`person_age` of 123 and 144; `person_emp_length` of 123). These are set to `NaN` and imputed inside the pipeline — **no row is dropped for having one**. The only rows removed are the 165 exact duplicates, and that removal *lowers* the scores rather than raising them.

---

## 1. Leakage audit

Run `python leakage_audit.py`. Three checks:

### A. Preprocessing fitted before the split — **LEAK CONFIRMED**

`EDA.ipynb` fills missing values in cells 6–7 and writes `clean_data.csv` in cell 21, *before* the train/test split in cell 23:

```python
df['person_emp_length'].fillna(df['person_emp_length'].mean(), inplace=True)   # cell 6
df['loan_int_rate'].fillna(df['loan_int_rate'].mean(), inplace=True)           # cell 7
df.to_csv("clean_data.csv")                                                    # cell 21
```

Verified in the committed data: **895 rows** of `person_emp_length` hold exactly `4.789686` and **3,114 rows** of `loan_int_rate` hold exactly `11.011695` — the full-dataset means, computed over every future test row. 4,009 cells affected.

**Fix:** all scripts now read `credit_risk_dataset.csv` and impute inside a `Pipeline`, so every fitted statistic is refit per fold.

### B. Duplicate rows spanning train and test — **LEAK CONFIRMED**

165 exact duplicate rows (0.51%), forming 165 groups, **none with a conflicting target**. Under the 5-fold split, an average of **51.6 test rows per fold** have an identical twin in their training fold — the model can memorise rather than generalise on them.

**Fix:** `drop_duplicates()` before splitting.

### C. Features recorded after the outcome — **NONE FOUND**, two flagged

| Feature | Recorded | Verdict |
|---|---|---|
| `person_age` | Application | Stated at application. No outcome information. **OK** |
| `person_income` | Application | Stated/verified at application. **OK** |
| `person_home_ownership` | Application | Status at application. **OK** |
| `person_emp_length` | Application | Employment history at application. **OK** |
| `loan_intent` | Application | Stated purpose. **OK** |
| `loan_amnt` | Origination | Amount approved at origination, before outcome. **OK** |
| `loan_percent_income` | Derived | `loan_amnt / person_income`, both known at origination. Redundant, not leaky. **OK** |
| `cb_person_default_on_file` | Bureau, pre-application | *Prior* defaults, strictly before this loan. **OK** |
| `cb_person_cred_hist_length` | Bureau, pre-application | History length at pull. **OK** |
| `loan_grade` | **Underwriting output** | The lender's own risk verdict, produced *by* underwriting. Pre-outcome, but target-adjacent. **CAUTION** |
| `loan_int_rate` | **Underwriting output** | Priced off `loan_grade`, i.e. off expected default. Pre-outcome, but target-adjacent. **CAUTION** |

No feature is recorded after the loan outcome. The two flagged features are legitimate *if* the model scores already-underwritten loans, and circular if it is meant to decide new applications.

### What the leaks were actually worth

| Configuration | Model | CV ROC-AUC | KS | Rows |
|---|---|---|---|---|
| 1. Leaky baseline (pre-imputed + duplicates) | Random Forest | 0.9352 ± 0.0039 | 0.7280 | 32,571 |
| 2. In-fold imputation, duplicates remain | Random Forest | 0.9339 ± 0.0051 | 0.7246 | 32,581 |
| 3. **Corrected** (in-fold + deduped) | Random Forest | **0.9342 ± 0.0022** | 0.7255 | 32,416 |
| 4. Corrected, application-only | Random Forest | 0.8740 ± 0.0042 | 0.5864 | 32,416 |

**Both leaks are real and both are nearly worthless.** Fixing them moves Random Forest by **−0.0009** (0.9352 → 0.9342), smaller than the ±0.0022 fold standard deviation. Pre-split imputation touched 1.1% of cells and the fold means barely differ from the full-data means; 165 duplicates are 0.5% of the data. The original 0.929–0.935 figure was **not** produced by leakage.

### What actually produces the near-vertical ROC

The steep rise is genuine, not an artefact:

| | TPR |
|---|---|
| at FPR = 0.001 | **0.6280** |
| at FPR = 0.01 | 0.7186 |
| at FPR = 0.05 | 0.7729 |
| at FPR = 0.10 | 0.8175 |

63% of all defaults are caught while touching 0.1% of good loans, because `loan_grade` is close to deterministic at the tail — grade G defaults at **98.4%** (63 of 64), F at 70.5%, E at 64.5%, D at 59.1%, against 10.0% for grade A.

Removing `loan_grade` and `loan_int_rate` costs **−0.0602** AUC (0.9342 → 0.8740) — 65x the combined cost of both leaks. The model's strength rests on the underwriter's own risk assessment, which is the finding that matters here.

---

## 2. Cross-validated results

Stratified 5-fold, `random_state=42`, every fitted statistic refit per fold. KS from out-of-fold predictions.

| Model | CV ROC-AUC | KS |
|---|---|---|
| Logistic Regression | 0.8707 ± 0.0034 | 0.6036 |
| Random Forest | 0.9342 ± 0.0022 | 0.7255 |
| LightGBM *(deployed in the app)* | 0.9473 ± 0.0025 | 0.7456 |

LightGBM on the leaky data scored 0.9482; corrected, 0.9473. Same −0.0009 shift.

---

## 3. WoE scorecard

Run `python build_scorecard.py`. Writes `iv_table.csv`, `scorecard.csv`, `cutoff_analysis.csv`.

WoE = `ln(%good / %bad)`, so higher WoE means lower risk and more points. Bin edges and WoE values are **fitted from the target**, so they are implemented as a scikit-learn transformer (`woe.py`) and refit inside every CV fold — the same discipline the audit above demanded of imputation. Missing values get their own bin rather than being imputed.

**Scorecard CV ROC-AUC 0.8700 ± 0.0033 · KS 0.6119** (11 characteristics, 57 bins)

### Information Value

| Feature | IV | Flag |
|---|---|---|
| `loan_grade` | 0.8790 | **SUSPICIOUS (>0.5)** |
| `loan_percent_income` | 0.7079 | **SUSPICIOUS (>0.5)** |
| `loan_int_rate` | 0.6192 | **SUSPICIOUS (>0.5)** |
| `person_income` | 0.4607 | ok (strong) |
| `person_home_ownership` | 0.3753 | ok (strong) |
| `cb_person_default_on_file` | 0.1643 | ok (medium) |
| `loan_intent` | 0.0959 | ok (weak) |
| `loan_amnt` | 0.0737 | ok (weak) |
| `person_emp_length` | 0.0603 | ok (weak) |
| `person_age` | 0.0065 | **WEAK (<0.02)** |
| `cb_person_cred_hist_length` | 0.0040 | **WEAK (<0.02)** |

The three suspicious flags are not a bug in the binning — they are the same finding as audit item C. Two of them are underwriting outputs, and `loan_percent_income` is a deterministic function of two other columns.

`loan_int_rate` is worth noting: IV 0.6192 univariately, but once `loan_grade` is in the model its points range collapses to 45–50 out of ~600. High IV, almost no independent contribution — it is a repricing of the grade.

### Points scaling

```
factor = PDO / ln(2)        = 20 / 0.6931 = 28.8539
offset = base - factor*ln(odds) = 600 - 28.8539*ln(50) = 487.1229
points = -(beta_i * woe_ij + alpha/n) * factor + offset/n
```

Base 600 at 50:1 odds, PDO 20. `build_scorecard.py` asserts the summed points reproduce `offset + factor*ln(odds)` and that a score of 600 returns exactly 50.0:1. Fitted score range **275–661**, mean 543.

Sample rows (`loan_grade`, the strongest characteristic):

| Bin | Count | Bad rate | WoE | Points |
|---|---|---|---|---|
| A | 10,703 | 0.0996 | 0.9283 | **+78** |
| B | 10,387 | 0.1632 | 0.3615 | +59 |
| C | 6,438 | 0.2075 | 0.0667 | +50 |
| D | 3,620 | 0.5906 | −1.6393 | −5 |
| E | 963 | 0.6449 | −1.8688 | −12 |
| F | 241 | 0.7054 | −2.1420 | −21 |
| G | 64 | 0.9844 | −5.0185 | **−114** |

---

## 4. Cut-off

Selected at the **KS-maximising score of 515**, from out-of-fold scores so the figures are not fitted on their own test data.

| | |
|---|---|
| **Approval rate** | **72.09%** |
| **Defaults captured** (declined) | **75.67%** |
| Approved-book bad rate | **7.38%** vs 21.87% baseline |
| Good applicants declined | 14.54% |

Accepting 72% of applicants removes **two thirds of the loss rate** (21.87% → 7.38%), at the cost of turning away 14.5% of borrowers who would have repaid.

Alternatives from `cutoff_analysis.csv`:

| Cut-off | Approval | Defaults captured | Approved bad rate |
|---|---|---|---|
| 480 | 86.65% | 47.45% | 13.26% |
| 500 | 78.96% | 65.13% | 9.66% |
| **515** | **72.09%** | **75.67%** | **7.38%** |
| 540 | 56.73% | 85.60% | 5.55% |
| 560 | 42.54% | 91.51% | 4.37% |

---

## Caveat on the headline numbers

Every figure above uses `loan_grade` and `loan_int_rate`. On an **application-only** basis — dropping both, as a scorecard deciding *new* applications would have to — performance falls materially:

| | Full (11) | Application-only (9) |
|---|---|---|
| Random Forest CV ROC-AUC | 0.9342 ± 0.0022 | 0.8740 ± 0.0042 |
| Logistic Regression CV ROC-AUC | 0.8707 ± 0.0034 | 0.8062 ± 0.0038 |
| Random Forest KS | 0.7255 | 0.5864 |
| Logistic Regression KS | 0.6036 | 0.4589 |

Both are reported because which one is honest depends on the use case: scoring an existing book, the full model is correct; underwriting new applications, the application-only model is the defensible number.

---

## Project structure

```
.
├── credit_risk_dataset.csv         # Raw Kaggle data (32,581 rows) - the modelling source
├── clean_data.csv                  # Legacy pre-imputed file; kept ONLY to reproduce the leaky baseline
├── credit_data.py                  # Shared loader + leak-free Pipeline builder
├── woe.py                          # WoE/IV binning as a fold-safe sklearn transformer
├── leakage_audit.py                # Audit A/B/C + ablation          -> leakage_ablation.csv, feature_audit.csv
├── build_scorecard.py              # IV, scorecard, points, cut-off  -> iv_table.csv, scorecard.csv, cutoff_analysis.csv
├── recheck_models.py               # LR / RF / LightGBM, leaky vs corrected
├── compare_models.py               # Earlier 4-model benchmark (reads clean_data.csv - see audit A)
├── app1.py                         # Streamlit app (LightGBM, tunable threshold)
├── EDA.ipynb                       # Exploratory analysis; also the source of leak A
├── requirements.txt / requirements-dev.txt / packages.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt     # app
streamlit run app1.py               # http://localhost:8501

pip install -r requirements-dev.txt # adds xgboost + catboost for the benchmark
python leakage_audit.py             # audit + ablation  (~4 min)
python build_scorecard.py           # scorecard + cut-off (~2 min)
python recheck_models.py            # leaky vs corrected comparison
```

The app has four sections — **Home**, **Data Overview**, **EDA**, and **Predict Risk**, which scores an applicant with a decision-threshold slider that re-scores cached probabilities instantly instead of retraining.

> **Note:** `app1.py` and `compare_models.py` still read `clean_data.csv`. Per the ablation that costs ~0.0009 AUC, so the app's displayed metrics are very slightly optimistic. Migrating them to `credit_data.load_raw()` is the outstanding change.

## Author

Yash Shelke — [GitHub](https://github.com/yashshelke2106)

## License

Public dataset; code is open source.
