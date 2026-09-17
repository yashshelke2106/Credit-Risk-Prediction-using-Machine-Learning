# Loan Portfolio Analytics

Collections and risk analytics on a loan portfolio: NPA rate, roll-rate
matrix, and vintage curves, ending in a set of business recommendations.
Built on top of the [Credit Risk Prediction](../README.md) project's dataset,
extended with a synthesized monthly repayment history so the data supports
time-series analysis a flat one-row-per-loan dataset can't.

**[Open the dashboard](dashboard/index.html)** (or view the hosted version
shared alongside this project).

## Why this exists

The credit-risk project answers "will this applicant default?" with a
classifier. This project answers a different, equally real question: "how is
the *existing* loan book performing, and what should collections/underwriting
do about it?" That's a business-analyst question, not a data-science one, and
it needs a different kind of evidence — SQL against a warehouse, KPIs tied to
what a portfolio manager actually tracks, and a written recommendation, not a
model accuracy score.

## Data

The source dataset (`credit_risk_dataset.csv` / `clean_data.csv`, one row per
loan, no time series) is extended into three tables:

| Table | Grain | What's new here |
|---|---|---|
| `dim_borrower` | one row per borrower | unchanged applicant attributes |
| `dim_loan` | one row per loan | adds `origination_month` (spread over the last 36 months, for vintage cohorts), `term_months`, `monthly_installment` |
| `fact_loan_payments` | one row per loan per month on book | **new** — a simulated monthly delinquency path (`Current → DPD30 → DPD60 → DPD90 → Default`), calibrated so per-grade default rates match the real dataset |

**Known simplification:** each borrower has exactly one loan, because the
source data is one row per applicant. `borrower_id` and `loan_id` are kept as
separate keys so the schema *could* support multiple loans per borrower —
it just isn't populated that way. Worth naming yourself in an interview
before it's asked.

Regenerate the raw tables with `python generate_synthetic_data.py` (seeded,
reproducible — `RNG_SEED = 42`).

## The simulation, and a bug worth knowing about

Each loan's monthly delinquency path is a stochastic Markov simulation:
roll-forward risk increases with `loan_grade`, and cure/roll-forward
probabilities govern how a delinquent loan either recovers or worsens.
Critically, **the path is forced to be consistent with the real
`loan_status` label** from the source dataset — a loan the original data says
actually defaulted will default somewhere in its simulated life; one that
didn't, won't.

An earlier version of this simulation had a real bug: it computed the forced
default timing against the *observed* (censored) window instead of the
loan's full contractual term, which forced young true-defaulters into
default within their first month or two — regardless of grade. The result
was obviously wrong (100% NPA on grade-A loans). The fix was to simulate
against the full term first, then censor to what's actually observable as of
the warehouse's as-of date. Sanity-checking the output against the known
grade-ordering pattern (NPA should climb monotonically from A to G) is what
caught it — worth having that verification instinct ready to describe.

## Warehouse

DuckDB (`data/warehouse/portfolio.duckdb`) — a real embedded analytical
database, not a toy, and it needs no cloud account. `scripts/load_warehouse.py`
loads the three raw CSVs into a `raw` schema; re-run any time the raw data
changes (`CREATE OR REPLACE`, idempotent).

## KPIs

`scripts/compute_kpis.py` runs the SQL behind every number on the dashboard:

- **NPA rate** — count-weighted and balance-weighted, on the current "on
  book" population (excludes only loans that voluntarily paid off in full)
- **NPA / default rate by grade and by loan purpose**
- **Roll-rate matrix** — monthly transition probability between delinquency
  buckets (self-join on `(loan_id, months_on_book)` vs. `(loan_id,
  months_on_book + 1)`)
- **Vintage curves** — cumulative default rate by months-on-book, grouped by
  origination quarter, compared at equal *age* rather than equal calendar
  date (younger cohorts are right-censored — they haven't had time to
  default yet, and comparing them to matured cohorts at face value is a
  classic vintage-analysis mistake)

Run `python scripts/compute_kpis.py` to print them; `dashboard/index.html`
embeds a snapshot of the same output.

## Headline numbers (this run)

- 32,571 loans originated, 26,367 currently on book
- NPA rate: 21.6% by count, 26.8% by balance (larger loans skew worse)
- NPA rate by grade: A 9.8% → D 48.4% → G 68.3% — the steepest jump is C→D
- DPD30 cure rate: 29.9%. DPD90 cure rate: 4.7% (with 24.8%/month rolling
  straight to default) — collections leverage drops off sharply past DPD60

Full write-up and the "so what" for each of these is in the dashboard.

## Project structure

```
.
├── generate_synthetic_data.py   # raw table generation (seeded)
├── scripts/
│   ├── load_warehouse.py        # raw CSVs -> DuckDB
│   └── compute_kpis.py          # the SQL/pandas behind every dashboard number
├── data/
│   ├── raw/                     # dim_borrower, dim_loan, fact_loan_payments
│   └── warehouse/portfolio.duckdb
└── dashboard/index.html         # standalone dashboard (also hosted separately)
```

## Honesty about what this is

The repayment history is simulated — the real dataset has no time series —
calibrated to match real grade-level default patterns. This project
demonstrates the analysis method (SQL against a warehouse, KPI definitions,
vintage/roll-rate technique, business recommendations from the numbers), not
a live production portfolio. Say that plainly if asked; don't let it be
discovered.
