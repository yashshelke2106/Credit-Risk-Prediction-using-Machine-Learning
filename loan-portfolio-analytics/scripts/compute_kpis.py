"""
Compute the portfolio KPIs behind the dashboard, straight from the DuckDB
warehouse: NPA rate (count- and balance-weighted), NPA by grade, NPA by loan
purpose, the monthly roll-rate matrix, current delinquency-bucket composition,
and vintage curves (cumulative default rate by months-on-book, by origination
quarter).

Run it stand-alone to print the numbers (e.g. before writing them up), or
import `compute_all()` to get the same dict the dashboard's JS was seeded
with. This is deliberately plain SQL + pandas, not dbt -- the project's
target is analyst/BA roles, where the SQL and the business read on it matter
more than the orchestration around it.
"""
import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = ROOT / "data" / "warehouse" / "portfolio.duckdb"


def _snapshot(con):
    """Last observed row per loan = its status as of the warehouse's as-of date."""
    snap = con.execute("""
        with ranked as (
            select f.*, row_number() over (partition by loan_id order by months_on_book desc) as rn
            from raw.fact_loan_payments f
        )
        select loan_id, dpd_bucket, is_closed, balance_outstanding
        from ranked where rn = 1
    """).df()
    dim_loan = con.execute("select * from raw.dim_loan").df()
    snap = snap.merge(dim_loan[["loan_id", "loan_grade", "loan_intent"]], on="loan_id")
    # "on book" excludes only loans that voluntarily paid off in full; a written-off
    # (Default) loan stays on book for NPA purposes until the business writes it off.
    on_book = snap[~((snap["is_closed"]) & (snap["dpd_bucket"] == "Current"))]
    return snap, on_book, dim_loan


def compute_all():
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    snap, on_book, dim_loan = _snapshot(con)

    npa_count_pct = float(on_book["dpd_bucket"].isin(["DPD90", "Default"]).mean())
    npa_balance_pct = float(
        on_book.loc[on_book["dpd_bucket"].isin(["DPD90", "Default"]), "balance_outstanding"].sum()
        / on_book["balance_outstanding"].sum()
    )

    by_grade = (
        on_book.groupby("loan_grade")
        .agg(
            n_loans=("loan_id", "count"),
            npa_pct=("dpd_bucket", lambda s: s.isin(["DPD90", "Default"]).mean()),
            default_pct=("dpd_bucket", lambda s: (s == "Default").mean()),
        )
        .reset_index()
        .sort_values("loan_grade")
    )

    by_intent = (
        on_book.groupby("loan_intent")
        .agg(n_loans=("loan_id", "count"), npa_pct=("dpd_bucket", lambda s: s.isin(["DPD90", "Default"]).mean()))
        .reset_index()
        .sort_values("npa_pct", ascending=False)
    )

    bucket_dist = on_book["dpd_bucket"].value_counts(normalize=True).round(4).to_dict()

    pairs = con.execute("""
        select a.dpd_bucket as from_state, b.dpd_bucket as to_state, count(*) as n
        from raw.fact_loan_payments a
        join raw.fact_loan_payments b
          on a.loan_id = b.loan_id and b.months_on_book = a.months_on_book + 1
        where a.dpd_bucket != 'Default'
        group by 1, 2
    """).df()
    roll = {}
    for s in ["Current", "DPD30", "DPD60", "DPD90"]:
        sub = pairs[pairs.from_state == s]
        total = sub["n"].sum()
        roll[s] = {row.to_state: round(row.n / total, 4) for row in sub.itertuples()}

    dl = dim_loan.copy()
    dl["orig_q"] = pd.PeriodIndex(dl["origination_month"], freq="M").asfreq("Q").astype(str)
    def_month = con.execute("""
        select loan_id, min(months_on_book) as default_month
        from raw.fact_loan_payments where is_default_event
        group by 1
    """).df()
    dl = dl.merge(def_month, on="loan_id", how="left")

    vintage_rows = []
    for q, grp in dl.groupby("orig_q"):
        n = len(grp)
        if n < 300:
            continue
        max_m = min(24, int(grp["term_months"].median()))
        for m in range(0, max_m + 1):
            cum_def = (grp["default_month"] <= m).sum()
            vintage_rows.append({"vintage": q, "month": m, "cum_default_pct": round(float(cum_def / n), 4), "n": int(n)})

    con.close()
    return {
        "npa_count_pct": round(npa_count_pct, 4),
        "npa_balance_pct": round(npa_balance_pct, 4),
        "on_book": int(len(on_book)),
        "total_loans": int(len(snap)),
        "by_grade": by_grade.round(4).to_dict(orient="records"),
        "by_intent": by_intent.round(4).to_dict(orient="records"),
        "roll_rate_matrix": roll,
        "vintage": vintage_rows,
        "bucket_dist": bucket_dist,
    }


if __name__ == "__main__":
    out = compute_all()
    print(json.dumps({k: v for k, v in out.items() if k != "vintage"}, indent=2))
    print(f"\nvintage cohorts: {sorted(set(r['vintage'] for r in out['vintage']))}")
