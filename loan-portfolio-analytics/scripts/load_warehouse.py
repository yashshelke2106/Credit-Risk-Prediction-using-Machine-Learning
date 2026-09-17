"""Load the raw CSVs (produced by generate_synthetic_data.py) into the DuckDB
warehouse. Re-run any time the raw data changes -- CREATE OR REPLACE keeps this
idempotent."""
import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
WAREHOUSE = ROOT / "data" / "warehouse" / "portfolio.duckdb"
WAREHOUSE.parent.mkdir(parents=True, exist_ok=True)

TABLES = ["dim_borrower", "dim_loan", "fact_loan_payments"]

def main():
    con = duckdb.connect(str(WAREHOUSE))
    con.execute("create schema if not exists raw")
    for tbl in TABLES:
        con.execute(
            f"create or replace table raw.{tbl} as select * from read_csv_auto('{RAW / (tbl + '.csv')}')"
        )
        n = con.execute(f"select count(*) from raw.{tbl}").fetchone()[0]
        print(f"{tbl}: {n:,} rows")
    con.close()

if __name__ == "__main__":
    main()
