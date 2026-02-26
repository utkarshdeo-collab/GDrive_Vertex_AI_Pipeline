"""
Domo Test Fetch Script
======================

Purpose:
    Simulates the Domo API pattern (authenticate → queryDataSet → JSON output)
    using a local CSV for testing. In production, flip testMode = False and
    provide real Domo credentials + dataset ID to hit the live API.

Output:
    domo_output.json  (written to the same folder as this script)

Usage:
    pip install pandas pandasql
    python test_domo_fetch.py
"""

import json
import os
import requests
import pandas as pd
import pandasql as pdsql
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

_config = {
    # Domo OAuth endpoint (used in production only)
    "domoAuthURL": "https://api.domo.com/oauth/token?grant_type=client_credentials&scope=data",

    # Domo API credentials — fill these in when going live
    "domoClientId":     "YOUR_DOMO_CLIENT_ID",
    "domoClientSecret": "YOUR_DOMO_CLIENT_SECRET",

    # Domo Dataset ID — the ID of the dataset in Domo (used in production only)
    "domoDatasetId": "YOUR_DOMO_DATASET_ID",

    # Local CSV path (used in test mode only)
    "csvPath": r"D:\Symphony_Agents\Drive_Pipeline - Personal\SF_and_DOMO\TEST_Pod_Appended_40000_Rows_FIXED.csv",

    # testMode = True  → reads local CSV (no real API call)
    # testMode = False → calls real Domo API (needs credentials above)
    "testMode": True,
}

# Output JSON is written next to this script
OUTPUT_PATH = Path(__file__).resolve().parent / "domo_output.json"

# ─────────────────────────────────────────────────────────────────────────────
# SQL QUERY
# Select only the fields needed by the Nexus pipeline (Phase 1 golden records).
# Table name must be "domo_data" — that is the pandas DataFrame name used below.
# ─────────────────────────────────────────────────────────────────────────────

SQL_QUERY = """
SELECT
    pod_id,
    pretty_name,
    meau,
    active_users,
    avg_daily_msg_sent,
    provisioned_users,
    health_score,
    risk_ratio_for_next_renewal,
    contracted_licenses,
    month,
    latest_health_score,
    next_renewal_date,
    account_owner,
    total_arr
FROM domo_data
"""

# ─────────────────────────────────────────────────────────────────────────────
# AUTH FUNCTION (manager's pattern)
# ─────────────────────────────────────────────────────────────────────────────

domoSession = None  # global token (matches manager's pattern)

def domoAuthenticate() -> str:
    """
    Authenticate with Domo API and return access token.
    In test mode: returns a dummy token (no real API call).
    In production: calls real Domo OAuth endpoint.
    """
    global domoSession

    if _config["testMode"]:
        print("[TEST MODE] Skipping real Domo auth — using dummy token.")
        domoSession = "TEST_DUMMY_TOKEN"
        return domoSession

    print("[AUTH] Authenticating with Domo API...")
    r = requests.get(
        _config["domoAuthURL"],
        auth=(_config["domoClientId"], _config["domoClientSecret"])
    )
    r.raise_for_status()
    access_token = json.loads(r.text)["access_token"]
    domoSession = access_token
    print("[AUTH] Successfully obtained access token.")
    return access_token


# ─────────────────────────────────────────────────────────────────────────────
# QUERY FUNCTION (manager's pattern)
# ─────────────────────────────────────────────────────────────────────────────

def queryDataSet(data_id: str, sql: str) -> dict:
    """
    Execute a SQL query against a Domo dataset.
    In test mode: reads local CSV and executes SQL via pandasql.
    In production: calls real Domo API endpoint.

    Returns a dict matching the Domo API response format:
        {
            "columns": ["col1", "col2", ...],
            "rows":    [[val, val, ...], ...],
            "num_rows": N
        }
    """
    global domoSession

    # ── TEST MODE: read local CSV, run SQL via pandasql ──────────────────────
    if _config["testMode"]:
        print(f"[TEST MODE] Reading local CSV: {_config['csvPath']}")
        csv_path = _config["csvPath"]

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at: {csv_path}")

        # Load CSV into DataFrame — named 'domo_data' to match the SQL above
        domo_data = pd.read_csv(csv_path, low_memory=False)
        print(f"[TEST MODE] Loaded {len(domo_data):,} rows, {len(domo_data.columns)} columns.")

        # Run SQL against the DataFrame
        print("[TEST MODE] Executing SQL query via pandasql...")
        result_df = pdsql.sqldf(sql, {"domo_data": domo_data})
        print(f"[TEST MODE] Query returned {len(result_df):,} rows.")

        # Convert to Domo API response format
        columns = list(result_df.columns)
        rows = result_df.where(pd.notnull(result_df), None).values.tolist()

        return {
            "columns":  columns,
            "rows":     rows,
            "num_rows": len(rows),
        }

    # ── PRODUCTION MODE: real Domo API call ──────────────────────────────────
    print(f"[API] Querying Domo dataset: {data_id}")
    url = "https://api.domo.com/v1/datasets/query/execute/" + data_id
    payload = {"sql": sql}
    headers = {
        "Content-Type":  "application/json",
        "Authorization": "Bearer " + domoSession,
    }
    r = requests.post(url, json=payload, headers=headers)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DOMO TEST FETCH")
    print(f"  Mode: {'TEST (local CSV)' if _config['testMode'] else 'PRODUCTION (live API)'}")
    print("=" * 60)

    # Step 1: Authenticate
    domoAuthenticate()

    # Step 2: Query dataset
    result = queryDataSet(_config["domoDatasetId"], SQL_QUERY)

    # Step 3: Save JSON output
    print(f"\n[OUTPUT] Writing JSON to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"[OUTPUT] Done. {result['num_rows']:,} rows saved.")
    print(f"\nJSON structure:")
    print(f"  columns  : {result['columns']}")
    print(f"  num_rows : {result['num_rows']:,}")
    print(f"  rows[0]  : {result['rows'][0] if result['rows'] else 'empty'}")
    print("\n" + "=" * 60)
    print("  SUCCESS — domo_output.json is ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
