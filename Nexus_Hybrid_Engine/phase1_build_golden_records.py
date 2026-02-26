"""
Phase 1: Build Golden Records (Decoupled Architecture)

This script processes Salesforce and Domo data into two separate types of vector records.
It adheres to the following principles:
- Decoupled Architecture: Salesforce and Domo records are kept distinct.
- Dynamic Content: Includes all available columns in the embedding text.
- Prioritized Information: Critical fields are placed at the top of the embedding text.

Usage:
    python Nexus_Hybrid_Engine/phase1_build_golden_records.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import requests
import pandasql as pdsql

# Ensure project root is in sys.path for config import
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from Nexus_Hybrid_Engine import nexus_config as config
except ImportError:
    import nexus_config as config

# --- DOMO API CONFIGURATION ---
# testMode = True  → reads local CSV (no real API call). Safe for dev/demo.
# testMode = False → calls real Domo API. Fill in the 3 credentials below before going live.
_DOMO_CONFIG = {
    "testMode": True,

    # Real Domo credentials — only needed when testMode = False
    "domoClientId":     "YOUR_DOMO_CLIENT_ID",
    "domoClientSecret": "YOUR_DOMO_CLIENT_SECRET",
    "domoDatasetId":    "YOUR_DOMO_DATASET_ID",

    # Local CSV fallback — used when testMode = True
    "csvPath": str(Path(__file__).resolve().parent.parent / "SF_and_DOMO" / "TEST_Pod_Appended_40000_Rows_FIXED.csv"),
}

# SQL query — selects exact fields needed for golden records.
# Column names must match the live Domo dataset when testMode = False.
_DOMO_SQL = """
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

_domo_session = None  # module-level token cache

# --- SALESFORCE BIGQUERY CONFIGURATION ---
# Client fills in these 3 values to point to their Salesforce BQ table.
# Auth uses the same service account already configured in nexus_config.py.
_BQ_CONFIG = {
    "projectId": "YOUR_GCP_PROJECT_ID",
    "dataset":   "YOUR_BQ_DATASET",
    "table":     "YOUR_BQ_TABLE_NAME",
}

# Selects only the exact 10 columns needed for golden records.
# Column names must match the client's BigQuery table schema.
_SF_SQL = """
SELECT
    Salesforce_Account_ID,
    Account_Owner,
    Contract_End_Date,
    Total_ARR,
    Client_Engagement_Lead,
    Renewal_Date,
    Customer_Name,
    Firm_ID,
    POD_Internal_Id__c,
    Task_Count
FROM `{project}.{dataset}.{table}`
"""


def query_salesforce_bq() -> pd.DataFrame:
    """
    Fetch Salesforce data from BigQuery and return as a DataFrame.
    Uses the service account credentials already configured in nexus_config.py.

    Returns:
        pd.DataFrame: Query results with the 10 required Salesforce columns.
    """
    from google.cloud import bigquery

    project = _BQ_CONFIG["projectId"]
    dataset = _BQ_CONFIG["dataset"]
    table   = _BQ_CONFIG["table"]

    sql = _SF_SQL.format(project=project, dataset=dataset, table=table)

    print(f"[Salesforce] Querying BigQuery: `{project}.{dataset}.{table}`")
    client = bigquery.Client(project=project)
    df = client.query(sql).to_dataframe()
    print(f"[Salesforce] BigQuery returned {len(df):,} rows.")
    return df


def domoAuthenticate() -> str:
    """
    Authenticate with Domo API and return an access token.
    In testMode: skips real auth, returns dummy token.
    In production: calls real Domo OAuth endpoint.
    """
    global _domo_session

    if _DOMO_CONFIG["testMode"]:
        print("[Domo] testMode=True — skipping real auth, using dummy token.")
        _domo_session = "TEST_DUMMY_TOKEN"
        return _domo_session

    print("[Domo] Authenticating with Domo API...")
    r = requests.get(
        "https://api.domo.com/oauth/token?grant_type=client_credentials&scope=data",
        auth=(_DOMO_CONFIG["domoClientId"], _DOMO_CONFIG["domoClientSecret"])
    )
    r.raise_for_status()
    _domo_session = json.loads(r.text)["access_token"]
    print("[Domo] Auth successful.")
    return _domo_session


def queryDataSet(data_id: str, sql: str) -> pd.DataFrame:
    """
    Execute a SQL query against a Domo dataset and return a DataFrame.
    In testMode: reads local CSV and runs SQL via pandasql.
    In production: calls the real Domo API endpoint.
    """
    if _DOMO_CONFIG["testMode"]:
        csv_path = _DOMO_CONFIG["csvPath"]
        print(f"[Domo] testMode=True — reading local CSV: {csv_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[Domo] CSV not found at: {csv_path}")
        domo_data = pd.read_csv(csv_path, low_memory=False)
        print(f"[Domo] Loaded {len(domo_data):,} rows from local CSV.")
        result_df = pdsql.sqldf(sql, {"domo_data": domo_data})
        print(f"[Domo] SQL returned {len(result_df):,} rows.")
        return result_df

    # Production: real Domo API call
    print(f"[Domo] Querying live Domo dataset: {data_id}")
    url = "https://api.domo.com/v1/datasets/query/execute/" + data_id
    headers = {
        "Content-Type":  "application/json",
        "Authorization": "Bearer " + _domo_session,
    }
    r = requests.post(url, json={"sql": sql}, headers=headers)
    r.raise_for_status()
    response = r.json()
    df = pd.DataFrame(response["rows"], columns=response["columns"])
    print(f"[Domo] API returned {len(df):,} rows.")
    return df


# --- TEMPLATES ---
# Templates define the core, high-priority fields that appear at the top of the embedding text.

SALESFORCE_CORE_TEMPLATE = """
[ACCOUNT] {Customer_Name}
[OWNER] {Account_Owner}
[TOTAL_ARR] {Total_ARR}
[RENEWAL_DATE] {Renewal_Date}
[TASK_COUNT] {Task_Count}
[ENGAGEMENT_SIGNAL] {Calculated_Engagement}
[POD_ID] {POD_Internal_Id__c}
"""

DOMO_CORE_TEMPLATE = """
[POD_ID] {pod_id}
[POD_NAME] {pretty_name}
[MEAU] {meau}
[HEALTH_SCORE] {health_score}
[EXPANSION_SIGNAL] {Calculated_Expansion}
[RISK_RATIO] {risk_ratio_for_next_renewal}
[PROVISIONED_USERS] {provisioned_users}
[CONTRACTED_LICENSES] {contracted_licenses}
"""


def clean_currency_column(series: pd.Series) -> pd.Series:
    """
    Cleans a currency column to a consistent string format (e.g., "$1,000").

    Args:
        series (pd.Series): The pandas Series containing currency data (mixed types).

    Returns:
        pd.Series: A Series of formatted strings.
    """
    def _format_value(val: Any) -> str:
        if pd.isna(val):
            return "N/A"
        if isinstance(val, (int, float)):
            return f"${int(val):,}"
        
        # Remove symbols and whitespace
        val_str = str(val).replace("$", "").replace(",", "").strip()
        try:
            return f"${int(float(val_str)):,}"
        except ValueError:
            return str(val)

    return series.apply(_format_value)


def calculate_engagement_signal(task_count: Any) -> str:
    """
    Calculates the Engagement Signal based on Task Count (Salesforce logic).

    Logic:
        - Task Count >= 4: Positive
        - Task Count 1-3: Neutral
        - Task Count 0/Null: Negative

    Args:
        task_count (Any): The raw task count value.

    Returns:
        str: "Positive", "Neutral", or "Negative".
    """
    if pd.isna(task_count):
        return "Negative"
    try:
        count = int(task_count)
        if count >= 4:
            return "Positive"
        elif 1 <= count <= 3:
            return "Neutral"
        else:
            return "Negative"
    except (ValueError, TypeError):
        return "Negative"


def calculate_expansion_signal(provisioned_users: Any, contracted_licenses: Any) -> str:
    """
    Calculates the Expansion Signal based on usage (Domo logic).

    Logic:
        - Provisioned > 90% of Contracted: Positive
        - Otherwise: Negative (or N/A if data missing)

    Args:
        provisioned_users (Any): Number of provisioned users.
        contracted_licenses (Any): Number of contracted licenses.

    Returns:
        str: "Positive", "Negative", or "N/A".
    """
    if pd.isna(contracted_licenses) or contracted_licenses == 0:
        return "N/A"
    if pd.isna(provisioned_users):
        return "N/A"
    
    try:
        prov = float(provisioned_users)
        cont = float(contracted_licenses)
        
        if cont == 0:
            return "N/A"
        
        if prov > (0.9 * cont):
            return "Positive"
        else:
            return "Negative"
    except (ValueError, TypeError):
        return "N/A"


def safe_str(val: Any) -> str:
    """
    Safely converts a value to a stripped string, handling NaNs.

    Args:
        val (Any): Input value.

    Returns:
        str: String representation or "N/A".
    """
    if pd.isna(val):
        return "N/A"
    return str(val).strip()


def build_dynamic_embedding_text(row: pd.Series, core_template: str, core_fields: List[str]) -> str:
    """
    Constructs the full text for embedding.
    
    Structure:
    1. Core Fields (Formatted via template)
    2. Separator
    3. All Other Fields (Key-Value pairs)

    Args:
        row (pd.Series): The data row.
        core_template (str): The template string for core fields.
        core_fields (List[str]): List of column names used in the core template.

    Returns:
        str: The complete, formatted text for embedding.
    """
    # 1. Prepare Core Fields Data
    format_dict = {k: safe_str(row.get(k)) for k in core_fields}
    
    # Add calculated fields explicitly if they exist in the row
    if 'Calculated_Engagement' in row:
        format_dict['Calculated_Engagement'] = safe_str(row['Calculated_Engagement'])
    if 'Calculated_Expansion' in row:
        format_dict['Calculated_Expansion'] = safe_str(row['Calculated_Expansion'])
        
    # Handle specific field overrides (e.g., Clean_ARR instead of Total_ARR)
    if 'Total_ARR' in core_fields:
        format_dict['Total_ARR'] = row.get('Clean_ARR', safe_str(row.get('Total_ARR')))

    # Generate Core Text
    core_text = core_template.format(**format_dict).strip()
    
    text_parts = [core_text, "\n--- ADDITIONAL DATA ---"]

    # 2. Append All Other Fields
    # Define keys to exclude from the dynamic section (already in core)
    exclude_keys = {k.lower() for k in core_fields}
    exclude_keys.update({'calculated_engagement', 'calculated_expansion', 'clean_arr'})

    for col in sorted(row.index):
        if not isinstance(col, str):
            continue
            
        if col.lower() in exclude_keys:
            continue
            
        val = row[col]
        if pd.isna(val) or val == "":
            continue  # Skip empty values to save context window tokens
            
        # Format Date Objects
        if isinstance(val, (pd.Timestamp, np.datetime64)):
            val_str = val.strftime('%Y-%m-%d')
        else:
            val_str = str(val).strip()
            
        text_parts.append(f"[{col.upper()}] {val_str}")

    return "\n".join(text_parts)


def process_salesforce_data() -> List[Dict[str, Any]]:
    """
    Fetches and processes Salesforce data from BigQuery into vector records.

    Returns:
        List[Dict[str, Any]]: List of dictionary records ready for JSONL output.
    """
    print("\n[Salesforce] Starting data fetch from BigQuery...")
    try:
        df = query_salesforce_bq()
    except Exception as e:
        print(f"[Salesforce] ERROR fetching data from BigQuery: {e}")
        sys.exit(1)

    print(f"[Salesforce] Loaded {len(df)} rows.")

    # Pre-calculate derived columns
    df['Calculated_Engagement'] = df['Task_Count'].apply(calculate_engagement_signal)
    df['Clean_ARR'] = clean_currency_column(df['Total_ARR'])

    # Define fields used in the core template
    core_fields = [
        'Customer_Name', 'Account_Owner', 'Total_ARR', 'Renewal_Date', 
        'Task_Count', 'POD_Internal_Id__c'
    ]

    records = []
    for _, row in df.iterrows():
        # Generate Embedding Text
        text_content = build_dynamic_embedding_text(row, SALESFORCE_CORE_TEMPLATE, core_fields)
        
        # Generate Metadata (for filtering)
        metadata = {
            "type": "salesforce",
            "account_name": safe_str(row.get('Customer_Name')),
            "owner": safe_str(row.get('Account_Owner')),
            "pod_id": safe_str(row.get('POD_Internal_Id__c')),
            "engagement": row['Calculated_Engagement'],
            "total_arr": row['Clean_ARR']
        }

        # Create Record ID
        safe_id = f"sf_{safe_str(row.get('Customer_Name'))}".replace(" ", "_")

        records.append({
            "id": safe_id,
            "content": text_content,
            "metadata": metadata
        })
    
    print(f"[Salesforce] Generated {len(records)} records.")
    return records


def process_domo_data() -> List[Dict[str, Any]]:
    """
    Fetches, deduplicates, and processes Domo data into vector records.
    Uses testMode (local CSV via pandasql) or live Domo API depending on _DOMO_CONFIG.

    Returns:
        List[Dict[str, Any]]: List of dictionary records ready for JSONL output.
    """
    print("\n[Domo] Starting data fetch...")

    domoAuthenticate()

    try:
        df = queryDataSet(_DOMO_CONFIG["domoDatasetId"], _DOMO_SQL)
    except Exception as e:
        print(f"[Domo] ERROR fetching data: {e}")
        sys.exit(1)

    print(f"[Domo] Loaded {len(df)} rows.")

    # Deduplicate: Keep most recent month per pod_id
    if 'month' in df.columns:
        df['month'] = pd.to_datetime(df['month'], errors='coerce')
        df = df.sort_values('month', ascending=False)
        df_dedup = df.drop_duplicates(subset=['pod_id'], keep='first')
        print(f"[Domo] Deduplicated: {len(df)} -> {len(df_dedup)} unique pods.")
    else:
        df_dedup = df
        print("[Domo] Warning: No 'month' column found, skipping sort.")

    # Define fields used in the core template
    core_fields = [
        'pod_id', 'pretty_name', 'meau', 'health_score', 
        'risk_ratio_for_next_renewal', 'provisioned_users', 'contracted_licenses'
    ]

    records = []
    for _, row in df_dedup.iterrows():
        # Calculate derived metrics
        expansion = calculate_expansion_signal(
            row.get('provisioned_users'), 
            row.get('contracted_licenses')
        )
        
        # Add to a copy for text generation
        row_copy = row.copy()
        row_copy['Calculated_Expansion'] = expansion
        
        # Generate Embedding Text
        text_content = build_dynamic_embedding_text(row_copy, DOMO_CORE_TEMPLATE, core_fields)
        
        # Generate Metadata
        metadata = {
            "type": "domo",
            "pod_id": safe_str(row.get('pod_id')),
            "pod_name": safe_str(row.get('pretty_name')),
            "health_score": safe_str(row.get('health_score')),
            "expansion_signal": expansion
        }

        # Create Record ID
        safe_id = f"domo_pod_{safe_str(row.get('pod_id'))}"

        records.append({
            "id": safe_id,
            "content": text_content,
            "metadata": metadata
        })

    print(f"[Domo] Generated {len(records)} records.")
    return records


def save_to_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Saves a list of records to a JSONL file.

    Args:
        records (List[Dict[str, Any]]): Processed records.
        output_path (Path): Destination file path.
    """
    print(f"\n[Output] Saving {len(records)} records to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
            
    print(f"[Output] Successfully saved file.")


def main():
    """
    Main orchestration function for Phase 1.
    """
    print("=" * 80)
    print("PHASE 1: Build Golden Records (Refactored)")
    print("=" * 80)

    # Paths
    output_file = Path("nexus_vectors_input.jsonl")

    # Processing
    salesforce_records = process_salesforce_data()
    domo_records = process_domo_data()

    # Combine and Save
    all_records = salesforce_records + domo_records
    save_to_jsonl(all_records, output_file)

    print("\nPhase 1 Complete. Ready for Batch Embedding.")


if __name__ == "__main__":
    main()
