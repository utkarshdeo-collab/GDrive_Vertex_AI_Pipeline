"""
Phase 1: Build Golden Records from CSV Data

This script:
1. Loads Salesforce and Domo Excel files
2. Cleans and joins data (POD_Internal_Id__c = pod_id)
3. Deduplicates Domo (keeps most recent month per pod)
4. Calculates business logic (Engagement, Expansion Signal)
5. Generates "Golden Record" text for each account
6. Outputs:
   - nexus_analytics.parquet (for in-memory calculations)
   - nexus_vectors_input.jsonl (for embedding in Phase 2)

Run from project root:
  python Nexus_Hybrid_Engine/phase1_build_golden_records.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from google.cloud import storage

# Add parent to path for config import
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nexus_Hybrid_Engine import nexus_config as config


def clean_arr_column(series: pd.Series) -> pd.Series:
    """Clean Total_ARR: remove $, commas, convert to float, NaN → None (for Parquet compatibility)."""
    def clean_value(val):
        if pd.isna(val):
            return None  # Use None instead of "N/A" for Parquet compatibility
        if isinstance(val, str):
            # Remove $ and commas
            val = val.replace("$", "").replace(",", "").strip()
            try:
                return float(val)
            except ValueError:
                return None  # Use None instead of "N/A"
        return float(val)
    
    return series.apply(clean_value)


def calculate_engagement(task_count) -> str:
    """
    Calculate Engagement from Task_Count:
    - >= 4: Positive
    - 1-3: Neutral
    - 0 or NULL: Negative
    """
    if pd.isna(task_count):
        return "Negative"
    try:
        count = int(task_count)
        if count >= config.ENGAGEMENT_POSITIVE_THRESHOLD:
            return "Positive"
        elif config.ENGAGEMENT_NEUTRAL_MIN <= count <= config.ENGAGEMENT_NEUTRAL_MAX:
            return "Neutral"
        else:
            return "Negative"
    except (ValueError, TypeError):
        return "Negative"


def calculate_expansion(provisioned, contracted) -> str:
    """
    Calculate Expansion Signal:
    - provisioned_users > 90% of contracted_licenses: Positive
    - contracted = 0 or NULL: N/A
    - Otherwise: Negative
    """
    if pd.isna(contracted) or contracted == 0:
        return "N/A"
    if pd.isna(provisioned):
        return "N/A"
    
    try:
        prov = float(provisioned)
        cont = float(contracted)
        if cont == 0:
            return "N/A"
        if prov > (config.EXPANSION_THRESHOLD * cont):
            return "Positive"
        else:
            return "Negative"
    except (ValueError, TypeError):
        return "N/A"


def format_date(date_val) -> str:
    """Format date as YYYY-MM-DD or 'N/A'."""
    if pd.isna(date_val):
        return "N/A"
    if isinstance(date_val, str):
        return date_val
    try:
        return pd.to_datetime(date_val).strftime("%Y-%m-%d")
    except:
        return str(date_val)


def safe_str(val, default="N/A") -> str:
    """Convert value to string, handling NaN."""
    if pd.isna(val):
        return default
    return str(val)


def create_golden_record(row: pd.Series) -> str:
    """
    Create Golden Record text from row using template.
    This is the searchable text that will be embedded.
    """
    return config.GOLDEN_RECORD_TEMPLATE.format(
        Customer_Name=safe_str(row.get("Customer_Name")),
        Account_Owner=safe_str(row.get("Account_Owner")),
        Total_ARR=safe_str(row.get("Total_ARR")),
        Renewal_Date=safe_str(row.get("Renewal_Date")),
        Calculated_Engagement=safe_str(row.get("Calculated_Engagement")),
        Calculated_Expansion=safe_str(row.get("Calculated_Expansion")),
        Mapped_ORBIT=safe_str(row.get("Mapped_ORBIT")),
        Mapped_Churn=safe_str(row.get("Mapped_Churn")),
        meau=safe_str(row.get("meau")),
        POD_Internal_Id__c=safe_str(row.get("POD_Internal_Id__c")),
        provisioned_users=safe_str(row.get("provisioned_users")),
        contracted_licenses=safe_str(row.get("contracted_licenses")),
        health_score=safe_str(row.get("health_score")),
        active_users=safe_str(row.get("active_users")),
    )


def main():
    print("\n" + "=" * 80)
    print("  NEXUS HYBRID ENGINE - Phase 1: Build Golden Records")
    print("=" * 80)
    print(f"\n  Project: {config.PROJECT_ID}")
    print(f"  Region:  {config.LOCATION}")
    print(f"  Bucket:  {config.GCS_BUCKET_NAME}")
    
    # Step 1: Load Salesforce data
    print("\n[Step 1] Loading Salesforce data...")
    sf_path = config.DATA_DIR / config.SALESFORCE_FILE
    if not sf_path.exists():
        print(f"  ERROR: File not found: {sf_path}")
        sys.exit(1)
    
    sf_df = pd.read_excel(sf_path)
    print(f"  Loaded {len(sf_df):,} Salesforce records")
    print(f"  Columns: {list(sf_df.columns)}")
    
    # Step 2: Load Domo data
    print("\n[Step 2] Loading Domo data...")
    domo_path = config.DATA_DIR / config.DOMO_FILE
    if not domo_path.exists():
        print(f"  ERROR: File not found: {domo_path}")
        sys.exit(1)
    
    domo_df = pd.read_excel(domo_path)
    print(f"  Loaded {len(domo_df):,} Domo records (multiple months per pod)")
    
    # Step 3: Deduplicate Domo (keep most recent month per pod_id)
    print("\n[Step 3] Deduplicating Domo data (most recent month per pod)...")
    domo_df['month'] = pd.to_datetime(domo_df['month'], errors='coerce')
    domo_df = domo_df.sort_values('month', ascending=False)
    domo_dedup = domo_df.drop_duplicates(subset=['pod_id'], keep='first')
    print(f"  After dedup: {len(domo_dedup):,} unique pods")
    
    # Step 4: Clean Salesforce data
    print("\n[Step 4] Cleaning Salesforce data...")
    sf_df['Total_ARR'] = clean_arr_column(sf_df['Total_ARR'])
    sf_df['Renewal_Date'] = sf_df['Renewal_Date'].apply(format_date)
    print(f"  Cleaned Total_ARR and Renewal_Date")
    
    # Step 5: Merge Salesforce + Domo
    print("\n[Step 5] Merging Salesforce + Domo...")
    merged_df = sf_df.merge(
        domo_dedup,
        left_on='POD_Internal_Id__c',
        right_on='pod_id',
        how='left',
        suffixes=('_sf', '_domo')
    )
    print(f"  Merged: {len(merged_df):,} rows")
    
    # Step 6: Calculate business logic
    print("\n[Step 6] Calculating business logic...")
    merged_df['Calculated_Engagement'] = merged_df['Task_Count'].apply(calculate_engagement)
    merged_df['Calculated_Expansion'] = merged_df.apply(
        lambda row: calculate_expansion(row.get('provisioned_users'), row.get('contracted_licenses')),
        axis=1
    )
    merged_df['Mapped_ORBIT'] = merged_df['health_score']
    merged_df['Mapped_Churn'] = merged_df['risk_ratio_for_next_renewal']
    
    # Show distribution
    print(f"  Engagement distribution:")
    print(merged_df['Calculated_Engagement'].value_counts().to_string())
    print(f"\n  Expansion distribution:")
    print(merged_df['Calculated_Expansion'].value_counts().to_string())
    
    # Step 7: Generate Golden Records
    print("\n[Step 7] Generating Golden Records...")
    merged_df['golden_record'] = merged_df.apply(create_golden_record, axis=1)
    print(f"  Created {len(merged_df):,} Golden Records")
    
    # Show sample
    print(f"\n  Sample Golden Record (first account):")
    print("  " + "-" * 76)
    sample = merged_df.iloc[0]['golden_record']
    for line in sample.split('\n'):
        print(f"  {line}")
    print("  " + "-" * 76)
    
    # Step 8: Save Parquet (Analytics Engine)
    print("\n[Step 8] Saving Parquet file for Analytics Engine...")
    local_parquet = Path("nexus_analytics.parquet")
    merged_df.to_parquet(local_parquet, index=False)
    print(f"  Saved locally: {local_parquet}")
    
    # Upload to GCS
    print(f"  Uploading to GCS: {config.ANALYTICS_PARQUET_GCS}")
    storage_client = storage.Client(project=config.PROJECT_ID)
    try:
        bucket = storage_client.get_bucket(config.GCS_BUCKET_NAME)
    except Exception:
        print(f"  Creating bucket: {config.GCS_BUCKET_NAME}")
        bucket = storage_client.create_bucket(config.GCS_BUCKET_NAME, location=config.LOCATION)
    
    blob_path = f"{config.GCS_ANALYTICS_PREFIX}/{config.ANALYTICS_PARQUET_FILE}"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_parquet)
    print(f"  Uploaded to: gs://{config.GCS_BUCKET_NAME}/{blob_path}")
    
    # Step 9: Save JSONL (Vector Engine input)
    print("\n[Step 9] Saving JSONL file for Vector Engine...")
    local_jsonl = Path("nexus_vectors_input.jsonl")
    
    with open(local_jsonl, 'w', encoding='utf-8') as f:
        for idx, row in merged_df.iterrows():
            record = {
                "id": f"account_{row['POD_Internal_Id__c']}",
                "content": row['golden_record'],
                "metadata": {
                    # Salesforce fields
                    "account_name": safe_str(row.get('Customer_Name')),
                    "owner": safe_str(row.get('Account_Owner')),
                    "calculated_engagement": safe_str(row.get('Calculated_Engagement')),
                    "pod_id": str(row.get('POD_Internal_Id__c')),
                    # Domo fields (for Domo agent filtering)
                    "meau": safe_str(row.get('meau')),
                    "health_score": safe_str(row.get('health_score')),
                    "provisioned_users": safe_str(row.get('provisioned_users')),
                    "contracted_licenses": safe_str(row.get('contracted_licenses')),
                    "active_users": safe_str(row.get('active_users')),
                    "risk_ratio_for_next_renewal": safe_str(row.get('risk_ratio_for_next_renewal')),
                }
            }
            f.write(json.dumps(record) + '\n')
    
    print(f"  Saved locally: {local_jsonl}")
    
    # Upload to GCS
    print(f"  Uploading to GCS: {config.VECTORS_INPUT_GCS}")
    blob_path = f"{config.GCS_EMBEDDINGS_PREFIX}/{config.VECTORS_INPUT_JSONL}"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_jsonl)
    print(f"  Uploaded to: gs://{config.GCS_BUCKET_NAME}/{blob_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("  Phase 1 Complete!")
    print("=" * 80)
    print(f"  Total accounts: {len(merged_df):,}")
    print(f"  Analytics file: {config.ANALYTICS_PARQUET_GCS}")
    print(f"  Vector input:   {config.VECTORS_INPUT_GCS}")
    print("\n  Next: Run Phase 2 to generate embeddings")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
