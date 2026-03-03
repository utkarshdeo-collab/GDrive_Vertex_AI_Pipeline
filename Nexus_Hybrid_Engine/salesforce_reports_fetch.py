import os
import pandas as pd
from simple_salesforce import Salesforce

# --------------------------------------------------------------------
# 1. SALESFORCE CONFIG (copied from bulk2.0.py lines 8–12)
# --------------------------------------------------------------------
SF_CONFIG = {
    "username": "utkarsh.deo@antino.com.antinodev",
    "password": "Protein@2026",
    "security_token": "a8SpU2K3EUzVVNDX5Sus4nUtM",
    "domain": "test",
}

# --------------------------------------------------------------------
# 2. REPORT SETTINGS
# --------------------------------------------------------------------
# Put the Salesforce Report Id here (from the URL: 00Oxxxxxxxxxxxx)
REPORT_ID = "00OWe000014tfZB"

# Output directory / file
OUTPUT_DIR = r"d:\Old_Pipeline_Scripts\Salesforce_BulkAPI_Pipeline\symphony_csv"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "salesforce_report_export.csv")
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "salesforce_report_export.xlsx")


def fetch_report_details(sf: Salesforce, report_id: str) -> dict:
    """Run a Salesforce report synchronously and return full JSON (with detail rows)."""
    return sf.restful(
        f"analytics/reports/{report_id}",
        params={"includeDetails": "true"},
    )


def report_json_to_dataframe(report_json: dict) -> pd.DataFrame:
    """
    Convert a simple (non-grouped) tabular report JSON into a pandas DataFrame.
    For grouped/matrix reports this may need adjustment.
    """
    # Column definitions
    detail_columns = report_json["reportMetadata"]["detailColumns"]
    detail_info = report_json["reportExtendedMetadata"]["detailColumnInfo"]

    column_api_names = []
    column_labels = []
    for key in detail_columns:
        column_api_names.append(key)
        label = detail_info.get(key, {}).get("label", key)
        column_labels.append(label)

    # Flat report rows are under factMap['T!T']['rows']
    fact_map = report_json["factMap"]
    if "T!T" not in fact_map:
        raise ValueError("Unexpected report structure: 'T!T' not found in factMap.")
    rows = fact_map["T!T"].get("rows", [])

    # If there are no detail rows, return an empty DataFrame with the
    # expected columns, so the caller can handle the "no data" case cleanly.
    if not rows:
        return pd.DataFrame(columns=column_labels)

    data = []
    for row in rows:
        cells = row["dataCells"]
        record = {}
        for col_name, cell in zip(column_api_names, cells):
            value = cell.get("value")
            if value is None:
                value = cell.get("label")
            record[col_name] = value
        data.append(record)

    df = pd.DataFrame(data, columns=column_labels)
    return df


def main():
    try:
        print("Authenticating to Salesforce for report export...")
        sf = Salesforce(**SF_CONFIG)

        print(f"Running report {REPORT_ID} with includeDetails=true ...")
        report_json = fetch_report_details(sf, REPORT_ID)

        print("Converting report JSON to DataFrame...")
        df = report_json_to_dataframe(report_json)

        if df.empty:
            print("Report returned no rows.")
            return

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print(f"Saving report to CSV: {OUTPUT_CSV}")
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

        print(f"Saving report to Excel: {OUTPUT_XLSX}")
        df.to_excel(OUTPUT_XLSX, index=False)

        print("Report export completed successfully.")

    except Exception as e:
        print(f"System Failure while fetching report: {e}")


if __name__ == "__main__":
    main()

