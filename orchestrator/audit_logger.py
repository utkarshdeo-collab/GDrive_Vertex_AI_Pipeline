"""
Audit logger: write orchestrator audit rows to BigQuery.
Creates dataset and table if they do not exist (same region as config).
One row per tool invocation: timestamp, user_question, assistant_response, tool_call,
sql_generated, turn_id, bigquery_bytes_processed, session_id, routing_hints, error_messages.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from google.cloud import bigquery


# Table schema: one row per tool invocation
AUDIT_SCHEMA = [
    bigquery.SchemaField("timestamp_utc", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("user_question", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("assistant_response", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("tool_call", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("sql_generated", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("turn_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("bigquery_bytes_processed", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("session_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("routing_hints", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("error_messages", "STRING", mode="NULLABLE"),
]


def ensure_audit_dataset_and_table(
    project_id: str,
    dataset_id: str,
    table_id: str,
    location: str,
    credentials=None,
) -> None:
    """Create the audit dataset and table if they do not exist. Same region as other data."""
    client = bigquery.Client(project=project_id, credentials=credentials)
    full_dataset_id = f"{project_id}.{dataset_id}"
    try:
        client.get_dataset(full_dataset_id)
    except Exception:
        dataset = bigquery.Dataset(full_dataset_id)
        dataset.location = location
        client.create_dataset(dataset)
    full_table_id = f"{full_dataset_id}.{table_id}"
    try:
        client.get_table(full_table_id)
    except Exception:
        table = bigquery.Table(full_table_id, schema=AUDIT_SCHEMA)
        client.create_table(table)


def write_audit_rows(
    rows: List[Dict[str, Any]],
    project_id: str,
    dataset_id: str,
    table_id: str,
    credentials=None,
) -> None:
    """Insert audit rows into BigQuery. On failure, log and do not raise."""
    if not rows:
        return
    try:
        client = bigquery.Client(project=project_id, credentials=credentials)
        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        errors = client.insert_rows_json(table_ref, rows)
        if errors:
            print(f"\n  [Audit] BigQuery insert_rows_json reported errors: {errors}", flush=True)
    except Exception as e:
        print(f"\n  [Audit] Failed to write to BigQuery: {e}", flush=True)


def build_audit_rows(
    timestamp_utc: datetime,
    user_question: str,
    assistant_response: str,
    turn_id: str,
    session_id: Optional[str],
    routing_hints: Optional[str],
    entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build one BigQuery row per audit entry. If entries is empty, return one row with tool_call='orchestrator'."""
    ts_str = timestamp_utc.isoformat()
    if not entries:
        return [{
            "timestamp_utc": ts_str,
            "user_question": user_question,
            "assistant_response": assistant_response,
            "tool_call": "orchestrator",
            "sql_generated": None,
            "turn_id": turn_id,
            "bigquery_bytes_processed": None,
            "session_id": session_id,
            "routing_hints": routing_hints,
            "error_messages": None,
        }]
    rows = []
    for e in entries:
        rows.append({
            "timestamp_utc": ts_str,
            "user_question": user_question,
            "assistant_response": assistant_response,
            "tool_call": e.get("tool_call", ""),
            "sql_generated": e.get("sql_generated"),
            "turn_id": turn_id,
            "bigquery_bytes_processed": e.get("bigquery_bytes_processed"),
            "session_id": session_id,
            "routing_hints": routing_hints,
            "error_messages": e.get("error_messages"),
        })
    return rows
