"""
Domo sub-agent: self-contained BigQuery agent using domo_test_dataset schema.
Schema path: orchestrator/domo_schema.json (next to this file).
Uses root config.py for project, location, and model.
Exports create_domo_agent(credentials) for the orchestrator.
Uses a custom execute_sql tool that records BigQuery bytes for cost display.
"""


import json
import sys
from pathlib import Path

# Ensure project root is on path (for config)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from google.cloud import bigquery
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools.function_tool import FunctionTool

# Schema lives next to this module in orchestrator/
SCHEMA_FILE = Path(__file__).resolve().parent / "domo_schema.json"

# Max rows to return (match typical ADK default)
_MAX_QUERY_ROWS = 1000

# Credentials for BigQuery (set by create_domo_agent so tool uses same creds as orchestrator)
_bq_credentials = None


def execute_sql(project_id: str, query: str) -> dict:
    """Run a read-only BigQuery SQL query. Records bytes processed for cost display.
    Use fully qualified names: `project_id.domo_test_dataset.TABLE_NAME`.
    """
    from .usage_collector import record_bigquery
    from .audit_context import append_audit_entry
    import google.auth
    try:
        if project_id != config.PROJECT_ID:
            err = (
                f"Tool is restricted to project {config.PROJECT_ID}. "
                f"Use project_id={config.PROJECT_ID}."
            )
            append_audit_entry("execute_sql", query, None, err)
            return {"status": "ERROR", "error_details": err}
        creds = _bq_credentials
        if creds is None:
            creds, _ = google.auth.default()
        # Do not set location so BigQuery uses dataset location (domo_test_dataset may be in US multi-region)
        client = bigquery.Client(project=project_id, credentials=creds)
        # Dry run to enforce SELECT-only
        dry_run_job = client.query(
            query,
            project=project_id,
            job_config=bigquery.QueryJobConfig(dry_run=True),
        )
        if dry_run_job.statement_type != "SELECT":
            err = "Read-only mode only supports SELECT statements."
            append_audit_entry("execute_sql", query, None, err)
            return {"status": "ERROR", "error_details": err}
        job = client.query(
            query,
            project=project_id,
        )
        rows_iter = job.result(max_results=_MAX_QUERY_ROWS)
        rows = []
        for row in rows_iter:
            row_values = {}
            for key, val in row.items():
                try:
                    json.dumps(val)
                except (TypeError, ValueError):
                    val = str(val)
                row_values[key] = val
            rows.append(row_values)
        bytes_processed = job.total_bytes_processed or 0
        record_bigquery(bytes_processed)
        append_audit_entry("execute_sql", query, bytes_processed, None)
        result = {"status": "SUCCESS", "rows": rows}
        if len(rows) == _MAX_QUERY_ROWS:
            result["result_is_likely_truncated"] = True
        return result
    except Exception as ex:
        append_audit_entry("execute_sql", query, None, str(ex))
        return {"status": "ERROR", "error_details": str(ex)}


def get_pod_data_by_id(pod_id: int) -> dict:
    """Fetch pod data from test_pod table using pod_id. Returns MEAU, ORBIT score, Churn Risk, and Expansion Signal data.
    Use this when you have a pod_id from Salesforce data and need to get the corresponding Domo metrics."""
    from .usage_collector import record_bigquery
    from .audit_context import append_audit_entry
    import google.auth
    try:
        creds = _bq_credentials
        if creds is None:
            creds, _ = google.auth.default()
        client = bigquery.Client(project=config.PROJECT_ID, credentials=creds)

        q_pod = f"""
        SELECT pretty_name, meau, provisioned_users, active_users, health_score,
               risk_ratio_for_next_renewal, contracted_licenses
        FROM `{config.PROJECT_ID}.domo_test_dataset.test_pod`
        WHERE pod_id = {pod_id}
        ORDER BY `month` DESC
        LIMIT 1
        """
        try:
            job = client.query(q_pod, project=config.PROJECT_ID)
            pod_rows = list(job.result(max_results=1))
            record_bigquery(job.total_bytes_processed or 0)
            append_audit_entry("get_pod_data_by_id", q_pod, job.total_bytes_processed or 0, None)

            if pod_rows:
                pod = pod_rows[0]
                return {
                    "status": "SUCCESS",
                    "pretty_name": pod.get("pretty_name"),
                    "meau": pod.get("meau"),
                    "provisioned_users": pod.get("provisioned_users"),
                    "active_users": pod.get("active_users"),
                    "health_score": pod.get("health_score"),
                    "risk_ratio_for_next_renewal": pod.get("risk_ratio_for_next_renewal"),
                    "contracted_licenses": pod.get("contracted_licenses")
                }
            else:
                return {
                    "status": "NOT_FOUND",
                    "pod_id": pod_id
                }
        except Exception as e:
            append_audit_entry("get_pod_data_by_id", q_pod, None, str(e))
            return {
                "status": "ERROR",
                "error": str(e),
                "pod_id": pod_id
            }
    except Exception as ex:
        append_audit_entry("get_pod_data_by_id", None, None, str(ex))
        return {
            "status": "ERROR",
            "error": str(ex),
            "pod_id": pod_id
        }


def get_schema_context(path: Path = None) -> str:
    """Load domo_test_dataset schema from JSON for the agent instruction (full schema for SQL)."""
    path = path or SCHEMA_FILE
    if not path.exists():
        return "Note: domo_schema.json not found. Use standard Domo naming conventions."
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tables = data.get("datasets", [{}])[0].get("tables", [])
        lines = ["# DATABASE SCHEMA (domo_test_dataset)"]
        for t in tables:
            lines.append(f"## Table: {t['table_id']}")
            for c in t.get("schema", []):
                lines.append(f"- {c['column_name']} ({c['data_type']})")
        return "\n".join(lines)
    except Exception as e:
        return f"Schema load error: {e}"


def create_domo_agent(credentials):
    """
    Create the Domo/BigQuery sub-agent.
    credentials: from google.auth.default(); passed to execute_sql so BigQuery uses same creds.
    """
    global _bq_credentials
    _bq_credentials = credentials
    model = Gemini(
        model_name=config.GEMINI_MODEL,
        project=config.PROJECT_ID,
        location=config.LOCATION,
        vertexai=True,
    )
    schema_context = get_schema_context()
    instruction = f"""You answer questions about Domo data stored in BigQuery.
{schema_context}
Use the execute_sql tool to query data. Always use fully qualified names: `{config.PROJECT_ID}.domo_test_dataset.TABLE_NAME`.
Return clear summaries and numbers. Do not run write/delete statements.

USAGE METRICS (test_pod): For questions about "average daily message sent", "avg_daily_msg_sent", "messages sent" for an account (e.g. Symphony1), query test_pod: SELECT pretty_name, avg_daily_msg_sent FROM `{config.PROJECT_ID}.domo_test_dataset.test_pod` WHERE LOWER(pretty_name) LIKE '%account_name%'. Use pretty_name to match account names."""

    return LlmAgent(
        model=model,
        name="domo_agent",
        instruction=instruction,
        tools=[FunctionTool(execute_sql), FunctionTool(get_pod_data_by_id)],
    )
