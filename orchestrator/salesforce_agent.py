"""
Salesforce sub-agent: self-contained BigQuery agent using nexus_data schema.
Schema path: orchestrator/nexus_schema.json (next to this file).
Uses root config.py for project, location, and model.
Exports create_salesforce_agent(credentials) for the orchestrator.
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
SCHEMA_FILE = Path(__file__).resolve().parent / "nexus_schema.json"

# Max rows to return (match typical ADK default)
_MAX_QUERY_ROWS = 1000

# Credentials for BigQuery (set by create_salesforce_agent so tool uses same creds as orchestrator)
_bq_credentials = None


def execute_sql(project_id: str, query: str) -> dict:
    """Run a read-only BigQuery SQL query. Records bytes processed for cost display.
    Use fully qualified names: `project_id.nexus_data.TABLE_NAME`.
    """
    from .usage_collector import record_bigquery
    import google.auth
    try:
        if project_id != config.PROJECT_ID:
            return {
                "status": "ERROR",
                "error_details": (
                    f"Tool is restricted to project {config.PROJECT_ID}. "
                    f"Use project_id={config.PROJECT_ID}."
                ),
            }
        creds = _bq_credentials
        if creds is None:
            creds, _ = google.auth.default()
        # Do not set location so BigQuery uses dataset location (nexus_data may be in US multi-region)
        client = bigquery.Client(project=project_id, credentials=creds)
        # Dry run to enforce SELECT-only
        dry_run_job = client.query(
            query,
            project=project_id,
            job_config=bigquery.QueryJobConfig(dry_run=True),
        )
        if dry_run_job.statement_type != "SELECT":
            return {
                "status": "ERROR",
                "error_details": "Read-only mode only supports SELECT statements.",
            }
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
        result = {"status": "SUCCESS", "rows": rows}
        if len(rows) == _MAX_QUERY_ROWS:
            result["result_is_likely_truncated"] = True
        return result
    except Exception as ex:
        return {"status": "ERROR", "error_details": str(ex)}


def get_schema_context(path: Path = None) -> str:
    """Load nexus_data schema from JSON for the agent instruction (full schema for SQL)."""
    path = path or SCHEMA_FILE
    if not path.exists():
        return "Note: nexus_schema.json not found. Use standard Salesforce naming conventions."
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tables = data.get("datasets", [{}])[0].get("tables", [])
        lines = ["# DATABASE SCHEMA (nexus_data)"]
        for t in tables:
            lines.append(f"## Table: {t['table_id']}")
            for c in t.get("schema", []):
                lines.append(f"- {c['column_name']} ({c['data_type']})")
        return "\n".join(lines)
    except Exception as e:
        return f"Schema load error: {e}"


def create_salesforce_agent(credentials):
    """
    Create the Salesforce/BigQuery sub-agent.
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
    instruction = f"""You answer questions about Salesforce data stored in BigQuery.
{schema_context}
Use the execute_sql tool to query data. Always use fully qualified names: `{config.PROJECT_ID}.nexus_data.TABLE_NAME`.
Return clear summaries and numbers. Do not run write/delete statements."""

    return LlmAgent(
        model=model,
        name="salesforce_agent",
        instruction=instruction,
        tools=[FunctionTool(execute_sql)],
    )
