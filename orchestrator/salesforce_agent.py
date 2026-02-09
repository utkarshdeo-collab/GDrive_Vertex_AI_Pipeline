"""
Salesforce sub-agent: self-contained BigQuery agent using nexus_data schema.
Schema path: orchestrator/nexus_schema.json (next to this file).
Uses root config.py for project, location, and model.
Exports create_salesforce_agent(credentials) for the orchestrator.
Uses a custom execute_sql tool that records BigQuery bytes for cost display.
"""
import json
import sys
from datetime import date, datetime
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


def _engagement_from_task_count(task_count_val) -> str:
    """Compute Engagement string from Task_Count. If Task_Count >= 4: 'Sentiment: Positive.', if 0: 'Sentiment: Negative.', if 1-3: 'Sentiment: Neutral.'"""
    if task_count_val is None:
        return "Engagement: N/A"
    try:
        task_count = int(float(str(task_count_val).strip()))
        if task_count >= 4:
            return f"Engagement: Task Count: {task_count}. Sentiment: Positive."
        elif task_count == 0:
            return f"Engagement: Task Count: {task_count}. Sentiment: Negative."
        elif 1 <= task_count <= 3:
            return f"Engagement: Task Count: {task_count}. Sentiment: Neutral."
        else:
            return "Engagement: N/A"
    except (ValueError, TypeError, AttributeError):
        return "Engagement: N/A"


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


def get_salesforce_account_data(account_name: str) -> dict:
    """Fetch Salesforce account data from test_dataset2 and return pod_id for Domo lookup.
    Returns a dict with Salesforce data and pod_id. Use this when you need to get pod_id to query Domo agent."""
    from .usage_collector import record_bigquery
    import google.auth
    try:
        creds = _bq_credentials
        if creds is None:
            creds, _ = google.auth.default()
        client = bigquery.Client(project=config.PROJECT_ID, credentials=creds)

        # Query test_dataset2 for Salesforce data
        safe_name = account_name.replace("'", "''")
        q_account = f"""
        SELECT Customer_Name, Total_ARR, Renewal_Date, Account_Owner, POD_Internal_Id__c, Task_Count
        FROM `{config.PROJECT_ID}.nexus_data.test_dataset2`
        WHERE LOWER(TRIM(Customer_Name)) LIKE LOWER(TRIM('%{safe_name}%'))
        LIMIT 1
        """
        commercial_rows = []
        pod_internal_id = None
        try:
            job = client.query(q_account, project=config.PROJECT_ID)
            commercial_rows = list(job.result(max_results=1))
            record_bigquery(job.total_bytes_processed or 0)
            if commercial_rows:
                pod_internal_id = commercial_rows[0].get("POD_Internal_Id__c")
        except Exception:
            pass

        # Return structured data
        if commercial_rows:
            r = commercial_rows[0]
            return {
                "status": "SUCCESS",
                "customer_name": str(r.get("Customer_Name") or account_name),
                "total_arr": r.get("Total_ARR"),
                "renewal_date": r.get("Renewal_Date"),
                "account_owner": r.get("Account_Owner"),
                "task_count": r.get("Task_Count"),
                "pod_id": pod_internal_id  # This will be passed to domo_agent
            }
        else:
            return {
                "status": "NOT_FOUND",
                "customer_name": account_name,
                "pod_id": None
            }
    except Exception as ex:
        return {
            "status": "ERROR",
            "error": str(ex),
            "pod_id": None
        }


def get_nexus_account_snapshot(account_name: str) -> str:
    """Fetch Nexus Account Snapshot for a specific account. 
    This function orchestrates: 
    1. Gets Salesforce data from test_dataset2 (including pod_id)
    2. Calls domo_agent's get_pod_data_by_id to fetch pod metrics
    3. Merges and formats the combined data
    
    Returns formatted snapshot in the Nexus Account Snapshot format. 
    Use when user asks for account snapshot/overview for a specific account 
    (e.g. 'account for ABC Capital', 'snapshot for Antino Bank')."""
    try:
        # Step 1: Get Salesforce data (including pod_id)
        salesforce_data = get_salesforce_account_data(account_name)
        
        if salesforce_data.get("status") != "SUCCESS":
            return f"Error fetching Salesforce data: {salesforce_data.get('error', 'Account not found')}"
        
        # Step 2: Get pod_id and call domo_agent's function
        pod_id = salesforce_data.get("pod_id")
        pod_data = {}
        
        if pod_id is not None:
            pod_id_str = str(pod_id).strip()
            if pod_id_str and pod_id_str.upper() not in ("NULL", "NONE", ""):
                try:
                    # Import domo_agent's function
                    from .domo_agent import get_pod_data_by_id
                    pod_id_int = int(float(pod_id_str))
                    pod_data_result = get_pod_data_by_id(pod_id_int)
                    
                    if pod_data_result.get("status") == "SUCCESS":
                        pod_data = pod_data_result
                except (ValueError, TypeError, ImportError):
                    # If import or conversion fails, pod_data remains empty
                    pass
        
        # Step 3: Format and merge the data
        client_name = salesforce_data.get("customer_name", account_name)
        parts = []

        # Commercial (from Salesforce)
        arr = salesforce_data.get("total_arr")
        renewal = salesforce_data.get("renewal_date") or "N/A"
        owner = salesforce_data.get("account_owner") or "N/A"
        arr_str = f"${int(arr):,}" if arr is not None and arr != "" else "N/A"
        parts.append(f"• Commercial: {arr_str} ARR | Renewal: {renewal} | Owner: {owner}")

        # Adoption — N/A for now
        parts.append("• Adoption: N/A")

        # MEAU — from Domo pod data
        meau_val = pod_data.get("meau")
        if meau_val is not None:
            parts.append(f"• MEAU: {meau_val}")
        else:
            parts.append("• MEAU: N/A")

        # Support placeholder
        parts.append("• Support: N/A (no Jira linkage in data)")

        # Engagement from Task_Count (from Salesforce)
        task_count = salesforce_data.get("task_count")
        parts.append(f"• {_engagement_from_task_count(task_count)}")

        # Summary & Insights — ORBIT score = health_score from test_pod (Domo)
        orbit = pod_data.get("health_score")
        risk = pod_data.get("risk_ratio_for_next_renewal")
        
        # Expansion Signal: provisioned_users > 90% of contracted_licenses from test_pod
        expansion_signal = "N/A"
        provisioned_users = pod_data.get("provisioned_users")
        contracted_licenses = pod_data.get("contracted_licenses")
        # Check if both values exist and are valid numbers
        if provisioned_users is not None and contracted_licenses is not None:
            try:
                provisioned_users_val = float(provisioned_users)
                contracted_licenses_val = float(contracted_licenses)
                # Only calculate if contracted_licenses > 0 (avoid division by zero)
                if contracted_licenses_val > 0:
                    if provisioned_users_val > (0.9 * contracted_licenses_val):
                        expansion_signal = "Positive"
                    else:
                        expansion_signal = "Negative"
                elif contracted_licenses_val == 0 and provisioned_users_val > 0:
                    # If contracted is 0 but provisioned > 0, it's positive
                    expansion_signal = "Positive"
                elif contracted_licenses_val == 0:
                    # If both are 0, it's negative
                    expansion_signal = "Negative"
            except (ValueError, TypeError, AttributeError):
                # If conversion fails, keep as N/A
                pass
        
        insights = []
        # ORBIT score = health_score from test_pod
        if orbit is not None:
            insights.append(f"○ ORBIT score: {orbit}")
        else:
            insights.append("○ ORBIT score: N/A")
        # Churn Risk = risk_ratio_for_next_renewal from test_pod
        if risk is not None:
            # Format as percentage if it's a decimal (0-1), otherwise show as-is
            if isinstance(risk, (int, float)):
                if 0 <= risk <= 1:
                    risk_pct = risk * 100
                    insights.append(f"○ Churn Risk: {risk_pct:.1f}% (risk_ratio_for_next_renewal)")
                else:
                    insights.append(f"○ Churn Risk: {risk} (risk_ratio_for_next_renewal)")
            else:
                insights.append(f"○ Churn Risk: {risk} (risk_ratio_for_next_renewal)")
        else:
            insights.append("○ Churn Risk: N/A")
        insights.append("○ Support Escalation: N/A")
        # Expansion Signal: provisioned_users > 90% of contracted_licenses from test_pod
        insights.append(f"○ Expansion Signal: {expansion_signal}")
        insights.append("○ Suggested Action: N/A")
        parts.append("• Summary & Insights:")
        parts.append("\n  ".join(insights))

        return f"Nexus Account Snapshot: {client_name}\n\n" + "\n".join(parts)
    except Exception as ex:
        return f"Error fetching Nexus Account Snapshot: {ex}"


def _format_single_snapshot(client_name: str, account_row: dict, pod_row: dict) -> str:
    """Format one account's snapshot (Commercial, MEAU, ORBIT, Churn Risk, Expansion Signal)."""
    parts = []
    arr = account_row.get("Total_ARR")
    renewal = account_row.get("Renewal_Date") or "N/A"
    owner = account_row.get("Account_Owner") or "N/A"
    arr_str = f"${int(arr):,}" if arr is not None and arr != "" else "N/A"
    parts.append(f"• Commercial: {arr_str} ARR | Renewal: {renewal} | Owner: {owner}")
    parts.append("• Adoption: N/A")
    meau_val = pod_row.get("meau") if pod_row else None
    parts.append(f"• MEAU: {meau_val}" if meau_val is not None else "• MEAU: N/A")
    parts.append("• Support: N/A (no Jira linkage in data)")
    task_count = account_row.get("Task_Count")
    parts.append(f"• {_engagement_from_task_count(task_count)}")
    orbit = pod_row.get("health_score") if pod_row else None
    risk = pod_row.get("risk_ratio_for_next_renewal") if pod_row else None
    
    # Expansion Signal: provisioned_users > 90% of contracted_licenses from test_pod
    expansion_signal = "N/A"
    if pod_row:
        provisioned_users = pod_row.get("provisioned_users")
        contracted_licenses = pod_row.get("contracted_licenses")
        # Check if both values exist and are valid numbers
        if provisioned_users is not None and contracted_licenses is not None:
            try:
                provisioned_users_val = float(provisioned_users)
                contracted_licenses_val = float(contracted_licenses)
                # Only calculate if contracted_licenses > 0 (avoid division by zero)
                if contracted_licenses_val > 0:
                    if provisioned_users_val > (0.9 * contracted_licenses_val):
                        expansion_signal = "Positive"
                    else:
                        expansion_signal = "Negative"
                elif contracted_licenses_val == 0 and provisioned_users_val > 0:
                    # If contracted is 0 but provisioned > 0, it's positive
                    expansion_signal = "Positive"
                elif contracted_licenses_val == 0:
                    # If both are 0, it's negative
                    expansion_signal = "Negative"
            except (ValueError, TypeError, AttributeError):
                # If conversion fails, keep as N/A
                pass
    
    insights = []
    # ORBIT score = health_score from test_pod
    insights.append(f"○ ORBIT score: {orbit}" if orbit is not None else "○ ORBIT score: N/A")
    # Churn Risk = risk_ratio_for_next_renewal from test_pod
    if risk is not None and isinstance(risk, (int, float)):
        # Format as percentage if it's a decimal (0-1), otherwise show as-is
        if 0 <= risk <= 1:
            risk_pct = risk * 100
            insights.append(f"○ Churn Risk: {risk_pct:.1f}% (risk_ratio_for_next_renewal)")
        else:
            insights.append(f"○ Churn Risk: {risk} (risk_ratio_for_next_renewal)")
    elif risk is not None:
        insights.append(f"○ Churn Risk: {risk} (risk_ratio_for_next_renewal)")
    else:
        insights.append("○ Churn Risk: N/A")
    insights.extend(["○ Support Escalation: N/A", f"○ Expansion Signal: {expansion_signal}", "○ Suggested Action: N/A"])
    parts.append("• Summary & Insights:")
    parts.append("\n  ".join(insights))
    return f"Nexus Account Snapshot: {client_name}\n\n" + "\n".join(parts)


def get_all_nexus_account_snapshots() -> str:
    """Fetch Nexus Account Snapshot for ALL accounts. Queries test_dataset2 and maps to test_pod
    via POD_Internal_Id__c = pod_id. Returns full snapshot (Commercial, MEAU, ORBIT, Churn Risk, Expansion Signal)
    for each account. Use when user asks for 'snapshot of all accounts', 'all accounts snapshot',
    'give me all accounts', etc."""
    from .usage_collector import record_bigquery
    import google.auth
    try:
        creds = _bq_credentials
        if creds is None:
            creds, _ = google.auth.default()
        client = bigquery.Client(project=config.PROJECT_ID, credentials=creds)

        # 1) All accounts from test_dataset2
        q_account = f"""
        SELECT Customer_Name, Total_ARR, Renewal_Date, Account_Owner, POD_Internal_Id__c, Task_Count
        FROM `{config.PROJECT_ID}.nexus_data.test_dataset2`
        ORDER BY Customer_Name
        """
        account_rows = []
        try:
            job = client.query(q_account, project=config.PROJECT_ID)
            account_rows = list(job.result(max_results=500))
            record_bigquery(job.total_bytes_processed or 0)
        except Exception as ex:
            return f"Error fetching accounts: {ex}"

        if not account_rows:
            return "No accounts found in test_dataset2."

        # 2) Collect valid pod_ids and fetch test_pod data
        pod_ids = []
        for r in account_rows:
            pid = r.get("POD_Internal_Id__c")
            if pid is not None and str(pid).strip():
                try:
                    pod_ids.append(int(float(str(pid).strip())))
                except (ValueError, TypeError):
                    pass
        pod_id_to_row = {}
        if pod_ids:
            ids_str = ",".join(str(i) for i in pod_ids)
            q_pod = f"""
            SELECT pod_id, meau, health_score, risk_ratio_for_next_renewal, provisioned_users, contracted_licenses
            FROM (
                SELECT pod_id, meau, health_score, risk_ratio_for_next_renewal, provisioned_users, contracted_licenses, `month`,
                       ROW_NUMBER() OVER (PARTITION BY pod_id ORDER BY `month` DESC) as rn
                FROM `{config.PROJECT_ID}.domo_test_dataset.test_pod`
                WHERE pod_id IN ({ids_str})
            )
            WHERE rn = 1
            """
            try:
                job = client.query(q_pod, project=config.PROJECT_ID)
                for row in job.result(max_results=500):
                    pod_id_to_row[row["pod_id"]] = dict(row)
                record_bigquery(job.total_bytes_processed or 0)
            except Exception:
                pass

        # 3) Format each account
        outputs = []
        for r in account_rows:
            client_name = str(r.get("Customer_Name") or "Unknown")
            pod_id_val = r.get("POD_Internal_Id__c")
            pod_row = None
            if pod_id_val is not None and str(pod_id_val).strip():
                try:
                    pid = int(float(str(pod_id_val).strip()))
                    pod_row = pod_id_to_row.get(pid)
                except (ValueError, TypeError):
                    pass
            outputs.append(_format_single_snapshot(client_name, r, pod_row))

        return "\n\n---\n\n".join(outputs)
    except Exception as ex:
        return f"Error fetching all Nexus Account Snapshots: {ex}"


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

GENERAL RULES:
- Use the execute_sql tool to run READ-ONLY SQL (SELECT only).
- Always use fully qualified table names: `{config.PROJECT_ID}.nexus_data.TABLE_NAME`.
- Never run INSERT, UPDATE, DELETE, or DDL statements.

INTENT: ACCOUNT vs SNAPSHOT
- **Single-account intent**: User asks about one specific account by name. → Use get_nexus_account_snapshot for full format (Commercial + MEAU + ORBIT + Churn + Expansion Signal).
- **All-accounts snapshot intent**: User asks for "snapshot of all accounts", "all accounts snapshot", "give me all accounts", "account overview for everyone". → Use **get_all_nexus_account_snapshots** (NOT execute_sql). This returns the full Nexus format for each account, with data from test_dataset2 and test_pod mapped via POD_Internal_Id__c = pod_id.

SINGLE-ACCOUNT (one account by name):
- Use **get_nexus_account_snapshot(account_name="...")** — this function orchestrates the flow:
  1. Calls get_salesforce_account_data to fetch Salesforce data from test_dataset2 (returns pod_id)
  2. Calls domo_agent's get_pod_data_by_id with the pod_id to fetch Domo metrics
  3. Merges and formats the combined data
- The function returns the full format with Engagement based on Task_Count and Expansion Signal based on provisioned_users vs contracted_licenses.
- Do NOT use execute_sql for single-account snapshot.

ALL ACCOUNTS SNAPSHOT:
- Use **get_all_nexus_account_snapshots()** — it fetches all accounts from test_dataset2, maps each to test_pod via POD_Internal_Id__c = pod_id, and returns the full Nexus Account Snapshot format (Commercial, MEAU, ORBIT score, Churn Risk, Expansion Signal) for each account.
- Do NOT use execute_sql for "snapshot of all accounts" or "all accounts" — get_all_nexus_account_snapshots provides the correct data with proper pod mapping.

EXECUTE_SQL (other queries only):
- Use execute_sql only for ad-hoc questions that are NOT account snapshot requests (e.g. pipeline totals, opportunity counts, custom queries).
"""

    return LlmAgent(
        model=model,
        name="salesforce_agent",
        instruction=instruction,
        tools=[FunctionTool(execute_sql), FunctionTool(get_salesforce_account_data), FunctionTool(get_nexus_account_snapshot), FunctionTool(get_all_nexus_account_snapshots)],
    )
