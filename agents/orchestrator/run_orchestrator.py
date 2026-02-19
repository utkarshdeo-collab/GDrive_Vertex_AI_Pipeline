"""
Orchestrator Agent (Nexus Hybrid Engine)
========================================

The Orchestrator acts as the Master Agent (Router) for the Nexus Hybrid Engine.
It uses a single Gemini model to route user queries to the appropriate sub-agent:
1.  Salesforce Agent: For account metadata, pipeline, and contract info.
2.  Domo Agent: For product usage metrics and health scores.

Features:
-   **Orchestrated Workflow**: Handles "Account Snapshot" requests by chaining
    Salesforce lookup -> Pod ID extraction -> Domo lookup -> Unified Response.
-   **Routing Hints**: Pre-processes user input to strongly guide the LLM
    towards the correct tool/agent.
-   **Robust Imports**: Handles execution both as a module and a script.

Run usage:
    python agents/orchestrator/run_orchestrator.py
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --- Configuration Import Logic ---
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    # Enable importing Nexus_Hybrid_Engine if running from agents/orchestrator
    sys.path.insert(0, str(_ROOT))

try:
    from Nexus_Hybrid_Engine import nexus_config as config
except ImportError:
    # Fallback to local import if needed
    sys.path.append(str(_ROOT / "Nexus_Hybrid_Engine"))
    import nexus_config as config

# --- Third-Party Imports ---
import google.auth
from google.auth.transport import requests as google_requests
from google.genai import types
import vertexai

# --- ADK Imports ---
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.apps.app import App
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from google.adk.utils.context_utils import Aclosing
from google.adk.tools.function_tool import FunctionTool

# --- Local Imports (Robust Handling) ---
# Add the directory containing this script to sys.path to allow sibling imports
_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

try:
    from salesforce_agent import create_salesforce_agent, get_salesforce_account_data
    from domo_agent import create_domo_agent, get_pod_data_by_id
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    print("Ensure salesforce_agent.py and domo_agent.py are in the same folder.")
    sys.exit(1)


# --- Constants & Keywords ---

SALESFORCE_SCHEMA_FILE = Path(__file__).resolve().parent / "nexus_schema.json"
DOMO_SCHEMA_FILE = Path(__file__).resolve().parent / "domo_schema.json"

_SALESFORCE_KEYWORDS = (
    "arr", "pipeline", "opportunity", "opportunities", "customer", "customers",
    "account", "accounts", "salesforce", "nexus_data",
    "total_arr", "customer_name", "contract", "renewal", "license", "licenses",
    "stage", "closed won", "close date", "owner", "contracted", "antino bank",
    "abc capital", "sf_account", "sf_opportunity",
    "all accounts", "commercial snapshot",
)

_DOMO_KEYWORDS = (
    "domo", "domo_test_dataset", "domo_test", "domo data", "domo dataset",
    "health score", "health_score", "num_total_accounts",
    "total accounts", "accounts owned", "owned by",
    "how many total accounts", "how many accounts",
    "average daily message", "avg_daily_msg_sent", "avg daily message",
    "message sent", "daily message", "messages sent",
    "active users", "provisioned users", "power users",
    "test_pod", "pretty_name", "att_sent",
)

_DOCUMENT_KEYWORDS = (
    "document", "pdf", "report", "case study", "implementation cost", "total implementation",
    "change management", "budget", "post-implementation", "lessons learned", "executive summary",
    "milestone", "phase 3", "phase 2", "technology stack", "financial benefits", "readmission",
    "telehealth", "resolution strategy", "risk", "mitigate", "recommendation from the document",
    "sym_1pgr", "1pgr_", "1pgr ", "insurtech", "ark capital", "symphony for wealth", "symphony for insurance",
    "federation", "insurance giants", "wealth management",
)


# --- Helper Functions ---

def _is_likely_salesforce_question(text: str) -> bool:
    lower = text.lower().strip()
    return any(kw in lower for kw in _SALESFORCE_KEYWORDS)


def _is_likely_document_question(text: str) -> bool:
    lower = text.lower().strip()
    return any(kw in lower for kw in _DOCUMENT_KEYWORDS)


def _is_likely_domo_question(text: str) -> bool:
    lower = text.lower().strip()
    return any(kw in lower for kw in _DOMO_KEYWORDS)


def _maybe_add_routing_hint(user_message: str) -> str:
    """
    Prepend a routing hint to the user message to guide the LLM's tool selection.
    This acts as a 'System 1' classifier before the 'System 2' LLM processing.

    Args:
        user_message (str): The raw input from the user.

    Returns:
        str: The message potentially prefixed with [ROUTING: ...] instructions.
    """
    if _is_likely_document_question(user_message):
        return (
            "[ROUTING: This question is about Documents/PDFs. "
            "NOTE: The PDF Agent is currently DISABLED in config. "
            "Only answer if general knowledge suffices.]\n\n"
            + user_message
        )
    if _is_likely_domo_question(user_message):
        return (
            "[ROUTING: This question is about Domo data. "
            "You MUST delegate to domo_agent only.]\n\n"
            + user_message
        )
    if _is_likely_salesforce_question(user_message):
        return (
            "[ROUTING: This question is about Salesforce data. "
            "You MUST delegate to salesforce_agent only.]\n\n"
            + user_message
        )
    return user_message


def get_routing_data_dictionary(salesforce_path: Path, domo_path: Path) -> str:
    """
    Build a concise data dictionary from schema files to include in the system prompt.
    This helps the Master Agent understand what data matches which sub-agent.
    """
    lines = []

    # Salesforce schema
    if salesforce_path.exists():
        try:
            with open(salesforce_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tables = data.get("datasets", [{}])[0].get("tables", [])
            lines.append("# Salesforce data -> use salesforce_agent")
            for t in tables:
                cols = [c["column_name"] for c in t.get("schema", [])]
                key_cols = cols[:15] if len(cols) > 15 else cols
                lines.append(f"- {t['table_id']}: {', '.join(key_cols)}")
        except Exception as e:
            lines.append(f"Salesforce data schema error: {e}")

    lines.append("")

    # Domo schema
    if domo_path.exists():
        try:
            with open(domo_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tables = data.get("datasets", [{}])[0].get("tables", [])
            lines.append("# Domo data -> use domo_agent")
            for t in tables:
                cols = [c["column_name"] for c in t.get("schema", [])]
                key_cols = cols[:15] if len(cols) > 15 else cols
                lines.append(f"- {t['table_id']}: {', '.join(key_cols)}")
        except Exception as e:
            lines.append(f"Domo data schema error: {e}")

    return "\n".join(lines)


# --- Orchestrated Workflow Tools ---

def get_nexus_account_snapshot_orchestrated(account_name: str) -> str:
    """
    Orchestrate fetching a complete Account Snapshot.
    Combines Salesforce data (metadata) and Domo data (usage metrics).

    Workflow:
    1.  Call Salesforce Agent -> Get Metadata & Pod ID.
    2.  Call Domo Agent -> Get Pod Metrics using the Pod ID.
    3.  Merge & Format -> Return strict template.

    Args:
        account_name (str): Name of the account (e.g., 'Antino Bank').

    Returns:
        str: Formatted Snapshot text or error message.
    """
    try:
        # Step 1: Get Salesforce data (including pod_id)
        sf_result = get_salesforce_account_data(account_name)

        if sf_result.get("status") != "SUCCESS":
            return f"Error fetching Salesforce data: {sf_result.get('error', 'Account not found')}"

        sf_record = sf_result.get("data", {})
        sf_meta = sf_record.get("metadata", {})

        # Step 2: Get pod_id and call domo_agent
        pod_id = sf_meta.get("pod_id")
        pod_data = {}
        domo_meta = {}

        if pod_id:
            pod_id_str = str(pod_id).strip()
            # Basic validation
            if pod_id_str and pod_id_str.upper() not in ("NULL", "NONE", "", "NAN"):
                try:
                    # Call Domo Agent (Direct GCS Lookup)
                    domo_result = get_pod_data_by_id(pod_id_str)
                    if domo_result.get("status") == "SUCCESS":
                        pod_data = domo_result.get("data", {})
                        domo_meta = pod_data.get("metadata", {})
                    else:
                        print(f"Domo lookup failed/not found: {domo_result}")
                except Exception as e:
                    print(f"Domo lookup exception: {e}")

        # Step 3: Format Output (Strict User Template)
        client_name = sf_meta.get("account_name", account_name)

        # Helper to extract from raw content text if missing in metadata
        def _extract_field(content: Optional[str], field_name: str) -> str:
            import re
            if not content:
                return "N/A"
            match = re.search(f"\\[{field_name.upper()}\\] (.*?)(?:\n|$)", content)
            return match.group(1).strip() if match else "N/A"

        # Commercial Fields
        arr = sf_meta.get("total_arr", "N/A")
        owner = sf_meta.get("owner", "N/A")
        renewal = _extract_field(sf_record.get("content", ""), "Renewal_Date")

        parts = []
        parts.append(f"Nexus Account Snapshot: {client_name}")
        parts.append(f"• Commercial: {arr} ARR | Renewal: {renewal} | Owner: {owner}")
        parts.append("• Adoption: N/A")

        # MEAU (from Domo)
        meau = domo_meta.get("meau", "N/A")
        parts.append(f"• MEAU: {meau}")

        parts.append("• Support: N/A (no Jira linkage in data)")

        # Engagement (from Salesforce)
        task_count = _extract_field(sf_record.get("content", ""), "Task_Count")
        sentiment = sf_meta.get("engagement", "N/A")
        parts.append("• Engagement:")
        parts.append(f"    Task Count: {task_count}")
        parts.append(f"    Sentiment: {sentiment}")

        # Summary & Insights (from Domo)
        parts.append("• Summary & Insights:")

        orbit = domo_meta.get("health_score", "N/A")

        # Risk Ratio
        risk_raw = domo_meta.get("risk_ratio_for_next_renewal", "N/A")
        if risk_raw == "N/A":
            risk_raw = _extract_field(pod_data.get("content", ""), "risk_ratio_for_next_renewal")

        # Format Risk % if numeric
        try:
            risk_val = float(risk_raw)
            if 0 <= risk_val <= 1:
                risk_fmt = f"{risk_val * 100:.1f}%"
            else:
                risk_fmt = str(risk_raw)
        except (ValueError, TypeError):
            risk_fmt = str(risk_raw)

        parts.append(f"    • ORBIT Score: {orbit}")
        parts.append(f"    • Churn Risk: {risk_fmt} (risk_ratio_for_next_renewal)")
        parts.append("    • Support Escalation: N/A")

        expansion = domo_meta.get("expansion_signal", "N/A")
        parts.append(f"    • Expansion Signal: {expansion}")
        parts.append("    • Suggested Action: N/A")

        return "\n".join(parts)

    except Exception as ex:
        return f"Error fetching Nexus Account Snapshot: {ex}"


# --- Agent Builder ---

def build_agents(credentials, routing_data_dict: str):
    """
    Construct the Master Orchestrator Agent and its Sub-Agents.
    """
    model = Gemini(
        model_name=config.GEMINI_MODEL,
        project=config.PROJECT_ID,
        location=config.LOCATION,
        vertexai=True,
    )

    # Sub-agents
    salesforce_agent = create_salesforce_agent(credentials)
    domo_agent = create_domo_agent(credentials)

    # Master agent instruction
    master_instruction = f"""You are the master assistant. Route each user question to the specialists.

{routing_data_dict}

ACCOUNT SNAPSHOT REQUESTS (Orchestrated Flow):
- If the user asks for a SNAPSHOT, OVERVIEW, or ACCOUNT INFO for a specific client (e.g. "snapshot for Antino Bank", "tell me about ABC Capital"):
  You MUST use the **get_nexus_account_snapshot_orchestrated(account_name="...")** tool.
  This tool handles the full Salesforce -> Domo lookup process.

OTHER ROUTING:
- If question is about General Salesforce Data (pipeline, opportunities) -> salesforce_agent.
- If question is about General Domo Data (usage metrics) -> domo_agent.
- If question is about PDFs/Documents -> Explain that PDF search is temporarily disabled.

After you get the answer, present it clearly to the user.
"""

    # Build tools list
    tools = [
        FunctionTool(get_nexus_account_snapshot_orchestrated),
    ]

    master_agent = LlmAgent(
        model=model,
        name="orchestrator",
        instruction=master_instruction,
        tools=tools,
        sub_agents=[salesforce_agent, domo_agent],
    )

    return master_agent


# --- Main Application Loop ---

async def main():
    """
    Main entry point for the CLI application.
    Initializes environment, authenticates, and runs the agent loop.
    """
    print("\n" + "=" * 60)
    print("  ORCHESTRATOR — Nexus Hybrid Engine (GCS Lookup Activated)")
    print("=" * 60)
    print(f"  Project: {config.PROJECT_ID}")

    # Initialize Vertex AI environment variables
    os.environ["GOOGLE_CLOUD_PROJECT"] = config.PROJECT_ID
    os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = config.PROJECT_ID
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"

    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)

    # Authentication
    try:
        credentials, _ = google.auth.default()
        if not credentials.valid:
            credentials.refresh(google_requests.Request())
    except Exception as e:
        print(f"  Authentication failed: {e}")
        sys.exit(1)

    # Setup Session and App
    session_id = str(uuid.uuid4())
    routing_data_dict = get_routing_data_dictionary(SALESFORCE_SCHEMA_FILE, DOMO_SCHEMA_FILE)
    root_agent = build_agents(credentials, routing_data_dict)

    app = App(name="orchestrator_app", root_agent=root_agent)
    runner = Runner(
        app=app,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        credential_service=InMemoryCredentialService(),
    )

    session = await runner.session_service.create_session(
        app_name="orchestrator_app",
        user_id="user",
    )

    print("\n  Ready. Ask: 'Show me snapshot for Antino Bank'")
    print("  Type 'exit' to quit.\n")

    # Interactive Loop
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue

            message_to_send = _maybe_add_routing_hint(user_input)

            msg = types.Content(role="user", parts=[types.Part(text=message_to_send)])
            print("Assistant: ", end="", flush=True)

            async with Aclosing(
                runner.run_async(
                    user_id="user",
                    session_id=session.id,
                    new_message=msg,
                )
            ) as stream:
                async for event in stream:
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                print(part.text, end="", flush=True)
                            if getattr(part, "function_call", None):
                                print(f"\n  [Calling {getattr(part.function_call, 'name', 'tool')}...]", flush=True)

            print("")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n  Error: {e}")


# Global handles for ADK web loader
root_agent = None
app = None

if __name__ == "__main__":
    asyncio.run(main())
