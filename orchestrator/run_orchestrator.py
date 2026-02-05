"""
Orchestrator: Master agent with three sub-agents (PDF + Salesforce + Domo).
Uses one Gemini model. Routes to pdf_agent, salesforce_agent, or domo_agent using a routing-focused data dictionary.
Combined questions (PDF + Salesforce + Domo in one) are answered with: ask separately.
"""
import asyncio
import json
import sys
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
import google.auth
from google.auth.transport import requests as google_requests
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.apps.app import App
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from google.adk.utils.context_utils import Aclosing
from google.genai import types

from orchestrator.pdf_agent import create_pdf_agent
from orchestrator.salesforce_agent import create_salesforce_agent
from orchestrator.domo_agent import create_domo_agent
from orchestrator.usage_collector import clear as usage_clear
from orchestrator.usage_collector import get_and_clear as usage_get_and_clear
from orchestrator.usage_collector import record_gemini as usage_record_gemini

# Schema files for routing summary (nexus_schema.json and domo_schema.json in orchestrator folder)
SALESFORCE_SCHEMA_FILE = Path(__file__).resolve().parent / "nexus_schema.json"
DOMO_SCHEMA_FILE = Path(__file__).resolve().parent / "domo_schema.json"


def _print_cost_breakdown(tasks: list) -> None:
    """Print per-task cost breakdown and total (terminal only). Uses config pricing."""
    if not tasks:
        return
    total_cost = 0.0
    lines = []
    for i, t in enumerate(tasks):
        kind = t.get("kind")
        label = t.get("label", "")
        # Fix mis-attribution: first Gemini event is always the orchestrator (routing), but ADK may set author to a sub-agent.
        if i == 0 and kind == "gemini" and label in ("salesforce_agent", "pdf_agent", "domo_agent"):
            label = "orchestrator"
        if kind == "gemini":
            pin = t.get("prompt_token_count", 0) or 0
            pout = t.get("candidates_token_count", 0) or 0
            cost = (pin / 1_000_000) * config.GEMINI_INPUT_PER_1M_TOKENS + (pout / 1_000_000) * config.GEMINI_OUTPUT_PER_1M_TOKENS
            total_cost += cost
            lines.append((f"  {label} (Gemini)", f"{pin:,} in / {pout:,} out tokens", cost))
        elif kind == "embedding":
            chars = t.get("chars", 0) or 0
            cost = (chars / 1000) * config.EMBEDDING_PER_1K_CHARS
            total_cost += cost
            lines.append((f"  {label} (Embedding)", f"{chars:,} chars", cost))
        elif kind == "bigquery":
            bp = t.get("bytes_processed", 0) or 0
            cost = (bp / 1e12) * config.BIGQUERY_PER_TB
            total_cost += cost
            size_str = f"{bp:,} B" if bp < 1024 else f"{bp / 1024:,.1f} KB" if bp < 1024**2 else f"{bp / 1024**2:,.1f} MB"
            lines.append((f"  {label} (BigQuery)", f"{size_str} processed", cost))
    print("\n--- Cost for this question ---")
    for (label, detail, cost) in lines:
        print(f"{label:36} : {detail:28} → ${cost:.6f}")
    print("  " + "-" * 58)
    print(f"  {'Total':36} : {'':28} → ${total_cost:.6f}\n")


# Keywords that indicate the question is about Salesforce/BigQuery (not the PDF or Domo)
_SALESFORCE_KEYWORDS = (
    "arr", "pipeline", "opportunity", "opportunities", "customer", "customers",
    "account", "accounts", "salesforce", "nexus_data",
    "total_arr", "customer_name", "contract", "renewal", "license", "licenses",
    "stage", "closed won", "close date", "owner", "contracted", "antino bank",
    "abc capital", "sf_account", "sf_opportunity",
    "snapshot", "all accounts", "commercial snapshot", "account overview",
)

# Keywords that indicate the question is about Domo/BigQuery (not the PDF or Salesforce).
# Domo = aggregate/count by owner (num_Total_accounts, Health_Score). Do NOT include "account owner"
# so that "Who is the account owner for ABC Capital?" routes to Salesforce (single-account lookup).
_DOMO_KEYWORDS = (
    "domo", "domo_test_dataset", "domo_test", "domo data", "domo dataset",
    "health score", "health_score", "num_total_accounts",
    "total accounts", "accounts owned", "owned by",
    "how many total accounts", "how many accounts",
)

# Keywords that indicate the question is about the UPLOADED DOCUMENT/PDF only
_DOCUMENT_KEYWORDS = (
    "document", "pdf", "report", "case study", "implementation cost", "total implementation",
    "change management", "budget", "post-implementation", "lessons learned", "executive summary",
    "milestone", "phase 3", "phase 2", "technology stack", "financial benefits", "readmission",
    "telehealth", "resolution strategy", "risk", "mitigate", "recommendation from the document",
)


def _is_likely_salesforce_question(text: str) -> bool:
    """True if the question clearly asks about Salesforce/BigQuery data (not the PDF)."""
    lower = text.lower().strip()
    return any(kw in lower for kw in _SALESFORCE_KEYWORDS)


def _is_likely_document_question(text: str) -> bool:
    """True if the question is clearly about the uploaded document/PDF (not Salesforce or Domo)."""
    lower = text.lower().strip()
    return any(kw in lower for kw in _DOCUMENT_KEYWORDS)


def _is_likely_domo_question(text: str) -> bool:
    """True if the question clearly asks about Domo/BigQuery data (not the PDF or Salesforce)."""
    lower = text.lower().strip()
    return any(kw in lower for kw in _DOMO_KEYWORDS)


def _maybe_add_routing_hint(user_message: str) -> str:
    """Prepend a routing hint so the master delegates to the correct sub-agent.
    Check Domo before Salesforce so questions like 'how many total accounts owned by X'
    route to domo_agent (num_Total_accounts, Account_Owner in domo_test), not salesforce_agent."""
    if _is_likely_document_question(user_message):
        return (
            "[ROUTING: This question is about the UPLOADED DOCUMENT or PDF (implementation, budget, lessons learned, etc.). "
            "You MUST delegate to pdf_agent only.]\n\n"
            + user_message
        )
    if _is_likely_domo_question(user_message):
        return (
            "[ROUTING: This question is about Domo/BigQuery data (domo_test_dataset). "
            "You MUST delegate to domo_agent only.]\n\n"
            + user_message
        )
    if _is_likely_salesforce_question(user_message):
        return (
            "[ROUTING: This question is about Salesforce/BigQuery data (e.g. ARR, customers, opportunities). "
            "You MUST delegate to salesforce_agent only.]\n\n"
            + user_message
        )
    return user_message


def get_routing_data_dictionary(salesforce_path: Path, domo_path: Path) -> str:
    """Build a short routing-focused summary from schema files for the master agent.
    Dataset + table names + key columns per table so the master can recognize Salesforce/BigQuery and Domo questions."""
    lines = []
    
    # Salesforce schema
    if salesforce_path.exists():
        try:
            with open(salesforce_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tables = data.get("datasets", [{}])[0].get("tables", [])
            lines.append("# Salesforce / BigQuery data (nexus_data) — use this to decide if the user is asking about Salesforce/BigQuery.")
            lines.append("If the question mentions any of these topics or columns, route to salesforce_agent:")
            lines.append("")
            for t in tables:
                cols = [c["column_name"] for c in t.get("schema", [])]
                # Keep first 15 columns for routing; that's enough for ARR, customer, opportunity, etc.
                key_cols = cols[:15] if len(cols) > 15 else cols
                lines.append(f"- {t['table_id']}: {', '.join(key_cols)}")
        except Exception as e:
            lines.append(f"Salesforce data: BigQuery dataset nexus_data (routing summary error: {e})")
    else:
        lines.append("Salesforce data is in BigQuery dataset nexus_data (schema file not found).")
    
    lines.append("")
    lines.append("")
    
    # Domo schema
    if domo_path.exists():
        try:
            with open(domo_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tables = data.get("datasets", [{}])[0].get("tables", [])
            lines.append("# Domo / BigQuery data (domo_test_dataset) — use this to decide if the user is asking about Domo/BigQuery.")
            lines.append("If the question mentions any of these topics or columns, route to domo_agent:")
            lines.append("")
            for t in tables:
                cols = [c["column_name"] for c in t.get("schema", [])]
                # Keep first 15 columns for routing
                key_cols = cols[:15] if len(cols) > 15 else cols
                lines.append(f"- {t['table_id']}: {', '.join(key_cols)}")
        except Exception as e:
            lines.append(f"Domo data: BigQuery dataset domo_test_dataset (routing summary error: {e})")
    else:
        lines.append("Domo data is in BigQuery dataset domo_test_dataset (schema file not found).")
    
    return "\n".join(lines)


def build_agents(credentials, routing_data_dict: str):
    """Build PDF agent, Salesforce agent, Domo agent, and master orchestrator from sub-agent modules."""
    model = Gemini(
        model_name=config.GEMINI_MODEL,
        project=config.PROJECT_ID,
        location=config.LOCATION,
        vertexai=True,
    )

    # Sub-agents from self-contained modules
    pdf_agent = create_pdf_agent()
    salesforce_agent = create_salesforce_agent(credentials)
    domo_agent = create_domo_agent(credentials)

    # Master agent: route using data dictionary; no combined questions
    master_instruction = f"""You are the master assistant. Route each user question to exactly ONE specialist.

{routing_data_dict}

ROUTING RULES:
- If the user message starts with "[ROUTING: ... salesforce_agent only.]": you MUST delegate to salesforce_agent (do not use pdf_agent or domo_agent).
- If the user message starts with "[ROUTING: ... domo_agent only.]": you MUST delegate to domo_agent (do not use pdf_agent or salesforce_agent).
- If the user message starts with "[ROUTING: ... pdf_agent only.]": you MUST delegate to pdf_agent (do not use salesforce_agent or domo_agent).
- If the question is about the UPLOADED DOCUMENT or PDF (reports, case studies, implementation cost, change management, budget, lessons learned, post-implementation, executive summary, tables in the document): delegate to pdf_agent.
- If the question is about SALESFORCE or BIGQUERY data (ARR, pipeline, opportunities, customers, licenses, Total_ARR, Customer_Name, Opportunity_Name, Stage, CloseDate, nexus_data, or any Salesforce tables/columns above): delegate to salesforce_agent.
- If the question is about DOMO or BIGQUERY data (domo_test_dataset, domo_test, or any Domo tables/columns above): delegate to domo_agent. This includes "how many total accounts (are) owned by [person]" or aggregate counts per owner (num_Total_accounts). For "who is the account owner for [specific account name]" (e.g. ABC Capital) or single-account lookup by customer name, use salesforce_agent, not domo_agent.
- If the question clearly needs MULTIPLE data sources (document + Salesforce + Domo) in one question: do NOT call multiple agents. Reply with exactly: "Please ask about the document, Salesforce data, or Domo data separately."

Always delegate to exactly one sub-agent (pdf_agent, salesforce_agent, or domo_agent). After you get the answer, present it clearly to the user. Do not mention "Salesforce data" when the user asked about the document or Domo."""

    master_agent = LlmAgent(
        model=model,
        name="orchestrator",
        instruction=master_instruction,
        sub_agents=[pdf_agent, salesforce_agent, domo_agent],
    )

    return master_agent


async def main():
    print("\n" + "=" * 60)
    print("  ORCHESTRATOR — PDF + Salesforce + Domo (Master + 3 Sub-Agents)")
    print("=" * 60)
    print(f"  Project: {config.PROJECT_ID}")
    print(f"  Region:  {config.LOCATION}")
    print(f"  Model:   {config.GEMINI_MODEL}")
    print("=" * 60)

    # Env for Vertex/Gemini
    import os
    os.environ["GOOGLE_CLOUD_PROJECT"] = config.PROJECT_ID
    os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = config.PROJECT_ID
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"

    # Init Vertex for PDF search (phase4_adk uses this)
    import vertexai
    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)

    try:
        credentials, _ = google.auth.default()
        if not credentials.valid:
            credentials.refresh(google_requests.Request())
    except Exception as e:
        print(f"  Authentication failed: {e}")
        sys.exit(1)

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

    print("\n  Ready. Ask about documents (PDF), Salesforce/BigQuery data, or Domo/BigQuery data.")
    print("  Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue

            # Pre-route hint: if question is clearly Salesforce/BigQuery, hint so master routes to salesforce_agent
            message_to_send = _maybe_add_routing_hint(user_input)
            msg = types.Content(role="user", parts=[types.Part(text=message_to_send)])
            print("Assistant: ", end="", flush=True)

            usage_clear()
            async with Aclosing(
                runner.run_async(
                    user_id="user",
                    session_id=session.id,
                    new_message=msg,
                )
            ) as stream:
                async for event in stream:
                    if getattr(event, "usage_metadata", None) and getattr(event, "author", None):
                        um = event.usage_metadata
                        usage_record_gemini(
                            event.author,
                            getattr(um, "prompt_token_count", None) or 0,
                            getattr(um, "candidates_token_count", None) or 0,
                        )
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                print(part.text, end="", flush=True)
                            if getattr(part, "function_call", None):
                                print(f"\n  [Calling {getattr(part.function_call, 'name', 'tool')}...]", flush=True)
            tasks = usage_get_and_clear()
            _print_cost_breakdown(tasks)
            print("")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n  Error: {e}")
            import traceback
            traceback.print_exc()

    print("  Done.\n")


if __name__ == "__main__":
    asyncio.run(main())
