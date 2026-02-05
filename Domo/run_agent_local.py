"""
Simplified Local Version - Domo BigQuery Assistant
(Parallel to Nexus Salesforce Assistant)
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import google.auth
from google.auth import credentials as auth_credentials
from google.auth.transport import requests as google_requests

from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.apps.app import App
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import (
    InMemoryCredentialService,
)
from google.adk.utils.context_utils import Aclosing
from google.adk.tools.bigquery import BigQueryCredentialsConfig, BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig, WriteMode
from google.genai import types

# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PROJECT_ID = "sym-dev-mr-agents-01"
REGION = "us-central1"
MODEL_NAME = "gemini-1.5-pro"
SCHEMA_FILE = Path(__file__).parent / "domo_schema.json"

# --------------------------------------------------
# ADC WRAPPER (UNCHANGED)
# --------------------------------------------------

class _ADCCredentialsWrapper(auth_credentials.Credentials):
    def __init__(self, credentials):
        super().__init__()
        self._credentials = credentials
        self.token = getattr(credentials, "token", None)
        self.expiry = getattr(credentials, "expiry", None)

    def refresh(self, request):
        if hasattr(self._credentials, "refresh"):
            self._credentials.refresh(request)
            self.token = self._credentials.token
            self.expiry = self._credentials.expiry

    def before_request(self, request, method, url, headers):
        self._credentials.before_request(request, method, url, headers)
        self.token = self._credentials.token

# --------------------------------------------------
# SCHEMA GROUNDING (UNCHANGED)
# --------------------------------------------------

class SchemaGrounding:
    @staticmethod
    def get_context(path: Path) -> str:
        if not path.exists():
            return "Note: domo_schema.json not found."
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tables = data.get("datasets", [{}])[0].get("tables", [])
            context = ["# DATABASE SCHEMA (DOMO)"]
            for t in tables:
                context.append(f"## Table: {t['table_id']}")
                for c in t.get("schema", []):
                    context.append(f"- {c['column_name']} ({c['data_type']})")
            return "\n".join(context)
        except Exception as e:
            return f"Schema load error: {e}"

# --------------------------------------------------
# MAIN
# --------------------------------------------------

async def main():
    print("=" * 60)
    print("Domo BigQuery Assistant - Local Test")
    print("=" * 60)

    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    # Auth
    try:
        credentials, _ = google.auth.default()
        if not credentials.valid:
            credentials.refresh(google_requests.Request())
        logger.info(f"Authenticated for project: {PROJECT_ID}")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return

    # Shared services (same pattern as Salesforce)
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    credential_service = InMemoryCredentialService()

    # BigQuery Tool
    credentials_for_tool = _ADCCredentialsWrapper(credentials)
    bigquery_toolset = BigQueryToolset(
        credentials_config=BigQueryCredentialsConfig(credentials=credentials_for_tool),
        bigquery_tool_config=BigQueryToolConfig(
            write_mode=WriteMode.BLOCKED,
            compute_project_id=PROJECT_ID,
        ),
    )

    schema_context = SchemaGrounding.get_context(SCHEMA_FILE)

    # Model
    model = Gemini(
        model_name=MODEL_NAME,
        project=PROJECT_ID,
        location=REGION,
        vertexai=True,
    )

    # 🔑 DOMO-SPECIFIC INSTRUCTIONS
    instructions = f"""
You are the Domo Data Assistant, an expert BigQuery analyst.

{schema_context}

RULES:
1. ALWAYS translate the user question into SQL
2. ALWAYS call execute_sql
3. NEVER answer without executing SQL
4. Use fully qualified table names:
   `{PROJECT_ID}.domo_test_dataset.domo_test`
5. CAST DATE and TIMESTAMP to STRING
6. Writes are forbidden

Examples:
- Count rows:
  SELECT COUNT(*) AS total FROM `{PROJECT_ID}.domo_test_dataset.domo_test`

- Filter by account owner or month:
  SELECT Month, Account_Owner, num_Total_accounts, `0_Health_Score`
  FROM `{PROJECT_ID}.domo_test_dataset.domo_test`
  WHERE Account_Owner = 'Some Owner'
"""

    agent = LlmAgent(
        model=model,
        instruction=instructions,
        tools=[bigquery_toolset],
        name="domo_agent",
    )

    app = App(name="domo_local_app", root_agent=agent)

    runner = Runner(
        app=app,
        artifact_service=artifact_service,
        session_service=session_service,
        credential_service=credential_service,
    )

    session = await session_service.create_session(
        app_name="domo_local_app",
        user_id="local_user",
    )

    print("\n✓ Agent initialized successfully!")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input:
                continue

            print("Assistant: ", end="", flush=True)
            msg = types.Content(role="user", parts=[types.Part(text=user_input)])

            async with Aclosing(
                runner.run_async(
                    user_id="local_user",
                    session_id=session.id,
                    new_message=msg,
                )
            ) as stream:
                async for event in stream:
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                print(part.text, end="", flush=True)
                            if part.function_call:
                                print(
                                    f"\n[SYSTEM: Calling {part.function_call.name}...]",
                                    flush=True,
                                )
            print("\n")

        except Exception as e:
            print(f"\n[Error]: {e}")
            logger.exception("Processing error")

if __name__ == "__main__":
    asyncio.run(main())
