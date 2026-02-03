"""
Simplified Local Version - Nexus Salesforce Assistant
Fixed: Session Sharing and Credential Wrapper
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
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from google.adk.utils.context_utils import Aclosing
from google.adk.tools.bigquery import BigQueryCredentialsConfig, BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig, WriteMode
from google.genai import types

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = "sym-dev-mr-agents-01"
REGION = "us-central1"
MODEL_NAME = "gemini-1.5-pro"
SCHEMA_FILE = Path(__file__).parent / "nexus_schema.json"

class _ADCCredentialsWrapper(auth_credentials.Credentials):
    """Wraps Application Default Credentials safely for the ADK."""
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

class SchemaGrounding:
    @staticmethod
    def get_context(path: Path) -> str:
        if not path.exists():
            return "Note: nexus_schema.json not found. Use standard Salesforce naming conventions."
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            tables = data.get("datasets", [{}])[0].get("tables", [])
            context = ["# DATABASE SCHEMA (nexus_data)"]
            for t in tables:
                context.append(f"## Table: {t['table_id']}")
                for c in t.get('schema', []):
                    context.append(f"- {c['column_name']} ({c['data_type']})")
            return "\n".join(context)
        except Exception as e:
            return f"Schema load error: {e}"

async def main():
    print("="*60)
    print("Nexus Salesforce Assistant - Local Test")
    print("="*60)
    
    os.environ['GOOGLE_CLOUD_PROJECT'] = PROJECT_ID
    os.environ['GOOGLE_CLOUD_QUOTA_PROJECT'] = PROJECT_ID
    os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = "True"
    
    try:
        credentials, _ = google.auth.default()
        if not credentials.valid:
            credentials.refresh(google_requests.Request())
        logger.info(f"Authenticated for project: {PROJECT_ID}")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return

    # 1. Initialize Services (Keep these instances to share)
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    credential_service = InMemoryCredentialService()

    # 2. Setup BigQuery Tool
    credentials_for_tool = _ADCCredentialsWrapper(credentials)
    bigquery_toolset = BigQueryToolset(
        credentials_config=BigQueryCredentialsConfig(credentials=credentials_for_tool),
        bigquery_tool_config=BigQueryToolConfig(write_mode=WriteMode.BLOCKED, compute_project_id=PROJECT_ID)
    )
    schema_context = SchemaGrounding.get_context(SCHEMA_FILE)

    # 3. Model & Agent
    model = Gemini(model_name=MODEL_NAME, project=PROJECT_ID, location=REGION, vertexai=True)
    
    instructions = f"""
    You are the Nexus Salesforce Assistant, an expert data analyst.
    {schema_context}
    When users ask for data, use the execute_sql tool immediately. 
    Always use fully qualified names: `{PROJECT_ID}.nexus_data.TABLE_NAME`.
    If the user asks for a count, run: SELECT count(*) as total FROM `{PROJECT_ID}.nexus_data.sf_account_info`
    """

    agent = LlmAgent(model=model, instruction=instructions, tools=[bigquery_toolset], name="nexus_agent")
    app = App(name="nexus_salesforce_local", root_agent=agent)
    
    # 4. Setup Runner with the PRE-DEFINED services
    runner = Runner(
        app=app,
        artifact_service=artifact_service,
        session_service=session_service,
        credential_service=credential_service
    )

    # Create the session in the SAME service instance the runner uses
    session = await session_service.create_session(app_name="nexus_salesforce_local", user_id="local_user")

    print("\n✓ Agent initialized successfully!")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['exit', 'quit']: break
            if not user_input: continue
            
            print("Assistant: ", end="", flush=True)
            msg = types.Content(role='user', parts=[types.Part(text=user_input)])
            
            # Using the specific session.id created above
            async with Aclosing(runner.run_async(user_id="local_user", session_id=session.id, new_message=msg)) as stream:
                async for event in stream:
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text: 
                                print(part.text, end="", flush=True)
                            if part.function_call:
                                # This helps you see if it's actually calling BigQuery
                                print(f"\n[SYSTEM: Calling {part.function_call.name}...]", flush=True)
            print("\n")
        except Exception as e:
            print(f"\n[Error]: {e}")
            logger.exception("Processing error")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")