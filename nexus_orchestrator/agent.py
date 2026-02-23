"""
ADK Web Entry Point
===================

Exposes `root_agent` for ADK Web UI.
Run: adk web .  (from project root)

Auth is handled automatically:
  - If credentials/service_account.json exists → uses it (via nexus_config.py)
  - Else → falls back to gcloud auth (google.auth.default())
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# nexus_config sets GOOGLE_APPLICATION_CREDENTIALS if service_account.json exists
# This must be imported before any google.auth calls
from Nexus_Hybrid_Engine import nexus_config  # noqa: F401

import google.auth
from google.auth.transport import requests as google_requests
from agents.orchestrator.run_orchestrator import (
    build_agents,
    get_routing_data_dictionary,
    SALESFORCE_SCHEMA_FILE,
    DOMO_SCHEMA_FILE,
)

# Initialize credentials (picks up service account if GOOGLE_APPLICATION_CREDENTIALS is set)
# IMPORTANT: Must pass scopes explicitly for service account JSON files.
_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
try:
    credentials, _ = google.auth.default(scopes=_SCOPES)
    if not credentials.valid:
        credentials.refresh(google_requests.Request())
except Exception as e:
    print(f"[nexus_orchestrator] Auth Warning: {e}")
    credentials = None

# Build the root agent — CRITICAL: ADK Web looks for this variable at module level
routing_data = get_routing_data_dictionary(SALESFORCE_SCHEMA_FILE, DOMO_SCHEMA_FILE)
root_agent = build_agents(credentials, routing_data)
