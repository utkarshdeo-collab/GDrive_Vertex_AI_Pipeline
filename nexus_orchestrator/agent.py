"""
ADK Web Entry Point (Project Root)
==================================

This file exposes the 'root_agent' for the ADK Web UI.
With this file at the root, you can simply run 'adk web' without arguments.
"""
import sys
from pathlib import Path
import google.auth
from google.auth.transport import requests as google_requests

# Ensure project root is in sys.path
# Since this file is in nexus_orchestrator/agent.py, root is 2 levels up relative to file? No, just 1 level up.
# Update: we moved it to D:\...\nexus_orchestrator\agent.py
# So ROOT is D:\...\Drive_Pipeline - Personal (parent)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.orchestrator.run_orchestrator import (
    build_agents, 
    get_routing_data_dictionary, 
    SALESFORCE_SCHEMA_FILE, 
    DOMO_SCHEMA_FILE
)

# Initialize Credentials
try:
    credentials, _ = google.auth.default()
    if not credentials.valid:
        credentials.refresh(google_requests.Request())
except Exception as e:
    print(f"Auth Error: {e}")
    credentials = None

# Build Agent
# CRITICAL: ADK Web looks for a variable named 'root_agent' at module level
routing_data = get_routing_data_dictionary(SALESFORCE_SCHEMA_FILE, DOMO_SCHEMA_FILE)
root_agent = build_agents(credentials, routing_data)

# Print confirmation for CLI debugging
if __name__ == "__main__":
    print(f"Root Agent loaded: {root_agent.name}")
