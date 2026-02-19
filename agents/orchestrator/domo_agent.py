"""
Domo Sub-Agent (Nexus Hybrid Engine)
====================================

This module implements the Domo Agent for the Nexus Hybrid Engine.
It is responsible for retrieving product usage metrics (Pod Data) from GCS.

Workflow:
1.  Receive a 'Pod ID' (e.g., 888) from the Orchestrator.
    (Note: The Pod ID usually originates from the Salesforce Agent's metadata).
2.  Construct the unique record ID: "domo_pod_{Pod_ID}".
3.  Perform a direct O(1) lookup in the GCS Content Library.
4.  Return structured metrics (MEAU, Health Score, etc.).

Usage:
    The agent is initialized with `create_domo_agent` and
    uses `get_pod_data_by_id` as its primary tool.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Third-party imports
from google.cloud import storage
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools.function_tool import FunctionTool

# --- Configuration Import Logic ---
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from Nexus_Hybrid_Engine import nexus_config as config
except ImportError:
    # Fallback to local import
    sys.path.append(str(_ROOT / "Nexus_Hybrid_Engine"))
    import nexus_config as config

# Alias for compatibility
nex_config = config


# --- Global Resources (Lazy Initialization) ---
_storage_client: Optional[storage.Client] = None
_bucket: Optional[storage.Bucket] = None


def _init_resources() -> None:
    """
    Initialize the Google Cloud Storage client lazily.
    """
    global _storage_client, _bucket

    if _storage_client is None:
        _storage_client = storage.Client(project=nex_config.PROJECT_ID)
        _bucket = _storage_client.bucket(nex_config.NEXUS_GCS_BUCKET)


def _fetch_record_from_gcs(record_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the full JSON record from the GCS Content Library.

    Args:
        record_id (str): Unique record ID (e.g., 'domo_pod_888').

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON data if found, else None.
    """
    _init_resources()
    try:
        blob_path = f"lookup_records/{record_id}.json"
        blob = _bucket.blob(blob_path)

        if not blob.exists():
            return None

        data_str = blob.download_as_text()
        return json.loads(data_str)
    except Exception as e:
        print(f"[Error] GCS Lookup failed for {record_id}: {e}")
        return None


def get_pod_data_by_id(pod_id: Any) -> Dict[str, Any]:
    """
    Fetch Domo pod data using direct GCS lookup.

    This function bypasses vector search because Pod IDs are deterministic
    and unique keys, allowing for efficient O(1) retrieval.

    Args:
        pod_id (Any): The Pod ID to search for (e.g., 888 or "888").

    Returns:
        Dict[str, Any]: A dictionary containing status and data.
                        Returns {'status': 'NOT_FOUND'} if the ID doesn't exist.
    """
    try:
        if not pod_id:
            return {"status": "ERROR", "error": "No Pod ID provided"}

        # Normalize Pod ID (matches Phase 1 ID generation logic)
        safe_pod_id = str(pod_id).strip()

        # Handle cases where pod_id might be "888.0" (float string from JSON)
        if safe_pod_id.replace('.', '', 1).isdigit() and '.' in safe_pod_id:
            safe_pod_id = str(int(float(safe_pod_id)))

        record_id = f"domo_pod_{safe_pod_id}"

        print(f"[DomoAgent] Looking up: {record_id}")

        # Direct GCS Lookup
        record_data = _fetch_record_from_gcs(record_id)

        if not record_data:
            print(f"[DomoAgent] Record {record_id} not found in GCS.")
            return {"status": "NOT_FOUND", "pod_id": pod_id}

        return {
            "status": "SUCCESS",
            "data": record_data,
            "source": "Nexus_Hybrid_Engine"
        }

    except Exception as e:
        print(f"[DomoAgent] Error: {e}")
        return {"status": "ERROR", "error": str(e)}


def create_domo_agent(credentials: Any) -> LlmAgent:
    """
    Factory function to create the Domo LlmAgent instance.

    Args:
        credentials (Any): Google Cloud credentials.

    Returns:
        LlmAgent: Configured Domo agent.
    """
    model = Gemini(
        model_name=config.GEMINI_MODEL,
        project=config.PROJECT_ID,
        location=config.LOCATION,
        vertexai=True,
    )

    return LlmAgent(
        model=model,
        name="domo_agent",
        instruction=(
            "You are a Domo data assistant. "
            "Retrieve pod metrics using get_pod_data_by_id(pod_id)."
        ),
        tools=[FunctionTool(get_pod_data_by_id)],
    )
