"""
Salesforce Sub-Agent (Nexus Hybrid Engine)
==========================================

This module implements the Salesforce Agent for the Nexus Hybrid Engine.
It retrieves account data using a hybrid approach:
1.  Semantic Vector Search (Vertex AI) to find potential matches.
2.  Exact ID Re-Ranking to ensure precision.
3.  Direct GCS Lookup for retrieving the full record content.

Usage:
    The agent is initialized with `create_salesforce_agent` and
    uses `get_salesforce_account_data` as its primary tool.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Third-party imports
from google.cloud import aiplatform
from google.cloud import storage
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools.function_tool import FunctionTool

# --- Configuration Import Logic ---
# Ensure project root is on path to allow importing Nexus_Hybrid_Engine
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from Nexus_Hybrid_Engine import nexus_config as config
except ImportError:
    # Fallback to local import if running from a different context
    sys.path.append(str(_ROOT / "Nexus_Hybrid_Engine"))
    import nexus_config as config

# Alias for backward compatibility / cleaner access
nex_config = config


# --- Global Resources (Lazy Initialization) ---
_embedding_model: Optional[TextEmbeddingModel] = None
_vector_endpoint: Optional[aiplatform.MatchingEngineIndexEndpoint] = None
_storage_client: Optional[storage.Client] = None
_bucket: Optional[storage.Bucket] = None


def _init_resources() -> None:
    """
    Initialize Vertex AI and Google Cloud Storage clients lazily.
    This prevents overhead if the agent module is imported but not used.
    """
    global _embedding_model, _vector_endpoint, _storage_client, _bucket

    if _embedding_model is None:
        vertexai.init(project=nex_config.PROJECT_ID, location=nex_config.LOCATION)
        _embedding_model = TextEmbeddingModel.from_pretrained(nex_config.NEXUS_EMBEDDING_MODEL)

    if _vector_endpoint is None:
        aiplatform.init(project=nex_config.PROJECT_ID, location=nex_config.LOCATION)
        _vector_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            nex_config.NEXUS_INDEX_ENDPOINT_RESOURCE_NAME
        )

    if _storage_client is None:
        _storage_client = storage.Client(project=nex_config.PROJECT_ID)
        _bucket = _storage_client.bucket(nex_config.NEXUS_GCS_BUCKET)


def _generate_query_embedding(query: str) -> List[float]:
    """
    Generate a text embedding for the search query using the configured model.

    Args:
        query (str): The search query text.

    Returns:
        List[float]: The 768-dimensional embedding vector.
    """
    _init_resources()
    embedding_input = TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")
    result = _embedding_model.get_embeddings([embedding_input])
    return result[0].values


def _fetch_record_from_gcs(record_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the full JSON record from the GCS Content Library by its ID.

    Args:
        record_id (str): The unique record ID (e.g., 'sf_Antino_Bank').

    Returns:
        Optional[Dict[str, Any]]: The parsed JSON data if found, else None.
    """
    _init_resources()
    try:
        blob_path = f"lookup_records/{record_id}.json"
        blob = _bucket.blob(blob_path)

        if not blob.exists():
            print(f"[Warning] Record {record_id} found in Index but missing in GCS ({blob_path}).")
            return None

        data_str = blob.download_as_text()
        return json.loads(data_str)
    except Exception as e:
        print(f"[Error] GCS Lookup failed for {record_id}: {e}")
        return None


def _safe_str(s: str) -> str:
    """
    Normalize a string for ID comparison by removing special characters.
    Used for the 'Exact ID Match' logic.

    Args:
        s (str): Input string.

    Returns:
        str: Normalized string (alphanumeric only, underscores).
    """
    return re.sub(r'[^a-zA-Z0-9]', '_', str(s)).strip('_')


def get_salesforce_account_data(account_name: str) -> Dict[str, Any]:
    """
    Fetch Salesforce account data using a Hybrid Search strategy.

    Strategy:
    1.  Vector Search: Find top 50 semantic matches.
    2.  Re-Ranking: Check if any match has an ID corresponding to the exact account name.
    3.  Retrieval: Fetch the full record content from GCS.

    Args:
        account_name (str): The name of the account to search for.

    Returns:
        Dict[str, Any]: A dictionary containing status, data, and metadata.
    """
    print(f"\n[SalesforceAgent] Searching for: {account_name}")

    try:
        _init_resources()

        # Step 1: Semantic Vector Search
        # -------------------------------------------------------------
        query = f"Account information for {account_name}"
        query_vec = _generate_query_embedding(query)

        response = _vector_endpoint.find_neighbors(
            deployed_index_id=nex_config.NEXUS_DEPLOYED_INDEX_ID,
            queries=[query_vec],
            num_neighbors=50  # High recall window for synthetic data robustness
        )

        if not response or not response[0]:
            print("[SalesforceAgent] No vector match found.")
            return {"status": "NOT_FOUND", "query": account_name}

        matches = response[0]
        best_semantic_match = matches[0]

        # Step 2: Exact ID Re-Ranking
        # -------------------------------------------------------------
        # Expected ID format: "sf_{Safe_Name}"
        expected_id_suffix = _safe_str(account_name)
        exact_match = None

        for m in matches:
            # Check if the ID ends with the normalized account name
            if m.id.lower().endswith(expected_id_suffix.lower()):
                exact_match = m
                print(f"[SalesforceAgent] found EXACT ID match: {m.id} (Score: {m.distance:.4f})")
                break
            # Logic Note: We could add fuzzy matching here if needed in future

        if exact_match:
            final_match = exact_match
        else:
            print(f"[SalesforceAgent] No exact ID match found in Top 50. Using best semantic match: {best_semantic_match.id}")
            final_match = best_semantic_match

        record_id = final_match.id
        score = final_match.distance

        # Step 3: GCS Content Retrieval
        # -------------------------------------------------------------
        record_data = _fetch_record_from_gcs(record_id)

        if not record_data:
            return {
                "status": "NOT_FOUND",
                "query": account_name,
                "reason": "Content Missing in GCS Library"
            }

        return {
            "status": "SUCCESS",
            "data": record_data,
            "source": "Nexus_Hybrid_Engine",
            "match_score": score
        }

    except Exception as e:
        print(f"[SalesforceAgent] Error: {e}")
        return {"status": "ERROR", "error": str(e)}


def create_salesforce_agent(credentials: Any) -> LlmAgent:
    """
    Factory function to create the Salesforce LlmAgent instance.

    Args:
        credentials (Any): Google Cloud credentials.

    Returns:
        LlmAgent: Configured Salesforce agent.
    """
    model = Gemini(
        model_name=config.GEMINI_MODEL,
        project=config.PROJECT_ID,
        location=config.LOCATION,
        vertexai=True,
    )

    return LlmAgent(
        model=model,
        name="salesforce_agent",
        instruction=(
            "You are a Salesforce data assistant. "
            "Retrieve account data using get_salesforce_account_data()."
        ),
        tools=[FunctionTool(get_salesforce_account_data)],
    )
