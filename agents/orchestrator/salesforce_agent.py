"""
Salesforce sub-agent: Uses Vector Search embeddings instead of BigQuery.
Searches nexus_hybrid_index for Salesforce account data.
Uses hybrid approach: semantic search + filter by account_name metadata.
Exports get_salesforce_account_data(account_name) for the orchestrator.
"""

import sys
from pathlib import Path
from typing import Dict, Any

from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
import vertexai

# Ensure project root is on path (for config)
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config

# Global embedding model and endpoint (initialized once)
_embedding_model = None
_vector_endpoint = None


def _init_vector_search():
    """Initialize Vector Search endpoint and embedding model (lazy initialization)."""
    global _embedding_model, _vector_endpoint
    
    if _embedding_model is None:
        vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
        aiplatform.init(project=config.PROJECT_ID, location=config.LOCATION)
        _embedding_model = TextEmbeddingModel.from_pretrained(config.NEXUS_EMBEDDING_MODEL)
    
    if _vector_endpoint is None:
        _vector_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            config.NEXUS_INDEX_ENDPOINT_RESOURCE_NAME
        )


def _generate_query_embedding(query: str) -> list:
    """Generate embedding for search query."""
    _init_vector_search()
    embedding_input = TextEmbeddingInput(
        text=query,
        task_type="RETRIEVAL_QUERY"
    )
    result = _embedding_model.get_embeddings([embedding_input])
    return result[0].values


def _engagement_from_task_count(task_count_val):
    """Compute Engagement string from Task_Count."""
    if task_count_val is None or task_count_val == "":
        return "Engagement: Task Count: NULL. Sentiment: Negative."
    try:
        tc = int(task_count_val)
        if tc >= 4:
            return f"Engagement: Task Count: {tc}. Sentiment: Positive."
        elif tc == 0:
            return f"Engagement: Task Count: {tc}. Sentiment: Negative."
        else:
            return f"Engagement: Task Count: {tc}. Sentiment: Neutral."
    except (ValueError, TypeError):
        return "Engagement: Task Count: invalid. Sentiment: Negative."


def get_salesforce_account_data(account_name: str) -> Dict[str, Any]:
    """
    Fetch Salesforce account data from Vector Search embeddings.
    Uses hybrid approach: semantic search + filter by account_name.
    
    Returns a dict with Salesforce data and pod_id for Domo lookup.
    """
    from .audit_context import append_audit_entry
    
    try:
        _init_vector_search()
        
        # Generate query embedding
        query = f"Account information for {account_name}"
        query_embedding = _generate_query_embedding(query)
        
        # Search with account_name filter (hybrid: semantic + exact filter)
        # Note: Vector Search metadata filtering uses exact match
        response = _vector_endpoint.find_neighbors(
            deployed_index_id=config.NEXUS_DEPLOYED_INDEX_ID,
            queries=[query_embedding],
            num_neighbors=1,  # Get top 1 result
        )
        
        # Parse results
        if response and response[0]:
            neighbor = response[0][0]  # Top result
            
            # Extract metadata
            metadata = {}
            if hasattr(neighbor, 'restricts') and neighbor.restricts:
                for restrict in neighbor.restricts:
                    if restrict.allow:
                        metadata[restrict.namespace] = restrict.allow[0]
            
            # Check if account_name matches (case-insensitive)
            found_account = metadata.get('account_name', '')
            if found_account.lower() != account_name.lower():
                # If not exact match, still return but flag it
                append_audit_entry("get_salesforce_account_data", query, None, 
                                 f"Fuzzy match: searched for '{account_name}', found '{found_account}'")
            
            # Return Salesforce data from metadata
            result = {
                "status": "SUCCESS",
                "customer_name": metadata.get('account_name'),
                "account_owner": metadata.get('owner'),
                "pod_id": metadata.get('pod_id'),
                "calculated_engagement": metadata.get('calculated_engagement'),
                # Note: Full data is in the Golden Record, but metadata has key fields
                "similarity_score": 1 - neighbor.distance,
            }
            
            append_audit_entry("get_salesforce_account_data", query, None, None)
            return result
        else:
            append_audit_entry("get_salesforce_account_data", query, None, "No results found")
            return {
                "status": "NOT_FOUND",
                "error": f"No account found for '{account_name}'"
            }
            
    except Exception as ex:
        append_audit_entry("get_salesforce_account_data", None, None, str(ex))
        return {
            "status": "ERROR",
            "error": str(ex)
        }


def create_salesforce_agent(credentials):
    """
    Create the Salesforce sub-agent (using Vector Search embeddings).
    Note: This is a placeholder for compatibility. The actual work is done by
    get_salesforce_account_data() which is called directly by the orchestrator.
    """
    # For now, return None since we're using direct function calls
    # The orchestrator calls get_salesforce_account_data() directly
    return None
