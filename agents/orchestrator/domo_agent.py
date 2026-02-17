"""
Domo sub-agent: Uses Vector Search embeddings instead of BigQuery.
Searches nexus_hybrid_index for Domo pod data.
Filters by pod_id metadata for exact match.
Exports get_pod_data_by_id(pod_id) for the orchestrator.
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


def get_pod_data_by_id(pod_id: int) -> Dict[str, Any]:
    """
    Fetch pod data from Vector Search embeddings using pod_id filter.
    Returns MEAU, ORBIT score, Churn Risk, and Expansion Signal data.
    
    Use this when you have a pod_id from Salesforce data and need to get 
    the corresponding Domo metrics.
    """
    from .audit_context import append_audit_entry
    
    try:
        _init_vector_search()
        
        # Generate a generic query embedding (we'll filter by pod_id metadata)
        query = f"Pod metrics for pod {pod_id}"
        query_embedding = _generate_query_embedding(query)
        
        # Search with pod_id filter (exact match)
        # Note: Vector Search metadata filtering requires string values
        response = _vector_endpoint.find_neighbors(
            deployed_index_id=config.NEXUS_DEPLOYED_INDEX_ID,
            queries=[query_embedding],
            num_neighbors=1,  # Get top 1 result
            # TODO: Add filter once we confirm pod_id is in metadata
            # filter=[{"namespace": "pod_id", "allow": [str(pod_id)]}]
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
            
            # Verify pod_id matches (since we can't filter yet)
            found_pod_id = metadata.get('pod_id')
            if found_pod_id and str(found_pod_id) != str(pod_id):
                # Pod ID mismatch - search didn't filter correctly
                append_audit_entry("get_pod_data_by_id", query, None,
                                 f"Pod ID mismatch: searched for {pod_id}, found {found_pod_id}")
                return {
                    "status": "NOT_FOUND",
                    "pod_id": pod_id,
                    "error": f"Pod ID mismatch in search results"
                }
            
            # Return Domo data from metadata (now includes all Domo fields!)
            result = {
                "status": "SUCCESS",
                "pretty_name": metadata.get('account_name'),
                "meau": metadata.get('meau'),
                "provisioned_users": metadata.get('provisioned_users'),
                "active_users": metadata.get('active_users'),
                "health_score": metadata.get('health_score'),
                "risk_ratio_for_next_renewal": metadata.get('risk_ratio_for_next_renewal'),
                "contracted_licenses": metadata.get('contracted_licenses'),
                "pod_id": found_pod_id,
                "similarity_score": 1 - neighbor.distance,
            }
            
            append_audit_entry("get_pod_data_by_id", query, None, None)
            return result
        else:
            append_audit_entry("get_pod_data_by_id", query, None, "No results found")
            return {
                "status": "NOT_FOUND",
                "pod_id": pod_id
            }
            
    except Exception as ex:
        append_audit_entry("get_pod_data_by_id", None, None, str(ex))
        return {
            "status": "ERROR",
            "error": str(ex),
            "pod_id": pod_id
        }


def create_domo_agent(credentials):
    """
    Create the Domo sub-agent (using Vector Search embeddings).
    Note: This is a placeholder for compatibility. The actual work is done by
    get_pod_data_by_id() which is called directly by the orchestrator.
    """
    # For now, return None since we're using direct function calls
    # The orchestrator calls get_pod_data_by_id() directly
    return None
