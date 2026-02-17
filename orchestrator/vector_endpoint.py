"""
Shared Vector Endpoint module for querying Vertex AI Vector Search index.
Used by Salesforce and Domo agents to query the same index with different filters.
"""
import json
import sys
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from google.api_core.exceptions import NotFound
from google.cloud import aiplatform
from google.cloud.aiplatform import matching_engine
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    MatchingEngineIndexEndpoint,
    Namespace,
)
import vertexai
from vertexai.language_models import TextEmbeddingModel

# Lazy-loaded state
_embedding_model = None
_endpoint = None
_chunks_lookup = None

# GCS path for chunks lookup (id -> {source, data}); must match index_sf_domo_data.CHUNKS_PREFIX
CHUNKS_LOOKUP_PREFIX = "sf_domo_chunks"
LOOKUP_BLOB_NAME = f"{CHUNKS_LOOKUP_PREFIX}/lookup.json"

# Default top_k for queries
DEFAULT_TOP_K = 10


def _get_embedding_model():
    """Get or create the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
        _embedding_model = TextEmbeddingModel.from_pretrained(config.EMBEDDING_MODEL)
    return _embedding_model


def _get_endpoint():
    """Get the index endpoint by display name (case-insensitive)."""
    global _endpoint
    if _endpoint is None:
        aiplatform.init(project=config.PROJECT_ID, location=config.LOCATION)
        endpoints = list(matching_engine.MatchingEngineIndexEndpoint.list())
        target = (config.INDEX_ENDPOINT_DISPLAY_NAME or "").strip()
        resource_name = None
        for ep in endpoints:
            name = getattr(getattr(ep, "_gca_resource", None), "display_name", None) or ""
            if name.strip().lower() == target.lower():
                resource_name = ep.resource_name
                break
        if resource_name is None:
            available = [getattr(getattr(ep, "_gca_resource", None), "display_name", None) for ep in endpoints]
            available = [n for n in available if n]
            raise RuntimeError(
                f"No index endpoint with display name '{config.INDEX_ENDPOINT_DISPLAY_NAME}' found. "
                f"Available in {config.LOCATION}: {available or '(none)'}. "
                "Set INDEX_ENDPOINT_DISPLAY_NAME in config.py."
            )
        _endpoint = MatchingEngineIndexEndpoint(resource_name)
    return _endpoint


def query_vector_index(
    query_text: str,
    filter_type: str = None,
    top_k: int = DEFAULT_TOP_K
) -> dict:
    """
    Query the Vertex AI Vector Search index with optional filter.
    
    Args:
        query_text: Text to embed and search for
        filter_type: Filter type - "SF" for Salesforce, "Domo" for Domo, None for no filter
        top_k: Number of neighbors to return
    
    Returns:
        dict with:
            - status: "SUCCESS" or "ERROR"
            - results: List of neighbor results (each with id, distance, and optional metadata)
            - error: Error message if status is "ERROR"
    """
    from .usage_collector import record_embedding
    from .audit_context import append_audit_entry
    
    try:
        record_embedding(len(query_text))
        append_audit_entry("query_vector_index", query_text, None, None)
        
        # Step 1: Embed the query text
        model = _get_embedding_model()
        embeddings = model.get_embeddings([query_text])
        query_embedding = list(embeddings[0].values)
        
        # Step 2: Get endpoint
        endpoint = _get_endpoint()
        endpoint._sync_gca_resource()
        deployed_ids = [d.id for d in (getattr(endpoint, "deployed_indexes", None) or [])]
        
        if deployed_ids and config.DEPLOYED_INDEX_ID not in deployed_ids:
            return {
                "status": "ERROR",
                "error": f"Deployed index '{config.DEPLOYED_INDEX_ID}' not found. Available: {deployed_ids}.",
                "results": []
            }
        
        # Step 3: Build filter namespace if filter_type is specified
        filter_ns = None
        if filter_type:
            filter_type_upper = filter_type.upper()
            if filter_type_upper == "SF" or filter_type_upper == "SALESFORCE":
                # Filter for Salesforce data
                filter_ns = [Namespace(name="source", allow_tokens=["SF", "Salesforce"], deny_tokens=[])]
            elif filter_type_upper == "DOMO":
                # Filter for Domo data
                filter_ns = [Namespace(name="source", allow_tokens=["Domo"], deny_tokens=[])]
        
        # Step 4: Query the index
        try:
            if filter_ns:
                results = endpoint.find_neighbors(
                    deployed_index_id=config.DEPLOYED_INDEX_ID,
                    queries=[query_embedding],
                    num_neighbors=top_k,
                    filter=filter_ns,
                )
            else:
                results = endpoint.find_neighbors(
                    deployed_index_id=config.DEPLOYED_INDEX_ID,
                    queries=[query_embedding],
                    num_neighbors=top_k,
                )
            neighbors = results[0] if results else []
            
            # Convert neighbors to dict format
            results_list = []
            for neighbor in neighbors:
                neighbor_dict = {
                    "id": neighbor.id if isinstance(neighbor.id, str) else str(neighbor.id),
                    "distance": getattr(neighbor, "distance", None),
                }
                # Add any metadata if available
                if hasattr(neighbor, "metadata"):
                    neighbor_dict["metadata"] = neighbor.metadata
                results_list.append(neighbor_dict)
            
            return {
                "status": "SUCCESS",
                "results": results_list,
                "query_text": query_text,
                "filter_type": filter_type,
                "num_results": len(results_list)
            }
            
        except NotFound as e:
            append_audit_entry("query_vector_index", query_text, None, str(e))
            return {
                "status": "ERROR",
                "error": f"Deployed index '{config.DEPLOYED_INDEX_ID}' not found. Details: {e}",
                "results": []
            }
            
    except Exception as ex:
        append_audit_entry("query_vector_index", query_text, None, str(ex))
        return {
            "status": "ERROR",
            "error": str(ex),
            "results": []
        }


def load_chunks_lookup():
    """
    Load id -> {source, data} lookup from GCS (used to resolve vector result IDs without BigQuery).
    Cached in memory after first load.
    """
    global _chunks_lookup
    if _chunks_lookup is not None:
        return _chunks_lookup
    from google.cloud import storage
    try:
        client = storage.Client(project=config.PROJECT_ID)
        bucket = client.bucket(config.GCS_BUCKET_NAME)
        blob = bucket.blob(LOOKUP_BLOB_NAME)
        data = blob.download_as_string()
        _chunks_lookup = json.loads(data)
        return _chunks_lookup
    except Exception as e:
        # Return empty dict so callers can fall back to BigQuery
        return {}


def get_data_from_vector_results(vector_results: dict, data_store: dict = None) -> dict:
    """
    Extract structured data from vector search results.
    
    Args:
        vector_results: Results from query_vector_index
        data_store: Optional dict mapping id -> data (e.g., from GCS or BigQuery)
    
    Returns:
        dict with extracted data
    """
    if vector_results.get("status") != "SUCCESS":
        return {
            "status": "ERROR",
            "error": vector_results.get("error", "Vector query failed"),
            "data": []
        }
    
    results = vector_results.get("results", [])
    extracted_data = []
    
    for result in results:
        result_id = result.get("id")
        data_item = {
            "id": result_id,
            "distance": result.get("distance"),
        }
        
        # If data_store is provided, try to get full data
        if data_store and result_id in data_store:
            data_item["data"] = data_store[result_id]
        elif result.get("metadata"):
            # Use metadata if available
            data_item["data"] = result["metadata"]
        else:
            # Just use the ID
            data_item["data"] = {"id": result_id}
        
        extracted_data.append(data_item)
    
    return {
        "status": "SUCCESS",
        "data": extracted_data,
        "num_results": len(extracted_data)
    }
