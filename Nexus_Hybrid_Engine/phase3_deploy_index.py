"""
Phase 3: Deploy Vector Search Index

This script:
1. Creates a new Vertex AI Vector Search index from embedded JSONL
2. Creates an Index Endpoint (or reuses existing)
3. Deploys the index to the endpoint
4. Prints the endpoint resource name for use in orchestrator

Run from project root:
  python Nexus_Hybrid_Engine/phase3_deploy_index.py
"""
import sys
import time
from pathlib import Path

from google.cloud import aiplatform
from google.api_core.exceptions import AlreadyExists, NotFound

# Add parent to path for config import
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nexus_Hybrid_Engine import nexus_config as config


def create_vector_index() -> str:
    """
    Create a new Vector Search index from embedded JSONL in GCS.
    Returns the index resource name.
    """
    print(f"\n[Step 1] Creating Vector Search index...")
    print(f"  Display name: {config.VECTOR_INDEX_DISPLAY_NAME}")
    print(f"  Dimensions:   {config.EMBEDDING_DIM}")
    print(f"  Distance:     {config.DISTANCE_MEASURE}")
    print(f"  Input data:   {config.VECTORS_EMBEDDED_GCS}")
    
    aiplatform.init(project=config.PROJECT_ID, location=config.LOCATION)
    
    # Create index configuration
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=config.VECTOR_INDEX_DISPLAY_NAME,
        contents_delta_uri=config.VECTORS_EMBEDDED_GCS,
        dimensions=config.EMBEDDING_DIM,
        approximate_neighbors_count=10,
        distance_measure_type=config.DISTANCE_MEASURE,
        leaf_node_embedding_count=500,
        leaf_nodes_to_search_percent=7,
        description="Nexus Hybrid Engine - Account snapshots with calculated business logic",
    )
    
    print(f"  Index created: {index.resource_name}")
    print(f"  Index ID: {index.name}")
    
    return index.resource_name


def create_or_get_endpoint() -> aiplatform.MatchingEngineIndexEndpoint:
    """
    Create a new Index Endpoint or get existing one.
    Returns the endpoint object.
    """
    print(f"\n[Step 2] Creating Index Endpoint...")
    print(f"  Display name: {config.INDEX_ENDPOINT_DISPLAY_NAME}")
    
    aiplatform.init(project=config.PROJECT_ID, location=config.LOCATION)
    
    # Check if endpoint already exists
    try:
        endpoints = aiplatform.MatchingEngineIndexEndpoint.list(
            filter=f'display_name="{config.INDEX_ENDPOINT_DISPLAY_NAME}"'
        )
        if endpoints:
            endpoint = endpoints[0]
            print(f"  Using existing endpoint: {endpoint.resource_name}")
            return endpoint
    except Exception as e:
        print(f"  No existing endpoint found: {e}")
    
    # Create new endpoint
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=config.INDEX_ENDPOINT_DISPLAY_NAME,
        description="Nexus Hybrid Engine endpoint for account search",
        public_endpoint_enabled=True,
    )
    
    print(f"  Endpoint created: {endpoint.resource_name}")
    return endpoint


def deploy_index_to_endpoint(index_resource_name: str, endpoint: aiplatform.MatchingEngineIndexEndpoint) -> None:
    """
    Deploy the index to the endpoint.
    If already deployed with same ID, undeploy first.
    """
    print(f"\n[Step 3] Deploying index to endpoint...")
    print(f"  Deployed index ID: {config.DEPLOYED_INDEX_ID}")
    
    # Check if already deployed
    try:
        deployed_indexes = endpoint.deployed_indexes
        for deployed in deployed_indexes:
            if deployed.id == config.DEPLOYED_INDEX_ID:
                print(f"  Undeploying existing index: {config.DEPLOYED_INDEX_ID}")
                endpoint.undeploy_index(deployed_index_id=config.DEPLOYED_INDEX_ID)
                print(f"  Waiting for undeploy to complete...")
                time.sleep(30)  # Wait for undeploy
                break
    except Exception as e:
        print(f"  No existing deployment found: {e}")
    
    # Deploy index
    print(f"  Deploying index...")
    endpoint.deploy_index(
        index=index_resource_name,
        deployed_index_id=config.DEPLOYED_INDEX_ID,
        display_name=config.DEPLOYED_INDEX_ID,
        machine_type="e2-standard-2",
        min_replica_count=1,
        max_replica_count=1,
    )
    
    print(f"  ✓ Index deployed successfully")


def main():
    print("\n" + "=" * 80)
    print("  NEXUS HYBRID ENGINE - Phase 3: Deploy Vector Search Index")
    print("=" * 80)
    print(f"\n  Project: {config.PROJECT_ID}")
    print(f"  Region:  {config.LOCATION}")
    
    # Verify embedded JSONL exists in GCS
    print(f"\n  Verifying input data: {config.VECTORS_EMBEDDED_GCS}")
    from google.cloud import storage
    try:
        storage_client = storage.Client(project=config.PROJECT_ID)
        parts = config.VECTORS_EMBEDDED_GCS.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(parts[0])
        blob = bucket.blob(parts[1])
        if not blob.exists():
            print(f"  ERROR: Embedded JSONL not found in GCS")
            print(f"         Run Phase 2 first to generate embeddings")
            sys.exit(1)
        print(f"  ✓ Input data found")
    except Exception as e:
        print(f"  ERROR checking GCS: {e}")
        sys.exit(1)
    
    # Step 1: Create index
    try:
        index_resource_name = create_vector_index()
    except Exception as e:
        print(f"\n  ERROR creating index: {e}")
        print(f"\n  If index already exists, you can skip this step")
        print(f"  and manually set VECTOR_INDEX_ID in nexus_config.py")
        sys.exit(1)
    
    # Step 2: Create or get endpoint
    try:
        endpoint = create_or_get_endpoint()
    except Exception as e:
        print(f"\n  ERROR creating endpoint: {e}")
        sys.exit(1)
    
    # Step 3: Deploy index
    try:
        deploy_index_to_endpoint(index_resource_name, endpoint)
    except Exception as e:
        print(f"\n  ERROR deploying index: {e}")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("  Phase 3 Complete!")
    print("=" * 80)
    print(f"\n  Index Resource Name:")
    print(f"    {index_resource_name}")
    print(f"\n  Endpoint Resource Name:")
    print(f"    {endpoint.resource_name}")
    print(f"\n  Deployed Index ID:")
    print(f"    {config.DEPLOYED_INDEX_ID}")
    
    # Extract index ID from resource name
    index_id = index_resource_name.split("/")[-1]
    
    print("\n" + "-" * 80)
    print("  IMPORTANT: Update nexus_config.py with these values:")
    print("-" * 80)
    print(f'  VECTOR_INDEX_ID = "{index_id}"')
    print(f'  INDEX_ENDPOINT_RESOURCE_NAME = "{endpoint.resource_name}"')
    print("\n  Or set environment variables:")
    print(f'  set NEXUS_VECTOR_INDEX_ID="{index_id}"')
    print(f'  set NEXUS_INDEX_ENDPOINT_RESOURCE="{endpoint.resource_name}"')
    print("=" * 80)
    
    print("\n  Next: Verify deployment in GCP Console, then integrate with orchestrator")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
