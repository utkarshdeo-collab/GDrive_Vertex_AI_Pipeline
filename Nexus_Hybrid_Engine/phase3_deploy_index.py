"""
Phase 3: Deploy Vector Search Index (Production Quality)

This script automates the creation and deployment of the Vertex AI Vector Search Index.
It adheres to strict coding standards:
- PEP 8 compliance
- Modular function design
- Robust error handling
- Type hinting

Workflow:
1. Create a Vector Search Index from the GCS data (output of Phase 2).
2. Create (or retrieve) an Index Endpoint.
3. Deploy the Index to the Endpoint.

Usage:
    python Nexus_Hybrid_Engine/phase3_deploy_index.py
"""

import sys
import time
from pathlib import Path
from typing import Optional

from google.cloud import aiplatform, storage
from google.api_core import exceptions

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from Nexus_Hybrid_Engine import nexus_config as config
except ImportError:
    import nexus_config as config


class VectorDeployer:
    """Manages the lifecycle of Vertex AI Vector Search Index deployment."""

    def __init__(self):
        """Initialize the deployer with configuration."""
        self.project_id = config.PROJECT_ID
        self.location = config.LOCATION
        self.bucket_name = config.GCS_BUCKET_NAME
        self.gcs_uri = config.VECTORS_EMBEDDED_GCS
        
        # Index Settings
        self.display_name = "nexus-hybrid-index"
        self.dimensions = 768 # text-embedding-004
        self.approx_neighbors_count = 150 # Higher for better recall
        
        # Endpoint Settings
        self.endpoint_name = "nexus-hybrid-endpoint"
        self.deployed_index_id = "nexus_deployed_index_v1"

    def verify_gcs_data(self) -> bool:
        """Verifies that the input data exists in GCS."""
        print(f"[Check] Verifying data at: {self.gcs_uri}")
        
        # Extract bucket and prefix
        if not self.gcs_uri.startswith("gs://"):
            print(f"[Error] Invalid GCS URI: {self.gcs_uri}")
            return False
            
        try:
            storage_client = storage.Client(project=self.project_id)
            parts = self.gcs_uri.replace("gs://", "").split("/", 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            
            bucket = storage_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
            
            if not blobs:
                print(f"[Error] No files found at {self.gcs_uri}")
                return False
                
            print(f"[Check] Data found! First file: {blobs[0].name}")
            return True
            
        except Exception as e:
            print(f"[Error] Failed to access GCS: {e}")
            return False

    def create_or_get_index(self) -> aiplatform.MatchingEngineIndex:
        """Get existing index or create new one."""
        print(f"\n[Step 1] Setting up Vector Search Index '{self.display_name}'...")
        
        aiplatform.init(project=self.project_id, location=self.location)
        
        # Check for existing index
        indexes = aiplatform.MatchingEngineIndex.list(
            filter=f'display_name="{self.display_name}"'
        )
        
        if indexes:
            print(f"[Info] Found existing index: {indexes[0].resource_name}")
            return indexes[0]
            
        print(f"[Action] Creating new index (this takes ~30 mins)...")
        try:
            index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                display_name=self.display_name,
                contents_delta_uri=self.gcs_uri,
                dimensions=self.dimensions,
                approximate_neighbors_count=self.approx_neighbors_count,
                distance_measure_type="DOT_PRODUCT_DISTANCE",
                leaf_node_embedding_count=500,
                leaf_nodes_to_search_percent=7,
                description="Nexus Hybrid Engine Index (80k records)",
                sync=True
            )
            print(f"[Success] Index created: {index.resource_name}")
            return index
        except Exception as e:
            print(f"[Error] Failed to create index: {e}")
            raise

    def get_or_create_endpoint(self) -> aiplatform.MatchingEngineIndexEndpoint:
        """Retrieves an existing endpoint or creates a new one."""
        print(f"\n[Step 2] Setting up Endpoint '{self.endpoint_name}'...")
        
        aiplatform.init(project=self.project_id, location=self.location)
        
        endpoints = aiplatform.MatchingEngineIndexEndpoint.list(
            filter=f'display_name="{self.endpoint_name}"'
        )
        
        if endpoints:
            print(f"[Info] Found existing endpoint: {endpoints[0].resource_name}")
            return endpoints[0]
            
        print("[Info] Creating new endpoint...")
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=self.endpoint_name,
            public_endpoint_enabled=True,
            description="Endpoint for Nexus Hybrid Engine"
        )
        print(f"[Success] Endpoint created: {endpoint.resource_name}")
        return endpoint

    def deploy_index(
        self, 
        index: aiplatform.MatchingEngineIndex, 
        endpoint: aiplatform.MatchingEngineIndexEndpoint
    ) -> None:
        """Deploys the index to the endpoint."""
        print(f"\n[Step 3] Deploying Index to Endpoint...")
        
        # Check if already deployed
        for deployed in endpoint.deployed_indexes:
            if deployed.id == self.deployed_index_id:
                print(f"[Info] Undeploying existing index '{self.deployed_index_id}' to update...")
                endpoint.undeploy_index(self.deployed_index_id)
                print("[Info] Undeploy complete.")
                break
        
        print(f"[Action] Deploying new index (this takes ~15-20 mins)...")
        endpoint.deploy_index(
            index=index,
            deployed_index_id=self.deployed_index_id,
            display_name=self.deployed_index_id,
            machine_type="e2-standard-16",
            min_replica_count=1,
            max_replica_count=1
        )
        print(f"[Success] Index deployed successfully!")

    def run(self):
        """Main execution flow."""
        print("="*80)
        print("PHASE 3: Deploy Vector Search Index")
        print("="*80)
        
        if not self.verify_gcs_data():
            sys.exit(1)
            
        try:
            index = self.create_or_get_index() # Returns OBJECT
            endpoint = self.get_or_create_endpoint()
            self.deploy_index(index, endpoint) # Passes OBJECT
            
            print("\n" + "="*80)
            print("DEPLOYMENT COMPLETE! 🚀")
            print(f"Index:    {index.resource_name}")
            print(f"Endpoint: {endpoint.resource_name}")
            print("="*80)
            
        except Exception as e:
            print(f"\n[Fatal Error] Deployment failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    deployer = VectorDeployer()
    deployer.run()
