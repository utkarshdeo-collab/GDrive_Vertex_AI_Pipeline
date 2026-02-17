"""
Phase 4a: Deploy Vector Search index to an endpoint

Step 1: Create an Index Endpoint (or reuse existing) with public access.
Step 2: If the same deployed_index_id is already in use, undeploy it first.
Step 3: Deploy the current index (config.VECTOR_INDEX_ID) to that endpoint.

After this, the deployed index can be queried for Phase 4 (ADK) retrieval.
"""
import sys
import time
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from google.api_core.exceptions import AlreadyExists, FailedPrecondition
from google.cloud import aiplatform
from google.cloud.aiplatform import matching_engine

# After undeploy, backend may need time to free the deployed_index_id slot.
WAIT_AFTER_UNDEPLOY_SECONDS = 120
# Retry deploy this many times if still "being undeployed".
DEPLOY_RETRIES = 5
DEPLOY_RETRY_WAIT_SECONDS = 60


def get_or_create_endpoint():
    """Return an existing endpoint with our display name, or create a new one (public)."""
    aiplatform.init(project=config.PROJECT_ID, location=config.LOCATION)
    endpoints = list(matching_engine.MatchingEngineIndexEndpoint.list())
    for ep in endpoints:
        name = getattr(getattr(ep, "_gca_resource", None), "display_name", None)
        if name == config.INDEX_ENDPOINT_DISPLAY_NAME:
            return ep
    print("   Creating new index endpoint (public)...")
    return matching_engine.MatchingEngineIndexEndpoint.create(
        display_name=config.INDEX_ENDPOINT_DISPLAY_NAME,
        public_endpoint_enabled=True,
        description="Document pipeline endpoint for Phase 4 (ADK)",
        sync=True,
    )


def main():
    print("\n" + "=" * 60)
    print("  PHASE 4a: Deploy index to endpoint")
    print("=" * 60)

    aiplatform.init(project=config.PROJECT_ID, location=config.LOCATION)

    print("\n[Step 1] Get or create index endpoint...")
    endpoint = get_or_create_endpoint()
    print(f"   Endpoint: {endpoint.resource_name}")

    print("\n[Step 2] Deploy index to endpoint...")
    index = matching_engine.MatchingEngineIndex(config.VECTOR_INDEX_ID)

    def do_deploy():
        endpoint.deploy_index(
            index=index,
            deployed_index_id=config.DEPLOYED_INDEX_ID,
            display_name=config.VECTOR_INDEX_DISPLAY_NAME,
            sync=True,
        )

    def retry_deploy_loop():
        """Retry deploy up to DEPLOY_RETRIES on FailedPrecondition (slot busy/being undeployed)."""
        for attempt in range(DEPLOY_RETRIES):
            try:
                if attempt > 0:
                    print(f"   Deploying (attempt {attempt + 1}/{DEPLOY_RETRIES})...")
                do_deploy()
                return
            except FailedPrecondition as e:
                if "retry again later" in str(e).lower() or "being undeployed" in str(e).lower():
                    if attempt < DEPLOY_RETRIES - 1:
                        print(f"   Slot not ready yet, waiting {DEPLOY_RETRY_WAIT_SECONDS}s before retry...")
                        time.sleep(DEPLOY_RETRY_WAIT_SECONDS)
                    else:
                        raise
                else:
                    raise

    try:
        do_deploy()
    except AlreadyExists:
        print(f"   Existing deployment '{config.DEPLOYED_INDEX_ID}' found. Undeploying (may take a few minutes)...")
        endpoint.undeploy_index(deployed_index_id=config.DEPLOYED_INDEX_ID)
        print("   Undeployed. Waiting for backend to free the slot...")
        time.sleep(WAIT_AFTER_UNDEPLOY_SECONDS)
        retry_deploy_loop()
    except FailedPrecondition as e:
        if "retry again later" in str(e).lower() or "being undeployed" in str(e).lower():
            print("   Slot busy (being undeployed). Waiting and retrying...")
            retry_deploy_loop()
        else:
            raise
    print(f"   Deployed index ID: {config.DEPLOYED_INDEX_ID}")

    print("\n" + "=" * 60)
    print("  Phase 4a complete. Use for ADK retrieval:")
    print(f"    Endpoint: {endpoint.resource_name}")
    print(f"    Deployed index ID: {config.DEPLOYED_INDEX_ID}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
