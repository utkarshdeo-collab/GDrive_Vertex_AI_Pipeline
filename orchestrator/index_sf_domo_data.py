"""
Script to index Salesforce and Domo data into Vertex AI Vector Search index.

This script:
1. Reads Salesforce account data from BigQuery (nexus_data.test_dataset2)
2. Reads Domo pod data from BigQuery (domo_test_dataset.test_pod)
3. Converts each record to a text representation
4. Generates embeddings using text-embedding-004
5. Uploads to GCS in the format required for Vector Search indexing
6. Creates/updates the Vector Search index with namespace filters (SF, Domo)

The indexed data will be queryable via the vector_endpoint module with filters.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from google.cloud import bigquery, storage
import google.auth
import google.auth.transport.requests
import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
from google.cloud.aiplatform import matching_engine
from google.cloud.aiplatform.matching_engine import matching_engine_index_config as me_config

# Embedding model
EMBEDDING_MODEL = config.EMBEDDING_MODEL
EMBEDDING_DIM = 768  # text-embedding-004 dimension

# GCS paths
EMBEDDINGS_PREFIX = "sf_domo_embeddings"
CHUNKS_PREFIX = "sf_domo_chunks"


def build_salesforce_text(account_row: dict) -> str:
    """Build a text representation of Salesforce account data."""
    parts = [
        f"Account: {account_row.get('Customer_Name', '')}",
        f"Account ID: {account_row.get('Salesforce_Account_ID', '')}",
        f"Total ARR: {account_row.get('Total_ARR', '')}",
        f"Renewal Date: {account_row.get('Renewal_Date', '')}",
        f"Account Owner: {account_row.get('Account_Owner', '')}",
        f"POD ID: {account_row.get('POD_Internal_Id__c', '')}",
        f"Task Count: {account_row.get('Task_Count', '')}",
    ]
    return " | ".join([p for p in parts if p.split(": ")[1]])


def build_domo_text(pod_row: dict) -> str:
    """Build a text representation of Domo pod data."""
    parts = [
        f"POD ID: {pod_row.get('pod_id', '')}",
        f"Pretty Name: {pod_row.get('pretty_name', '')}",
        f"MEAU: {pod_row.get('meau', '')}",
        f"Health Score: {pod_row.get('health_score', '')}",
        f"Provisioned Users: {pod_row.get('provisioned_users', '')}",
        f"Active Users: {pod_row.get('active_users', '')}",
        f"Contracted Licenses: {pod_row.get('contracted_licenses', '')}",
        f"Risk Ratio: {pod_row.get('risk_ratio_for_next_renewal', '')}",
    ]
    return " | ".join([p for p in parts if p.split(": ")[1]])


def fetch_salesforce_data() -> list:
    """Fetch Salesforce account data from BigQuery."""
    client = bigquery.Client(project=config.PROJECT_ID)
    query = f"""
    SELECT Customer_Name, Salesforce_Account_ID, Total_ARR, Renewal_Date, 
           Account_Owner, POD_Internal_Id__c, Task_Count
    FROM `{config.PROJECT_ID}.nexus_data.test_dataset2`
    """
    job = client.query(query)
    rows = []
    for row in job.result():
        rows.append(dict(row))
    return rows


def fetch_domo_data() -> list:
    """Fetch Domo pod data from BigQuery (latest month per pod)."""
    client = bigquery.Client(project=config.PROJECT_ID)
    query = f"""
    SELECT pod_id, pretty_name, meau, provisioned_users, active_users, 
           health_score, risk_ratio_for_next_renewal, contracted_licenses
    FROM (
        SELECT pod_id, pretty_name, meau, provisioned_users, active_users,
               health_score, risk_ratio_for_next_renewal, contracted_licenses,
               ROW_NUMBER() OVER (PARTITION BY pod_id ORDER BY `month` DESC) as rn
        FROM `{config.PROJECT_ID}.domo_test_dataset.test_pod`
    )
    WHERE rn = 1
    """
    job = client.query(query)
    rows = []
    for row in job.result():
        rows.append(dict(row))
    return rows


def generate_embeddings(texts: list, batch_size: int = 100) -> list:
    """Generate embeddings for a list of texts."""
    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Embedding batch {i//batch_size + 1}/{total_batches} ({len(batch)} texts)...", flush=True)
        embeddings = model.get_embeddings(batch)
        all_embeddings.extend([list(e.values) for e in embeddings])
    
    return all_embeddings


def create_vector_index_entries(sf_data: list, domo_data: list) -> tuple:
    """
    Create vector index entries for Salesforce and Domo data.
    Returns: (chunks_dict, embeddings_jsonl_lines)
    """
    chunks_dict = {}
    embeddings_lines = []
    
    # Index Salesforce data
    print("\n[1] Processing Salesforce data...")
    sf_texts = []
    for i, row in enumerate(sf_data):
        text = build_salesforce_text(row)
        chunk_id = f"sf_{i}_{row.get('Salesforce_Account_ID', 'unknown')}"
        chunks_dict[chunk_id] = {
            "id": chunk_id,
            "text": text,
            "data": row,
            "source": "SF"
        }
        sf_texts.append(text)
    
    print(f"  Generated {len(sf_texts)} Salesforce text entries")
    
    # Index Domo data
    print("\n[2] Processing Domo data...")
    domo_texts = []
    for i, row in enumerate(domo_data):
        text = build_domo_text(row)
        pod_id = row.get('pod_id', 'unknown')
        chunk_id = f"domo_{i}_pod{pod_id}"
        chunks_dict[chunk_id] = {
            "id": chunk_id,
            "text": text,
            "data": row,
            "source": "Domo"
        }
        domo_texts.append(text)
    
    print(f"  Generated {len(domo_texts)} Domo text entries")
    
    # Generate embeddings
    print("\n[3] Generating embeddings...")
    all_texts = sf_texts + domo_texts
    all_embeddings = generate_embeddings(all_texts)
    
    # Create embeddings JSONL
    print("\n[4] Creating embeddings JSONL...")
    chunk_ids = list(chunks_dict.keys())
    for i, (chunk_id, embedding) in enumerate(zip(chunk_ids, all_embeddings)):
        chunk_info = chunks_dict[chunk_id]
        entry = {
            "id": chunk_id,
            "embedding": embedding,
            "restricts": [
                {
                    "namespace": "source",
                    "allow": [chunk_info["source"]]
                }
            ]
        }
        # Only include numeric_restricts if needed (empty array is fine)
        # sparse_embedding should be omitted entirely if not used (not null)
        embeddings_lines.append(json.dumps(entry))
    
    return chunks_dict, embeddings_lines


def ensure_bucket(bucket_name: str):
    """Ensure GCS bucket exists, create if not."""
    from google.cloud.exceptions import NotFound, Conflict
    storage_client = storage.Client(project=config.PROJECT_ID)
    try:
        bucket = storage_client.get_bucket(bucket_name)
        print(f"  Bucket exists: gs://{bucket_name}")
        return bucket
    except NotFound:
        print(f"  Creating bucket: gs://{bucket_name} in {config.LOCATION}...")
        try:
            bucket = storage_client.create_bucket(bucket_name, location=config.LOCATION)
            print(f"  ✓ Bucket created successfully!")
            return bucket
        except Conflict:
            print(f"  ERROR: Bucket name '{bucket_name}' is already taken globally.")
            raise
    except Exception as e:
        print(f"  ERROR creating bucket: {e}")
        raise


def upload_embeddings_to_gcs(embeddings_lines: list):
    """Upload embeddings to GCS (required for Vector Search index creation)."""
    import os
    import tempfile
    from google.cloud.storage import Client
    
    bucket_name = config.GCS_BUCKET_NAME
    
    # Ensure bucket exists
    print(f"\n[5a] Ensuring bucket exists: {bucket_name}")
    bucket = ensure_bucket(bucket_name)
    
    # Upload embeddings (only file needed for index creation)
    print("\n[5] Uploading embeddings to GCS...")
    embeddings_jsonl = "\n".join(embeddings_lines)
    file_size_mb = len(embeddings_jsonl.encode('utf-8')) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB ({len(embeddings_lines)} embeddings)")
    
    # Write to temp file first (better for large files)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.json') as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(embeddings_jsonl)
    
    try:
        print("  Uploading via file-based upload (handles large files better)...")
        
        # Create storage client
        storage_client = Client(project=config.PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        embeddings_blob = bucket.blob(f"{EMBEDDINGS_PREFIX}/embeddings.json")
        
        # Upload from file - this automatically uses resumable upload for large files
        # Use a very long timeout to handle slow/unstable connections
        print("  Starting upload (may take 5-10 minutes for 93MB file)...")
        print("  Note: If upload fails due to network issues, the file is saved locally for manual upload")
        
        # Try with file handle approach (more reliable for large files)
        with open(tmp_path, 'rb') as f:
            embeddings_blob.upload_from_file(
                f,
                content_type="application/jsonl",
                rewind=True,
                timeout=1800,  # 30 minutes total timeout
            )
        
        embeddings_gcs = f"gs://{bucket_name}/{EMBEDDINGS_PREFIX}/embeddings.json"
        print(f"  ✓ Uploaded embeddings to {embeddings_gcs}")
        return embeddings_gcs
    except Exception as e:
        error_msg = str(e)
        print(f"\n  ⚠ Upload error: {error_msg}")
        print(f"\n  File saved locally at: {tmp_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"\n  Options:")
        print(f"  1. Retry the upload: python orchestrator/index_sf_domo_data.py --upload-only")
        print(f"  2. Upload manually via Google Cloud Console:")
        print(f"     - Go to: https://console.cloud.google.com/storage/browser/{bucket_name}")
        print(f"     - Navigate to: {EMBEDDINGS_PREFIX}/")
        print(f"     - Upload: {tmp_path}")
        print(f"  3. Check network connection and retry")
        
        # Don't delete temp file on error so user can manually upload
        if "timeout" in error_msg.lower() or "ssl" in error_msg.lower():
            print(f"\n  ⚠ Network/SSL issue detected. Keeping file for manual upload.")
            return None
        raise
    finally:
        # Only clean up if upload succeeded
        # On error, keep the file for manual upload
        pass


def _serialize_value(v):
    """Make a value JSON-serializable (e.g. date/datetime)."""
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_serialize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _serialize_value(x) for k, x in v.items()}
    return str(v)


def upload_chunks_lookup_to_gcs(chunks_dict: dict):
    """Upload id -> {source, data} lookup to GCS so agents can resolve vector result IDs without BigQuery."""
    bucket_name = config.GCS_BUCKET_NAME
    bucket = ensure_bucket(bucket_name)
    lookup = {}
    for cid, info in chunks_dict.items():
        data = info.get("data") or {}
        data_ser = {k: _serialize_value(v) for k, v in data.items()}
        lookup[cid] = {"source": info.get("source", ""), "data": data_ser}
    blob = bucket.blob(f"{CHUNKS_PREFIX}/lookup.json")
    blob.upload_from_string(
        json.dumps(lookup),
        content_type="application/json",
    )
    print(f"  ✓ Uploaded chunks lookup to gs://{bucket_name}/{CHUNKS_PREFIX}/lookup.json ({len(lookup)} entries)")


def create_or_update_index(sync: bool = False):
    """
    Create or update the Vector Search index.
    
    Args:
        sync: If True, wait for index creation to complete (blocks for 30+ minutes).
              If False, submit async and return immediately (default).
    
    Note: Index creation time is the same regardless of sync mode. The difference is:
    - sync=False: Submit and return immediately (fastest to get started)
    - sync=True: Wait until completion (useful for scripts that need the index ID immediately)
    
    There is no faster way to create the index - the backend build process takes 30+ minutes
    regardless of whether you use batch or streaming index types.
    """
    print("\n[7] Creating/updating Vector Search index...")
    aiplatform.init(project=config.PROJECT_ID, location=config.LOCATION)
    
    contents_uri = f"gs://{config.GCS_BUCKET_NAME}/{EMBEDDINGS_PREFIX}/"
    
    # Check if index exists
    try:
        existing_index = matching_engine.MatchingEngineIndex(config.VECTOR_INDEX_ID)
        print(f"  Using existing index: {config.VECTOR_INDEX_ID}")
        print(f"  To create a new index, update config.VECTOR_INDEX_ID or create manually in Console")
        return
    except Exception:
        pass
    
    # Create new index
    print("  Creating new Vector Search index...")
    if sync:
        print("  Using sync=True - this will block until index creation completes (30+ minutes)...")
    else:
        print("  Using sync=False - index will be created asynchronously (submission is quick)")
    
    index = matching_engine.MatchingEngineIndex.create_tree_ah_index(
        display_name=f"{config.VECTOR_INDEX_DISPLAY_NAME}-sf-domo",
        contents_delta_uri=contents_uri,
        dimensions=EMBEDDING_DIM,
        approximate_neighbors_count=10,
        leaf_node_embedding_count=500,
        leaf_nodes_to_search_percent=10,
        distance_measure_type=me_config.DistanceMeasureType.DOT_PRODUCT_DISTANCE,
        feature_norm_type=me_config.FeatureNormType.UNIT_L2_NORM,
        description="SF and Domo data index",
        sync=sync,
    )
    
    if sync:
        print("  ✓ Index creation completed!")
        try:
            index_resource = getattr(index, "resource_name", None) or str(index)
            if index_resource and "/indexes/" in str(index_resource):
                index_id = index_resource.split("/indexes/")[-1].split("/")[0].strip()
                print(f"  Index ID: {index_id}")
                print(f"  Update config.VECTOR_INDEX_ID with: {index_id}")
        except Exception as e:
            print(f"  Note: Could not extract index ID: {e}")
    else:
        print("  Index creation submitted (async). Build may take 30+ minutes.")
        print("  Check status in Console: Vertex AI → Vector Search.")
        print(f"  Embeddings URI: {contents_uri}")
        
        # Try to get resource name (may not be available immediately with sync=False)
        try:
            index_resource = getattr(index, "resource_name", None) or str(index)
            if index_resource and "/indexes/" in str(index_resource):
                index_id = index_resource.split("/indexes/")[-1].split("/")[0].strip()
                print(f"  Index ID: {index_id}")
                print(f"  Update config.VECTOR_INDEX_ID with: {index_id}")
        except RuntimeError as re:
            if "has not been created" in str(re):
                print("  Index is still building. Resource name will be available after creation completes.")
            else:
                raise


def check_authentication():
    """Check if user is authenticated (BigQuery/Vertex or gcloud CLI)."""
    import subprocess
    
    # Check BigQuery auth (needed for fetching data)
    try:
        bigquery.Client(project=config.PROJECT_ID).query("SELECT 1").result()
        print("  ✓ Authentication verified (BigQuery)")
        return True
    except Exception:
        pass
    
    # Fallback: check gcloud CLI
    try:
        result = subprocess.run(
            ["gcloud", "auth", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and "ACTIVE" in result.stdout:
            print("  ✓ gcloud CLI authentication verified")
            return True
    except FileNotFoundError:
        pass
    except Exception:
        pass
    
    # No authentication found
    print("  ⚠ No authentication found")
    return False


def upload_existing_embeddings():
    """Upload existing embeddings file from temp_embeddings/ to GCS, then create index."""
    import os
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "temp_embeddings")
    embeddings_path = os.path.join(output_dir, "embeddings.json")
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    print(f"  Found embeddings file: {embeddings_path}")
    
    # Read embeddings file
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        embeddings_jsonl = f.read()
    
    # Parse embeddings to get embeddings_lines format
    embeddings_lines = []
    for line in embeddings_jsonl.strip().split('\n'):
        if line:
            embeddings_lines.append(line)
    
    print(f"  Loaded {len(embeddings_lines)} embeddings")
    
    # Upload embeddings to GCS (required for index creation)
    result = upload_embeddings_to_gcs(embeddings_lines)
    if result is None:
        # Upload failed, but file is saved locally for manual upload
        raise RuntimeError("Upload failed. Please upload manually or retry with better network connection.")


def main():
    """Main function to index SF and Domo data."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Index SF and Domo data to Vector Search")
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Only upload existing files from temp_embeddings/ to GCS, skip data fetching and embedding generation"
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip file preparation, assume files are already uploaded to GCS"
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Wait for index creation to complete (blocks for 30+ minutes). Default is async (faster to start)"
    )
    parser.add_argument(
        "--upload-lookup-only",
        action="store_true",
        help="Fetch SF+Domo from BigQuery, build chunks lookup, upload to GCS only (so orchestrator uses vector search without BQ)"
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  INDEX SALESFORCE AND DOMO DATA TO VECTOR SEARCH")
    print("=" * 60)
    
    # Check authentication (needed for BigQuery and Vertex AI)
    print("\n[0] Checking authentication...")
    if not check_authentication():
        print("\n  Please authenticate:")
        print("    gcloud auth application-default login")
        print(f"    gcloud config set project {config.PROJECT_ID}")
        print("\n  Then run this script again.")
        return
    
    if args.upload_only:
        # Just upload existing embeddings and create index
        print("\n[1] Uploading existing embeddings to GCS...")
        try:
            upload_existing_embeddings()
            print("\n[2] Creating/updating Vector Search index...")
            create_or_update_index(sync=args.sync)
        except FileNotFoundError as e:
            print(f"\n  ERROR: {e}")
            print("  Run the script without --upload-only to generate files first.")
            return
        return
    
    if args.skip_upload:
        # Skip upload - proceed to index creation
        print("\n[1] Skipping upload (assuming files are already in GCS)...")
        print("\n[2] Creating/updating Vector Search index...")
        create_or_update_index()
        return
    
    if args.upload_lookup_only:
        # Build and upload only the chunks lookup so orchestrator can use vector search (no BQ for SF/Domo)
        print("\n[1] Fetching data from BigQuery...")
        sf_data = fetch_salesforce_data()
        domo_data = fetch_domo_data()
        if not sf_data and not domo_data:
            print("  No data. Exiting.")
            return
        print("\n[2] Building chunks lookup...")
        chunks_dict = {}
        for i, row in enumerate(sf_data):
            chunk_id = f"sf_{i}_{row.get('Salesforce_Account_ID', 'unknown')}"
            chunks_dict[chunk_id] = {"id": chunk_id, "text": "", "data": row, "source": "SF"}
        for i, row in enumerate(domo_data):
            chunk_id = f"domo_{i}_pod{row.get('pod_id', 'unknown')}"
            chunks_dict[chunk_id] = {"id": chunk_id, "text": "", "data": row, "source": "Domo"}
        print("\n[3] Uploading chunks lookup to GCS...")
        upload_chunks_lookup_to_gcs(chunks_dict)
        print("\n  Done. Orchestrator will now use vector search + this lookup (no BigQuery for SF/Domo).")
        return
    
    # Full flow: fetch data, generate embeddings, upload, create index
    # Fetch data
    print("\n[1] Fetching data from BigQuery...")
    sf_data = fetch_salesforce_data()
    print(f"  Fetched {len(sf_data)} Salesforce accounts")
    
    domo_data = fetch_domo_data()
    print(f"  Fetched {len(domo_data)} Domo pods")
    
    if not sf_data and not domo_data:
        print("  No data to index. Exiting.")
        return
    
    # Create vector index entries
    print("\n[2] Creating vector index entries...")
    chunks_dict, embeddings_lines = create_vector_index_entries(sf_data, domo_data)
    
    # Upload embeddings to GCS (required for index creation)
    print("\n[3] Uploading embeddings to GCS...")
    upload_embeddings_to_gcs(embeddings_lines)
    
    # Upload chunks lookup so agents can resolve vector IDs without BigQuery
    print("\n[3b] Uploading chunks lookup to GCS...")
    upload_chunks_lookup_to_gcs(chunks_dict)
    
    # Create/update index
    print("\n[4] Creating/updating Vector Search index...")
    create_or_update_index()
    
    print("\n" + "=" * 60)
    print("  INDEXING COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Wait for index creation to complete (check Console)")
    print(f"2. Deploy the index to an endpoint (use phase4_deploy_index.py)")
    print(f"3. Update config.DEPLOYED_INDEX_ID with the deployed index ID")
    print(f"4. The orchestrator will now use vector search for SF and Domo queries")


if __name__ == "__main__":
    main()
