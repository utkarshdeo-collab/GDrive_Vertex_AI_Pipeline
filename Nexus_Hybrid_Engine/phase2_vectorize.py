"""
Phase 2: Vectorization (Generate Embeddings)

This script:
1. Reads nexus_vectors_input.jsonl from GCS
2. Generates embeddings using Vertex AI text-embedding-004
3. Outputs nexus_vectors_embedded.jsonl to GCS (ready for index creation)

Two modes:
  - Batch mode (recommended): Uses Vertex AI Batch Prediction for large datasets
  - Streaming mode: Generates embeddings one-by-one (slower, for small datasets)

Run from project root:
  python Nexus_Hybrid_Engine/phase2_vectorize.py [--mode batch|streaming]
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from google.cloud import storage
from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

# Add parent to path for config import
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nexus_Hybrid_Engine import nexus_config as config


def download_from_gcs(gcs_uri: str, local_path: Path) -> None:
    """Download file from GCS to local path."""
    # Parse gs://bucket/path
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1]
    
    storage_client = storage.Client(project=config.PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)


def upload_to_gcs(local_path: Path, gcs_uri: str) -> None:
    """Upload local file to GCS."""
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1]
    
    storage_client = storage.Client(project=config.PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)


def generate_embeddings_streaming(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate embeddings one-by-one using Vertex AI SDK.
    Slower but simpler for small datasets (<1000 records).
    """
    print(f"\n  Initializing embedding model: {config.EMBEDDING_MODEL}")
    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
    model = TextEmbeddingModel.from_pretrained(config.EMBEDDING_MODEL)
    
    embedded_records = []
    total = len(records)
    
    print(f"  Generating embeddings for {total:,} records...")
    for i, record in enumerate(records, 1):
        if i % 50 == 0 or i == total:
            print(f"    Progress: {i:,}/{total:,} ({100*i/total:.1f}%)")
        
        try:
            # Generate embedding for content
            embedding_input = TextEmbeddingInput(
                text=record['content'],
                task_type="RETRIEVAL_DOCUMENT"
            )
            result = model.get_embeddings([embedding_input])
            embedding_vector = result[0].values
            
            # Add embedding to record with FULL metadata (including all Domo fields)
            embedded_record = {
                "id": record['id'],
                "embedding": embedding_vector,
                "restricts": [
                    # Salesforce fields
                    {"namespace": "account_name", "allow": [record['metadata']['account_name']]},
                    {"namespace": "owner", "allow": [record['metadata']['owner']]},
                    {"namespace": "calculated_engagement", "allow": [record['metadata']['calculated_engagement']]},
                    {"namespace": "pod_id", "allow": [record['metadata']['pod_id']]},
                    # Domo fields (for filtering by Domo agent)
                    {"namespace": "meau", "allow": [str(record['metadata'].get('meau', 'N/A'))]},
                    {"namespace": "health_score", "allow": [str(record['metadata'].get('health_score', 'N/A'))]},
                    {"namespace": "provisioned_users", "allow": [str(record['metadata'].get('provisioned_users', 'N/A'))]},
                    {"namespace": "contracted_licenses", "allow": [str(record['metadata'].get('contracted_licenses', 'N/A'))]},
                    {"namespace": "active_users", "allow": [str(record['metadata'].get('active_users', 'N/A'))]},
                    {"namespace": "risk_ratio_for_next_renewal", "allow": [str(record['metadata'].get('risk_ratio_for_next_renewal', 'N/A'))]},
                ]
            }
            embedded_records.append(embedded_record)
            
        except Exception as e:
            print(f"    ERROR on record {i}: {e}")
            continue
        
        # Rate limiting (avoid quota errors)
        if i % 100 == 0:
            time.sleep(1)
    
    return embedded_records


def generate_embeddings_batch(input_gcs_uri: str, output_gcs_uri: str) -> None:
    """
    Generate embeddings using Vertex AI Batch Prediction.
    Faster and more efficient for large datasets (>1000 records).
    
    Note: Batch prediction requires specific JSONL format and may take 10-30 minutes.
    """
    print(f"\n  Using Vertex AI Batch Prediction (this may take 10-30 minutes)...")
    print(f"  Input:  {input_gcs_uri}")
    print(f"  Output: {output_gcs_uri}")
    
    aiplatform.init(project=config.PROJECT_ID, location=config.LOCATION)
    
    # Create batch prediction job
    print(f"\n  Creating batch prediction job...")
    
    # Parse output URI to get bucket and prefix
    output_parts = output_gcs_uri.replace("gs://", "").split("/")
    output_bucket = output_parts[0]
    output_prefix = "/".join(output_parts[1:-1])  # Remove filename
    output_gcs_dir = f"gs://{output_bucket}/{output_prefix}/batch_output"
    
    job = aiplatform.BatchPredictionJob.create(
        job_display_name=f"nexus-embedding-{int(time.time())}",
        model_name=f"publishers/google/models/{config.EMBEDDING_MODEL}",
        input_config_gcs_source=input_gcs_uri,
        output_config_gcs_destination_prefix=output_gcs_dir,
        machine_type=config.BATCH_PREDICTION_MACHINE_TYPE,
        max_replica_count=config.BATCH_PREDICTION_MAX_REPLICA_COUNT,
    )
    
    print(f"  Job created: {job.resource_name}")
    print(f"  Waiting for completion...")
    
    # Wait for job to complete
    job.wait()
    
    if job.state == aiplatform.gapic.JobState.JOB_STATE_SUCCEEDED:
        print(f"  ✓ Batch prediction completed successfully")
        print(f"  Output location: {output_gcs_dir}")
        print(f"\n  Note: You'll need to download and reformat the batch output")
        print(f"        to match Vector Search JSONL format (id, embedding, restricts)")
    else:
        print(f"  ERROR: Batch prediction failed with state: {job.state}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for Nexus Golden Records")
    parser.add_argument(
        "--mode",
        choices=["streaming", "batch"],
        default="streaming",
        help="Embedding generation mode (default: streaming)"
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("  NEXUS HYBRID ENGINE - Phase 2: Vectorization")
    print("=" * 80)
    print(f"\n  Project: {config.PROJECT_ID}")
    print(f"  Region:  {config.LOCATION}")
    print(f"  Mode:    {args.mode}")
    
    # Step 1: Download input JSONL from GCS
    print("\n[Step 1] Downloading input JSONL from GCS...")
    local_input = Path("nexus_vectors_input.jsonl")
    
    if not local_input.exists():
        print(f"  Downloading: {config.VECTORS_INPUT_GCS}")
        download_from_gcs(config.VECTORS_INPUT_GCS, local_input)
    else:
        print(f"  Using existing local file: {local_input}")
    
    # Step 2: Load records
    print("\n[Step 2] Loading records...")
    records = []
    with open(local_input, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records):,} records")
    
    # Step 3: Generate embeddings
    print(f"\n[Step 3] Generating embeddings ({args.mode} mode)...")
    
    if args.mode == "streaming":
        embedded_records = generate_embeddings_streaming(records)
        
        # Step 4: Save embedded JSONL
        print(f"\n[Step 4] Saving embedded JSONL...")
        local_output = Path("nexus_vectors_embedded.jsonl")
        with open(local_output, 'w', encoding='utf-8') as f:
            for record in embedded_records:
                f.write(json.dumps(record) + '\n')
        print(f"  Saved locally: {local_output} ({len(embedded_records):,} records)")
        
        # Step 5: Upload to GCS
        print(f"\n[Step 5] Uploading to GCS...")
        print(f"  Uploading: {config.VECTORS_EMBEDDED_GCS}")
        upload_to_gcs(local_output, config.VECTORS_EMBEDDED_GCS)
        print(f"  ✓ Upload complete")
        
    else:  # batch mode
        generate_embeddings_batch(config.VECTORS_INPUT_GCS, config.VECTORS_EMBEDDED_GCS)
        print(f"\n  Note: Batch mode output requires manual reformatting")
        print(f"        See Vertex AI Batch Prediction docs for output format")
    
    # Summary
    print("\n" + "=" * 80)
    print("  Phase 2 Complete!")
    print("=" * 80)
    print(f"  Embedded file: {config.VECTORS_EMBEDDED_GCS}")
    print("\n  Next: Run Phase 3 to create and deploy Vector Search index")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
