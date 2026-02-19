"""
Phase 2: Client-Side Batch Vectorization (Org Policy Workaround)

This script generates embeddings for the 80k records using the Vertex AI 
Online Prediction API (client-side batching) instead of a Batch Prediction Job.
This creates the final JSONL file required for Vector Search Index update.

Architecture:
1. Read 'nexus_vectors_input.jsonl' locally.
2. Process in batches (e.g., 10 records) to respect API quotas and token limits.
3. Call TextEmbeddingModel.get_embeddings().
4. Write output JSONL with fields: {"id": "...", "embedding": [...], "restricts": [...]}
5. Upload final file to GCS.

Features:
- RESUME CAPABILITY: Skips already processed records.
- TRUNCATION: Ensures content doesn't exceed model limits.
- BATCH OPTIMIZATION: Dynamic or small batch sizes.

Usage:
    python Nexus_Hybrid_Engine/phase2_vectorize.py
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from google.cloud import storage
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from Nexus_Hybrid_Engine import nexus_config as config
except ImportError:
    import nexus_config as config

# Configuration
BATCH_SIZE = 10  # Reduced from 50 to 10 to avoid 20k token limit per request
MAX_CHARS_PER_ITEM = 18000 # Approx 4k tokens, safe upper bound for single item (model limit is 2k tokens ~ 8k chars? No, 004 is 2048 tokens input. So ~8000 chars.)
# Actually text-embedding-004 input token limit is 2048. 
# 1 token ~= 4 chars. So 8192 chars. Let's be safe with 8000.
TRUNCATE_LIMIT = 8000 
SLEEP_BETWEEN_BATCHES = 0.1 

def ensure_bucket_exists(bucket_name: str, location: str) -> None:
    """Checks if a GCS bucket exists; if not, creates it."""
    from google.cloud import storage
    from google.api_core import exceptions
    storage_client = storage.Client(project=config.PROJECT_ID)
    try:
        bucket = storage_client.get_bucket(bucket_name)
        print(f"[Setup] Bucket '{bucket_name}' already exists.")
    except exceptions.NotFound:
        print(f"[Setup] Bucket '{bucket_name}' not found. Creating in {location}...")
        bucket = storage_client.bucket(bucket_name)
        bucket.create(location=location)
        print(f"[Setup] Bucket '{bucket_name}' created successfully.")

def truncate_text(text: str, limit: int) -> str:
    """Truncates text to limit characters."""
    if len(text) <= limit:
        return text
    return text[:limit]

def process_and_embed(
    input_file: Path, 
    output_file: Path, 
    model_name: str
):
    """
    Reads input JSONL, generates embeddings in batches, and writes output JSONL.
    Supports RESUMING from existing output file.
    """
    print(f"\n[Processing] Initializing Vertex AI in {config.LOCATION}...")
    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
    
    # Load Model
    model = TextEmbeddingModel.from_pretrained(model_name)
    print(f"[Processing] Model '{model_name}' loaded.")

    # 1. Check existing progress
    processed_ids = set()
    if output_file.exists():
        print(f"[Resume] Checking existing output in {output_file}...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        processed_ids.add(rec.get('id'))
                    except:
                        pass
        print(f"[Resume] Found {len(processed_ids)} already processed records. Will skip them.")

    # 2. Read Input
    print(f"[Processing] Reading {input_file}...")
    records_to_process = []
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record.get('id') in processed_ids:
                    skipped_count += 1
                else:
                    records_to_process.append(record)
    
    total_new = len(records_to_process)
    print(f"[Processing] {skipped_count} skipped. {total_new} new records to embed.")

    if total_new == 0:
        print("[Processing] All records processed!")
        return

    # 3. Batch Processing
    processed_count = 0
    start_time = time.time()
    
    # Append mode 'a'
    with open(output_file, 'a', encoding='utf-8') as out_f:
        for i in range(0, total_new, BATCH_SIZE):
            batch = records_to_process[i : i + BATCH_SIZE]
            inputs = []
            
            # Prepare inputs
            for record in batch:
                raw_content = record.get("content", "")
                # Truncate to avoid 2048 token limit per item
                content = truncate_text(raw_content, TRUNCATE_LIMIT)
                
                task_type = record.get("task_type", "RETRIEVAL_DOCUMENT")
                title = record.get("title", "")
                inputs.append(TextEmbeddingInput(text=content, task_type=task_type, title=title))

            try:
                # API Call
                embeddings = model.get_embeddings(inputs)
                
                # Write to output
                for record, embedding_obj in zip(batch, embeddings):
                    # Format for Vector Search: id, embedding, restricts
                    vector_record = {
                        "id": record["id"],
                        "embedding": embedding_obj.values,
                        "restricts": record.get("restricts", [])
                    }
                    out_f.write(json.dumps(vector_record) + "\n")
                
                processed_count += len(batch)
                
                # Progress
                elapsed = time.time() - start_time
                if elapsed > 0:
                    rate = processed_count / elapsed
                    print(f"  Processed {processed_count}/{total_new} (Total: {len(processed_ids)+processed_count}) ({rate:.1f} rec/s)...", end='\r')
                
                # Rate limiting sleep
                time.sleep(SLEEP_BETWEEN_BATCHES)

            except Exception as e:
                print(f"\n[Error] Batch failed at index {i}: {e}")
                print(f"  Skipping this batch to avoid stopping entire flow. (Data loss for {len(batch)} records)")
                # In strict mode we might stop, but for 80k rows, missing 10 is better than crashing?
                # Actually, let's try to continue.
                continue

    print(f"\n[Processing] Completed! {len(processed_ids) + processed_count} records saved to {output_file}")


def upload_to_gcs(local_path: Path, bucket_name: str, blob_name: str):
    """Uploads file to GCS."""
    ensure_bucket_exists(bucket_name, config.LOCATION)
    
    print(f"\n[Upload] Uploading {local_path} to gs://{bucket_name}/{blob_name}...")
    storage_client = storage.Client(project=config.PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    print(f"[Upload] Success!")
    return f"gs://{bucket_name}/{blob_name}"


def run_phase_2():
    print("=" * 80)
    print("PHASE 2: Client-Side Vectorization (Org Policy Workaround)")
    print("=" * 80)

    # 1. Setup Paths
    input_file = Path("nexus_vectors_input.jsonl")
    output_file = Path("nexus_vectors_output.jsonl")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        sys.exit(1)

    # 2. Run Embedding Process
    try:
        process_and_embed(input_file, output_file, config.NEXUS_EMBEDDING_MODEL)
    except KeyboardInterrupt:
        print("\n[Aborted] Process stopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] Embeddings generation failed: {e}")
        sys.exit(1)

    # 3. Upload to GCS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    gcs_blob = f"{config.GCS_EMBEDDINGS_PREFIX}/{timestamp}/final/nexus_vectors_output.jsonl"
    
    try:
        gcs_uri = upload_to_gcs(output_file, config.GCS_BUCKET_NAME, gcs_blob)
        print("\n" + "="*80)
        print("PHASE 2 COMPLETE")
        print(f"Vector data available at: {gcs_uri}")
        print("Ready for Phase 3 (Index Deployment).")
        print("="*80)
    except Exception as e:
        print(f"\n[Error] Upload failed: {e}")
        print(f"Note: Your local file '{output_file}' is safe.")
        sys.exit(1)

if __name__ == "__main__":
    run_phase_2()
