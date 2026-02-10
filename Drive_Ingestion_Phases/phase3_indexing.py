"""
Phase 3: Indexing (Embedding & Vector Search)

Step 1: Read chunks from GCS (chunks/chunks.jsonl).
Step 2: Generate embeddings in batches via Vertex AI text-embedding-004 (768-dim).
Step 3: Upload embedding JSONL to GCS (embeddings/embeddings.json).
Step 4: Create Vertex AI Vector Search index from that GCS path (async LRO).

By default every run starts from the beginning. To resume after a failure, use
--resume-file with the partial file (written every 500 chunks as embeddings_partial.jsonl).
Resume is reliable only when chunks.jsonl has not changed.

  python phase3_indexing.py
  python phase3_indexing.py --resume-file embeddings_partial.jsonl
"""
import argparse
import json
import os
import sys
import time

import config
from google.api_core.exceptions import (
    DeadlineExceeded,
    ResourceExhausted,
    ServiceUnavailable,
)
from google.cloud import storage

# Vertex AI after config (quota project set)
import vertexai
from vertexai.language_models import TextEmbeddingModel

from google.cloud import aiplatform
from google.cloud.aiplatform import matching_engine
from google.cloud.aiplatform.matching_engine import matching_engine_index_config as me_config

# text-embedding-004 dimension; smaller batch + delay to stay under quota
EMBEDDING_DIM = 768
BATCH_SIZE = 20
SLEEP_BETWEEN_BATCHES = 3.0  # stay under embedding quota (e.g. 60–300 req/min)
MAX_RETRIES_429 = 5
BACKOFF_SECONDS = 90
# Retry on 503/timeouts (connection or server temporarily unavailable)
TRANSIENT_EXCEPTIONS = (ResourceExhausted, ServiceUnavailable, DeadlineExceeded)
MAX_RETRIES_TRANSIENT = 5
# Checkpoint progress every N batches so we can resume after timeout/503
CHECKPOINT_FILE = "embeddings_partial.jsonl"
CHECKPOINT_INTERVAL_BATCHES = 25  # save every 500 chunks so we lose at most 500 on crash


def download_chunks(storage_client, bucket_name, chunks_path):
    """Download chunks JSONL from GCS; return list of chunk dicts."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(chunks_path)
    if not blob.exists():
        return []
    data = blob.download_as_text()
    chunks = []
    for line in data.strip().split("\n"):
        if not line:
            continue
        chunks.append(json.loads(line))
    return chunks


def embed_batch(model, texts):
    """Return list of embedding vectors (each a list of floats) for the given texts."""
    if not texts:
        return []
    embeddings = model.get_embeddings(texts)
    return [list(e.values) for e in embeddings]


def embed_batch_with_retry(model, texts):
    """Call embed_batch; retry on 429 (quota), 503 (unavailable), timeouts (DeadlineExceeded)."""
    for attempt in range(MAX_RETRIES_TRANSIENT + 1):
        try:
            return embed_batch(model, texts)
        except TRANSIENT_EXCEPTIONS as e:
            if attempt < MAX_RETRIES_TRANSIENT:
                kind = "Quota exceeded" if type(e).__name__ == "ResourceExhausted" else "Connection/timeout or server unavailable"
                print(f"   {kind}, waiting {BACKOFF_SECONDS}s before retry ({attempt + 1}/{MAX_RETRIES_TRANSIENT})...")
                time.sleep(BACKOFF_SECONDS)
            else:
                raise


def load_resume_file(path, chunks):
    """
    Load a partial embeddings JSONL and verify it matches the first N chunks.
    Returns (lines, start_index). Exits with error if IDs don't align.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if not lines:
        return [], 0
    n = len(lines)
    if n > len(chunks):
        print(f"   ERROR: Resume file has {n} lines but chunks only has {len(chunks)}. Wrong file or chunks changed.")
        sys.exit(1)
    # Ensure each line's "id" matches the corresponding chunk (reliable resume)
    for j, line in enumerate(lines):
        try:
            rec = json.loads(line)
            rid = rec.get("id")
        except (json.JSONDecodeError, TypeError):
            print(f"   ERROR: Resume file line {j + 1} is not valid JSON with 'id'.")
            sys.exit(1)
        if rid != chunks[j]["id"]:
            print(f"   ERROR: Resume file chunk ID at line {j + 1} does not match chunks.jsonl (chunks may have changed). Do not use an old checkpoint after re-running Phase 2.")
            sys.exit(1)
    return lines, n


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Embed chunks and create Vector Search index. Use --resume-file to resume from a partial JSONL.")
    parser.add_argument("--resume-file", type=str, default=None, help="Path to partial embeddings JSONL (same format as output). Script will only embed the remaining chunks. Reliable only if chunks.jsonl has not changed.")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  PHASE 3: Indexing (Embeddings → Vector Search)")
    print("=" * 60)
    print("\n  Chunks → embeddings (text-embedding-004) → GCS → Vector Search index (async).")
    print("  For ~1.7k chunks expect ~10–15 min for embedding; index build is async (30+ min).")
    print("=" * 60)

    chunks_prefix = config.CHUNK_OUTPUT_PREFIX
    embeddings_prefix = getattr(config, "EMBEDDINGS_GCS_PREFIX", "embeddings")

    storage_client = storage.Client(project=config.PROJECT_ID)

    # Use same bucket as Phase 2 chunks (CHUNKS_BUCKET = pdf-input bucket by default)
    bucket = getattr(config, "CHUNKS_BUCKET", config.GCS_PDF_INPUT_BUCKET)
    chunks_path = f"{chunks_prefix}/chunks.jsonl"

    print(f"\n  Chunks: gs://{bucket}/{chunks_path}")
    print(f"  Embeddings output: gs://{bucket}/{embeddings_prefix}/")
    print("\n[Step 1] Reading chunks from GCS...")
    chunks = download_chunks(storage_client, bucket, chunks_path)
    if not chunks:
        print(f"   No chunks at gs://{bucket}/{chunks_path}. Run Phase 2 first.")
        sys.exit(1)
    print(f"   Loaded {len(chunks)} chunks.")

    # Start from 0 unless user explicitly passes --resume-file
    vectors_jsonl_lines = []
    start_index = 0
    if args.resume_file:
        if not os.path.exists(args.resume_file):
            print(f"   ERROR: --resume-file not found: {args.resume_file}")
            sys.exit(1)
        vectors_jsonl_lines, start_index = load_resume_file(args.resume_file, chunks)
        print(f"   Resuming from --resume-file: {start_index} embeddings already done (validated against chunks).")

    print("\n[Step 2] Generating embeddings (text-embedding-004)...")
    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
    model = TextEmbeddingModel.from_pretrained(config.EMBEDDING_MODEL)

    for i in range(start_index, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        embs = embed_batch_with_retry(model, texts)
        for chunk_id, vec in zip(ids, embs):
            record = {
                "id": chunk_id,
                "embedding": vec,
                "restricts": [{"namespace": "source", "allow": ["doc-pipeline"]}],
            }
            vectors_jsonl_lines.append(json.dumps(record))
        print(f"   Embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")
        time.sleep(SLEEP_BETWEEN_BATCHES)
        # Checkpoint periodically so we can resume if the run fails later
        batch_num = (i - start_index) // BATCH_SIZE + 1
        if batch_num % CHECKPOINT_INTERVAL_BATCHES == 0:
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(vectors_jsonl_lines) + "\n")
            print(f"   Checkpoint saved ({len(vectors_jsonl_lines)} embeddings).")

    if len(vectors_jsonl_lines) != len(chunks):
        print("   WARNING: embedding count does not match chunk count.")
    print(f"   Total vectors: {len(vectors_jsonl_lines)}")

    # Remove checkpoint on success so next run starts fresh
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
        except OSError:
            pass

    print("\n[Step 3] Uploading embeddings to GCS...")
    embeddings_jsonl = "\n".join(vectors_jsonl_lines)
    blob_name = f"{embeddings_prefix}/embeddings.json"
    bucket_obj = storage_client.bucket(bucket)
    out_blob = bucket_obj.blob(blob_name)
    out_blob.upload_from_string(
        embeddings_jsonl,
        content_type="application/jsonl",
    )
    contents_uri = f"gs://{bucket}/{embeddings_prefix}/"
    print(f"   Uploaded to {contents_uri}")

    print("\n[Step 4] Creating Vector Search index...")
    aiplatform.init(project=config.PROJECT_ID, location=config.LOCATION)
    try:
        index = matching_engine.MatchingEngineIndex.create_tree_ah_index(
            display_name=config.VECTOR_INDEX_DISPLAY_NAME,
            contents_delta_uri=contents_uri,
            dimensions=EMBEDDING_DIM,
            approximate_neighbors_count=150,
            leaf_node_embedding_count=500,
            leaf_nodes_to_search_percent=10,
            distance_measure_type=me_config.DistanceMeasureType.COSINE_DISTANCE,
            feature_norm_type=me_config.FeatureNormType.UNIT_L2_NORM,
            description="Document pipeline index from Phase 3",
            sync=False,
        )
        print("   Index creation submitted (async). Build may take 30+ minutes.")
        print("   Check status in Console: Vertex AI → Vector Search.")
        print(f"   Embeddings URI: {contents_uri}")
        # With sync=False, resource_name is not available until the LRO completes; accessing it may raise.
        index_resource = None
        try:
            index_resource = getattr(index, "resource_name", None) or str(index)
        except RuntimeError as re:
            if "has not been created" in str(re):
                pass  # Expected when sync=False; index is still building.
            else:
                raise
        if index_resource and "/indexes/" in str(index_resource):
            index_id = index_resource.split("/indexes/")[-1].split("/")[0].strip()
            print(f"   New index resource: {index_resource}")
            print(f"   >>> Update config.py: VECTOR_INDEX_ID = \"{index_id}\"")
        else:
            print("   Once the index build completes, get the index ID from Console:")
            print("   Vertex AI → Vector Search → your index → copy the index ID.")
            print("   Then set in config.py: VECTOR_INDEX_ID = \"<index_id>\"")
    except Exception as e:
        print(f"   Index create failed: {e}")
        print("   Embeddings are at", contents_uri)
        print("   You can create the index manually in Console (Vector Search) using that URI.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Phase 3 complete. Deploy index to an endpoint for Phase 4 (ADK).")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
