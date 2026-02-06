"""
List which PDFs were used to build the current index (Phase 2 chunks).

The Vertex AI index does not store source file names. This script reads the
chunks that were used for indexing from GCS and reports unique source PDFs.

Usage:
  python list_indexed_pdfs.py

Output: Unique file_name values from chunks.jsonl (one per source document).
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from google.cloud import storage


def main():
    bucket_name = getattr(config, "CHUNKS_BUCKET", config.GCS_PDF_INPUT_BUCKET)
    chunks_path = f"{config.CHUNK_OUTPUT_PREFIX}/chunks.jsonl"
    print(f"Chunks: gs://{bucket_name}/{chunks_path}")
    print()

    client = storage.Client(project=config.PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(chunks_path)
    if not blob.exists():
        print("No chunks file found. Run Phase 2 (chunking) first.")
        sys.exit(1)

    data = blob.download_as_text()
    seen_files = set()
    chunk_count_per_file = {}
    for line in data.strip().split("\n"):
        if not line:
            continue
        obj = json.loads(line)
        meta = obj.get("metadata") or {}
        name = meta.get("file_name") or "(unknown)"
        seen_files.add(name)
        chunk_count_per_file[name] = chunk_count_per_file.get(name, 0) + 1

    if not seen_files:
        print("No documents found in chunks.")
        sys.exit(0)

    print("PDFs used for this index (from chunks):")
    print("-" * 50)
    for i, name in enumerate(sorted(seen_files), 1):
        count = chunk_count_per_file.get(name, 0)
        print(f"  {i}. {name}  ({count} chunks)")
    print("-" * 50)
    print(f"Total: {len(seen_files)} document(s)")


if __name__ == "__main__":
    main()
