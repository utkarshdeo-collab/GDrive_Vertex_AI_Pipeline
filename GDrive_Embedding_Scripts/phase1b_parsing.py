"""
Phase 1b: Parsing — Document AI Layout Parser (no standard OCR).

Uses Document AI Layout Parser to preserve tables, headers, and paragraphs.
Output is structured JSON per document in GCS. Chunking (Phase 2) will consume this.

Input:  PDFs in gs://{GCS_PDF_INPUT_BUCKET}/{GCS_INPUT_PREFIX}/
Output: gs://{GCS_PDF_INPUT_BUCKET}/{DOCAI_OUTPUT_PREFIX}/{doc_stem}/  (structured JSON)

Requires: DOCAI_PROCESSOR_ID set in config (Layout Parser processor).

  python phase1b_parsing.py
  python phase1b_parsing.py --only "SYM_2025_1PGR_Federation_via_Microsoft_Teams.pdf"   # retry one file
  python phase1b_parsing.py --only "file.pdf" --sync   # use sync (online) API instead of batch (15-page limit)
"""
import argparse
import os
import sys

import config
from google.cloud import storage

# Reuse batch and sync logic from Phase 1
from phase1_ingestion import run_batch_process, run_sync_process


def _stem(name: str) -> str:
    """Filename without extension, safe for GCS path (no spaces/slashes)."""
    base = os.path.splitext(name)[0]
    return base.replace(" ", "_").replace("/", "_").replace("\\", "_").strip() or "document"


def list_pdfs_in_bucket(storage_client, bucket_name: str, prefix: str):
    """List blob names under prefix that end with .pdf."""
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    return [b.name for b in blobs if b.name.lower().endswith(".pdf")]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1b: Document AI Layout Parser. Use --only to process specific PDF(s) only."
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Process only these PDF(s): filename or comma-separated list (e.g. 'failed.pdf' or 'a.pdf,b.pdf').",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use sync (online) API instead of batch. Use for single PDFs when batch fails. Limit: ~15 pages, 20MB.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  PHASE 1b: Parsing (Document AI Layout Parser)")
    print("=" * 60)
    print("\n  Uses Layout Parser only (no standard OCR).")
    print("  Output: structured JSON — tables, headers, paragraphs preserved.")
    print("=" * 60)
    print(f"\n  Project: {config.PROJECT_ID}")
    print(f"  Region:  {config.LOCATION}")
    print(f"  Bucket:  {config.GCS_PDF_INPUT_BUCKET}")
    print(f"  Input:   gs://{config.GCS_PDF_INPUT_BUCKET}/{config.GCS_INPUT_PREFIX}/")
    print(f"  Output:  gs://{config.GCS_PDF_INPUT_BUCKET}/{config.DOCAI_OUTPUT_PREFIX}/<doc_stem>/")

    if not config.DOCAI_PROCESSOR_ID:
        print("\nERROR: DOCAI_PROCESSOR_ID is not set.")
        print("  Create a Layout Parser processor in Document AI (Console) and set DOCAI_PROCESSOR_ID in config.py or env.")
        sys.exit(1)

    print("\n[Step 1] Connecting to GCS...")
    try:
        storage_client = storage.Client(project=config.PROJECT_ID)
    except Exception as e:
        print(f"\nERROR: Failed to connect to GCS: {e}")
        sys.exit(1)

    print("\n[Step 2] Listing PDFs in bucket...")
    bucket_name = config.GCS_PDF_INPUT_BUCKET
    prefix = config.GCS_INPUT_PREFIX.rstrip("/") + "/"
    all_pdf_names = list_pdfs_in_bucket(storage_client, bucket_name, prefix)
    if not all_pdf_names:
        print(f"   No PDFs found under gs://{bucket_name}/{prefix}")
        print("   Run phase1_ingestion.py first to upload PDFs.")
        sys.exit(1)

    if args.only:
        only_basenames = {s.strip() for s in args.only.split(",") if s.strip()}
        pdf_names = [n for n in all_pdf_names if os.path.basename(n) in only_basenames]
        missing = only_basenames - {os.path.basename(n) for n in pdf_names}
        if missing:
            print(f"   WARNING: Not found in bucket: {missing}")
        if not pdf_names:
            print("   No matching PDFs to process. Check --only filenames.")
            sys.exit(1)
        print(f"   Processing only: {[os.path.basename(n) for n in pdf_names]}")
    else:
        pdf_names = all_pdf_names
    if args.sync:
        print("   Mode: sync (online) — single-doc limit ~15 pages / 20MB")
    print(f"   Found {len(pdf_names)} PDF(s) to process")

    print("\n[Step 3] Running Document AI Layout Parser on each PDF...")
    out_prefix = config.DOCAI_OUTPUT_PREFIX.strip().rstrip("/")
    succeeded = []
    failed = []
    for i, blob_name in enumerate(pdf_names, 1):
        filename = os.path.basename(blob_name)
        stem = _stem(filename)
        gcs_input_uri = f"gs://{bucket_name}/{blob_name}"
        gcs_output_prefix = f"gs://{bucket_name}/{out_prefix}/{stem}/"
        print(f"\n   ({i}/{len(pdf_names)}) {filename}")
        print(f"   Input:  {gcs_input_uri}")
        print(f"   Output: {gcs_output_prefix}")
        try:
            if args.sync:
                run_sync_process(gcs_input_uri, gcs_output_prefix)
            else:
                run_batch_process(gcs_input_uri, gcs_output_prefix, timeout_seconds=1200)
            succeeded.append((filename, gcs_output_prefix))
        except Exception as e:
            failed.append((filename, str(e)))
            print(f"   [FAILED] {e}")
            print("   Continuing with next PDF.")

    if failed:
        print(f"\n   Failed ({len(failed)}):")
        for fn, err in failed:
            print(f"      - {fn}: {err}")
    print(f"\n   Succeeded: {len(succeeded)} | Failed: {len(failed)}")

    if succeeded:
        print("\n[Step 4] Output locations (structured JSON):")
        for fn, out_uri in succeeded:
            print(f"      {fn} → {out_uri}")

    print("\n" + "=" * 60)
    print("  Phase 1b complete. Parsed output is in GCS.")
    print("  Next: Phase 2 (chunking) will read this structured JSON.")
    print("=" * 60 + "\n")

    if failed and not succeeded:
        sys.exit(1)


if __name__ == "__main__":
    main()
