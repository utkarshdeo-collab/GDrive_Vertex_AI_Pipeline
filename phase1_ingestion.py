"""
Phase 1: Ingestion & Structural Parsing (Document AI)

Step 1: Pull PDF from your Google Drive folder → download to local.
Step 2: Upload PDF to GCS (staging bucket).
Step 3: Document AI Batch Process (Layout Parser) → structural JSON to GCS.
Step 4: List output JSON locations in GCS.

Set in config.py:
  DRIVE_FOLDER_ID = "your-drive-folder-id"   (e.g. test_data folder)
  DOCAI_PROCESSOR_ID = "your-document-ai-processor-id"

GCS bucket used: config.GCS_BUCKET_NAME (e.g. mcp-project-alpha-docai-staging)
"""
import io
import os
import sys

import config
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload


def ensure_bucket(storage_client, bucket_name):
    """Create GCS bucket if it does not exist."""
    try:
        storage_client.get_bucket(bucket_name)
        print(f"   Bucket exists: gs://{bucket_name}")
    except Exception:
        print(f"   Creating bucket: gs://{bucket_name}")
        storage_client.create_bucket(bucket_name, location=config.LOCATION)


def get_drive_service():
    import google.auth
    creds, _ = google.auth.default()
    return build("drive", "v3", credentials=creds)


def list_pdfs_in_folder(service, folder_id):
    """List PDF files in a Drive folder. Returns list of (file_id, name)."""
    q = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    try:
        r = service.files().list(q=q, fields="files(id,name)", pageSize=10).execute()
    except HttpError as e:
        if e.resp.status == 404:
            raise ValueError(
                "Drive folder not found (404). Check: (1) Folder ID is correct — open the folder in Drive and copy from the URL. "
                "(2) You used the same Google account for 'gcloud auth application-default login' that has access to this folder. "
                "Alternatively, use the PDF file ID: open the PDF in Drive, copy the ID from the URL (drive.google.com/file/d/<FILE_ID>/view) and set DRIVE_FILE_ID in config.py (clear DRIVE_FOLDER_ID)."
            ) from e
        raise
    files = r.get("files", [])
    return [(f["id"], f["name"]) for f in files]


def download_file_from_drive(service, file_id, local_path):
    """Download a file from Google Drive by file ID to local_path."""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    with open(local_path, "wb") as f:
        f.write(fh.read())
    print(f"   Downloaded to: {local_path}")


def upload_pdf_to_gcs(storage_client, local_path, bucket_name, gcs_prefix):
    """Upload local PDF to GCS; return gs:// URI."""
    blob_name = f"{gcs_prefix}/{os.path.basename(local_path)}"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path, content_type="application/pdf")
    uri = f"gs://{bucket_name}/{blob_name}"
    print(f"   Uploaded: {uri}")
    return uri


def run_batch_process(gcs_input_uri, gcs_output_uri_prefix):
    """Submit Document AI batch process (Layout Parser). Output prefix must end with /."""
    opts = ClientOptions(
        api_endpoint=f"{config.DOCAI_LOCATION}-documentai.googleapis.com"
    )
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    gcs_doc = documentai.GcsDocument(
        gcs_uri=gcs_input_uri,
        mime_type="application/pdf",
    )
    gcs_documents = documentai.GcsDocuments(documents=[gcs_doc])
    input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)

    if not gcs_output_uri_prefix.endswith("/"):
        gcs_output_uri_prefix = gcs_output_uri_prefix.rstrip("/") + "/"

    gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(
        gcs_uri=gcs_output_uri_prefix,
    )
    output_config = documentai.DocumentOutputConfig(
        gcs_output_config=gcs_output_config,
    )

    name = client.processor_path(
        config.PROJECT_ID,
        config.DOCAI_LOCATION,
        config.DOCAI_PROCESSOR_ID,
    )
    request = documentai.BatchProcessRequest(
        name=name,
        input_documents=input_config,
        document_output_config=output_config,
    )

    print("   Submitting batch process (Layout Parser)...")
    operation = client.batch_process_documents(request)
    print(f"   Operation: {operation.operation.name}")
    print("   Waiting for completion (may take several minutes for large PDFs)...")
    operation.result(timeout=600)
    return operation


def print_output_locations():
    """List JSON output files under the known GCS output prefix."""
    bucket = config.GCS_BUCKET_NAME
    prefix = config.GCS_OUTPUT_PREFIX
    storage_client = storage.Client(project=config.PROJECT_ID)
    blobs = list(storage_client.list_blobs(bucket, prefix=prefix))
    print(f"   Output prefix: gs://{bucket}/{prefix}/")
    if not blobs:
        print("   (No files yet; listing may be delayed.)")
        return
    for b in blobs[:25]:
        if b.name.endswith(".json"):
            print(f"      - {b.name}")
    if len(blobs) > 25:
        print(f"      ... and {len(blobs) - 25} more objects")


def main():
    print("\n" + "=" * 60)
    print("  PHASE 1: Ingestion (Drive → GCS → Document AI)")
    print("=" * 60)

    if not config.DOCAI_PROCESSOR_ID:
        print("\nERROR: DOCAI_PROCESSOR_ID is not set in config.py")
        print("  Create a Layout Parser (or Document OCR) processor in Document AI Console,")
        print("  then set DOCAI_PROCESSOR_ID in config.py or env.")
        sys.exit(1)

    storage_client = storage.Client(project=config.PROJECT_ID)
    ensure_bucket(storage_client, config.GCS_BUCKET_NAME)

    local_pdf = config.LOCAL_PDF_PATH

    # Step 1: Get PDF from Drive (folder or single file) or use local
    if config.DRIVE_FOLDER_ID:
        print("\n[Step 1] Pulling PDF from Drive folder...")
        service = get_drive_service()
        try:
            pdfs = list_pdfs_in_folder(service, config.DRIVE_FOLDER_ID)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        if not pdfs:
            print("ERROR: No PDF files found in that Drive folder.")
            sys.exit(1)
        file_id, name = pdfs[0]
        print(f"   Found PDF: {name}")
        download_file_from_drive(service, file_id, local_pdf)
    elif config.DRIVE_FILE_ID:
        print("\n[Step 1] Downloading from Google Drive (single file)...")
        service = get_drive_service()
        download_file_from_drive(service, config.DRIVE_FILE_ID, local_pdf)
    else:
        print("\n[Step 1] Using local PDF.")
        if not os.path.isfile(local_pdf):
            print(f"ERROR: File not found: {local_pdf}")
            print("  Set DRIVE_FOLDER_ID or DRIVE_FILE_ID in config.py, or put a PDF at LOCAL_PDF_PATH.")
            sys.exit(1)
        print(f"   Local file: {os.path.abspath(local_pdf)}")

    # Step 2: Upload to GCS
    print("\n[Step 2] Uploading PDF to GCS...")
    gcs_input_uri = upload_pdf_to_gcs(
        storage_client,
        local_pdf,
        config.GCS_BUCKET_NAME,
        config.GCS_INPUT_PREFIX,
    )

    # Step 3: Document AI batch process
    gcs_output_prefix = f"gs://{config.GCS_BUCKET_NAME}/{config.GCS_OUTPUT_PREFIX}"
    print("\n[Step 3] Document AI Batch Process (Layout Parser)...")
    run_batch_process(gcs_input_uri, gcs_output_prefix)

    # Step 4: Output locations
    print("\n[Step 4] Output locations:")
    print_output_locations()

    print("\n" + "=" * 60)
    print("  Phase 1 complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
