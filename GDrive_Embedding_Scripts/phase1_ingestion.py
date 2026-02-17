"""
Phase 1: Ingestion & Structural Parsing (Document AI)

Step 1: Ingest PDFs into GCS — from PDF_URL_LIST (HTTP(S) URLs), Google Drive, or local file.
Step 2: (Optional) Document AI Batch Process (Layout Parser) → structural JSON to GCS.
Step 3: List output locations in GCS.

Sources (first non-empty wins):
  PDF_URL_LIST / PDF_URLS = comma-separated URLs → stream directly to config.GCS_PDF_INPUT_BUCKET
  DRIVE_FILE_ID / DRIVE_FOLDER_ID = Google Drive
  LOCAL_PDF_PATH = local file

For URL streaming: config.GCS_PDF_INPUT_BUCKET (e.g. {project}-pdf-input). Others: config.GCS_BUCKET_NAME.
"""
import io
import os
import sys
from urllib.parse import urlparse, unquote
from urllib.request import Request, urlopen

import config
from google.cloud import storage
from google.cloud.exceptions import NotFound, Conflict
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload


def ensure_bucket(storage_client, bucket_name):
    """Create GCS bucket if it does not exist."""
    try:
        bucket = storage_client.get_bucket(bucket_name)
        print(f"   Bucket exists: gs://{bucket_name}")
        return bucket
    except NotFound:
        print(f"   Creating bucket: gs://{bucket_name} in {config.LOCATION}...")
        try:
            bucket = storage_client.create_bucket(bucket_name, location=config.LOCATION)
            print(f"   Bucket created successfully!")
            return bucket
        except Conflict:
            # Bucket name already taken globally
            print(f"   ERROR: Bucket name '{bucket_name}' is already taken globally.")
            print(f"   Try a different bucket name in config.py")
            sys.exit(1)
    except Exception as e:
        print(f"   ERROR creating bucket: {e}")
        sys.exit(1)


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


def get_file_metadata(service, file_id):
    """Get file name and size from Drive."""
    try:
        file_info = service.files().get(fileId=file_id, fields="name,size,mimeType").execute()
        return file_info
    except HttpError as e:
        if e.resp.status == 404:
            raise ValueError(
                f"Drive file not found (404). Check:\n"
                f"  1. File ID is correct: {file_id}\n"
                f"  2. File is shared with your account or 'Anyone with link'\n"
                f"  3. You authenticated with the correct Google account"
            ) from e
        raise


def stream_drive_to_gcs(service, file_id, storage_client, bucket_name, gcs_prefix, file_name=None):
    """Stream PDF directly from Google Drive to GCS without local download."""
    # Get file metadata if name not provided
    if not file_name:
        file_info = get_file_metadata(service, file_id)
        file_name = file_info.get("name", "document.pdf")
        file_size = int(file_info.get("size", 0))
        print(f"   File: {file_name} ({file_size / (1024*1024):.2f} MB)")
    
    # Stream download from Drive
    print(f"   Streaming from Drive to GCS (no local download)...")
    request = service.files().get_media(fileId=file_id)
    
    # Download to memory in chunks
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request, chunksize=10*1024*1024)  # 10MB chunks
    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            print(f"   Download progress: {int(status.progress() * 100)}%")
    
    # Upload to GCS
    fh.seek(0)
    blob_name = f"{gcs_prefix}/{file_name}"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    print(f"   Uploading to GCS...")
    blob.upload_from_file(fh, content_type="application/pdf", rewind=True)
    
    uri = f"gs://{bucket_name}/{blob_name}"
    print(f"   Uploaded: {uri}")
    return uri


def download_file_from_drive(service, file_id, local_path):
    """Download a file from Google Drive by file ID to local_path (legacy method)."""
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


def _filename_from_url(url: str) -> str:
    """Get a safe filename from URL path (strip query, unquote)."""
    parsed = urlparse(url)
    path = parsed.path or "/document.pdf"
    name = unquote(os.path.basename(path))
    if not name or not name.lower().endswith(".pdf"):
        name = name + ".pdf" if name else "document.pdf"
    return name


def stream_url_to_gcs(url: str, storage_client, bucket_name: str, gcs_prefix: str) -> str:
    """Stream PDF from HTTP(S) URL to GCS. Returns gs:// URI."""
    filename = _filename_from_url(url)
    blob_name = f"{gcs_prefix}/{filename}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; DocumentPipeline/1.0)"})
    print(f"   Streaming: {url[:70]}...")
    with urlopen(req, timeout=120) as resp:
        chunk_size = 8 * 1024 * 1024  # 8 MB
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        buf = io.BytesIO()
        total = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            buf.write(chunk)
            total += len(chunk)
        buf.seek(0)
        blob.upload_from_file(buf, content_type="application/pdf", rewind=True)
    uri = f"gs://{bucket_name}/{blob_name}"
    size_mb = total / (1024 * 1024)
    print(f"   Uploaded: {uri} ({size_mb:.2f} MB)")
    return uri


def run_batch_process(gcs_input_uri, gcs_output_uri_prefix, timeout_seconds=600):
    """Submit Document AI batch process (Layout Parser). Output prefix must end with /."""
    from google.api_core.client_options import ClientOptions
    from google.cloud import documentai_v1 as documentai
    
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
    operation.result(timeout=timeout_seconds)
    return operation


def run_sync_process(gcs_input_uri: str, gcs_output_uri_prefix: str):
    """Process a single document with Document AI sync (online) API and write result to GCS.
    Use when batch fails for one file. Sync has a 15-page / 20MB limit for Document OCR."""
    from google.api_core.client_options import ClientOptions
    from google.cloud import documentai_v1 as documentai
    from google.protobuf import json_format

    opts = ClientOptions(
        api_endpoint=f"{config.DOCAI_LOCATION}-documentai.googleapis.com"
    )
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    name = client.processor_path(
        config.PROJECT_ID,
        config.DOCAI_LOCATION,
        config.DOCAI_PROCESSOR_ID,
    )
    request = documentai.ProcessRequest(
        name=name,
        gcs_document=documentai.GcsDocument(
            gcs_uri=gcs_input_uri,
            mime_type="application/pdf",
        ),
    )
    print("   Submitting sync (online) process...")
    response = client.process_document(request=request)
    prefix = gcs_output_uri_prefix.rstrip("/") + "/"
    if not prefix.startswith("gs://"):
        raise ValueError("gcs_output_uri_prefix must be a gs:// URI")
    parts = prefix[5:].split("/", 1)
    bucket_name, path_prefix = parts[0], parts[1]
    blob_name = f"{path_prefix}document.json"
    json_str = json_format.MessageToJson(response.document)
    storage_client = storage.Client(project=config.PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json_str, content_type="application/json")
    print(f"   Uploaded: gs://{bucket_name}/{blob_name}")
    return response


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
    print("  PHASE 1: Ingestion (PDFs → GCS)")
    print("=" * 60)
    print(f"\n  Project: {config.PROJECT_ID}")
    print(f"  Bucket:  {config.GCS_BUCKET_NAME}")
    print(f"  Region:  {config.LOCATION}")

    # Initialize storage client
    print("\n[Step 0] Connecting to Google Cloud Storage...")
    try:
        storage_client = storage.Client(project=config.PROJECT_ID)
        print(f"   Connected to project: {config.PROJECT_ID}")
    except Exception as e:
        print(f"\nERROR: Failed to connect to GCS: {e}")
        print("\nMake sure you've run:")
        print("  gcloud auth login")
        print("  gcloud auth application-default login")
        print(f"  gcloud config set project {config.PROJECT_ID}")
        sys.exit(1)

    # Ensure bucket: PDF URL path uses dedicated bucket; Drive/local use staging bucket
    if config.PDF_URL_LIST:
        bucket_name = getattr(config, "GCS_PDF_INPUT_BUCKET", config.GCS_BUCKET_NAME)
        print("\n[Step 1] Ensuring PDF bucket exists...")
        ensure_bucket(storage_client, bucket_name)
    else:
        print("\n[Step 1] Ensuring GCS bucket exists...")
        ensure_bucket(storage_client, config.GCS_BUCKET_NAME)

    # Step 2: Ingest PDFs into GCS (URLs > Drive > local)
    num_pdfs_uploaded = 1
    if config.PDF_URL_LIST:
        bucket_name = getattr(config, "GCS_PDF_INPUT_BUCKET", config.GCS_BUCKET_NAME)
        print("\n[Step 2] Streaming PDFs from URLs to GCS (no local download)...")
        gcs_uris = []
        failed = []
        for i, url in enumerate(config.PDF_URL_LIST, 1):
            filename = _filename_from_url(url)
            print(f"\n   ({i}/{len(config.PDF_URL_LIST)}) {filename}")
            try:
                uri = stream_url_to_gcs(
                    url,
                    storage_client,
                    bucket_name,
                    config.GCS_INPUT_PREFIX,
                )
                gcs_uris.append(uri)
            except Exception as e:
                failed.append((filename, url, str(e)))
                print(f"   [SKIP] {filename}: {e}")
                print(f"   Logged; continuing with next URL.")
        if failed:
            print(f"\n   Failed ({len(failed)}):")
            for fn, u, err in failed:
                print(f"      - {fn}: {err}")
        if not gcs_uris:
            print("\nERROR: No PDFs were uploaded. Fix URLs or network and retry.")
            sys.exit(1)
        gcs_input_uri = gcs_uris[0]
        num_pdfs_uploaded = len(gcs_uris)
        print(f"\n   Total uploaded: {num_pdfs_uploaded} PDF(s) to gs://{bucket_name}/{config.GCS_INPUT_PREFIX}/")
    elif config.DRIVE_FILE_ID:
        print("\n[Step 2] Streaming PDF from Google Drive to GCS...")
        service = get_drive_service()
        try:
            gcs_input_uri = stream_drive_to_gcs(
                service,
                config.DRIVE_FILE_ID,
                storage_client,
                config.GCS_BUCKET_NAME,
                config.GCS_INPUT_PREFIX,
            )
        except ValueError as e:
            print(f"\nERROR: {e}")
            sys.exit(1)
        except HttpError as e:
            print(f"\nERROR accessing Google Drive: {e}")
            print("\nMake sure:")
            print("  1. The file is shared with your authenticated account")
            print("  2. You have Google Drive API enabled in your project")
            sys.exit(1)
    elif config.DRIVE_FOLDER_ID:
        print("\n[Step 2] Pulling PDF from Drive folder...")
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
        gcs_input_uri = stream_drive_to_gcs(
            service,
            file_id,
            storage_client,
            config.GCS_BUCKET_NAME,
            config.GCS_INPUT_PREFIX,
            file_name=name,
        )
    else:
        # Fallback to local file
        print("\n[Step 2] Using local PDF...")
        local_pdf = config.LOCAL_PDF_PATH
        if not os.path.isfile(local_pdf):
            print(f"ERROR: File not found: {local_pdf}")
            print("  Set DRIVE_FILE_ID in config.py, or put a PDF at LOCAL_PDF_PATH.")
            sys.exit(1)
        print(f"   Local file: {os.path.abspath(local_pdf)}")
        gcs_input_uri = upload_pdf_to_gcs(
            storage_client,
            local_pdf,
            config.GCS_BUCKET_NAME,
            config.GCS_INPUT_PREFIX,
        )

    # Step 3: Document AI batch process (optional; single-file only)
    if config.DOCAI_PROCESSOR_ID and not config.PDF_URL_LIST:
        # Import Document AI only if needed
        from google.cloud import documentai_v1 as documentai
        gcs_output_prefix = f"gs://{config.GCS_BUCKET_NAME}/{config.GCS_OUTPUT_PREFIX}"
        print("\n[Step 3] Document AI Batch Process (Layout Parser)...")
        run_batch_process(gcs_input_uri, gcs_output_prefix)

        # Step 4: Output locations
        print("\n[Step 4] Output locations:")
        print_output_locations()
    elif config.PDF_URL_LIST:
        print("\n[Step 3] Streaming-only run: PDFs are in GCS. Chunking/indexing in a later step.")
    else:
        print("\n[Step 3] Skipping Document AI (DOCAI_PROCESSOR_ID not set). PDF(s) uploaded to GCS.")
        print("   To run Document AI later, set DOCAI_PROCESSOR_ID in config.py")

    print("\n" + "=" * 60)
    print("  Phase 1 complete!")
    if config.PDF_URL_LIST:
        pdf_bucket = getattr(config, "GCS_PDF_INPUT_BUCKET", config.GCS_BUCKET_NAME)
        print(f"  PDFs in GCS: gs://{pdf_bucket}/{config.GCS_INPUT_PREFIX}/ ({num_pdfs_uploaded} file(s))")
    else:
        print(f"  PDF location: {gcs_input_uri}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
