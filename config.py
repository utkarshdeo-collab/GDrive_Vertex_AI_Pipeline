"""
Shared configuration for the Document Pipeline.
Aligns with: Ingestion → Processing → Indexing → Interface (ADK).
"""
import os

# ================= PROJECT =================
# Your Vertex AI project (MCP-Project-Alpha). Override with GCP_PROJECT_ID env if needed.
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "mcp-project-alpha")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

# Document AI uses regional endpoints: "us" or "eu" (not us-central1).
# Map Vertex region to Document AI region.
DOCAI_LOCATION = "us" if LOCATION.startswith("us") else "eu"

# Fix quota/billing 403: set before any Vertex/Document AI imports.
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = PROJECT_ID
# ADK / google.genai Client: use Vertex AI with this project and location.
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION

# ================= PHASE 1: INGESTION =================
# GCS bucket for staging PDFs (Document AI requires files in GCS for batch).
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", f"{PROJECT_ID}-docai-staging")
# Prefix inside bucket for input PDFs and Document AI output.
GCS_INPUT_PREFIX = "input"
GCS_OUTPUT_PREFIX = "document-ai-output"

# Document AI processor ID (your "test" Document OCR processor, region us).
DOCAI_PROCESSOR_ID = os.environ.get("DOCAI_PROCESSOR_ID", "711e5322a9d89b88")

# Google Drive: folder ID (list PDFs and use first) or file ID (single PDF).
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "")  # leave empty when using file ID
DRIVE_FILE_ID = os.environ.get("DRIVE_FILE_ID", "1RaWmZPygasvTlQ0Otg4g5YZF1FZYZ8GB")  # PDF from Drive URL

# Local path where PDF is downloaded (or used if not from Drive).
LOCAL_PDF_PATH = os.environ.get("LOCAL_PDF_PATH", "input.pdf")

# ================= PHASE 2: CHUNKING (later) =================
CHUNK_OUTPUT_PREFIX = "chunks"

# ================= PHASE 3: INDEXING =================
VECTOR_INDEX_DISPLAY_NAME = "doc-pipeline-index"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDINGS_GCS_PREFIX = "embeddings"

# ================= PHASE 4: DEPLOY & ADK =================
# Vector Search index (from Phase 3 output).
VECTOR_INDEX_ID = os.environ.get("VECTOR_INDEX_ID", "17005046835183616")
INDEX_ENDPOINT_DISPLAY_NAME = os.environ.get("INDEX_ENDPOINT_DISPLAY_NAME", "doc-pipeline-endpoint")
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID", "doc_pipeline_deployed_1770029483421")
# Endpoint resource (for query script). Use full name or leave empty to lookup by display name.
INDEX_ENDPOINT_RESOURCE_NAME = os.environ.get(
    "INDEX_ENDPOINT_RESOURCE_NAME",
    "projects/970885760464/locations/us-central1/indexEndpoints/6571185063914897408",
)
# Gemini model for answering from retrieved chunks.
# Run check_gemini_models.py to see available models in your project.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-001")
