"""
Shared configuration for the Document Pipeline.
Aligns with: Ingestion → Processing → Indexing → Interface (ADK).
"""
import os

# ================= PROJECT =================
# Your Vertex AI project. Override with GCP_PROJECT_ID env if needed.
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "sym-dev-mr-agents-01")
LOCATION = os.environ.get("GCP_LOCATION", "us-east4")  # Northern Virginia (allowed by org policy)

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

# Document AI processor ID (your Document OCR processor, region us).
# Leave empty to skip Document AI processing in Phase 1.
DOCAI_PROCESSOR_ID = os.environ.get("DOCAI_PROCESSOR_ID", "")

# Google Drive: folder ID (list PDFs and use first) or file ID (single PDF).
# Leave both empty to use local file (LOCAL_PDF_PATH)
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "")
DRIVE_FILE_ID = os.environ.get("DRIVE_FILE_ID", "")  # Empty = use local file

# Local path where PDF is downloaded (or used if not from Drive).
LOCAL_PDF_PATH = os.environ.get("LOCAL_PDF_PATH", "input.pdf")

# ================= PHASE 2: CHUNKING (later) =================
CHUNK_OUTPUT_PREFIX = "chunks"

# ================= PHASE 3: INDEXING =================
VECTOR_INDEX_DISPLAY_NAME = "doc-pipeline-index"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDINGS_GCS_PREFIX = "embeddings"

# ================= PHASE 4: DEPLOY & ADK =================
# Vector Search index (from Phase 3 output). Update after running Phase 3.
VECTOR_INDEX_ID = os.environ.get("VECTOR_INDEX_ID", "4165552578387509248")
INDEX_ENDPOINT_DISPLAY_NAME = os.environ.get("INDEX_ENDPOINT_DISPLAY_NAME", "doc-pipeline-endpoint")
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID", "doc_pipeline_deployed")
# Endpoint resource (for query script). Use full name or leave empty to lookup by display name.
INDEX_ENDPOINT_RESOURCE_NAME = os.environ.get(
    "INDEX_ENDPOINT_RESOURCE_NAME",
    "projects/913936335566/locations/us-east4/indexEndpoints/5152614952967602176",
)
# Gemini model for answering from retrieved chunks.
# Run check_gemini_models.py to see available models in your project.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-001")

# ================= PRICING (public list prices, for cost display in orchestrator) =================
# Override with env if needed. See cloud.google.com/vertex-ai/pricing and cloud.google.com/bigquery/pricing.
GEMINI_INPUT_PER_1M_TOKENS = float(os.environ.get("GEMINI_INPUT_PER_1M_TOKENS", "0.075"))   # Gemini 2.0 Flash input $/1M tokens
GEMINI_OUTPUT_PER_1M_TOKENS = float(os.environ.get("GEMINI_OUTPUT_PER_1M_TOKENS", "0.30"))  # Gemini 2.0 Flash output $/1M tokens
EMBEDDING_PER_1K_CHARS = float(os.environ.get("EMBEDDING_PER_1K_CHARS", "0.000025"))        # text-embedding-004 $/1K characters
BIGQUERY_PER_TB = float(os.environ.get("BIGQUERY_PER_TB", "5.0"))                           # BigQuery on-demand $/TB processed
