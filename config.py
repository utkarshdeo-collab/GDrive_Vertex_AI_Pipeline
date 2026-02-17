"""
Shared configuration for the Document Pipeline.
Aligns with: Ingestion → Processing → Indexing → Interface (ADK).
"""
import os

# ================= PROJECT =================
# Your Vertex AI project. Override with GCP_PROJECT_ID env if needed.
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "sym-dev-mr-agents-01")
LOCATION = os.environ.get("GCP_LOCATION", "us-east4")  # Northern Virginia (allowed by org policy)

# Document AI region: must match where your processor was created (e.g. us, eu, asia-southeast1).
# Override with DOCAI_LOCATION env. Default matches processor in asia-southeast1.
DOCAI_LOCATION = os.environ.get("DOCAI_LOCATION", "asia-southeast1")

# Fix quota/billing 403: set before any Vertex/Document AI imports.
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = PROJECT_ID
# ADK / google.genai Client: use Vertex AI with this project and location.
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION

# ================= PHASE 1: INGESTION =================
# GCS bucket for staging PDFs (Document AI requires files in GCS for batch).
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", f"{PROJECT_ID}-docai-staging")
# Dedicated bucket for PDF URL streaming (no laptop download). Created automatically if missing.
GCS_PDF_INPUT_BUCKET = os.environ.get("GCS_PDF_INPUT_BUCKET", f"{PROJECT_ID}-pdf-input")
# Prefix inside bucket for input PDFs and Document AI output.
GCS_INPUT_PREFIX = "input"
GCS_OUTPUT_PREFIX = "document-ai-output"

# Document AI: use Layout Parser / Document OCR for tables, headers, paragraphs.
# Processor created in asia-southeast1. Override with DOCAI_PROCESSOR_ID env if needed.
DOCAI_PROCESSOR_ID = os.environ.get("DOCAI_PROCESSOR_ID", "bd0e4f1d48aba276")

# Parsing phase (phase1b): Layout Parser output under the PDF bucket. One subfolder per document.
DOCAI_OUTPUT_PREFIX = os.environ.get("DOCAI_OUTPUT_PREFIX", "document-ai-output")

# Google Drive: folder ID (list PDFs and use first) or file ID (single PDF).
# Leave both empty to use local file (LOCAL_PDF_PATH)
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "")
DRIVE_FILE_ID = os.environ.get("DRIVE_FILE_ID", "")  # Empty = use local file

# Local path where PDF is downloaded (or used if not from Drive).
LOCAL_PDF_PATH = os.environ.get("LOCAL_PDF_PATH", "input.pdf")

# PDF URLs: stream from HTTP(S) links into GCS (e.g. goto.symphony.com, shared links).
# Set env PDF_URLS (comma-separated) to override. When unset, uses the default list below.
_PDF_URLS_RAW = os.environ.get("PDF_URLS", "").strip()
PDF_URL_LIST = (
    [u.strip() for u in _PDF_URLS_RAW.split(",") if u.strip()]
    if _PDF_URLS_RAW
    else [
    "https://goto.symphony.com/rs/945-HBF-959/images/SYM_1PGR_Federation.pdf",
    "https://goto.symphony.com/rs/945-HBF-959/images/SYM_2025_1PGR_Federation_via_Microsoft_Teams.pdf?version=0",
    "https://goto.symphony.com/rs/945-HBF-959/images/ARK%20Capital%20Case%20Study.pdf?version=0",
    "https://goto.symphony.com/rs/945-HBF-959/images/SYM_1PGR_Insurance.pdf?version=5",
    "https://goto.symphony.com/rs/945-HBF-959/images/InsurTech100-Report-2025-Mike.pdf?version=0",
    "https://goto.symphony.com/rs/945-HBF-959/images/InsurTech100_2025%20Case%20Study.pdf?version=0",
    "https://goto.symphony.com/rs/945-HBF-959/images/Why_insurance_giants_are_racing_to_harness_AI_to_reinvent_communication.pdf?version=1",
    "https://goto.symphony.com/rs/945-HBF-959/images/SYM_2025_1PGR_Symphony_for_Wealth%20Management.pdf?version=0",
    ]
)

# ================= PHASE 2: CHUNKING =================
# Bucket/prefix for chunks (Phase 2 writes here; Phase 3 reads from here).
CHUNKS_BUCKET = os.environ.get("CHUNKS_BUCKET", GCS_PDF_INPUT_BUCKET)
CHUNK_OUTPUT_PREFIX = "chunks"
# Overlap between consecutive chunks (0.10 = 10%, 0.15 = 15%).
CHUNK_OVERLAP_RATIO = float(os.environ.get("CHUNK_OVERLAP_RATIO", "0.12"))

# ================= PHASE 3: INDEXING =================
VECTOR_INDEX_DISPLAY_NAME = "doc-pipeline-index"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDINGS_GCS_PREFIX = "embeddings"

# ================= PHASE 4: DEPLOY & ADK =================
# Vector Search index (from Phase 3 output). Latest: 1,744 chunks → index 8645508307714310144.
VECTOR_INDEX_ID = os.environ.get("VECTOR_INDEX_ID", "8645508307714310144")
# Endpoint where the index is deployed (must match display name in Vector Search → Index endpoints).
INDEX_ENDPOINT_DISPLAY_NAME = os.environ.get("INDEX_ENDPOINT_DISPLAY_NAME", "doc-pipeline-endpoint")
# Deployed index ID on that endpoint (must match the deployment that has our 1,744-chunk index).
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID", "test23_1770204137582")
# Endpoint resource (optional). Leave empty to lookup by display name above.
INDEX_ENDPOINT_RESOURCE_NAME = os.environ.get("INDEX_ENDPOINT_RESOURCE_NAME", "")
# Gemini model for answering from retrieved chunks.
# Run check_gemini_models.py to see available models in your project.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-001")

# ================= PRICING (public list prices, for cost display in orchestrator) =================
# Override with env if needed. See cloud.google.com/vertex-ai/pricing and cloud.google.com/bigquery/pricing.
GEMINI_INPUT_PER_1M_TOKENS = float(os.environ.get("GEMINI_INPUT_PER_1M_TOKENS", "0.075"))   # Gemini 2.0 Flash input $/1M tokens
GEMINI_OUTPUT_PER_1M_TOKENS = float(os.environ.get("GEMINI_OUTPUT_PER_1M_TOKENS", "0.30"))  # Gemini 2.0 Flash output $/1M tokens
EMBEDDING_PER_1K_CHARS = float(os.environ.get("EMBEDDING_PER_1K_CHARS", "0.000025"))        # text-embedding-004 $/1K characters
BIGQUERY_PER_TB = float(os.environ.get("BIGQUERY_PER_TB", "5.0"))                           # BigQuery on-demand $/TB processed

# ================= ORCHESTRATOR AUDIT (BigQuery log table) =================
# Audit log: one row per tool invocation (question, response, tool_call, sql_generated, etc.).
AUDIT_ENABLED = os.environ.get("AUDIT_ENABLED", "true").lower() in ("true", "1", "yes")
AUDIT_DATASET = os.environ.get("AUDIT_DATASET", "orchestrator_logs")
AUDIT_TABLE = os.environ.get("AUDIT_TABLE", "orchestrator_audit")
# Region for audit dataset (create if not exists). Use same as LOCATION.
AUDIT_REGION = os.environ.get("AUDIT_REGION", LOCATION)

# ================= BIGQUERY VECTOR SEARCH (embedding creation only) =================
# New dataset for embedded tables (same rows as source + embedding column). Used by bq_vector_setup scripts only.
EMBEDDING_DATASET = os.environ.get("EMBEDDING_DATASET", "embedding_tables")

# ================= NEXUS HYBRID ENGINE (Salesforce + Domo Account Snapshots) =================
# Separate from PDF pipeline - handles CSV-based account snapshots with dual engines:
#   - Vector Engine: Semantic search via Vertex AI Vector Search
#   - Analytics Engine: In-memory Pandas for exact calculations

# Data Sources (local CSV/Excel files)
from pathlib import Path
NEXUS_DATA_DIR = Path(__file__).parent / "Sf_and_Domo"
NEXUS_SALESFORCE_FILE = "Nexus_Account_Info_Exact_Columns.xlsx"
NEXUS_DOMO_FILE = "TEST_Pod Summary Insights - Monthly Pod Metrics.csv.xlsx"

# GCS Storage (dedicated bucket for Nexus data)
NEXUS_GCS_BUCKET = os.environ.get("NEXUS_GCS_BUCKET", "symphony-nexus-data-core")
NEXUS_GCS_RAW_PREFIX = "raw_inputs"
NEXUS_GCS_EMBEDDINGS_PREFIX = "embeddings"
NEXUS_GCS_ANALYTICS_PREFIX = "analytics"

# Vector Search Index (separate from PDF doc-pipeline-index)
NEXUS_VECTOR_INDEX_DISPLAY_NAME = "nexus_hybrid_index"
NEXUS_VECTOR_INDEX_ID = os.environ.get("NEXUS_VECTOR_INDEX_ID", "1379935471370502144")
NEXUS_INDEX_ENDPOINT_DISPLAY_NAME = "nexus-hybrid-endpoint"
NEXUS_INDEX_ENDPOINT_RESOURCE_NAME = os.environ.get("NEXUS_INDEX_ENDPOINT_RESOURCE", "projects/913936335566/locations/us-east4/indexEndpoints/3048976530148425728")
NEXUS_DEPLOYED_INDEX_ID = "nexus_deployed_v1"

# Embedding Configuration (reuses text-embedding-004 from PDF pipeline)
NEXUS_EMBEDDING_MODEL = EMBEDDING_MODEL  # "text-embedding-004"
NEXUS_EMBEDDING_DIM = 768
NEXUS_DISTANCE_MEASURE = "COSINE_DISTANCE"

# Output Files
NEXUS_ANALYTICS_PARQUET_FILE = "nexus_analytics.parquet"
NEXUS_ANALYTICS_PARQUET_GCS = f"gs://{NEXUS_GCS_BUCKET}/{NEXUS_GCS_ANALYTICS_PREFIX}/{NEXUS_ANALYTICS_PARQUET_FILE}"
NEXUS_VECTORS_INPUT_JSONL = "nexus_vectors_input.jsonl"
NEXUS_VECTORS_EMBEDDED_JSONL = "nexus_vectors_embedded.jsonl"
NEXUS_VECTORS_INPUT_GCS = f"gs://{NEXUS_GCS_BUCKET}/{NEXUS_GCS_EMBEDDINGS_PREFIX}/{NEXUS_VECTORS_INPUT_JSONL}"
NEXUS_VECTORS_EMBEDDED_GCS = f"gs://{NEXUS_GCS_BUCKET}/{NEXUS_GCS_EMBEDDINGS_PREFIX}/{NEXUS_VECTORS_EMBEDDED_JSONL}"

# Business Logic Rules
NEXUS_ENGAGEMENT_POSITIVE_THRESHOLD = 4  # Task_Count >= 4 → Positive
NEXUS_ENGAGEMENT_NEUTRAL_MIN = 1         # Task_Count 1-3 → Neutral
NEXUS_ENGAGEMENT_NEUTRAL_MAX = 3
NEXUS_EXPANSION_THRESHOLD = 0.9          # provisioned_users > 90% of contracted → Positive

# Deduplication Strategy
NEXUS_DOMO_DEDUP_STRATEGY = "most_recent"  # Keep row with latest 'month' per pod_id

# Metadata Fields for Vector Search Filtering
NEXUS_VECTOR_METADATA_FIELDS = ["account_name", "owner", "calculated_engagement", "pod_id"]

# Golden Record Template
NEXUS_GOLDEN_RECORD_TEMPLATE = """[ACCOUNT] {Customer_Name}
[OWNER] {Account_Owner}
[TOTAL_ARR] {Total_ARR}
[RENEWAL_DATE] {Renewal_Date}
[ENGAGEMENT_STATUS] {Calculated_Engagement}
[EXPANSION_SIGNAL] {Calculated_Expansion}
[ORBIT_SCORE] {Mapped_ORBIT}
[CHURN_RISK] {Mapped_Churn}
[MEAU] {meau}
[POD_ID] {POD_Internal_Id__c}
[PROVISIONED_USERS] {provisioned_users}
[CONTRACTED_LICENSES] {contracted_licenses}
[HEALTH_SCORE] {health_score}
[ACTIVE_USERS] {active_users}"""

