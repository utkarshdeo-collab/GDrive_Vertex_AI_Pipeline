"""
Configuration for the Nexus Hybrid Engine (Salesforce + Domo).
"""
import os
from pathlib import Path

# ================= PROJECT =================
# Your Vertex AI project. Override with GCP_PROJECT_ID env if needed.
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "sym-dev-mr-agents-01")
LOCATION = os.environ.get("GCP_LOCATION", "us-east4")  # Northern Virginia (allowed by org policy)

# Fix quota/billing 403: set before any Vertex/Document AI imports.
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = PROJECT_ID
# ADK / google.genai Client: use Vertex AI with this project and location.
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION

# ================= PRICING (public list prices) =================
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

# ================= NEXUS HYBRID ENGINE (Salesforce + Domo Account Snapshots) =================
# Separate from PDF pipeline - handles CSV-based account snapshots with dual engines:
#   - Vector Engine: Semantic search via Vertex AI Vector Search
#   - Analytics Engine: In-memory Pandas for exact calculations

# Data Sources (local CSV/Excel files)
NEXUS_DATA_DIR = Path(__file__).parent / "Sf_and_Domo"
NEXUS_SALESFORCE_FILE = "Nexus_Account_Info_Exact_Columns.xlsx"
NEXUS_DOMO_FILE = "TEST_Pod Summary Insights - Monthly Pod Metrics.csv.xlsx"

# GCS Storage (dedicated bucket for Nexus data)
NEXUS_GCS_BUCKET = os.environ.get("NEXUS_GCS_BUCKET", "nexus-hybrid-engine-80k-prod") # Updated to verified bucket
NEXUS_GCS_RAW_PREFIX = "raw_inputs"
NEXUS_GCS_EMBEDDINGS_PREFIX = "embeddings"
NEXUS_GCS_ANALYTICS_PREFIX = "analytics"

# Vector Search Index (separate from PDF doc-pipeline-index)
NEXUS_VECTOR_INDEX_DISPLAY_NAME = "nexus_hybrid_index"
# Updated after Phase 3 COMPLETE Deployment (2026-02-19)
NEXUS_VECTOR_INDEX_ID = os.environ.get("NEXUS_VECTOR_INDEX_ID", "9140763529236709376")
NEXUS_INDEX_ENDPOINT_DISPLAY_NAME = "nexus-hybrid-endpoint"
NEXUS_INDEX_ENDPOINT_RESOURCE_NAME = os.environ.get("NEXUS_INDEX_ENDPOINT_RESOURCE", "projects/913936335566/locations/us-east4/indexEndpoints/346816753726128128")
NEXUS_DEPLOYED_INDEX_ID = "nexus_deployed_index_v1"

# Embedding Configuration (reuses text-embedding-004 from PDF pipeline)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-001")
NEXUS_EMBEDDING_MODEL = "text-embedding-004"
NEXUS_EMBEDDING_DIM = 768
NEXUS_DISTANCE_MEASURE = "COSINE_DISTANCE"

# Output Files
NEXUS_ANALYTICS_PARQUET_FILE = "nexus_analytics.parquet"
NEXUS_ANALYTICS_PARQUET_GCS = f"gs://{NEXUS_GCS_BUCKET}/{NEXUS_GCS_ANALYTICS_PREFIX}/{NEXUS_ANALYTICS_PARQUET_FILE}"
NEXUS_VECTORS_INPUT_JSONL = "nexus_vectors_input.jsonl"
NEXUS_VECTORS_EMBEDDED_JSONL = "nexus_vectors_embedded.jsonl"
# Updated after Phase 2 success - POINTING TO DIRECTORY
NEXUS_VECTORS_EMBEDDED_GCS_DIR = f"gs://{NEXUS_GCS_BUCKET}/{NEXUS_GCS_EMBEDDINGS_PREFIX}/batch_init/20260219_0034/final"

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
