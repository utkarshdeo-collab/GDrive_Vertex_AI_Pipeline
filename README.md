# Nexus Hybrid Engine — Pipeline Documentation

## Overview

The **Nexus Hybrid Engine** is a 4-phase data pipeline that:
1. Fetches account data from **Salesforce (BigQuery)** and product usage metrics from **Domo (API)**
2. Builds structured "Golden Records" combining both sources
3. Generates vector embeddings and deploys them to **Vertex AI Vector Search**
4. Splits records into individual GCS files for fast O(1) agent lookups

The end result powers an AI orchestration layer where agents can answer questions like *"Give me a snapshot for Antino Bank"* by combining Salesforce commercial data with Domo usage metrics in real time.

---

## Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Service Account
Place your GCP service account key at:
```
credentials/service_account.json
```
The pipeline auto-detects and uses it. No manual `gcloud auth` needed.

### 3. GCP Roles Required on the Service Account
| Role | Needed For |
|---|---|
| `BigQuery Data Viewer` | Phase 1 — Salesforce fetch |
| `BigQuery Job User` | Phase 1 — Salesforce fetch |
| `Storage Object Admin` | All phases — GCS read/write |
| `Vertex AI User` | Phase 2 — Embeddings, Phase 3 — Index deployment |

---

## Quick Reference — All Client Config Changes

Before running anything, fill in these values across two files:

### `Nexus_Hybrid_Engine/phase1_build_golden_records.py`

**Salesforce — BigQuery (`_BQ_CONFIG`):**
```python
_BQ_CONFIG = {
    "projectId": "YOUR_GCP_PROJECT_ID",   # ← your GCP project
    "dataset":   "YOUR_BQ_DATASET",       # ← BigQuery dataset name
    "table":     "YOUR_BQ_TABLE_NAME",    # ← BigQuery table name
}
```

**Domo — API (`_DOMO_CONFIG`):**
```python
_DOMO_CONFIG = {
    "testMode":       True,                    # ← flip to False for live API
    "domoClientId":   "YOUR_DOMO_CLIENT_ID",   # ← Domo OAuth client ID
    "domoClientSecret": "YOUR_DOMO_CLIENT_SECRET", # ← Domo OAuth secret
    "domoDatasetId":  "YOUR_DOMO_DATASET_ID",  # ← Domo dataset ID
    ...
}
```
> **Note:** While `testMode = True`, the pipeline reads the local CSV at `SF_and_DOMO/TEST_Pod_Appended_40000_Rows_FIXED.csv`. Flip to `False` and fill in credentials to use the live Domo API.

### `Nexus_Hybrid_Engine/nexus_config.py`

**Vector Search Index & Endpoint** — update these after first deployment (Phase 3 generates them):
```python
NEXUS_VECTOR_INDEX_ID                = "your-index-id"
NEXUS_INDEX_ENDPOINT_RESOURCE_NAME   = "projects/.../indexEndpoints/..."
NEXUS_DEPLOYED_INDEX_ID              = "nexus_deployed_index_v1"  # can rename if needed
```

---

## Phase 1 — Build Golden Records

**Script:** `Nexus_Hybrid_Engine/phase1_build_golden_records.py`

**What it does:**
- Fetches Salesforce data from BigQuery (10 columns: account info, ARR, renewal date, task count, pod ID, etc.)
- Fetches Domo data via API or local CSV in testMode (14 columns: pod metrics, health score, MEAU, provisioned users, etc.)
- Calculates derived fields:
  - `Calculated_Engagement` from `Task_Count` (≥4 = Positive, 1-3 = Neutral, 0 = Negative)
  - `Calculated_Expansion` from `provisioned_users / contracted_licenses` (>90% = Positive)
- Deduplicates Domo records by `pod_id` — keeps the most recent month per pod
- Builds a text embedding document per record combining core fields + all additional fields
- Outputs `nexus_vectors_input.jsonl` — one JSON record per line

**Client config to fill in:** `_BQ_CONFIG` and `_DOMO_CONFIG` (see Quick Reference above)

**Run:**
```bash
python Nexus_Hybrid_Engine/phase1_build_golden_records.py
```

**Output:** `nexus_vectors_input.jsonl` in the current working directory

---

## Phase 2 — Vectorize (Generate Embeddings)

**Script:** `Nexus_Hybrid_Engine/phase2_vectorize.py`

**What it does:**
- Reads `nexus_vectors_input.jsonl` (output of Phase 1)
- Generates 768-dimensional embeddings using `text-embedding-004` via Vertex AI
- Processes in batches of 10 (respects API token limits)
- Has **resume capability** — if interrupted, re-running skips already-processed records
- Uploads the final embedded JSONL to GCS at:
  `gs://nexus-hybrid-engine-80k-prod/embeddings/{timestamp}/final/nexus_vectors_output.jsonl`

**Client config to fill in:** None — uses `nexus_config.py` automatically

**Run:**
```bash
python Nexus_Hybrid_Engine/phase2_vectorize.py
```

**Output:** Embedded JSONL uploaded to GCS

---

## Phase 3 — Deploy Vector Search Index

**Script:** `Nexus_Hybrid_Engine/phase3_deploy_index.py`

**What it does:**
- Automatically finds the latest Phase 2 output in GCS (no manual path update needed)
- Creates a new Vertex AI Vector Search Index from the embeddings (or reuses existing)
- Creates (or retrieves) an Index Endpoint
- Deploys the index to the endpoint
- First-time creation takes ~30 minutes; re-deployment takes ~15-20 minutes

**⚠️ Important — Index & Endpoint names:**
If you are deploying for the **first time** or want to use different names, update these in `nexus_config.py` before running:
```python
NEXUS_VECTOR_INDEX_DISPLAY_NAME      = "nexus_hybrid_index"       # index display name
NEXUS_INDEX_ENDPOINT_DISPLAY_NAME    = "nexus-hybrid-endpoint"    # endpoint display name
NEXUS_DEPLOYED_INDEX_ID              = "nexus_deployed_index_v1"  # deployed index ID
```
After Phase 3 completes, copy the generated resource names back into `nexus_config.py`:
```python
NEXUS_VECTOR_INDEX_ID                = "..."   # from output
NEXUS_INDEX_ENDPOINT_RESOURCE_NAME   = "..."   # from output
```

**Run:**
```bash
python Nexus_Hybrid_Engine/phase3_deploy_index.py
```

**Output:** Live Vector Search Index deployed and ready for queries

---

## Phase 4 — Split Records to GCS (Serverless Lookup)

**Script:** `Nexus_Hybrid_Engine/phase4_split_records_to_gcs.py`

**What it does:**
- Reads `nexus_vectors_input.jsonl`
- Uploads each record as an individual JSON file to GCS:
  `gs://nexus-hybrid-engine-80k-prod/lookup_records/{record_id}.json`
- Uses 50 parallel upload threads for speed
- Enables O(1) agent lookups — agents can retrieve any record instantly by ID without loading the full dataset

**Client config to fill in:** None — uses `nexus_config.py` automatically

**Run:**
```bash
python Nexus_Hybrid_Engine/phase4_split_records_to_gcs.py
```

**Output:** Individual record JSON files in GCS `lookup_records/` folder

---

## Running the Full Pipeline (In Order)

```bash
# Step 1 — Fill in _BQ_CONFIG and _DOMO_CONFIG in phase1_build_golden_records.py first

# Step 2 — Build golden records
python Nexus_Hybrid_Engine/phase1_build_golden_records.py

# Step 3 — Generate and upload embeddings
python Nexus_Hybrid_Engine/phase2_vectorize.py

# Step 4 — Deploy vector search index
python Nexus_Hybrid_Engine/phase3_deploy_index.py

# Step 5 — Split records for agent lookups
python Nexus_Hybrid_Engine/phase4_split_records_to_gcs.py

# Step 6 — Start the orchestrator agent
python agents/orchestrator/run_orchestrator.py
# OR for ADK Web UI (from project root):
# adk web .
```

---

## Running the Orchestrator Agent

**CLI:**
```bash
python agents/orchestrator/run_orchestrator.py
```

**ADK Web UI:**
```bash
adk web .
```
*(Run from the project root — `nexus_orchestrator/agent.py` is the entry point)*

**Example questions:**
- `"Give me a snapshot for Antino Bank"`
- `"What is the ARR for ABC Capital?"`
- `"Show me health score for pod 888"`

---

## GCS Bucket Structure

```
nexus-hybrid-engine-80k-prod/
├── embeddings/
│   └── {timestamp}/
│       └── final/
│           └── nexus_vectors_output.jsonl   ← Phase 2 output
├── lookup_records/
│   ├── sf_{Account_Name}.json               ← Salesforce records (Phase 4)
│   └── domo_pod_{pod_id}.json               ← Domo records (Phase 4)
└── analytics/
    └── nexus_analytics.parquet
```

---

---

## Function Reference — Every Script

---

### `Nexus_Hybrid_Engine/nexus_config.py`

| Function | What it does |
|---|---|
| `_resolve_latest_embeddings_dir(bucket, prefix)` | Scans the GCS `embeddings/` prefix, sorts all timestamp folders, returns the most recent one as the GCS path for Phase 3. Falls back to the last known good path if GCS is unreachable. |

---

### `Nexus_Hybrid_Engine/phase1_build_golden_records.py`

**Salesforce (BigQuery) functions:**

| Function | What it does |
|---|---|
| `query_salesforce_bq()` | Connects to BigQuery using the service account, runs `_SF_SQL` against `_BQ_CONFIG` table, returns a 10-column pandas DataFrame. |

**Domo (API / testMode) functions:**

| Function | What it does |
|---|---|
| `domoAuthenticate()` | In `testMode=True`: sets a dummy token and skips auth. In production: calls the Domo OAuth endpoint with client credentials and stores the real access token. |
| `queryDataSet(data_id, sql)` | In `testMode=True`: reads local CSV, runs the SQL via `pandasql`, returns a DataFrame. In production: POSTs SQL to Domo API, converts `{columns, rows}` response into a DataFrame. |

**Data transformation functions:**

| Function | What it does |
|---|---|
| `clean_currency_column(series)` | Converts a mixed-type currency column (strings like `"$1,000"`, floats, NaN) into a uniform `"$1,000"` string format. Returns `"N/A"` for nulls. |
| `calculate_engagement_signal(task_count)` | Converts raw `Task_Count` into a signal: `≥4 → "Positive"`, `1-3 → "Neutral"`, `0/null → "Negative"`. |
| `calculate_expansion_signal(provisioned_users, contracted_licenses)` | Checks if provisioned users exceed 90% of contracted licenses. Returns `"Positive"`, `"Negative"`, or `"N/A"` if data is missing. |
| `safe_str(val)` | Safely converts any value to a stripped string. Returns `"N/A"` for NaN/None values. Used throughout to prevent crashes on missing data. |
| `build_dynamic_embedding_text(row, core_template, core_fields)` | Builds the full text document for embedding. Section 1: formats the core priority fields using the template. Section 2: appends all remaining non-null fields as `[FIELD_NAME] value` pairs. Skips empty values to save token space. |

**Record building functions:**

| Function | What it does |
|---|---|
| `process_salesforce_data()` | Calls `query_salesforce_bq()`, calculates engagement + ARR formatting, loops through all rows to build vector records `{id, content, metadata}`. Record ID format: `sf_{Customer_Name}`. |
| `process_domo_data()` | Calls `domoAuthenticate()` + `queryDataSet()`, deduplicates by `pod_id` keeping most recent `month`, calculates expansion signal, builds vector records. Record ID format: `domo_pod_{pod_id}`. |
| `save_to_jsonl(records, output_path)` | Writes a list of record dicts to a `.jsonl` file, one JSON object per line. |
| `main()` | Entry point. Calls `process_salesforce_data()` + `process_domo_data()`, combines both lists, saves to `nexus_vectors_input.jsonl`. |

---

### `Nexus_Hybrid_Engine/phase2_vectorize.py`

| Function | What it does |
|---|---|
| `ensure_bucket_exists(bucket_name, location)` | Checks if the target GCS bucket exists. Creates it in the specified region if not found. |
| `truncate_text(text, limit)` | Truncates text to a character limit (default 8,000 chars ≈ 2,048 tokens). Prevents exceeding the `text-embedding-004` model input limit. |
| `process_and_embed(input_file, output_file, model_name)` | Core embedding loop. Reads input JSONL, skips already-processed IDs (resume support), batches records in groups of 10, calls Vertex AI `get_embeddings()`, writes `{id, embedding, restricts}` to output JSONL. |
| `upload_to_gcs(local_path, bucket_name, blob_name)` | Calls `ensure_bucket_exists()` then uploads a local file to the specified GCS path. |
| `run_phase_2()` | Entry point. Sets up input/output paths, calls `process_and_embed()`, then uploads the result to GCS with a timestamped folder name. |

---

### `Nexus_Hybrid_Engine/phase3_deploy_index.py`

All logic lives inside the `VectorDeployer` class:

| Method | What it does |
|---|---|
| `__init__()` | Loads config values: project ID, location, bucket name, GCS URI of embeddings, index display name, dimensions (768), endpoint name, deployed index ID. |
| `verify_gcs_data()` | Connects to GCS and checks that at least one file exists at the embeddings URI. Exits with error if nothing found. |
| `create_or_get_index()` | Searches for an existing Vertex AI index by display name. Returns it if found. Otherwise creates a new `tree_ah` index from the GCS embeddings data (~30 min). |
| `get_or_create_endpoint()` | Searches for an existing index endpoint by display name. Returns it if found. Otherwise creates a new public endpoint. |
| `deploy_index(index, endpoint)` | If the deployed index ID already exists on the endpoint, undeploys it first. Then deploys the (new) index to the endpoint (~15-20 min). |
| `run()` | Entry point. Calls `verify_gcs_data()` → `create_or_get_index()` → `get_or_create_endpoint()` → `deploy_index()` in sequence. |

---

### `Nexus_Hybrid_Engine/phase4_split_records_to_gcs.py`

All logic lives inside the `GCSRecordSplitter` class:

| Method | What it does |
|---|---|
| `__init__(project_id, bucket_name, source_file, destination_prefix, max_workers)` | Initialises GCS client, connects to the target bucket, stores config. Default: 50 parallel upload threads. |
| `_upload_single_record(record)` | Takes one record dict, serialises it to JSON, uploads to `{destination_prefix}/{record_id}.json` in GCS. Returns `True` on success, `False` on failure. |
| `process()` | Reads all records from the source JSONL file, submits all upload tasks to a `ThreadPoolExecutor` (50 threads), tracks progress with `tqdm`, prints a final summary (total uploaded, failed, time taken, rate). |

---

### `agents/orchestrator/salesforce_agent.py`

| Function | What it does |
|---|---|
| `_init_resources()` | Lazy initialisation — only runs once. Sets up the `TextEmbeddingModel`, `MatchingEngineIndexEndpoint`, and GCS `Client`/`Bucket`. Avoids startup overhead if the module is imported but not used. |
| `_generate_query_embedding(query)` | Takes a search query string, generates a 768-dimensional embedding using `text-embedding-004` with task type `RETRIEVAL_QUERY`. |
| `_fetch_record_from_gcs(record_id)` | Looks up `lookup_records/{record_id}.json` in GCS. Returns the parsed JSON dict, or `None` if not found. |
| `_safe_str(s)` | Normalises a string to alphanumeric + underscores only. Used to build and compare record IDs (e.g. `"Antino Bank"` → `"Antino_Bank"`). |
| `get_salesforce_account_data(account_name)` | **Main tool function.** 3-step hybrid search: (1) Vector search — gets top 50 semantic matches. (2) Re-ranking — checks if any match ID corresponds exactly to the account name. (3) GCS fetch — retrieves the full record content. Returns `{status, data, match_score}`. |
| `create_salesforce_agent(credentials)` | Factory function. Builds and returns an `LlmAgent` wired with `get_salesforce_account_data` as its tool. |

---

### `agents/orchestrator/domo_agent.py`

| Function | What it does |
|---|---|
| `_init_resources()` | Lazy initialisation — sets up GCS `Client` and `Bucket` on first call only. |
| `_fetch_record_from_gcs(record_id)` | Looks up `lookup_records/{record_id}.json` in GCS. Returns parsed JSON or `None`. |
| `get_pod_data_by_id(pod_id)` | **Main tool function.** Normalises the pod ID (handles float strings like `"888.0"` → `"888"`), constructs `domo_pod_{pod_id}` as the record ID, does a direct O(1) GCS lookup. Returns `{status, data}`. |
| `create_domo_agent(credentials)` | Factory function. Builds and returns an `LlmAgent` wired with `get_pod_data_by_id` as its tool. |

---

### `agents/orchestrator/run_orchestrator.py`

**Routing / classification functions:**

| Function | What it does |
|---|---|
| `_is_likely_salesforce_question(text)` | Checks if the user message contains any Salesforce keywords (ARR, pipeline, account, renewal, etc.). Returns `True/False`. |
| `_is_likely_domo_question(text)` | Checks if the user message contains any Domo keywords (health score, MEAU, provisioned users, active users, etc.). Returns `True/False`. |
| `_is_likely_document_question(text)` | Checks if the user message contains PDF/document keywords. Returns `True/False`. |
| `_maybe_add_routing_hint(user_message)` | Prepends a `[ROUTING: ...]` instruction to the user message before it reaches the LLM. Acts as a fast pre-classifier to strongly guide the model to the correct sub-agent. |
| `get_routing_data_dictionary(salesforce_path, domo_path)` | Reads `nexus_schema.json` and `domo_schema.json`, extracts column names from each table, returns a formatted string that is injected into the master agent's system prompt. |

**Orchestration functions:**

| Function | What it does |
|---|---|
| `get_nexus_account_snapshot_orchestrated(account_name)` | **Primary tool exposed to the master agent.** Step 1: calls `get_salesforce_account_data()` to get SF metadata + pod_id. Step 2: calls `get_pod_data_by_id()` with the pod_id to get Domo metrics. Step 3: merges both into a formatted snapshot (ARR, renewal, engagement, MEAU, ORBIT score, churn risk, expansion signal). |
| `build_agents(credentials, routing_data_dict)` | Constructs the master `LlmAgent` (orchestrator) with `get_nexus_account_snapshot_orchestrated` as a direct tool, and the Salesforce + Domo agents as sub-agents. |
| `main()` | Async CLI entry point. Initialises Vertex AI, authenticates, creates session, runs interactive input loop. Applies routing hints to each user message before sending to the agent. |

---

### `nexus_orchestrator/agent.py`

No functions — this is the **ADK Web entry point**. At module load time it:
1. Imports `nexus_config` to auto-set service account credentials
2. Authenticates using `google.auth.default()`
3. Calls `build_agents()` from `run_orchestrator.py`
4. Exposes `root_agent` at module level — the variable ADK Web looks for when you run `adk web .`

---

## Project Structure

```
Symphony_Final_Pipeline/
├── Nexus_Hybrid_Engine/
│   ├── nexus_config.py                  ← Central config (GCP, buckets, index IDs)
│   ├── phase1_build_golden_records.py   ← ETL: SF + Domo → JSONL
│   ├── phase2_vectorize.py              ← Embed JSONL → GCS
│   ├── phase3_deploy_index.py           ← Deploy Vertex AI Vector Search
│   └── phase4_split_records_to_gcs.py  ← Split records for O(1) lookup
├── agents/orchestrator/
│   ├── run_orchestrator.py              ← CLI entry point (Master Agent)
│   ├── salesforce_agent.py              ← Salesforce sub-agent
│   ├── domo_agent.py                    ← Domo sub-agent
│   ├── nexus_schema.json                ← Salesforce data dictionary
│   └── domo_schema.json                 ← Domo data dictionary
├── nexus_orchestrator/
│   └── agent.py                         ← ADK Web entry point
├── SF_and_DOMO/
│   ├── salesforce_40000_rows.xlsx       ← Local SF reference data
│   ├── TEST_Pod_Appended_40000_Rows_FIXED.csv  ← Local Domo CSV (testMode)
│   └── domo_test_fetch/
│       └── test_domo_fetch.py           ← Standalone Domo API test utility
├── credentials/
│   └── service_account.json             ← GCP service account key (do not commit)
└── requirements.txt
```
